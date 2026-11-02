"""CHR-P live scanner and $1,000 shadow book (research/prereg/CHR.md, "Live book"). Ledger only: no orders.

Every tick (5 minutes) on a regular day, between 9:50 ET and 30 minutes before the close:
1. Penny universe of the day (built once from Alpaca daily bars: previous raw close $1 to $5, 20-day median raw
   dollar volume >= $1M), with each stock's prior-day levels exactly as `ch_data.flag_days` makes them.
2. 5-minute SIP raw bars up to now - 16 minutes for the universe, SPY and the sector ETFs.
3. For every stock without a decision today: `ch_build.first_trigger`. A new trigger gets the 68 inputs
   (`ch_build.features`, `ch_data.market`), the frozen models' six predictions, the largest above the cut picks
   side and stop/target pair, size from the frozen scale. Entry = the latest IEX trade when the decision is made.
   Shorts only where Alpaca marks the stock shortable and easy to borrow. Open exposure (positions not yet settled,
   counted for the full 5 sessions) stays <= 100% of the book.
4. After the close: the same-day random control for each pick (as `ch_study.with_control`), the day record to the
   ledger (kind chrp_day), and every position whose 5 sessions are over is settled with the training code
   (`ch_build.labels`: first barrier touched on the bars after the decision, else the 5th close; costs as
   history's $250 position) as kind chrp_settle.

Usage: python -m tracks.main.chr_live --state state [--minutes 20]   (exit 3 = nothing more today)
       python -m tracks.main.chr_live --check                       (read every source once, no state)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.ledger import Ledger
from tracks.main import ch_build as B
from tracks.main import ch_data as C
from tracks.main import ch_study as S
from tracks.news import w1_data as D

ET = C.ET
MODEL_DIR = Path("models/chr_p")
CACHE = Path("cache/chrp")
BOOK = 1000.0
CAP = 1.0                                            # open exposure, share of the book
TICK_S = 300
LAST_DECISION_GAP = pd.Timedelta(minutes=45)        # a trigger seen later than this after its bar is logged "late"
SEED = S.SEED
HIST_DAYS = 420


# ---------- pure helpers (tested) ----------

def load_frozen(dirpath=MODEL_DIR):
    """The frozen version 1: manifest, six boosters (sha256 checked), cut, scale and the volume share curve."""
    import lightgbm as lgb
    man = json.loads((Path(dirpath) / "manifest.json").read_text())
    models = []
    for lab in man["labels"]:
        f = Path(dirpath) / f"{lab}.txt"
        if hashlib.sha256(f.read_bytes()).hexdigest() != man["sha256"][f.name]:
            raise SystemExit(f"[chr] {f} does not match the manifest")
        models.append(lgb.Booster(model_file=str(f)))
    man["models"] = models
    man["scale"] = np.array(man["scale"])
    return man


def predict(frozen, feats):
    """(n, 6) predictions for a list of feature dicts, in the manifest's input order."""
    X = np.array([[f.get(k, np.nan) for k in frozen["features"]] for f in feats], np.float32)
    return np.column_stack([m.predict(X) for m in frozen["models"]])


def choose(pred, frozen):
    """For one prediction row: (label, trades, size fraction)."""
    pick, trade, size = S.decide(np.atleast_2d(pred), frozen["cut"], frozen["scale"])
    return B.LABELS[int(pick[0])], bool(trade[0]), float(size[0])


def sector_of(sym, cik_of, sic_of):
    """SPDR sector ETF of a stock from its SEC SIC code (as history), SPY when the code is unknown."""
    return C.sector_etf(sic_of.get(cik_of.get(sym)))


def exposure_ok(open_sizes, size):
    return sum(open_sizes) + size <= CAP + 1e-9


def controls(day, pool, picks):
    """{pick symbol: control symbol}: another stock from the day's triggered `pool`, default_rng(SEED + date
    ordinal), picks in symbol order, as ch_study.with_control."""
    rng = np.random.default_rng(SEED + pd.Timestamp(day).toordinal())
    pool = sorted(pool)
    out = {}
    for s in sorted(picks):
        cand = [x for x in pool if x != s]
        if cand:
            out[s] = cand[int(rng.integers(len(cand)))]
    return out


def universe_ctx(raw, adj, day):
    """{symbol: ctx} for the penny universe on `day`, and {symbol: prior daily bars on today's raw basis}. A
    row for `day` itself (price unchanged, nothing known) lets `flag_days` shift yesterday's levels into it."""
    ctx, prior = {}, {}
    for sym in raw:
        if sym not in adj:
            continue
        d = D.daily_rows(sym, raw[sym], adj[sym])
        d = d[d["date"] < day]
        if len(d) < 30:
            continue
        last = d.iloc[-1]
        today = {c: last[c] for c in d.columns}
        today.update(date=day, o=last["c"], h=last["c"], l=last["c"], v=0.0, ao=last["ac"],
                     seg=last["seg"] if (day - last["date"]).days <= D.L1.GAP_DAYS else last["seg"] + 1)
        t = C.flag_days(pd.concat([d, pd.DataFrame([today])], ignore_index=True)).iloc[-1]
        if t["track"] != "penny":
            continue
        ctx[sym] = {k: float(t[k]) for k in ("prev_c", "prev_dv20", "prev_ac", "vol20", "hi20", "lo20", "hi252", "lo252")}
        g = d[d["seg"] == t["seg"]]
        k = g["ac"] / g["c"]
        prior[sym] = g.assign(o=g["o"] * k, h=g["h"] * k, l=g["l"] * k, c=g["c"] * k, v=g["v"] / k).tail(300)
    return ctx, prior


def sic_maps(ctx_syms):
    """({ticker: cik}, {cik: SIC}) from SEC's ticker file and EDGAR submissions (cached in CACHE/sic.json); empty
    maps when SEC can't be reached, which leaves every sector input at SPY's."""
    try:
        from core import edgar_live, industry
        by_cik, _ = edgar_live.ticker_map()
        cik_of = {t: c for c, ts in by_cik.items() for t in ts}
        have = {t: cik_of[t] for t in ctx_syms if t in cik_of}
        return have, industry.sic_codes(set(have.values()), CACHE)
    except Exception as e:
        print(f"[chr] SIC lookup failed ({e!r}); sector inputs use SPY")
        return {}, {}


# ---------- Alpaca ----------

class Api:
    def __init__(self):
        self.h = alpaca_headers()

    def assets(self):
        a = alpaca.active_assets()
        a = a[a["excl_reason"].isna() & a["tradable"].astype(bool)]
        ok = a["shortable"].fillna(False).astype(bool) & a["easy_to_borrow"].fillna(False).astype(bool)
        return sorted(set(a["base_symbol"])), set(a.loc[ok, "base_symbol"])

    def daily(self, syms, start, end, adjustment):
        return to_frames(fetch_daily_bars(syms, start, end, adjustment=adjustment, headers=self.h))

    def bars5(self, syms, day, end_utc=None):
        parts = []
        for k in range(0, len(syms), C.PER_REQ):
            b = C.day_bars(syms[k:k + C.PER_REQ], day, self.h, end=end_utc)
            if len(b):
                parts.append(b)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["symbol", "t", "o", "h", "l", "c", "v", "vw"])

    def last_trades(self, syms):
        out = {}
        for i in range(0, len(syms), 200):
            j = alpaca.get("https://data.alpaca.markets/v2/stocks/trades/latest",
                           {"symbols": ",".join(syms[i:i + 200]), "feed": "iex"}, headers=self.h)
            for s, tr in (j.get("trades") or {}).items():
                if tr and tr.get("p"):
                    out[s] = (float(tr["p"]), str(tr.get("t")))
        return out


def build_universe(api, day):
    """(ctx, prior) of the day: cheap price screen on the last sessions first, then the full history."""
    syms, _ = api.assets()
    end = alpaca.sip_end(day - pd.Timedelta(days=1))
    near = api.daily(syms, str((day - pd.Timedelta(days=12)).date()), end, "raw")
    cand = sorted(s for s, f in near.items() if C.PENNY_LO <= f["c"].iloc[-1] < C.MAIN_PRICE)
    start = str((day - pd.Timedelta(days=HIST_DAYS)).date())
    raw, adj = api.daily(cand, start, end, "raw"), api.daily(cand, start, end, "all")
    return universe_ctx(raw, adj, day)


# ---------- state ----------

class State:
    def __init__(self, root):
        self.root = Path(root) / "chrp"
        self.ledger = Ledger(self.root / "ledger")
        self.work = self.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)

    def load(self, name, default):
        f = self.work / name
        return json.loads(f.read_text()) if f.exists() else default

    def save(self, name, obj):
        f = self.work / name
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str))
        tmp.replace(f)


# ---------- one tick ----------

def scan(st, api, fz, now):
    """Decide every new trigger of the day. `now` is a naive ET timestamp."""
    day = now.normalize()
    cal = alpaca.trading_calendar(day, day)
    if cal.empty:
        return "closed"
    close_t = pd.Timedelta(hours=int(cal["close"].iloc[0][:2]), minutes=int(cal["close"].iloc[0][3:5]))
    if now < day + B.FIRST_S:
        return "early"
    ud = f"{day:%Y-%m-%d}"
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / f"{ud}_prior.pkl").exists():
        ctx, prior = build_universe(api, day)
        pd.to_pickle(prior, CACHE / f"{ud}_prior.pkl")
        st.save(f"{ud}/universe.json", ctx)
        print(f"[chr] universe {len(ctx)} penny stocks")
    ctx, prior = st.load(f"{ud}/universe.json", {}), pd.read_pickle(CACHE / f"{ud}_prior.pkl")
    trig = st.load(f"{ud}/triggers.json", {})
    openp = st.load("open.json", [])
    end_utc = (pd.Timestamp(now, tz=ET).tz_convert("UTC") - alpaca.SIP_DELAY)
    syms = sorted(s for s in ctx if s not in trig)
    etfs = C.ETFS
    bars = api.bars5(syms + etfs, day, end_utc)
    if bars.empty:
        return "no bars"
    bars["date"] = day
    by_sd = {k: g.sort_values("t") for k, g in bars.groupby(["symbol", "date"])}
    etf_raw = api.daily(etfs, str((day - pd.Timedelta(days=15)).date()), alpaca.sip_end(day - pd.Timedelta(days=1)), "raw")
    etf_all = api.daily(["SPY"], str((day - pd.Timedelta(days=15)).date()), alpaca.sip_end(day - pd.Timedelta(days=1)), "all")
    etf_raw = pd.concat([f.assign(symbol=k) for k, f in etf_raw.items()]).reset_index(names="date")
    etf_all = pd.concat([f.assign(symbol=k) for k, f in etf_all.items()]).reset_index(names="date")
    vix = pd.read_csv(CACHE / "vix.csv", parse_dates=["date"]) if (CACHE / "vix.csv").exists() else None
    if vix is None:
        import io
        import requests
        r = requests.get(C.VIX_URL, timeout=60)
        r.raise_for_status()
        vix = pd.read_csv(io.StringIO(r.text))
        vix.columns = ["date", "vix"]
        vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
        vix = vix.dropna()
        vix["date"] = pd.to_datetime(vix["date"])
        vix.to_csv(CACHE / "vix.csv", index=False)
    days = pd.DatetimeIndex(sorted(set(etf_all["date"])) + [day])
    fired = []
    for sym in syms:
        b = B.session(by_sd.get((sym, day), pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw"])), day, close_t)
        if b.empty:
            continue
        i, tname = B.first_trigger(b, ctx[sym], fz["curve"], "penny", close_t - pd.Timedelta(minutes=30))
        if i is not None:
            fired.append((sym, b, i, tname))
    cik_of, sic_of = sic_maps([f[0] for f in fired]) if fired else ({}, {})
    new = []
    for sym, b, i, tname in fired:
        s_end = b["end"].iloc[i]
        mkt = C.market(day, s_end, by_sd, etf_raw, etf_all, vix, sector_of(sym, cik_of, sic_of), days)
        pri = prior[sym]
        A = B.atr(pri["h"], pri["l"], pri["c"])
        try:
            f = B.features(b, i, pri[["o", "h", "l", "c", "v"]], ctx[sym], mkt, fz["curve"], tname)
        except Exception as e:                                   # a stock with a malformed history is skipped, logged
            trig[sym] = {"symbol": sym, "trig": tname, "s": str(s_end), "trade": False, "reason": f"inputs failed: {e!r}"}
            continue
        new.append({"symbol": sym, "trig": tname, "s": s_end, "atr": float(A), "f": f})
    if not new:
        st.save(f"{ud}/triggers.json", trig)
        return "no new triggers"
    new.sort(key=lambda r: (r["s"], r["symbol"]))
    pred = predict(fz, [r["f"] for r in new])
    px = api.last_trades([r["symbol"] for r in new])
    _, short_ok = api.assets()
    for r, p in zip(new, pred):
        lab, trade, size = choose(p, fz)
        sym = r["symbol"]
        rec = {"symbol": sym, "trig": r["trig"], "s": str(r["s"]), "decided": str(now), "atr": r["atr"],
               "pred": [float(x) for x in p], "label": lab, "size": size, "trade": False,
               "entry": px.get(sym, (np.nan, None))[0], "entry_t": px.get(sym, (None, None))[1]}
        if now - r["s"] > LAST_DECISION_GAP:
            rec["reason"] = "late"
        elif not trade:
            rec["reason"] = "below the cut"
        elif not np.isfinite(rec["entry"]):
            rec["reason"] = "no quote"
        elif lab.startswith("short") and sym not in short_ok:
            rec["reason"] = "not shortable"
        elif not exposure_ok([x["size"] for x in openp if x["role"] == "pick"], size):
            rec["reason"] = "exposure cap"
        else:
            rec["trade"] = True
            openp.append({"symbol": sym, "date": ud, "label": lab, "size": size, "entry": rec["entry"],
                          "atr": r["atr"], "decided": str(now), "role": "pick"})
        trig[sym] = rec
        print(f"[chr] {sym} {r['trig']}: {lab} {'TAKEN' if rec['trade'] else rec.get('reason')}")
    st.save(f"{ud}/triggers.json", trig)
    st.save("open.json", openp)
    return f"{len(new)} new"


def after_close(st, api, now):
    """Controls and the day record for today, then settle every position whose 5 sessions are over."""
    day = now.normalize()
    ud = f"{day:%Y-%m-%d}"
    trig = st.load(f"{ud}/triggers.json", {})
    if not st.ledger.has("chrp_day", ud):
        picks = [s for s, r in trig.items() if r.get("trade")]
        ctl = controls(day, [s for s, r in trig.items() if np.isfinite(r.get("entry", np.nan)) and "pred" in r], picks)
        openp = st.load("open.json", [])
        for s, c in ctl.items():
            pk = trig[s]
            cr = trig[c]
            openp.append({"symbol": c, "date": ud, "label": pk["label"], "size": pk["size"], "entry": cr["entry"],
                          "atr": cr["atr"], "decided": cr["decided"], "role": "control", "control_of": s})
        st.save("open.json", openp)
        st.ledger.write_json("chrp_day", ud, f"days/{ud}.json", {"triggers": trig, "controls": ctl, "book": BOOK})
    settle_due(st, api, day)


def settle_due(st, api, today):
    openp = st.load("open.json", [])
    keep, cal = [], alpaca.trading_calendar(today - pd.Timedelta(days=20), today)
    for p in openp:
        d0 = pd.Timestamp(p["date"])
        later_days = [d for d in cal["date"] if d > d0 and d <= today]
        if len(later_days) < B.HOLD - 1:
            keep.append(p)
            continue
        net = settle_one(api, p, d0, later_days[:B.HOLD - 1])
        if net is not None:
            st.save(f"settled/{p['date']}_{p['symbol']}_{p['role']}.json", {**p, "net": net})
        else:
            keep.append(p)
    st.save("open.json", keep)
    # one ledger record per decision day, once every position of that day is settled
    days = sorted({p["date"] for p in openp} - {p["date"] for p in keep})
    for ud in days:
        if st.ledger.has("chrp_settle", ud):
            continue
        rows = [json.loads(f.read_text()) for f in (st.work / "settled").glob(f"{ud}_*.json")]
        picks = [r["net"] for r in rows if r["role"] == "pick"]
        ctrl = [r["net"] for r in rows if r["role"] == "control"]
        st.ledger.write_json("chrp_settle", ud, f"settle/{ud}.json", {
            "positions": rows, "pick_mean": float(np.mean(picks)) if picks else None,
            "control_mean": float(np.mean(ctrl)) if ctrl else None})


def settle_one(api, p, d0, later_days):
    """Net return of one position with the training code: first barrier on the 5-minute bars after the decision,
    then the next sessions' daily highs/lows, else the last close."""
    t_dec = pd.Timestamp(p["decided"])
    b = api.bars5([p["symbol"]], d0)
    b = b[b["t"] >= t_dec.ceil("5min")]
    if b.empty:
        return None
    raw = api.daily([p["symbol"]], str(d0.date()), str((later_days[-1] + pd.Timedelta(days=1)).date()), "raw")
    adj = api.daily([p["symbol"]], str(d0.date()), str((later_days[-1] + pd.Timedelta(days=1)).date()), "all")
    if p["symbol"] not in raw or p["symbol"] not in adj:
        return None
    g = D.daily_rows(p["symbol"], raw[p["symbol"]], adj[p["symbol"]])
    g = g[g["date"] >= d0]
    if g.empty or g["date"].iloc[0] != d0:
        return None
    k = g["ac"] / g["c"] / (g["ac"].iloc[0] / g["c"].iloc[0])
    base = g.assign(h=g["h"] * k, l=g["l"] * k, c=g["c"] * k)
    later = base[base["date"] > d0].head(B.HOLD - 1)
    lab = B.labels(b[["o", "h", "l", "c"]], float(p["entry"]), p["atr"], later[["date", "h", "l", "c"]], "penny", d0)
    v = lab.get(p["label"])
    return None if v is None or not np.isfinite(v) else float(v)


# ---------- loop ----------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    api, fz = Api(), load_frozen()
    now = lambda: pd.Timestamp.now(tz=ET).tz_localize(None)
    if a.check:
        syms, short_ok = api.assets()
        ctx, prior = build_universe(api, now().normalize())
        print(f"[chr check] {len(syms)} assets, {len(short_ok)} easy to borrow, {len(ctx)} penny stocks today, "
              f"{len(fz['features'])} inputs, cut {fz['cut']:.4%}")
        return 0
    st = State(a.state)
    stop = time.monotonic() + 60 * a.minutes
    while time.monotonic() < stop:
        n = now()
        cal = alpaca.trading_calendar(n.normalize(), n.normalize())
        if cal.empty:
            return 3
        close_t = n.normalize() + pd.Timedelta(hours=int(cal["close"].iloc[0][:2]), minutes=int(cal["close"].iloc[0][3:5]))
        if n >= close_t + pd.Timedelta(minutes=20):
            after_close(st, api, n)
            return 3
        if n >= close_t - pd.Timedelta(minutes=30):
            time.sleep(60)
            continue
        res = scan(st, api, fz, n)
        print(f"[chr] {n:%H:%M} {res}")
        time.sleep(max(1, TICK_S - (now() - n).total_seconds()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
