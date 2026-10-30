"""NX1 (NEWS-XGB) nightly scorer and shadow books (research/prereg/NX1.md), run by nx1-nightly.yml after
10:00pm ET.

Each trading night d:
1. Settle both $5,000 ledger books (picks and random control) on d's SIP daily bars (raw): fill entries
   picked last night at d's open, stop out at 10% (a gap fills at the open), exit at the close of the third
   session, with adaptive/costs.py costs and borrow on shorts.
2. Build d's batch: Benzinga articles naming one symbol, created after 10:00pm ET on the previous trading
   day up to 10:00pm ET on d, deduplicated as in training; FinBERT at the training run's pinned commit.
3. Inputs exactly as in training (tracks/news/nx1_study.py builds them from the same rows), then the
   version 1 model: up to 3 longs and 3 shorts (shorts only where Alpaca marks the stock shortable and easy
   to borrow), the same number of random control events, and as many placebo events as real ones.
4. Log the night (inputs, probabilities, picks, control, placebo, fills; no headline text) to the
   hash-chained ledger in <state>/nx1.
5. NX2's near-miss book (research/prereg/FWD1.md), once its version 1 models exist: the same night's real
   events scored by NX2 (effect e = N(x) - N(x0), fired at |e| >= its training cut), up to 3 longs and 3
   shorts (shortable and easy to borrow), a same-size random control with the same sides, and the 5 largest
   |e| as the most-confident variant. Logged as kind NX2_picks; returns are read later from SIP bars.

Usage: python -m tracks.news.nx1_live --state <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive import costs, sizing
from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.ledger import Ledger
from tracks.main.longterm import news_history
from tracks.news import nx1_study as S
from tracks.news import w1_data as W

ET = "America/New_York"
MODEL = Path("research/results/NX1/nx1_v1.json")
META = Path("research/results/NX1/nx1_v1_meta.json")
BOOK_CASH = 5000.0
STOP = 0.10
LOOKBACK_DAYS = 420                 # calendar days of bars for the 250-return beta
NEWS_DAYS_KEEP = 40
BOOKS = ("picks", "control")
NX2_DIR = Path("research/results/NX2")
NX2_TOP = 5


# ---------- pure helpers (tested) ----------

def batch_day(now_et, cal_dates):
    """The trading day whose batch is complete at now (its 10:00pm ET has passed), else None."""
    d = pd.DatetimeIndex(cal_dates)
    done = d[d + S.CUTOFF <= now_et.tz_localize(None)]
    return done.max() if len(done) else None


def window(d, cal_dates):
    d_ = pd.DatetimeIndex(cal_dates)
    prev = d_[d_ < d].max()
    tz = lambda x: (x + S.CUTOFF).tz_localize(ET).tz_convert("UTC")
    return tz(prev), tz(d)


def new_book():
    return {"cash": BOOK_CASH, "pos": [], "pending": [], "closed": []}


def kind(price):
    return "main" if price >= 5 else "penny"


def equity(b, closes):
    return b["cash"] + sum(p["sign"] * p["shares"] * closes.get(p["symbol"], p["entry_px"]) for p in b["pos"])


def gross(b, closes):
    return (sum(p["shares"] * closes.get(p["symbol"], p["entry_px"]) for p in b["pos"])
            + sum(p["shares"] * p["ref_px"] for p in b["pending"]))


def _close(b, p, px, day, why):
    c = costs.fill_cost(kind(p["entry_px"]), p["shares"], px, p["sign"] > 0)
    if p["sign"] < 0:
        held = (pd.Timestamp(day) - pd.Timestamp(p["entry_day"])).days + 1
        c += costs.borrow_cost(kind(p["entry_px"]), p["shares"] * p["entry_px"], held)
    b["cash"] += p["sign"] * p["shares"] * px - c
    pnl = p["sign"] * p["shares"] * (px - p["entry_px"]) - p["entry_cost"] - c
    b["closed"].append(p | {"exit_px": px, "exit_on": day, "exit_cost": c, "why": why, "pnl": pnl,
                            "ret": pnl / (p["shares"] * p["entry_px"])})


def settle(b, bars, day):
    """bars: {symbol: (o, h, l, c)} for `day`. Fills pending entries dated `day`, then stops and exits."""
    fills = []
    for q in [q for q in b["pending"] if q["entry_day"] == day]:
        b["pending"].remove(q)
        if q["symbol"] not in bars:
            fills.append({"symbol": q["symbol"], "skipped": "no bar on the entry day"})
            continue
        o = bars[q["symbol"]][0]
        c = costs.fill_cost(kind(o), q["shares"], o, q["sign"] < 0)
        b["cash"] -= q["sign"] * q["shares"] * o + c
        p = q | {"entry_px": o, "entry_cost": c, "stop": o * (1 - STOP * q["sign"])}
        b["pos"].append(p)
        fills.append({"symbol": q["symbol"], "side": q["sign"], "shares": q["shares"], "px": o})
    for p in list(b["pos"]):
        if p["symbol"] not in bars:
            continue
        o, h, lo, c = bars[p["symbol"]]
        hit = lo <= p["stop"] if p["sign"] > 0 else h >= p["stop"]
        if hit:
            gap = o < p["stop"] if p["sign"] > 0 else o > p["stop"]
            px, why = (o, "stop (gap)") if gap and p["entry_day"] != day else (p["stop"], "stop")
        elif day >= p["exit_day"]:
            px, why = c, "3 sessions"
        else:
            continue
        b["pos"].remove(p)
        _close(b, p, px, day, why)
    return fills


def choose(ev, rng, picks_n=S.PICKS):
    """Scored real events (p_short, p_long, cls, short_ok) -> picks and a same-size random control. Shorts
    only where short_ok; each control slot takes its pick's side, drawn from events that side allows."""
    s = ev.assign(score=ev["p_long"] - ev["p_short"])
    lo = s[s["cls"] == 2].nlargest(picks_n, "score").assign(side=1)
    sh = s[(s["cls"] == 0) & s["short_ok"]].nsmallest(picks_n, "score").assign(side=-1)
    p = pd.concat([lo, sh])
    used, rows = set(), []
    for side in p["side"]:
        pool = [i for i in s.index if i not in used and (side > 0 or s.at[i, "short_ok"])]
        if not pool:
            continue
        i = pool[int(rng.integers(len(pool)))]
        used.add(i)
        rows.append(s.loc[i].to_dict() | {"side": int(side)})
    return p, pd.DataFrame(rows)


def nx2_record(real, news_model, cut, rng, day, entry_day, exit_day, closes):
    """The NX2_picks record for one night (research/prereg/NX2.md picks, FWD1's book rules)."""
    from tracks.news import nx2_study as N2
    X = real[S.INPUTS].to_numpy(float)
    e = news_model(X) - news_model(N2.no_news(X))
    r = real.assign(e=e, fired=(np.abs(e) >= cut) & (np.abs(e) > 0))
    f = r[r["fired"]]
    lo = f[f["e"] > 0].nlargest(S.PICKS, "e").assign(side=1)
    sh = f[(f["e"] < 0) & f["short_ok"]].nsmallest(S.PICKS, "e").assign(side=-1)
    p = pd.concat([lo, sh])
    used, ctrl = set(), []
    for side in p["side"]:
        pool = [i for i in r.index if i not in used and (side > 0 or r.at[i, "short_ok"])]
        if pool:
            i = pool[int(rng.integers(len(pool)))]
            used.add(i)
            ctrl.append({"symbol": r.at[i, "symbol"], "side": "long" if side > 0 else "short"})
    days = (pd.Timestamp(exit_day) - pd.Timestamp(entry_day)).days + 1

    def cost(sym, side):
        return float(S.cost_frac([closes.get(sym, np.nan)], [side < 0], [days])[0])
    return {"idea": "NX2", "book": "FWD1 near-miss shadow book", "as_of": day, "events": int(len(r)),
            "fired": int(r["fired"].sum()), "fire_cut": cut,
            "picks": [{"symbol": x["symbol"], "side": "long" if x["side"] > 0 else "short", "e": float(x["e"]),
                       "cost": cost(x["symbol"], x["side"])} for _, x in p.iterrows()],
            "control": [c | {"cost": cost(c["symbol"], 1 if c["side"] == "long" else -1)} for c in ctrl],
            "top5": p.reindex(p["e"].abs().sort_values(ascending=False).index)["symbol"].head(NX2_TOP).tolist(),
            "hold_sessions": S.HOLD, "entry": "next session's open", "exit": f"close of session {S.HOLD}"}


def plan(b, chosen, closes, entry_day, exit_day):
    """Sized pending entries for tomorrow's open at the SYS1 shadow limits (1% risk to a 10% stop, 10% name
    cap, 5% per short, 1x gross, 0.5% of median daily dollar volume)."""
    eq, g, out = equity(b, closes), gross(b, closes), []
    for r in chosen.to_dict("records"):
        side = "long" if r["side"] > 0 else "short"
        px = float(np.exp(r["log_px"]))
        n, why = sizing.size(eq, side, STOP, px, float(np.exp(r["log_dv"])), g, sizing.MAIN)
        if n <= 0:
            out.append({"symbol": r["symbol"], "side": int(r["side"]), "skipped": why})
            continue
        g += n * px
        q = {"symbol": r["symbol"], "sign": int(r["side"]), "shares": int(n), "ref_px": px,
             "entry_day": entry_day, "exit_day": exit_day}
        b["pending"].append(q)
        out.append(q | {"binding": why})
    return out


# ---------- data (network) ----------

class Api:
    def __init__(self):
        self.h = alpaca_headers()

    def calendar(self, start, end):
        return alpaca.trading_calendar(start, end)

    def assets(self):
        return alpaca.active_assets()

    def daily(self, symbols, start, end, adjustment):
        out = {}
        for i in range(0, len(symbols), 500):
            out |= to_frames(fetch_daily_bars(symbols[i:i + 500], start, alpaca.sip_end(end), adjustment=adjustment,
                                              headers=self.h))
        return out

    def news(self, start, end):
        return news_history.fetch_window(start, end)

    def finbert(self, meta_file):
        return S.finbert_scorer(meta_file)

    def model(self):
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(MODEL)
        return lambda X: m.predict(xgb.DMatrix(X))

    def nx2(self):
        """NX2's version 1 news model as a function of the 32 inputs, and its fire cut; None before training."""
        meta_f = NX2_DIR / "nx2_v1_meta.json"
        if not meta_f.exists():
            return None
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(NX2_DIR / "nx2_v1_news.json")
        meta = json.loads(meta_f.read_text())
        return (lambda X: m.predict(xgb.DMatrix(X))), meta


def news_days_of(news):
    n = news[news["symbols"].fillna("") != ""]
    x = pd.DataFrame({"date": W.et_time(n["created_at"]).dt.normalize(), "symbol": n["symbols"].str.split(",")})
    return x.explode("symbol").drop_duplicates()[["symbol", "date"]]


def save_parquet(df, f):
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


def run(state, api=None, now=None):
    api = api or Api()
    root = Path(state) / "nx1"
    root.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists():
        print("[nx1] model v1 not trained yet; nothing to do")
        return None
    meta = json.loads(META.read_text())
    now = now if now is not None else alpaca.utcnow()
    now_et = now.tz_convert(ET)
    cal = api.calendar((now_et - pd.Timedelta(days=LOOKBACK_DAYS + 10)).date(), (now_et + pd.Timedelta(days=10)).date())
    cal["date"] = pd.to_datetime(cal["date"])
    dates = pd.DatetimeIndex(cal["date"])
    d = batch_day(now_et, dates)
    nxt = dates[dates > d]
    if len(nxt) < S.HOLD or now_et.tz_localize(None) >= nxt[0] + pd.Timedelta(hours=9, minutes=30):
        print(f"[nx1] {d.date()}: run too late (next session already open); night skipped")
        return None
    day = f"{d:%Y-%m-%d}"
    led = Ledger(root)
    if led.has("nx1_night", day):
        print(f"[nx1] {day} already logged")
        return None
    books_f = root / "books.json"
    books = json.loads(books_f.read_text()) if books_f.exists() else {k: new_book() for k in BOOKS}

    # 1. settle the books on d's bars
    held = sorted({p["symbol"] for b in books.values() for p in b["pos"] + b["pending"]})
    raw_d = api.daily(held, day, day, "raw") if held else {}
    bars_d = {s: tuple(float(f.loc[d, k]) for k in "ohlc") for s, f in raw_d.items() if d in f.index}
    fills = {k: settle(b, bars_d, day) for k, b in books.items()}

    # 2. news: the batch, recent headline hashes (3-day dedup) and the 40-day news-day table
    start, end = window(d, dates)
    nd_f, heads_f = root / "news_days.parquet", root / "recent_heads.parquet"
    if nd_f.exists():
        nd = pd.read_parquet(nd_f)
        news = api.news(start, end)
    else:
        news = api.news(start - pd.Timedelta(days=NEWS_DAYS_KEEP), end)
        nd = pd.DataFrame(columns=["symbol", "date"])
    nd = pd.concat([x for x in (nd, news_days_of(news)) if len(x)]).drop_duplicates() if len(news) else nd
    nd = nd[pd.to_datetime(nd["date"]) >= d - pd.Timedelta(days=NEWS_DAYS_KEEP)]
    tm = W.et_time(news["created_at"]) if len(news) else pd.Series(dtype="datetime64[ns]")
    lo_, hi_ = (x.tz_convert(ET).tz_localize(None) for x in (start, end))
    batch = news[((tm > lo_) & (tm <= hi_)).to_numpy()] if len(news) else news
    rows = S.headline_rows(batch, api.finbert(_finbert_meta(root, meta)))
    prev = pd.read_parquet(heads_f) if heads_f.exists() else rows.iloc[0:0]
    allr = pd.concat([prev.assign(_old=True), rows.assign(_old=False)], ignore_index=True)
    kept = S.dedup(allr)
    rows = kept[~kept["_old"].astype(bool)].drop(columns="_old")
    news_ev = S.news_events(rows, cal) if len(rows) else pd.DataFrame(columns=["symbol", "d", *S.NEWS_INPUTS])
    news_ev = news_ev[news_ev["d"] == d]

    # 3. universe, inputs and the model
    a = api.assets()
    a = a[a["excl_reason"].isna() & a["tradable"].fillna(False).astype(bool)]
    short_ok = set(a.loc[a["shortable"].fillna(False).astype(bool) & a["easy_to_borrow"].fillna(False).astype(bool),
                         "symbol"])
    syms = sorted(a["symbol"])
    recent = api.daily(syms, (d - pd.Timedelta(days=45)).date(), day, "raw")
    pool = [s for s, f in recent.items() if d in f.index and len(f) >= 20 and f.loc[d, "c"] >= S.MIN_PRICE
            and float((f["c"] * f["v"]).iloc[-20:].median()) >= S.MIN_DV20]
    real_syms = sorted(set(news_ev["symbol"]) & set(pool))
    keys = pd.DataFrame({"symbol": [s for s in pool if s not in real_syms], "cik": 0.0, "date": d})
    quiet = keys[W.quiet_mask(keys, nd.assign(date=pd.to_datetime(nd["date"])),
                              pd.DataFrame({"cik": pd.Series(dtype=float), "date": pd.Series(dtype="datetime64[ns]")}),
                              back=S.QUIET_BACK, cal=dates)]["symbol"].tolist()
    rng = np.random.default_rng(S.SEED + int(d.strftime("%Y%m%d")))
    plac_syms = list(rng.choice(quiet, size=min(len(quiet), 2 * len(real_syms)), replace=False)) if quiet else []
    first = (d - pd.Timedelta(days=LOOKBACK_DAYS)).date()
    want = sorted(set(real_syms) | set(plac_syms) | {"SPY"})
    raw, adj = api.daily(want, first, day, "raw"), api.daily(want, first, day, "all")
    spy = adj["SPY"].rename(columns={"o": "ao", "c": "ac"})[["ao", "ac"]].rename_axis("date").reset_index()
    spy_in = S.spy_inputs(spy)
    feats = []
    for s in want:
        if s == "SPY" or s not in raw or s not in adj:
            continue
        p = S.stock_panel(W.daily_rows(s, raw[s], adj[s]), spy, cal)
        p = p[(p["d"] == d) & p["uni"]] if len(p) else p
        if len(p):
            feats.append(p.assign(symbol=s))
    pan = pd.concat(feats, ignore_index=True).merge(spy_in, on="d") if feats else pd.DataFrame()
    real = pan.merge(news_ev, on=["symbol", "d"]).assign(kind="real") if len(pan) else pd.DataFrame()
    plac = pan[pan["symbol"].isin(plac_syms)] if len(pan) else pd.DataFrame()
    plac = plac.iloc[rng.permutation(len(plac))[:len(real)]].assign(kind="placebo", **S.placebo_news())
    ev = pd.concat([real, plac], ignore_index=True)
    out = {"day": day, "model_sha256": meta["model_sha256"], "batch_headlines": int(len(rows)),
           "fills": fills, "events": [], "picks": [], "control": [], "plans": {}}
    closes = {s: float(f.loc[d, "c"]) for s, f in raw_d.items() if d in f.index}
    if len(ev):
        ev["novelty"] = S.novelty(ev[["symbol", "d"]].reset_index(drop=True), nd.assign(date=pd.to_datetime(nd["date"])))
        prob = api.model()(ev[S.INPUTS].to_numpy(float))
        ev = ev.assign(p_short=prob[:, 0], p_none=prob[:, 1], p_long=prob[:, 2], cls=prob.argmax(1),
                       short_ok=ev["symbol"].isin(short_ok))
        closes |= {r["symbol"]: float(np.exp(r["log_px"])) for r in ev.to_dict("records")}
        picks, control = choose(ev[ev["kind"] == "real"], rng)
        entry_day, exit_day = f"{nxt[0]:%Y-%m-%d}", f"{nxt[S.HOLD - 1]:%Y-%m-%d}"
        out["plans"] = {"picks": plan(books["picks"], picks, closes, entry_day, exit_day),
                        "control": plan(books["control"], control, closes, entry_day, exit_day)}
        cols = ["symbol", "kind", *S.INPUTS, "beta", "p_short", "p_none", "p_long", "cls", "short_ok"]
        out["events"] = ev[cols].to_dict("records")
        out["picks"] = picks[["symbol", "side", "p_long", "p_short"]].to_dict("records")
        out["control"] = control[["symbol", "side"]].to_dict("records") if len(control) else []
    out["books"] = {k: {"equity": equity(b, closes), "cash": b["cash"], "open": len(b["pos"]),
                        "pending": len(b["pending"]), "closed": len(b["closed"])} for k, b in books.items()}
    led.write_json("nx1_night", day, f"nights/{day}.json", json.loads(json.dumps(out, default=_js)))
    books_f.write_text(json.dumps(books, indent=1, default=_js))
    save_parquet(nd.reset_index(drop=True), nd_f)
    keep = kept[kept["t"] >= d - pd.Timedelta(days=4)].drop(columns="_old")
    save_parquet(keep.reset_index(drop=True), heads_f)
    (root / "README.md").write_text(summary(out))
    print(summary(out))
    nx2 = api.nx2() if len(ev) else None
    if nx2 is not None and not led.has("NX2_picks", day):
        try:
            rec = nx2_record(ev[ev["kind"] == "real"], nx2[0], float(nx2[1]["fire_cut"]),
                             np.random.default_rng(S.SEED + 2 + int(d.strftime("%Y%m%d"))), day,
                             f"{nxt[0]:%Y-%m-%d}", f"{nxt[S.HOLD - 1]:%Y-%m-%d}", closes)
            rec["model_sha256"] = nx2[1]["sha256"]
            led.write_json("NX2_picks", day, f"nx2/{day}.json", json.loads(json.dumps(rec, default=_js)))
            out["nx2"] = rec
            print(f"[nx2] {day}: {rec['fired']} of {rec['events']} fired; picks "
                  + ", ".join(f"{p['side']} {p['symbol']}" for p in rec["picks"]))
        except Exception as x:                      # NX1's night is already logged; NX2 must not break it
            print(f"::warning::[nx2] {day}: {type(x).__name__}: {x}")
    return out


def _finbert_meta(root, meta):
    f = root / "finbert_meta.json"
    if not f.exists():
        f.write_text(json.dumps(meta["finbert"]))
    return f


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return str(x)


def summary(out):
    lines = [f"# NX1 shadow books, night {out['day']}", "",
             f"Headlines in the batch: {out['batch_headlines']}. Events: "
             f"{sum(e['kind'] == 'real' for e in out['events'])} real, "
             f"{sum(e['kind'] == 'placebo' for e in out['events'])} placebo.", ""]
    for k, b in out["books"].items():
        lines.append(f"- {k}: equity ${b['equity']:,.2f}, {b['open']} open, {b['pending']} to enter, "
                     f"{b['closed']} closed")
    lines += ["", "Picks: " + (", ".join(f"{'long' if p['side'] > 0 else 'short'} {p['symbol']}" for p in out["picks"])
                              or "none")]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    run(a.state)
    print(f"[nx1] done in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
