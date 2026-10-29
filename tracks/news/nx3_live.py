"""NX3 live scorer and $5,000 shadow book (research/prereg/NX3.md, after its gate passed on 2026-10-24).
Ledger only: no orders.

Each trading day, from the open to 2 minutes before the close, every POLL_S seconds:
1. New Benzinga articles from Alpaca (real time) since the previous close: the training run's headline rows
   (LM counts, FinBERT at NX1's pinned commit, flags, the 3-day dedup); no text is kept.
2. Events as in training (tracks/news/nx3_study.session_events): per (symbol, d), decision time t = the first
   headline after the previous close, or the open when it came before; news inputs over the headlines up to t.
3. One minute after t (and no later than 10 minutes after), the event is decided: stock and market inputs at
   the previous close (training code), the intraday inputs from Alpaca's IEX real-time feed (the only free
   real-time source; see `intraday`), the version 1 news model's effect e, fired when |e| >= the frozen cut.
   Fired events are taken in time order (ties at the open in symbol order, as training sorts them): up to
   3 longs and 3 shorts; a short only where Alpaca marks the stock shortable and easy to borrow, else the
   slot passes to the next fired short.
4. After the close: the same-day random control (as many of the day's decided events, same sides,
   default_rng(SEED + yyyymmdd)) and the day record to the ledger (kind nx3_day).
5. After the next session's close: the trades settled from SIP bars with the training code
   (nx3_study.month_trades: entry = open of the first SIP minute bar >= t + 1 minute, exit = next close,
   minus SPY), costs as in training ($500 a trade), logged as kind nx3_settle with the day's pick minus
   control. The forward test (prereg): book minus control, Newey-West t >= 2 at 100 closed trades or
   2027-12-31.

Usage: python -m tracks.news.nx3_live --state state [--minutes 20]   (exit 3 = nothing more today)
"""
import argparse
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
from tracks.main.longterm import news_history
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S
from tracks.news import nx3_orders as O
from tracks.news import nxm_live as M
from tracks.news import w1_data as W

ET = S.ET
RES = Path("research/results/NX3")
MODEL, META = RES / "nx3_v1_news.json", RES / "nx3_v1_meta.json"
NX1_META = Path("research/results/NX1/nx1_v1_meta.json")   # FinBERT's pinned commit, as NX1 training used
POLL_S = 30
LOOKBACK_DAYS = 420
NEWS_DAYS_KEEP = 40
IEX_SHARE_DAYS = 5
SETTLE_AFTER = pd.Timedelta(minutes=20)     # SIP bars are free 15 minutes after the fact
# settled and kept, but left out of the forward test: no pre-open event was decided (NX3.md, bug fixes 2026-10-28)
BUG_DAYS = {"2026-10-26", "2026-10-27", "2026-10-28"}
BOOK_CASH = 5000.0
SEED = S.SEED


# ---------- pure helpers (tested) ----------

def session_row(cal, d):
    """nx3_study.sessions' row for day d (open_t, close_t, prev_close_t, prev, next), or None."""
    s = S.sessions(cal)
    s = s[s["date"] == d]
    return None if s.empty else s.iloc[0]


def due(ev, decided, now):
    """Events whose decision time has come (now >= t + 1 minute) and that aren't decided yet, split into
    (ready, missed): missed = more than 10 minutes past t, when the entry window has closed."""
    e = ev[~ev["key"].isin(decided) & (ev["t"] + S.WAIT <= now)]
    late = e["t"] + S.MAX_WAIT < now
    return e[~late], e[late]


def intraday(bars, t, open_t, entry, prev_c, dv20, iex_share):
    """The 4 stock intraday inputs as nx3_study.trade computes them, from today's IEX minute bars (t naive ET
    bar start, o, c, v) and `entry`, the latest IEX trade at decision time (the price a 1-minute entry pays;
    training used the first SIP bar's open). rv30 counts IEX volume, which is a small share of all trading,
    so it is divided by the stock's IEX share of SIP volume over the previous IEX_SHARE_DAYS sessions."""
    before = bars[bars["t"] + pd.Timedelta(minutes=1) <= t] if len(bars) else bars
    if t <= open_t or before.empty:
        first = bars[bars["t"] >= open_t] if len(bars) else bars
        px_t = float(first["o"].iloc[0]) if len(first) else np.nan
    else:
        px_t = float(before["c"].iloc[-1])
    if t <= open_t:
        rv = np.nan
    else:
        w = before[before["t"] >= t - pd.Timedelta(minutes=30)]
        base = dv20 / prev_c * 30 / 390 if dv20 > 0 and prev_c > 0 else np.nan
        ok = base and np.isfinite(base) and iex_share and np.isfinite(iex_share) and iex_share > 0
        rv = float(w["v"].sum()) / iex_share / base if ok else np.nan
    return {"mins_open": (t - open_t).total_seconds() / 60, "r_open": px_t / prev_c - 1 if prev_c > 0 else np.nan,
            "r_react": entry / px_t - 1 if px_t > 0 else np.nan, "rv30": rv}


def take(e, fired, symbol, short_ok, taken):
    """Side taken for one decided event (1, -1) or the reason it isn't (None, why). `taken` lists the day's
    sides so far."""
    if not fired:
        return None, "not fired"
    side = 1 if e > 0 else -1
    if sum(1 for s in taken if s == side) >= S.PICKS:
        return None, "side full"
    if side < 0 and symbol not in short_ok:
        return None, "not easy to borrow"
    return side, None


def control(decided, picks, seed):
    """The same-day control: as many of the day's decided real events as there are picks, at random, with
    the picks' sides (nx3_study.picks)."""
    if not picks or len(decided) < len(picks):
        return []
    rng = np.random.default_rng(seed)
    k = rng.choice(len(decided), size=len(picks), replace=False)
    return [dict(decided[i], side=p["side"]) for i, p in zip(k, picks)]


def day_seed(d):
    return SEED + int(pd.Timestamp(d).strftime("%Y%m%d"))


# ---------- network ----------

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

    def iex_bars(self, symbols, start, end, timeframe):
        """{symbol: DataFrame(t naive ET bar start, o, c, v)} from the IEX real-time feed (decision inputs
        only; every return is settled on SIP bars)."""
        out = {}
        s, e = (pd.Timestamp(x, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") for x in (start, end))
        for i in range(0, len(symbols), S.SYM_CHUNK):
            tok = None
            while True:
                p = {"symbols": ",".join(symbols[i:i + S.SYM_CHUNK]), "timeframe": timeframe, "start": s, "end": e,
                     "feed": "iex", "adjustment": "raw", "limit": 10000}
                if tok:
                    p["page_token"] = tok
                j = alpaca.get(S.BARS_URL, p, headers=self.h)
                for sym, bs in (j.get("bars") or {}).items():
                    out.setdefault(sym, []).extend((b["t"], b["o"], b["c"], b["v"]) for b in bs)
                tok = j.get("next_page_token")
                if not tok:
                    break
        res = {}
        for sym, rows in out.items():
            d = pd.DataFrame(rows, columns=["t", "o", "c", "v"])
            d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
            res[sym] = d.sort_values("t").reset_index(drop=True)
        return res

    def last_trades(self, symbols):
        """{symbol: (price, time)} of the latest IEX trade."""
        out = {}
        for i in range(0, len(symbols), 200):
            j = alpaca.get("https://data.alpaca.markets/v2/stocks/trades/latest",
                           {"symbols": ",".join(symbols[i:i + 200]), "feed": "iex"}, headers=self.h)
            for s, tr in (j.get("trades") or {}).items():
                if tr and tr.get("p"):
                    out[s] = (float(tr["p"]), str(tr.get("t")))
        return out

    def news(self, start, end):
        return news_history.fetch_window(start, end)

    def finbert(self, meta_file):
        return N.finbert_scorer(meta_file)

    def effect(self):
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(MODEL)

        def eff(X):
            return m.predict(xgb.DMatrix(X)) - m.predict(xgb.DMatrix(S.no_news(X)))
        return eff

    def settle_trades(self, ev, cal):
        return S.month_trades(ev, S.sessions(cal), self.h)


# ---------- state ----------

def _read_jsonl(f):
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def _append(f, rec):
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=_js) + "\n")


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def _save(df, f):
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


class Day:
    """One session's working files under <root>/days/<d>/: deduplicated headline rows (no text), the
    decision log (one line per decided or missed event) and the per-symbol inputs at the previous close."""

    def __init__(self, root, d):
        self.dir = Path(root) / "days" / f"{d:%Y-%m-%d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows_f, self.dec_f, self.ctx_f = (self.dir / x for x in ("rows.parquet", "decisions.jsonl", "ctx.json"))
        self.rows = pd.read_parquet(self.rows_f) if self.rows_f.exists() else None
        self.decisions = _read_jsonl(self.dec_f)
        self.ctx = json.loads(self.ctx_f.read_text()) if self.ctx_f.exists() else {}

    def decided(self):
        return {x["key"] for x in self.decisions}

    def taken(self):
        return [x["side"] for x in self.decisions if x.get("side")]

    def log(self, rec):
        _append(self.dec_f, rec)
        self.decisions.append(json.loads(json.dumps(rec, default=_js)))

    def save(self):
        if self.rows is not None:
            _save(self.rows, self.rows_f)
        self.ctx_f.write_text(json.dumps(self.ctx, default=_js))


# ---------- the day ----------

def stock_context(api, symbols, prev, cal):
    """Per symbol: the 6 stock inputs, universe flag, previous raw close and dv20 at the previous session
    (training's batch_events), and the IEX share of SIP volume over the last IEX_SHARE_DAYS sessions."""
    first = (prev - pd.Timedelta(days=LOOKBACK_DAYS)).date()
    want = sorted(set(symbols) | {"SPY"})
    raw, adj = api.daily(want, first, prev.date(), "raw"), api.daily(want, first, prev.date(), "all")
    spy = adj["SPY"].rename(columns={"o": "ao", "c": "ac"})[["ao", "ac"]].rename_axis("date").reset_index()
    sp = N.spy_inputs(spy).set_index("d")
    caldates = pd.DatetimeIndex(cal["date"])
    lo = caldates[max(caldates.searchsorted(prev) - IEX_SHARE_DAYS + 1, 0)]
    iex = api.iex_bars(want, lo, prev + pd.Timedelta(days=1), "1Day")
    out = {}
    for s in want:
        if s == "SPY" or s not in raw or s not in adj:
            continue
        rows = W.daily_rows(s, raw[s], adj[s])
        p = N.stock_panel(rows, spy, cal) if len(rows) else pd.DataFrame()
        p = p[p["d"] == prev] if len(p) else p
        if p.empty:
            continue
        r, g = p.iloc[0], rows.drop_duplicates("date").set_index("date")
        v_sip = g["v"][g.index >= lo].sum()
        v_iex = float(iex[s]["v"].sum()) if s in iex else 0.0
        out[s] = {**{k: float(r[k]) for k in N.STOCK_INPUTS}, "uni": bool(r["uni"]),
                  "prev_c": float(g.loc[prev, "c"]), "dv20": float(g.loc[prev, "dv20"]),
                  "iex_share": v_iex / v_sip if v_sip > 0 else np.nan}
    market = {"spy_r1": float(sp.loc[prev, "spy_r1"]), "spy_r5": float(sp.loc[prev, "spy_r5"]),
              "spy_prev_c": float(raw["SPY"].loc[prev, "c"])}
    return out, market


def news_days(root, api, d, cal):
    """(symbol, date) news days for novelty: NEWS_DAYS_KEEP days back, kept in <root>/news_days.parquet and
    topped up each morning with the sessions since the last run."""
    f = Path(root) / "news_days.parquet"
    nd = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["symbol", "date"])
    have = pd.to_datetime(nd["date"]).max() if len(nd) else d - pd.Timedelta(days=NEWS_DAYS_KEEP + 1)
    start = pd.Timestamp(have + pd.Timedelta(days=1), tz=ET).tz_convert("UTC")
    end = pd.Timestamp(d, tz=ET).tz_convert("UTC")
    if start < end:
        news = api.news(start, end)
        n = news[news["symbols"].fillna("") != ""]
        x = pd.DataFrame({"date": W.et_time(n["created_at"]).dt.normalize(), "symbol": n["symbols"].str.split(",")})
        x = x.explode("symbol").drop_duplicates()[["symbol", "date"]]
        nd = pd.concat([nd, x], ignore_index=True) if len(nd) else x
    nd["date"] = pd.to_datetime(nd["date"])
    nd = nd[nd["date"] >= d - pd.Timedelta(days=NEWS_DAYS_KEEP)].drop_duplicates().reset_index(drop=True)
    _save(nd, f)
    return nd


def recent_heads(root):
    """{(symbol, headline hash): last time} from the last 4 days of logged rows, for the 3-day dedup."""
    last = {}
    for f in sorted((Path(root) / "days").glob("*/rows.parquet"))[-5:]:
        r = pd.read_parquet(f, columns=["symbol", "hkey", "t"])
        for s, h, t in zip(r["symbol"], r["hkey"], r["t"]):
            last[(s, h)] = max(t, last.get((s, h), t))
    return last


def poll(day, api, ses, now, finbert, last):
    """Reads Alpaca's news since the last read (5 minutes of overlap; ids already kept are skipped), adds the
    new deduplicated headline rows to the day and returns its events."""
    since = pd.Timestamp(day.ctx.get("news_until") or ses["prev_close_t"])
    start = pd.Timestamp(since - pd.Timedelta(minutes=5), tz=ET).tz_convert("UTC")
    news = api.news(start, pd.Timestamp(now, tz=ET).tz_convert("UTC") + pd.Timedelta(seconds=1))
    if len(news):
        news = news[~news["id"].isin(set(day.ctx.get("ids", [])))]
    if len(news):
        rows = N.headline_rows(news, finbert)
        rows = rows[rows["t"] > ses["prev_close_t"]]
        day.ctx["ids"] = sorted(set(day.ctx.get("ids", [])) | set(news["id"].astype(int).tolist()))
        if len(rows):
            rows = N.dedup(rows, last)
            day.rows = rows if day.rows is None else pd.concat([day.rows, rows], ignore_index=True)
    day.ctx["news_until"] = now.isoformat()
    if day.rows is None or day.rows.empty:
        return pd.DataFrame(columns=["symbol", "d", "t", *N.NEWS_INPUTS, "key"])
    ev = S.session_events(day.rows[day.rows["t"] <= now], pd.DataFrame([ses]))
    return ev.assign(key=ev["symbol"] + "|" + ev["d"].dt.strftime("%Y-%m-%d"))


def decide(day, api, ses, ready, ctx, market, nd, short_ok, effect, now):
    """Decides the ready events (time order, ties in symbol order) and logs each one."""
    ready = ready.sort_values(["t", "symbol"], kind="stable")
    syms = sorted(set(ready["symbol"]) & set(ctx))
    trades = api.last_trades(syms + ["SPY"]) if syms else {}
    bars = api.iex_bars(syms + ["SPY"], ses["open_t"], now + pd.Timedelta(minutes=1), "1Min") if syms else {}
    spy_b = bars.get("SPY", pd.DataFrame(columns=["t", "o", "c", "v"]))
    for r in ready.to_dict("records"):
        base = {"key": r["key"], "symbol": r["symbol"], "d": r["d"], "t": r["t"], "decided_at": now}
        c = ctx.get(r["symbol"])
        if c is None or not c["uni"]:
            day.log(base | {"status": "not in universe"})
            continue
        if r["symbol"] not in trades or "SPY" not in trades:
            day.log(base | {"status": "no IEX trade"})
            continue
        entry = trades[r["symbol"]][0]
        x = intraday(bars.get(r["symbol"], spy_b.iloc[0:0]), r["t"], ses["open_t"], entry, c["prev_c"], c["dv20"],
                     c["iex_share"])
        sp = intraday(spy_b, r["t"], ses["open_t"], trades["SPY"][0], market["spy_prev_c"], 0, np.nan)
        if not (np.isfinite(x["r_open"]) and np.isfinite(sp["r_open"])) and now < r["t"] + S.MAX_WAIT - pd.Timedelta(
                seconds=POLL_S):
            continue        # the price at t isn't published yet (e.g. the open's first IEX bar): next read
        inp = {k: r[k] for k in N.NEWS_INPUTS} | {k: c[k] for k in N.STOCK_INPUTS}
        inp |= {"spy_r1": market["spy_r1"], "spy_r5": market["spy_r5"], "spy_open": sp["r_open"], **x}
        inp["novelty"] = float(N.novelty(pd.DataFrame({"symbol": [r["symbol"]], "d": [r["d"]]}), nd)[0])
        e = float(effect(np.array([[inp[k] for k in S.INPUTS]], dtype=float))[0])
        fired = bool(abs(e) >= day.ctx["cut"] and abs(e) > 0)
        side, why = take(e, fired, r["symbol"], short_ok, day.taken())
        day.log(base | {"status": "decided", "inputs": inp, "e": e, "fired": fired, "side": side, "skip": why,
                        "entry_iex": entry, "entry_iex_t": trades[r["symbol"]][1], "prev_c": c["prev_c"],
                        "dv20": c["dv20"], "etb": r["symbol"] in short_ok})


def close_day(root, day, d):
    """After the close: control and the day record (ledger kind nx3_day)."""
    led = Ledger(root)
    day_s = f"{d:%Y-%m-%d}"
    if led.has("nx3_day", day_s):
        return None
    dec = [x for x in day.decisions if x["status"] == "decided"]
    picks = [x for x in dec if x.get("side")]
    ctrl = control(dec, picks, day_seed(d))
    slim = ["symbol", "t", "side", "e", "prev_c", "dv20"]
    rec = {"day": day_s, "model_sha256": json.loads(META.read_text())["sha256"]["news"], "fire_cut": day.ctx["cut"],
           "events": day.decisions, "picks": [{k: p.get(k) for k in slim} for p in picks],
           "control": [{k: p.get(k) for k in slim} for p in ctrl],
           "counts": {"events": len(day.decisions), "decided": len(dec), "fired": sum(x["fired"] for x in dec),
                      "picks": len(picks), "missed": sum(x["status"] == "missed" for x in day.decisions)}}
    out = led.write_json("nx3_day", day_s, f"nx3/days/{day_s}.json", json.loads(json.dumps(rec, default=_js)))
    try:    # the NXM books (research/prereg/NXM.md) read this day's logged inputs; they never stop NX3
        M.close_day(root, day.decisions, d, day_seed(d))
    except Exception as ex:
        print(f"[nxm] day not logged: {type(ex).__name__}: {ex}")
    return out


def settle(root, api, cal, today):
    """Settles every logged day whose exit session has closed (nx3_settle), oldest first."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3_day"]:
        d = pd.Timestamp(e["date"])
        if led.has("nx3_settle", e["date"]):
            continue
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, role=role) for role in ("picks", "control") for p in rec[role]]
        res = {"day": e["date"], "trades": [], "pick_minus_control": None}
        if e["date"] in BUG_DAYS:
            res["excluded"] = "bug days, left out of the forward test (research/prereg/NX3.md, 2026-10-28)"
        if rows:
            ev = pd.DataFrame(rows).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                res["trades"] = t[["symbol", "role", "side", "t", "entry_t", "entry", "exit_ac", "ret", "spy_ret",
                                   "mret", "price", "days", "net"]].to_dict("records")
                p, c = t[t["role"] == "picks"], t[t["role"] == "control"]
                if len(p) and len(c):
                    res["pick_minus_control"] = float(p["net"].mean() - c["net"].mean())
                res["book_pnl"] = float((p["net"] * N.NOTIONAL).sum())
        out.append(led.write_json("nx3_settle", e["date"], f"nx3/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=_js))))
    try:
        M.settle(root, api, cal, today)
    except Exception as ex:
        print(f"[nxm] settle failed: {type(ex).__name__}: {ex}")
    return out


def run(state, minutes, api=None, clock=None, sleep=time.sleep, broker=O.broker):
    """Polls until the session's last decision, then closes and settles the day. 3 = nothing more today."""
    root = Path(state) / "nx3"
    root.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists():
        print("[nx3] model v1 missing; nothing to do")
        return 3
    api = api or Api()
    clock = clock or (lambda: alpaca.utcnow().tz_convert(ET).tz_localize(None))
    stop = time.monotonic() + minutes * 60
    now = clock()
    cal = api.calendar((now - pd.Timedelta(days=LOOKBACK_DAYS + 10)).date(), (now + pd.Timedelta(days=10)).date())
    cal["date"] = pd.to_datetime(cal["date"])
    d = now.normalize()
    ses = session_row(cal, d)
    if ses is None:
        settle(root, api, cal, d - pd.Timedelta(days=1))
        return 3
    if now >= ses["close_t"] + SETTLE_AFTER:
        day = Day(root, d)
        if day.ctx:
            close_day(root, day, d)
            if br := broker():
                O.guarded(O.log_day, root, day, d, br)
        settle(root, api, cal, d)
        return 3
    if now < ses["open_t"] - pd.Timedelta(minutes=5):
        while time.monotonic() < stop and clock() < ses["open_t"] - pd.Timedelta(minutes=5):
            sleep(POLL_S)
        if clock() < ses["open_t"] - pd.Timedelta(minutes=5):
            return 0
    day = Day(root, d)
    meta = json.loads(META.read_text())
    day.ctx.setdefault("cut", float(meta["fire_cut"]))
    a = api.assets()
    listed = set(a["symbol"])
    short_ok = set(a.loc[a["shortable"].fillna(False).astype(bool) & a["easy_to_borrow"].fillna(False).astype(bool),
                         "symbol"])
    nd = news_days(root, api, d, cal)
    fb = root / "finbert_meta.json"
    if not fb.exists():
        fb.write_text(json.dumps(json.loads(NX1_META.read_text())["finbert"]))
    finbert, effect, last = api.finbert(fb), api.effect(), recent_heads(root)
    br = broker()           # EL-NX3 paper orders (nx3_orders); None without its secrets
    ctx, market = {}, None
    while time.monotonic() < stop:
        now = clock()
        if now >= ses["close_t"] - S.LAST_T + S.MAX_WAIT:
            break
        start = time.monotonic()
        ev = poll(day, api, ses, now, finbert, last)
        ready, missed = due(ev, day.decided(), now)
        for r in missed.to_dict("records"):
            day.log({"key": r["key"], "symbol": r["symbol"], "d": r["d"], "t": r["t"], "status": "missed",
                     "decided_at": now})
        if len(ready):
            new = sorted(set(ready["symbol"]) - set(ctx) - set(day.ctx.get("no_ctx", [])))
            # only symbols Alpaca lists: one foreign symbol (TSX:CAE, 002594) made it refuse the whole request
            ask = [s for s in new if s in listed]
            if ask or market is None:
                c, market = stock_context(api, ask, ses["prev"], cal)
                ctx |= c
            day.ctx["no_ctx"] = sorted(set(day.ctx.get("no_ctx", [])) | (set(new) - set(ctx)))
            decide(day, api, ses, ready, ctx, market, nd, short_ok, effect, now)
        if br and now >= ses["open_t"]:
            O.guarded(O.send_exits, day, d, br)
            O.guarded(O.send_entries, day, d, br)
        day.save()
        sleep(max(0.0, POLL_S - (time.monotonic() - start)))
    day.save()
    while time.monotonic() < stop and clock() < ses["close_t"] + SETTLE_AFTER:
        sleep(POLL_S)
    if clock() >= ses["close_t"] + SETTLE_AFTER:
        close_day(root, day, d)
        if br:
            O.guarded(O.log_day, root, day, d, br)
        settle(root, api, cal, d)
        return 3
    return 0


def check(api=None):
    """Every network piece once on the last session (no ledger writes): news, FinBERT, IEX bars and trades,
    the stock context and one effect. Prints counts only."""
    api = api or Api()
    now = alpaca.utcnow().tz_convert(ET).tz_localize(None)
    cal = api.calendar((now - pd.Timedelta(days=LOOKBACK_DAYS + 10)).date(), now.date())
    cal["date"] = pd.to_datetime(cal["date"])
    ses = S.sessions(cal)
    ses = ses[ses["close_t"] <= now].iloc[-1]
    news = api.news(pd.Timestamp(ses["prev_close_t"], tz=ET).tz_convert("UTC"),
                    pd.Timestamp(ses["open_t"], tz=ET).tz_convert("UTC"))
    fb = Path("/tmp/nx3_finbert_meta.json")
    fb.write_text(json.dumps(json.loads(NX1_META.read_text())["finbert"]))
    rows = N.headline_rows(news.head(200), api.finbert(fb))
    ev = S.session_events(N.dedup(rows), pd.DataFrame([ses]))
    print(f"[check] {ses['date']:%Y-%m-%d}: {len(news)} pre-open articles, {len(rows)} single-symbol rows, "
          f"{len(ev)} events (first 200 articles)")
    listed = set(api.assets()["symbol"])
    syms = sorted(s for s in ev["symbol"].unique() if s in listed)[:5]
    ctx, market = stock_context(api, syms, ses["prev"], cal)
    print(f"[check] context for {len(ctx)} of {len(syms)}; in universe {sum(c['uni'] for c in ctx.values())}; "
          f"IEX shares {[round(c['iex_share'], 4) for c in ctx.values()]}; market {market}")
    b = api.iex_bars(syms + ["SPY"], ses["open_t"], ses["open_t"] + pd.Timedelta(minutes=15), "1Min")
    tr = api.last_trades(syms + ["SPY"])
    print(f"[check] IEX minute bars for {len(b)} symbols ({sum(len(x) for x in b.values())} bars); latest trades {len(tr)}")
    effect = api.effect()
    if len(ctx):
        s0 = next(iter(ctx))
        r = ev[ev["symbol"] == s0].iloc[0]
        x = intraday(b.get(s0, pd.DataFrame(columns=["t", "o", "c", "v"])), r["t"], ses["open_t"], tr.get(s0, (np.nan,))[0],
                     ctx[s0]["prev_c"], ctx[s0]["dv20"], ctx[s0]["iex_share"])
        inp = {k: r[k] for k in N.NEWS_INPUTS} | {k: ctx[s0][k] for k in N.STOCK_INPUTS} | x
        inp |= {"spy_r1": market["spy_r1"], "spy_r5": market["spy_r5"], "spy_open": 0.0, "novelty": 0.0}
        print(f"[check] effect for one event: {float(effect(np.array([[inp[k] for k in S.INPUTS]], float))[0]):.4f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    return check() if a.check else run(a.state, a.minutes)


if __name__ == "__main__":
    sys.exit(main())
