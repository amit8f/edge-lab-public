"""HI3 (research/prereg/HI3.md): LightGBM, CatBoost and Ensemble news models, plain and high bar, on 2026 data.

Stages (cache under <cache>, results under <out>):
  build   replay sessions: Benzinga news from Alpaca (2026-04 on), headline rows and events as nx3_live.poll makes
          them, inputs as nx3_live.decide makes them (SIP minute bars in place of IEX) -> one decided-event file a day
  live    the decided events NX3's live job logged (state-nx3 ledger files nx3/days/<d>.json) from 2026-10-29
  report  every arm's picks and control per day, settled with NX3's training code -> <out>/HI3.json

    python -m tracks.news.hi3_replay --stage build|live|report --cache cache/hi3 --out research/results/HI3 \
        [--state <state-nx3 checkout>]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main.longterm import news_history as H
from tracks.main.longterm.stats import nw_tstat
from tracks.news import nx1_study as N
from tracks.news import nx3_live as L
from tracks.news import nx3_study as S
from tracks.news import nxm_live as M

ET = S.ET
FIRST_MONTH, LIVE_FIRST, REPLAY_LAST = "2026-04", "2026-10-29", pd.Timestamp("2026-10-28")
MODELS = ("LGBM", "CAT", "ENS", "XGB", "LIN", "MLP", "XGBR", "RFR", "KRR")     # Amendment 1 added the last six
HI = 1.5
ARMS = [(m, v) for m in MODELS for v in ("plain", "HI")]     # arm numbers 1..18 in this order
MIN_DAYS_IN_MONTH = 0.9      # a month is "full" when articles appear on >= 90% of its weekdays
NOVELTY_GAP = 31


# ---------- pure helpers (tested) ----------

def news_start(month_days):
    """{month 'YYYY-MM': (days with articles, weekdays)} -> the first month from FIRST_MONTH on with articles on
    nearly every weekday, or None."""
    for m in sorted(month_days):
        if m < FIRST_MONTH:
            continue
        have, weekdays = month_days[m]
        if weekdays and have >= MIN_DAYS_IN_MONTH * weekdays:
            return m
    return None


def price_at(bars, t):
    """Close of the last SIP bar starting at or before t (the price an entry at t + 1 minute sees), or NaN."""
    if bars is None or len(bars) == 0:
        return np.nan
    b = bars[bars["t"] <= t]
    return float(b["c"].iloc[-1]) if len(b) else np.nan


def arm_book(decided, model, seed, hi):
    """One arm's picks and control (nxm_live.book's rule; with hi, an event only takes a slot when
    |e| >= HI x cut)."""
    dec = sorted(decided, key=lambda x: (pd.Timestamp(x["t"]), x["symbol"]))
    if not dec:
        return [], []
    X = np.array([[x["inputs"][k] if x["inputs"][k] is not None else np.nan for k in S.INPUTS] for x in dec], float)
    e = model.effect(X)
    bar = model.cut * (HI if hi else 1.0)
    picks, n = [], {1: 0, -1: 0}
    for x, ei in zip(dec, e):
        if not (abs(ei) >= bar and abs(ei) > 0):
            continue
        side = 1 if ei > 0 else -1
        if n[side] >= S.PICKS or (side < 0 and not x.get("etb", True)):
            continue
        n[side] += 1
        picks.append(dict(x, side=side, e=float(ei)))
    ctrl = []
    if picks and len(dec) >= len(picks):
        k = np.random.default_rng(seed).choice(len(dec), size=len(picks), replace=False)
        ctrl = [dict(dec[i], side=p["side"]) for i, p in zip(k, picks)]
    return picks, ctrl


def summarize(tr, days):
    """Settled trades of one arm (day, role, side, net) over the period's sessions -> the reported numbers."""
    p, c = tr[tr["role"] == "picks"], tr[tr["role"] == "control"]
    out = {"sessions": len(days), "trades": len(p)}
    if p.empty:
        return out | {"days_with_picks": 0}
    dp, dc = p.groupby("day")["net"].mean(), c.groupby("day")["net"].mean()
    diff = (dp - dc).dropna()
    m, _se, t = nw_tstat(diff.to_numpy(), 5) if len(diff) > 2 else (float(diff.mean()) if len(diff) else None, None, None)
    gain = p["net"] * N.NOTIONAL
    top5 = gain.sort_values(ascending=False).head(5).sum()
    out |= {"days_with_picks": int(dp.size), "pick_minus_control": m, "t": t, "passes": bool(m is not None and t is not None
            and m > 0 and t >= 1.5), "win": float((p["net"] > 0).mean()), "net_per_trade": float(p["net"].mean()),
            "pnl_usd": float(gain.sum()), "pnl_pct_of_5000": float(gain.sum() / 5000 * 100),
            "top5_share": float(top5 / gain.sum()) if gain.sum() > 0 else None,
            "longs": int((p["side"] > 0).sum()), "shorts": int((p["side"] < 0).sum()),
            "long_net": float(p.loc[p["side"] > 0, "net"].mean()) if (p["side"] > 0).any() else None,
            "short_net": float(p.loc[p["side"] < 0, "net"].mean()) if (p["side"] < 0).any() else None}
    return out


# ---------- build (replay) ----------

class SipApi(L.Api):
    """nx3_live's Api with SIP in place of IEX for the intraday inputs (history: IEX's own share is 1)."""

    def iex_bars(self, symbols, start, end, timeframe):
        out = {}
        s, e = (pd.Timestamp(x, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") for x in (start, end))
        for i in range(0, len(symbols), S.SYM_CHUNK):
            tok = None
            while True:
                p = {"symbols": ",".join(symbols[i:i + S.SYM_CHUNK]), "timeframe": timeframe, "start": s, "end": e,
                     "feed": "sip", "adjustment": "raw", "limit": 10000}
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


def months_news(cache, deadline):
    """Every month from FIRST_MONTH to now -> cache/news/news_<m>.parquet; {month: (days with news, weekdays)}."""
    out = Path(cache) / "news"
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    info = {}
    for m in pd.period_range(FIRST_MONTH, now.strftime("%Y-%m"), freq="M"):
        f = out / f"news_{m}.parquet"
        done = f.exists() and m.end_time.tz_localize("UTC") < now - pd.Timedelta(days=1)
        if not done:
            if time.monotonic() > deadline:
                return None
            N.atomic_parquet(H.fetch_month(str(m)), f)
        df = pd.read_parquet(f, columns=["created_at"])
        have = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET).dt.normalize().nunique() if len(df) else 0
        wd = pd.bdate_range(m.start_time, min(m.end_time, now.tz_localize(None))).size
        info[str(m)] = (int(have), int(wd))
        print(f"[hi3 news] {m}: {len(df)} articles, news on {have} of {wd} weekdays", flush=True)
    return info


def headline_file(cache, info, deadline):
    """All months' deduplicated headline rows (FinBERT at NX1's pinned commit) -> cache/rows.parquet."""
    f = Path(cache) / "rows.parquet"
    if f.exists():
        return pd.read_parquet(f)
    fb = Path(cache) / "finbert_meta.json"
    if not fb.exists():
        fb.write_text(json.dumps(json.loads(L.NX1_META.read_text())["finbert"]))
    finbert = N.finbert_scorer(fb)
    parts = []
    for m in sorted(info):
        g = Path(cache) / f"rows_{m}.parquet"
        if not g.exists():
            if time.monotonic() > deadline:
                return None
            news = pd.read_parquet(Path(cache) / "news" / f"news_{m}.parquet")
            N.atomic_parquet(N.headline_rows(news, finbert), g)
            print(f"[hi3 rows] {m} scored", flush=True)
        parts.append(pd.read_parquet(g))
    rows = N.dedup(pd.concat(parts, ignore_index=True))
    N.atomic_parquet(rows, f)
    return rows


def news_days_all(cache, info):
    parts = []
    for m in sorted(info):
        n = pd.read_parquet(Path(cache) / "news" / f"news_{m}.parquet", columns=["created_at", "symbols"])
        n = n[n["symbols"].fillna("") != ""]
        parts.append(pd.DataFrame({"date": S.W.et_time(n["created_at"]).dt.normalize(),
                                   "symbol": n["symbols"].str.split(",")}).explode("symbol"))
    nd = pd.concat(parts, ignore_index=True).drop_duplicates()
    nd["date"] = pd.to_datetime(nd["date"])
    return nd


def replay_day(api, cal, ses, ev, nd):
    """The day's decided events (nx3_live.decide's records, inputs from SIP)."""
    ctx, market = L.stock_context(api, sorted(set(ev["symbol"])), ses["prev"], cal)
    syms = sorted(set(ev["symbol"]) & set(ctx))
    bars = S.day_bars(syms + ["SPY"], ses["date"], ses["open_t"], ses["close_t"], api.h) if syms else {}
    spy_b = bars.get("SPY", pd.DataFrame(columns=["t", "o", "c", "v"]))
    out = []
    for r in ev.sort_values(["t", "symbol"], kind="stable").to_dict("records"):
        base = {"key": f"{r['symbol']}|{r['d']:%Y-%m-%d}", "symbol": r["symbol"], "d": r["d"], "t": r["t"]}
        c = ctx.get(r["symbol"])
        if c is None or not c["uni"]:
            out.append(base | {"status": "not in universe"})
            continue
        b = bars.get(r["symbol"])
        entry, spy_px = price_at(b, r["t"]), price_at(spy_b, r["t"])
        if not (np.isfinite(entry) and np.isfinite(spy_px)):
            out.append(base | {"status": "no SIP trade"})
            continue
        x = L.intraday(b, r["t"], ses["open_t"], entry, c["prev_c"], c["dv20"], 1.0)
        sp = L.intraday(spy_b, r["t"], ses["open_t"], spy_px, market["spy_prev_c"], 0, np.nan)
        inp = {k: r[k] for k in N.NEWS_INPUTS} | {k: c[k] for k in N.STOCK_INPUTS}
        inp |= {"spy_r1": market["spy_r1"], "spy_r5": market["spy_r5"], "spy_open": sp["r_open"], **x}
        inp["novelty"] = float(N.novelty(pd.DataFrame({"symbol": [r["symbol"]], "d": [r["d"]]}), nd)[0])
        inp = {k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in inp.items()}
        out.append(base | {"status": "decided", "inputs": inp, "prev_c": c["prev_c"], "dv20": c["dv20"], "etb": True})
    return out


def stage_build(cache, budget_min):
    deadline = time.monotonic() + budget_min * 60
    info = months_news(cache, deadline)
    if info is None:
        return False
    start = news_start(info)
    meta = {"months": info, "news_start": start}
    (Path(cache) / "window.json").write_text(json.dumps(meta, indent=1))
    if start is None:
        raise SystemExit("[hi3] no full news month from 2026-04 on")
    rows = headline_file(cache, info, deadline)
    if rows is None:
        return False
    api = SipApi()
    cal = api.calendar(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2026-12-31").date())
    cal["date"] = pd.to_datetime(cal["date"])
    ses = S.sessions(cal)
    first = pd.Timestamp(f"{start}-01") + pd.Timedelta(days=NOVELTY_GAP)
    days = ses[(ses["date"] >= first) & (ses["date"] <= REPLAY_LAST)]
    print(f"[hi3] news from {start}; replay {days['date'].min():%Y-%m-%d} .. {days['date'].max():%Y-%m-%d}, "
          f"{len(days)} sessions", flush=True)
    nd = news_days_all(cache, info)
    ev_all = S.session_events(rows, days.reset_index(drop=True))
    out = Path(cache) / "days"
    out.mkdir(parents=True, exist_ok=True)
    for s in days.to_dict("records"):
        f = out / f"{s['date']:%Y-%m-%d}.json"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        ev = ev_all[ev_all["d"] == s["date"]]
        dec = replay_day(api, cal, s, ev, nd) if len(ev) else []
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": f"{s['date']:%Y-%m-%d}", "part": "replay", "events": dec}, default=L._js))
        tmp.replace(f)
        print(f"[hi3 day] {s['date']:%Y-%m-%d}: {len(ev)} events, "
              f"{sum(x['status'] == 'decided' for x in dec)} decided", flush=True)
    return True


# ---------- live ----------

def stage_live(cache, state):
    out = Path(cache) / "days"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((Path(state) / "nx3" / "nx3" / "days").glob("*.json")):
        day = f.stem
        if day < LIVE_FIRST or day in L.BUG_DAYS:
            continue
        rec = json.loads(f.read_text())
        dec = [x for x in rec.get("events", []) if x.get("status") == "decided"]
        g = out / f"{day}.json"
        g.write_text(json.dumps({"day": day, "part": "live", "events": dec}, default=L._js))
        n += 1
    print(f"[hi3 live] {n} logged sessions", flush=True)
    return True


# ---------- report ----------

class Xgb:
    """NX3's own version 1 XGBoost news model with its frozen fire cut (nx3_live.Api.effect)."""

    def __init__(self):
        self.cut = float(json.loads(L.META.read_text())["fire_cut"])
        self._e = L.Api.effect(None)

    def effect(self, X):
        return self._e(X)


def stage_report(cache, out, parts=("replay", "live"), name="HI3"):
    """Settle every day file's arms; parts[0] is the deciding part (HI3K reuses this with gap/overlap)."""
    models = M.load() | {"XGB": Xgb()}
    api = L.Api()
    cal = api.calendar(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2026-12-31").date())
    cal["date"] = pd.to_datetime(cal["date"])
    caldates = pd.DatetimeIndex(cal["date"])
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    sdir = Path(cache) / "settled"
    sdir.mkdir(parents=True, exist_ok=True)
    trades, days = [], {p: [] for p in parts}
    for f in sorted((Path(cache) / "days").glob("*.json")):
        rec = json.loads(f.read_text())
        d = pd.Timestamp(rec["day"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] >= today:
            continue
        days[rec["part"]].append(rec["day"])
        g = sdir / f"{rec['day']}.parquet"
        if not g.exists():
            dec = [x for x in rec["events"] if x.get("status") == "decided"]
            rows = []
            for i, (m, v) in enumerate(ARMS, start=1):
                p, c = arm_book(dec, models[m], L.day_seed(d) + i, v == "HI")
                rows += [dict(x, arm=f"{m}-{v}", role=role) for role, xs in (("picks", p), ("control", c)) for x in xs]
            t = pd.DataFrame()
            if rows:
                ev = pd.DataFrame([{k: r.get(k) for k in ("symbol", "t", "side", "prev_c", "dv20", "arm", "role")}
                                   for r in rows]).assign(d=d, next=caldates[k + 1])
                ev["t"] = pd.to_datetime(ev["t"])
                t = api.settle_trades(ev, cal)
                if len(t):
                    t["net"] = N.net(t)
                    t = t[["symbol", "arm", "role", "side", "net"]].assign(day=rec["day"], part=rec["part"])
            N.atomic_parquet(t, g)
            print(f"[hi3 settle] {rec['day']} ({rec['part']}): {len(t)} trades", flush=True)
        t = pd.read_parquet(g)
        if len(t):
            trades.append(t)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["arm", "part", "role", "day", "side", "net"])
    N.atomic_parquet(tr, Path(cache) / "trades.parquet")
    res = {"window": json.loads((Path(cache) / "window.json").read_text()) if (Path(cache) / "window.json").exists()
           else None, "sessions": {k: [v[0], v[-1], len(v)] if v else None for k, v in days.items()}, "arms": {}}
    for m, v in ARMS:
        a = f"{m}-{v}"
        res["arms"][a] = {part: summarize(tr[(tr["arm"] == a) & (tr["part"] == part)], days[part])
                          for part in parts}
        res["arms"][a]["all"] = summarize(tr[tr["arm"] == a], [d for p in parts for d in days[p]])
    for m in MODELS:     # HI minus plain on shared pick days (replay), reported
        tp, th = (tr[(tr["arm"] == f"{m}-{v}") & (tr["role"] == "picks") & (tr["part"] == parts[0])]
                  .groupby("day")["net"].mean() for v in ("plain", "HI"))
        diff = (th - tp).dropna()
        res["arms"][f"{m}-HI"][f"hi_minus_plain_{parts[0]}"] = (
            dict(zip(("mean", "se", "t"), nw_tstat(diff.to_numpy(), 5))) if len(diff) > 2 else None)
    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = Path(out) / f"{name}.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(Path(out) / f"{name}.json")
    for a, r in res["arms"].items():
        print(f"[hi3] {a}: " + " | ".join(f"{p} {r[p].get('trades')} tr, pmc {r[p].get('pick_minus_control')}, "
                                          f"t {r[p].get('t')}, win {r[p].get('win')}, pnl {r[p].get('pnl_usd')}"
                                          for p in parts), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "live", "report"])
    ap.add_argument("--cache", default="cache/hi3")
    ap.add_argument("--out", default="research/results/HI3")
    ap.add_argument("--state", default="state-nx3")
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    alpaca_headers()
    if a.stage == "build":
        done = stage_build(a.cache, a.budget_min)
        print(f"[hi3] build {'done' if done else 'not finished (run again)'}", flush=True)
    elif a.stage == "live":
        stage_live(a.cache, a.state)
    else:
        stage_report(a.cache, a.out)


if __name__ == "__main__":
    main()
