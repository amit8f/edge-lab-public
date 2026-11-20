"""S0 (research/prereg/S0.md): how NX3's strong news events move in the first two hours after entry, 2019-2023.

Descriptive only. Refits NX3's walk-forward model per year, takes the real events that fire with confidence >= 1.5
(uncapped) and the day's random control, reads raw SIP 1-minute bars of each session and writes
<out>/S0_paths.json (summaries only).

    python -m tracks.news.s0_study --cache cache/nx1 --out research/results/S0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

YEARS = (2019, 2020, 2021, 2022, 2023)
HI = 1.5
DELAYS = (0, 5, 15, 30)
HORIZONS = (5, 15, 30, 60, 90, 120)
WINDOW = 60
BRACKETS = ((0.02, 0.01), (0.03, 0.015), (0.05, 0.025))


# ---------- pure helpers (tested) ----------

def path_stats(bars, entry_t, side, delay=0):
    """bars: one session's raw 1-minute bars (t naive ET bar start, o, h, l, c). Entry at the open of the first bar at
    or after entry_t + delay. Returns signed returns at HORIZONS, best / worst signed excursion within WINDOW
    minutes, and first-touch outcomes for BRACKETS (1 up first, -1 stop first, 0 neither; a bar touching both is
    the stop), or None without an entry bar."""
    if bars is None or bars.empty:
        return None
    t0 = entry_t + pd.Timedelta(minutes=delay)
    b = bars[bars["t"] >= t0]
    if b.empty or b["t"].iloc[0] > t0 + pd.Timedelta(minutes=5):
        return None
    start, entry = b["t"].iloc[0], float(b["o"].iloc[0])
    if not entry > 0:
        return None
    out = {"entry_t": start}
    for h in HORIZONS:
        w = b[b["t"] < start + pd.Timedelta(minutes=h)]
        out[f"r{h}"] = side * (float(w["c"].iloc[-1]) / entry - 1)
    w = b[b["t"] < start + pd.Timedelta(minutes=WINDOW)]
    up = (w["h"] / entry - 1) if side > 0 else (1 - w["l"] / entry)
    dn = (w["l"] / entry - 1) if side > 0 else (1 - w["h"] / entry)
    out["mfe"], out["mae"] = float(max(up.max(), 0.0)), float(min(dn.min(), 0.0))
    for tp, sl in BRACKETS:
        res = 0
        for u, d in zip(up.to_numpy(), dn.to_numpy(), strict=True):
            if d <= -sl:
                res = -1
                break
            if u >= tp:
                res = 1
                break
        out[f"b{int(tp * 1000)}"] = res
    return out


def summarize(t):
    """t: one row per event and delay (grp, delay, kind, side, r*, mfe, mae, b*) -> nested summary dict."""
    out = {}
    for keys, g in t.groupby(["grp", "delay", "when", "side"]):
        k = "/".join(str(x) for x in keys)
        r = {"n": int(len(g)), "days": int(g["d"].nunique())}
        for h in HORIZONS:
            x = g[f"r{h}"]
            r[f"r{h}"] = {"mean": float(x.mean()), "median": float(x.median()), "pos": float((x > 0).mean())}
        for c in ("mfe", "mae"):
            r[c] = {q: float(g[c].quantile(p)) for q, p in (("p05", .05), ("p25", .25), ("median", .5),
                                                             ("p75", .75), ("p95", .95))}
        for tp, sl in BRACKETS:
            x = g[f"b{int(tp * 1000)}"]
            r[f"tp{tp:.3f}_sl{sl:.3f}"] = {"up_first": float((x == 1).mean()), "stop_first": float((x == -1).mean()),
                                           "neither": float((x == 0).mean())}
        out[k] = r
    return out


# ---------- job ----------

def day_bars_hl(symbols, day, open_t, close_t, headers):
    s = pd.Timestamp(open_t, tz=X.ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = pd.Timestamp(close_t, tz=X.ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = {}
    for i in range(0, len(symbols), X.SYM_CHUNK):
        tok = None
        while True:
            p = {"symbols": ",".join(symbols[i:i + X.SYM_CHUNK]), "timeframe": "1Min", "start": s, "end": e,
                 "feed": "sip", "adjustment": "raw", "limit": 10000}
            if tok:
                p["page_token"] = tok
            j = alpaca.get(X.BARS_URL, p, headers=headers)
            for sym, bs in (j.get("bars") or {}).items():
                raw.setdefault(sym, []).extend((b["t"], b["o"], b["h"], b["l"], b["c"]) for b in bs)
            tok = j.get("next_page_token")
            if not tok:
                break
    res = {}
    for sym, rows in raw.items():
        d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"])
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(X.ET).dt.tz_localize(None)
        res[sym] = d.sort_values("t").reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/S0")
    a = ap.parse_args()
    d3 = X.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    cal = X.calendar(a.cache)
    ses = X.sessions(cal).set_index("date")
    caldates = pd.DatetimeIndex(cal["date"])
    headers = alpaca_headers()
    rows = []
    for yr in YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr, te = ev[ev["d"] <= cut_d], ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = X.Model(tr)
        real = te[te["kind"] == "real"]
        e = model.effect(real[X.INPUTS].to_numpy(float))
        real = real.assign(e=e, fired=model.fires(e))
        rng = np.random.default_rng(X.SEED + yr)
        for d, day in real.groupby("d", sort=True):
            p, c = X.picks(day, rng, cap=None, hi=HI, cut=model.cut)
            if p.empty or d not in ses.index:
                continue
            o_t = ses.loc[d, "open_t"]
            bars = day_bars_hl(sorted(set(p["symbol"]) | set(c["symbol"])), d, o_t, ses.loc[d, "close_t"], headers)
            for grp, fr in (("event", p), ("control", c)):
                for r in fr.itertuples(index=False):
                    when = "preopen" if r.entry_t <= o_t + pd.Timedelta(minutes=2) else "intraday"
                    for dl in DELAYS:
                        x = path_stats(bars.get(r.symbol), r.entry_t, int(r.side), dl)
                        if x is not None:
                            rows.append({"d": d, "year": yr, "grp": grp, "delay": dl, "when": when,
                                         "side": "long" if r.side > 0 else "short", **x})
        print(f"[s0] {yr}: {sum(1 for r in rows if r['year'] == yr)} event-delay rows", flush=True)
    t = pd.DataFrame(rows)
    both = pd.concat([t, t.assign(side="both")], ignore_index=True)
    both = pd.concat([both, both.assign(when="all")], ignore_index=True)
    rep = {"years": list(YEARS), "hi": HI, "delays": list(DELAYS), "horizons": list(HORIZONS), "window": WINDOW,
           "events_per_day": float(t[(t["grp"] == "event") & (t["delay"] == 0)].groupby("d").size().mean()),
           "cells": summarize(both),
           "note": "descriptive, no costs; signed returns are side x price change; keys grp/delay/when/side"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "S0_paths.tmp"
    tmp.write_text(json.dumps(rep, indent=1, default=float))
    tmp.replace(out / "S0_paths.json")
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, default=float))


if __name__ == "__main__":
    main()
