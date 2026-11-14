"""PICK-MAIN and PICK-PENNY (research/prereg/PICK.md): the decision rows, built from 5-minute SIP bars.

Stage "rows", month by month from 2018-12 (warm-up for the 20-session swing) to 2025-09:
- the day's universe from W1's daily cache (`tracks/news/w1_data.py`), using the PREVIOUS session's raw close and
  20-day median dollar volume: main = close >= $5 and dv20 >= $5M, penny = $1 <= close < $5 and dv20 >= $1M;
- the day's 5-minute bars 09:30-16:00 ET (feed=sip, adjustment=raw) of those symbols and SPY;
- at each decision time 09:35 .. 14:00 (every 5 minutes, bars complete before it), the chart inputs, and for each
  candidate stop width the trade result: buy at the open of the bar starting at the decision time, sell at the
  take profit (+8% main, +15% penny), the stop (width x the stock's usual 30-minute swing below the fill), or
  the close of the 15:50 bar. A bar that touches both counts as the stop (the cautious reading).
- kept: a fixed 3% sample of decision rows (hash of symbol, day and time), for fitting and validation. The
  frozen models score every row of the test period in a later pass.

The bars themselves are not kept. Each month is written atomically to <cache>/pick/rows/<YYYY-MM>.parquet with
the swing state needed to resume. News inputs are joined later from NX1's headline cache.

Usage: python -m tracks.main.pick_data --stage rows --cache cache/w1 --budget-min 330
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.news import w1_data as W

ET = "America/New_York"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
THREADS = 4
CHUNK = 100                     # symbols per request: 100 x 78 five-minute bars fit one 10,000-bar page
FIRST, LAST = "2018-12", "2025-09"
TRACKS = {"main": {"lo": 5.0, "hi": np.inf, "dv": 5e6, "tp": 0.08, "cost": 0.002, "widths": (1.0, 1.5, 2.0)},
          "penny": {"lo": 1.0, "hi": 5.0, "dv": 1e6, "tp": 0.15, "cost": 0.02, "widths": (1.5, 2.0, 3.0)}}
DEC_FIRST, DEC_LAST, EXIT_BAR = "09:35", "14:00", "15:50"
SAMPLE = 0.03
SWING_N = 20                    # sessions in the usual 30-minute swing


def pdir(cache):
    return Path(cache) / "pick"


def atomic(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


def universe(d):
    """{symbol: (track, prev close, dv20)} for the next session, from W1 rows of one date (the previous session)."""
    out = {}
    for r in d.itertuples(index=False):
        for tr, p in TRACKS.items():
            if p["lo"] <= r.c < p["hi"] and r.dv20 >= p["dv"]:
                out[r.symbol] = (tr, float(r.c), float(r.dv20))
    return out


def bars_5min(symbols, day, headers):
    """{symbol: DataFrame(t naive ET bar start, o, h, l, c, v)} of the regular session."""
    s, e = (pd.Timestamp(f"{day.date()} {x}", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            for x in ("09:30", "16:00"))
    def chunk(syms):
        got, tok = {}, None
        while True:
            p = {"symbols": ",".join(syms), "timeframe": "5Min", "start": s, "end": e,
                 "feed": "sip", "adjustment": "raw", "limit": 10000}
            if tok:
                p["page_token"] = tok
            j = alpaca.get(BARS_URL, p, headers=headers)
            for sym, bs in (j.get("bars") or {}).items():
                got.setdefault(sym, []).extend((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bs)
            tok = j.get("next_page_token")
            if not tok:
                return got

    rows = {}
    with ThreadPoolExecutor(THREADS) as ex:      # 4 at a time stays under Alpaca's 200 requests a minute
        for got in ex.map(chunk, [symbols[i:i + CHUNK] for i in range(0, len(symbols), CHUNK)]):
            for sym, r in got.items():
                rows.setdefault(sym, []).extend(r)
    out = {}
    for sym, r in rows.items():
        d = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v"])
        d = d[d["v"] > 0]
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        out[sym] = d.sort_values("t").reset_index(drop=True)
    return out


def swing30(b):
    """The day's median 30-minute range (high - low over 6 bars) as a share of the price; NaN if too short."""
    if len(b) < 12:
        return np.nan
    h = b["h"].rolling(6).max().to_numpy()[5::6]
    lo = b["l"].rolling(6).min().to_numpy()[5::6]
    c = b["c"].to_numpy()[5::6]
    return float(np.nanmedian((h - lo) / c))


def keep(sym, day, t):
    """The fixed sample: the same rows on every run."""
    h = hashlib.sha256(f"{sym}|{day.date()}|{t}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < SAMPLE


def trade(b, k, tp, stop_frac, cost):
    """Net result of a buy at the open of bar k: (net, exit kind). Exit at +tp, the stop, or the 15:50 bar's close."""
    entry = float(b["o"].iat[k])
    if not entry > 0:
        return np.nan, None
    stop, target = entry * (1 - stop_frac), entry * (1 + tp)
    last = b.index[b["t"].dt.strftime("%H:%M") <= EXIT_BAR]
    if not len(last) or last[-1] < k:
        return np.nan, None
    for i in range(k, last[-1] + 1):
        o, h, lo = float(b["o"].iat[i]), float(b["h"].iat[i]), float(b["l"].iat[i])
        if i > k and o <= stop:
            return o / entry - 1 - cost, "stop"
        if lo <= stop:
            return stop / entry - 1 - cost, "stop"
        if i > k and o >= target:
            return o / entry - 1 - cost, "tp"
        if h >= target:
            return tp - cost, "tp"
    return float(b["c"].iat[last[-1]]) / entry - 1 - cost, "close"


def inputs(b, k, prev_c, dv20, swing, spy):
    """Chart inputs at the decision time (bar k starts at it; bars before k are complete)."""
    a = b.iloc[:k]
    px = float(a["c"].iat[-1])
    o0 = float(a["o"].iat[0])
    vol = float((a["c"] * a["v"]).sum())
    mins = len(a) * 5
    r = lambda n: px / float(a["c"].iat[-n - 1]) - 1 if len(a) > n else np.nan     # noqa: E731
    sp = spy.iloc[:k] if spy is not None and len(spy) >= k else None
    return {"mins_open": mins, "gap": o0 / prev_c - 1, "r_open": px / o0 - 1, "r_prev": px / prev_c - 1,
            "r_5m": r(1), "r_15m": r(3), "r_30m": r(6), "hi_dist": px / float(a["h"].max()) - 1,
            "lo_dist": px / float(a["l"].min()) - 1, "range_open": float(a["h"].max() / a["l"].min() - 1),
            "vol_pace": vol / (dv20 * mins / 390) if dv20 > 0 else np.nan, "swing30": swing,
            "log_px": float(np.log(px)), "log_dv": float(np.log(dv20)),
            "spy_open": float(sp["c"].iat[-1] / sp["o"].iat[0] - 1) if sp is not None and len(sp) else np.nan}


def day_rows(day, uni, bars, swings):
    spy = bars.get("SPY")
    out = []
    for sym, (tr, prev_c, dv20) in uni.items():
        b = bars.get(sym)
        hist = swings.get(sym, [])
        sw = float(np.mean(hist)) if len(hist) >= 10 else np.nan
        if b is None or len(b) < 2 or not np.isfinite(sw) or b["t"].iat[0].strftime("%H:%M") != "09:30":
            continue                    # no 9:30 bar: no clean open for the gap and the inputs
        p = TRACKS[tr]
        hm = b["t"].dt.strftime("%H:%M")
        for k in np.flatnonzero((hm >= DEC_FIRST) & (hm <= DEC_LAST)):     # k >= 1: the 9:30 bar comes first
            t = b["t"].iat[k]
            if not keep(sym, day, t.strftime("%H:%M")):
                continue
            row = {"symbol": sym, "d": day, "t": t, "track": tr} | inputs(b, k, prev_c, dv20, sw, spy)
            for w in p["widths"]:
                net, kind = trade(b, k, p["tp"], w * sw, p["cost"])
                row[f"net_{w}"], row[f"exit_{w}"] = net, kind
            out.append(row)
    return out


def stage_rows(cache, deadline, cal, last=LAST):
    daily = W.read_rows(cache, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", pd.Timestamp(f"{FIRST}-01") - pd.Timedelta(days=10)), ("c", ">=", 1.0),
                         ("dv20", ">=", 1e6)])
    by_day = {d: g for d, g in daily.groupby("date")}
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    state = pdir(cache) / "swings.json"
    swings = {k: v for k, v in json.loads(state.read_text()).items()} if state.exists() else {}
    h = alpaca_headers()
    for m in pd.period_range(FIRST, last, freq="M"):
        f = pdir(cache) / "rows" / f"{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        out = []
        for day in dates[(dates >= m.start_time) & (dates <= m.end_time)]:
            i = dates.get_loc(day)
            if i == 0:
                continue
            uni = universe(by_day.get(dates[i - 1], daily.iloc[0:0]))
            bars = bars_5min(sorted(uni) + ["SPY"], day, h)
            if str(m) != FIRST:
                out += day_rows(day, uni, bars, swings)
            for sym, b in bars.items():
                s = swing30(b)
                if np.isfinite(s):
                    swings[sym] = (swings.get(sym, []) + [s])[-SWING_N:]
        atomic(pd.DataFrame(out), f)
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps(swings))
        tmp.replace(state)
        print(f"[pick rows] {m}: {len(out)} sampled decision rows", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="rows", choices=["rows"])
    ap.add_argument("--cache", default="cache/w1")
    ap.add_argument("--budget-min", type=float, default=330)
    ap.add_argument("--last", default=LAST, help="last month to build (YYYY-MM)")
    a = ap.parse_args()
    deadline = time.monotonic() + 60 * a.budget_min
    cal = alpaca.trading_calendar(pd.Timestamp(f"{FIRST}-01") - pd.Timedelta(days=10), pd.Timestamp("2025-10-01"))
    done = stage_rows(a.cache, deadline, cal, a.last)
    print("done" if done else "stopped on the time budget; rerun to continue", flush=True)


if __name__ == "__main__":
    main()
