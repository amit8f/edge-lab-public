"""C0: what trading really costs at our size (research/prereg/C0.md). A measurement, not an idea.

On 24 random trading days (2025-01-02 .. 2025-09-30), 100 random common stocks per liquidity bucket get
their NBBO quote as of five instants (09:31, 09:35, 10:00, 15:50, 15:59 ET), from Alpaca's SIP quotes.
Half-spread = (ask - bid) / 2 / mid. The same stock-days also get the Corwin-Schultz estimate from daily
highs and lows, so the quote-based numbers can be carried to years without quotes.

Usage: python -m core.costs --cache <dir> --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch

QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
FIRST, LAST = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-09-30")
N_DAYS, PER_BUCKET, SEED = 24, 100, 20261015
INSTANTS = ["09:31", "09:35", "10:00", "15:50", "15:59"]
BUCKETS = [  # (name, min price, max price, min dv20, max dv20); raw prices, 20-day median dollar volume
    ("penny", 1.0, 5.0, 5e5, np.inf),
    ("small", 5.0, np.inf, 2e6, 1e7),
    ("mid", 5.0, np.inf, 1e7, 5e7),
    ("large", 5.0, np.inf, 5e7, np.inf),
]
WINDOWS = [pd.Timedelta(seconds=2), pd.Timedelta(seconds=60)]   # look-back for the quote "as of" an instant


def bucket_of(price, dv20):
    for name, p0, p1, d0, d1 in BUCKETS:
        if p0 <= price < p1 and d0 <= dv20 < d1:
            return name
    return None


def sample(prev, days, seed=SEED):
    """prev: rows (day, symbol, price, dv20) known at the prior close. PER_BUCKET random names per bucket
    per day (fewer if a bucket is smaller)."""
    prev = prev.assign(bucket=[bucket_of(p, d) for p, d in zip(prev["price"], prev["dv20"])]).dropna(subset=["bucket"])
    out = []
    for day in days:
        g = prev[prev["day"] == day]
        rng = np.random.default_rng(seed + pd.Timestamp(day).toordinal())
        for b, gb in g.groupby("bucket", sort=True):
            gb = gb.sort_values("symbol")
            out.append(gb.iloc[rng.choice(len(gb), size=min(PER_BUCKET, len(gb)), replace=False)])
    return pd.concat(out, ignore_index=True)


def as_of(quotes, t):
    """Last valid quote (bid > 0, ask > bid, spread under half the mid) strictly before t, or None."""
    q = [x for x in quotes if pd.Timestamp(x["t"]) < t and x.get("bp", 0) > 0 and x.get("ap", 0) > x.get("bp", 0)
         and (x["ap"] - x["bp"]) / ((x["ap"] + x["bp"]) / 2) < 0.5]
    return max(q, key=lambda x: pd.Timestamp(x["t"])) if q else None


def half_spread(q):
    mid = (q["ap"] + q["bp"]) / 2
    return (q["ap"] - q["bp"]) / 2 / mid


def fetch_quotes(symbols, start, end, headers, pace=0.31):
    """{symbol: [quotes]} in [start, end) for many symbols, all pages (Alpaca free plan: 200 requests/min)."""
    out, token = {}, None
    while True:
        p = {"symbols": ",".join(symbols), "start": start.isoformat(), "end": end.isoformat(), "feed": "sip",
             "limit": 10000}
        if token:
            p["page_token"] = token
        js = alpaca.get(QUOTES_URL, p, headers=headers)
        for s, rows in (js.get("quotes") or {}).items():
            out.setdefault(s, []).extend(rows)
        token = js.get("next_page_token")
        time.sleep(pace)
        if not token:
            return out


def measure_day(day, syms, headers, fetch=fetch_quotes):
    """Rows (day, symbol, instant, half_spread, window_s) for one day's sample."""
    rows = []
    for ins in INSTANTS:
        t = pd.Timestamp(f"{day:%Y-%m-%d} {ins}", tz="America/New_York").tz_convert("UTC")
        todo = list(syms)
        for w in WINDOWS:
            if not todo:
                break
            got = {}
            for i in range(0, len(todo), 100):
                got.update(fetch(todo[i:i + 100], t - w, t, headers))
            left = []
            for s in todo:
                q = as_of(got.get(s, []), t)
                if q is None:
                    left.append(s)
                else:
                    rows.append({"day": day, "symbol": s, "instant": ins, "half_spread": half_spread(q),
                                 "window_s": int(w.total_seconds())})
            todo = left
        rows += [{"day": day, "symbol": s, "instant": ins, "half_spread": np.nan, "window_s": None} for s in todo]
    return rows


def summarize(m, cs):
    """Per bucket and instant: quote coverage and half-spread percentiles (bps); per bucket the round
    trip implied by entering at 09:35 and leaving at 15:59 (median and 75th percentile of the sum), and
    the Corwin-Schultz median for comparison."""
    out = {}
    for b, g in m.groupby("bucket", sort=True):
        piv = g.pivot_table(index=["day", "symbol"], columns="instant", values="half_spread")
        rt = (piv.get("09:35") + piv.get("15:59")).dropna()
        out[b] = {"n_stock_days": int(len(piv)),
                  "instants": {i: {"coverage": float(gi["half_spread"].notna().mean()),
                                   "p50_bps": float(gi["half_spread"].median() * 1e4),
                                   "p75_bps": float(gi["half_spread"].quantile(.75) * 1e4),
                                   "p90_bps": float(gi["half_spread"].quantile(.9) * 1e4)}
                               for i, gi in g.groupby("instant", sort=True)},
                  "round_trip_0935_1559": {"p50": float(rt.median()), "p75": float(rt.quantile(.75)), "n": int(len(rt))},
                  "corwin_schultz_full_spread_p50": float(cs.loc[cs["bucket"] == b, "cs"].median())}
    return out


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    h = alpaca_headers()
    cal = alpaca.trading_calendar(FIRST - pd.Timedelta(days=45), LAST)
    tdays = list(cal["date"])
    pool = [d for d in tdays if FIRST <= d <= LAST]
    days = sorted(pd.Timestamp(d) for d in np.random.default_rng(SEED).choice(pool, N_DAYS, replace=False))
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]))
    prevf = cache / "prev.parquet"
    if not prevf.exists():
        rows = []
        for b in range(0, len(syms), 500):
            fr = to_frames(fetch_daily_bars(syms[b:b + 500], f"{tdays[0]:%Y-%m-%d}", alpaca.sip_end(LAST),
                                            adjustment="raw", headers=h))
            for s, d in fr.items():
                dv = (d["c"] * d["v"]).rolling(20).median()
                cs = pd.Series(np.r_[np.nan, stats.corwin_schultz(d["h"].to_numpy(), d["l"].to_numpy())], index=d.index)
                for day in days:
                    k = d.index.searchsorted(day) - 1          # the prior trading day's bar
                    if k >= 19 and day in d.index and (day - d.index[k]).days <= 15:
                        rows.append({"day": day, "symbol": s, "price": d["c"].iloc[k], "dv20": dv.iloc[k],
                                     "cs": cs.loc[day]})
            print(f"[prev] {min(b + 500, len(syms))}/{len(syms)} symbols")
        tmp = prevf.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp)
        tmp.replace(prevf)
    prev = pd.read_parquet(prevf)
    smp = sample(prev, days)
    meas = []
    for day in days:
        f = cache / f"q_{day:%Y%m%d}.parquet"
        if not f.exists():
            tmp = f.with_suffix(".tmp")
            pd.DataFrame(measure_day(day, list(smp.loc[smp["day"] == day, "symbol"]), h)).to_parquet(tmp)
            tmp.replace(f)
        meas.append(pd.read_parquet(f))
        print(f"[quotes] {day:%Y-%m-%d} done")
    m = pd.concat(meas, ignore_index=True).merge(smp[["day", "symbol", "bucket"]], on=["day", "symbol"])
    res = {"study": "C0", "prereg": "research/prereg/C0.md", "days": [f"{d:%Y-%m-%d}" for d in days],
           "buckets": summarize(m, smp), "current_config": {"main_round_trip": 0.002, "penny_round_trip": 0.02}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "C0_costs.json").write_text(json.dumps(res, indent=1, default=str))
    m.to_csv(out / "C0_quotes.csv", index=False)
    print(json.dumps(res["buckets"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
