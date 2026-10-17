"""Shared data for W1 (research/prereg/W1.md), cached under <cache>/w1 in 500-symbol batches with atomic
writes:

- daily bars of every common stock named in SEC's insider data sets (the point-in-time ticker -> CIK map),
  2015-06 .. 2025-10: raw open/high/low/close/volume for the universe and the W1b signal, all-adjusted
  open and close for returns, the 20-day median raw dollar volume and the universe flag (L1's rule);
- each universe company's EDGAR filings 2015-10-01 .. 2025-09-30 (forms, filing dates, 8-K items);
- the Benzinga headline months, shared with G1 (<cache>/g1/news), and per-day sets of symbols named.

Nothing here reads a return; the studies do.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch, news_history, sec_data
from tracks.news import l1_study as L1

ET = "America/New_York"
BARS_START, BARS_END = "2015-06-01", "2025-10-15"
FILE_FIRST, FILE_LAST = pd.Timestamp("2015-10-01"), pd.Timestamp("2025-09-30")
NEWS_FIRST, NEWS_LAST = "2015-10", "2025-09"
BATCH = 500
COLS = ["symbol", "date", "o", "h", "l", "c", "v", "ao", "ac", "dv20", "uni", "seg", "cik"]


def w1dir(cache):
    return Path(cache) / "w1"


def daily_rows(sym, raw, adj):
    """One symbol's daily rows: raw OHLCV joined with all-adjusted open/close on the same dates, zero-volume
    bars dropped, segments split at >15-day gaps, dv20 inside each segment, and the universe flag (raw
    close >= $1 and dv20 >= $500k, as in L1)."""
    d = raw[["o", "h", "l", "c", "v"]].join(adj[["o", "c"]].rename(columns={"o": "ao", "c": "ac"}), how="inner")
    d = d[d["v"] > 0]
    if d.empty:
        return pd.DataFrame(columns=COLS[:-1])
    seg = (d.index.to_series().diff().dt.days > L1.GAP_DAYS).cumsum().to_numpy()
    dv = (d["c"] * d["v"]).groupby(seg).transform(lambda x: x.rolling(20).median())
    out = d.reset_index(names="date")
    out.insert(0, "symbol", sym)
    out["dv20"] = dv.to_numpy()
    out["uni"] = (out["c"] >= L1.MIN_PRICE) & (out["dv20"] >= L1.MIN_DV20)
    out["seg"] = seg
    return out


def stage_bars(cache):
    h = alpaca_headers()
    tmap = sec_data.dera_ticker_map(2014, pd.Period("2025Q3"), Path(cache) / "dera")
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]) & set(tmap["symbol"]))
    d = w1dir(cache) / "daily"
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(syms), BATCH):
        f = d / f"d_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        part = syms[b:b + BATCH]
        end = alpaca.sip_end(BARS_END)
        raw = to_frames(fetch_daily_bars(part, BARS_START, end, adjustment="raw", headers=h))
        adj = to_frames(fetch_daily_bars(part, BARS_START, end, adjustment="all", headers=h))
        rows = [daily_rows(s, raw[s], adj[s]) for s in part if s in raw and s in adj]
        u = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLS[:-1])
        u["cik"] = sec_data.map_ciks(u[["symbol", "date"]], tmap) if len(u) else []
        tmp = f.with_suffix(".tmp")
        u.to_parquet(tmp)
        tmp.replace(f)
        print(f"[w1 bars] batch {b // BATCH + 1}/{-(-len(syms) // BATCH)}: {len(u)} rows")


def batch_files(cache):
    return sorted((w1dir(cache) / "daily").glob("d_*.parquet"))


def read_rows(cache, columns=None, filters=None):
    """Concatenated daily rows matching pyarrow `filters`, read batch by batch (callers keep the slice
    small: one year, or a set of symbols)."""
    parts = [pd.read_parquet(p, columns=columns, filters=filters) for p in batch_files(cache)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns or COLS)


def universe_year(cache, year):
    """(symbol, date, close, dv20, cik, ac) universe rows of one calendar year (close is raw, ac the
    all-adjusted close)."""
    f = [("date", ">=", pd.Timestamp(f"{year}-01-01")), ("date", "<", pd.Timestamp(f"{year + 1}-01-01")),
         ("uni", "==", True)]
    u = read_rows(cache, ["symbol", "date", "c", "dv20", "cik", "ac"], f)
    return u.rename(columns={"c": "close"})


def stage_filings(cache):
    ciks = sorted({int(c) for p in batch_files(cache)
                   for c in pd.read_parquet(p, columns=["cik"], filters=[("uni", "==", True)])["cik"].dropna()})
    d = w1dir(cache) / "subs"
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), BATCH):
        f = d / f"s_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + BATCH] for r in L1.submissions_rows(c, first=FILE_FIRST, last=FILE_LAST)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=["cik", "accession", "form", "filed", "items"]).to_parquet(tmp)
        tmp.replace(f)
        print(f"[w1 filings] batch {b // BATCH + 1}/{-(-len(ciks) // BATCH)}: {len(rows)} filings")


def load_filings(cache, columns=None):
    parts = [pd.read_parquet(p, columns=columns) for p in sorted((w1dir(cache) / "subs").glob("s_*.parquet"))]
    f = pd.concat(parts, ignore_index=True)
    f["filed"] = pd.to_datetime(f["filed"])
    return f


def news_dir(cache):
    return Path(cache) / "g1" / "news"


def stage_news(cache, deadline=None):
    return news_history.fetch_range(NEWS_FIRST, NEWS_LAST, news_dir(cache), deadline)


def news_month(cache, per):
    return pd.read_parquet(news_dir(cache) / f"news_{per}.parquet")


def et_time(created_at):
    return pd.to_datetime(created_at, utc=True).dt.tz_convert(ET).dt.tz_localize(None)


def news_symbol_days(cache):
    """(symbol, date) pairs: each symbol named by a Benzinga article on that ET calendar date, cached."""
    f = w1dir(cache) / "news_days.parquet"
    if f.exists():
        return pd.read_parquet(f)
    parts = []
    for per in pd.period_range(NEWS_FIRST, NEWS_LAST, freq="M"):
        n = news_month(cache, per)
        n = n[n["symbols"].fillna("") != ""]
        x = pd.DataFrame({"date": et_time(n["created_at"]).dt.normalize(), "symbol": n["symbols"].str.split(",")})
        parts.append(x.explode("symbol").drop_duplicates())
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    out.to_parquet(tmp)
    tmp.replace(f)
    return out


def quiet_mask(keys, news_days, filing_days, back=3, cal=None):
    """True where neither a Benzinga article named the symbol nor the company filed anything on the
    trading days t-back .. t (keys: symbol, cik, date; calendar days between consecutive trading days
    count with the later one, so a Monday looks back over the weekend)."""
    cal = pd.DatetimeIndex(sorted(cal))
    pos = cal.searchsorted(keys["date"].to_numpy())
    lo = cal[np.maximum(pos - back, 0)]
    prev = cal[np.maximum(pos - back - 1, 0)]
    start = np.where(pos - back - 1 >= 0, prev + pd.Timedelta(days=1), lo)
    k = keys.assign(_lo=start, _i=np.arange(len(keys)))
    bad = np.zeros(len(keys), dtype=bool)
    for days, col in ((news_days, "symbol"), (filing_days, "cik")):
        j = k[[col, "_lo", "date", "_i"]].merge(days.rename(columns={"date": "_d"}), on=col)
        hit = j[(j["_d"] >= j["_lo"]) & (j["_d"] <= j["date"])]["_i"].unique()
        bad[hit] = True
    return ~bad


def calendar(start="2015-06-01", end="2025-10-31"):
    cal = alpaca.trading_calendar(start, end)
    cal["date"] = pd.to_datetime(cal["date"])
    return cal.reset_index(drop=True)


def run(cache, deadline):
    """The shared stages in order; returns False when the time budget ran out."""
    stage_bars(cache)
    stage_filings(cache)
    if not stage_news(cache, deadline):
        return False
    return time.monotonic() < deadline
