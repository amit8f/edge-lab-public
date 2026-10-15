"""Weekly feature panel shared as DATA by M10 (main) and P3 (penny): price/volume features, 21- and 5-day
forward returns, point-in-time CIK, and trailing headline / analyst-action counts for every common stock
with a $1+ close and $1M+ median dollar volume. Bars are fetched in ~500-symbol batches, each cached
with an atomic write, so a run that hits its time budget resumes where it stopped. No model is fitted
here.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, shortterm as st
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import analyst, bars_batch, news_history, sec_data
from tracks.main.longterm.prices import align, split_points

FIRST_SIGNAL = pd.Timestamp("2017-01-06")
LAST_SIGNAL = pd.Timestamp("2025-09-30")
LABEL_END = pd.Timestamp("2025-10-01")       # phase-2 holdout: no label window may reach it
BARS_START, BARS_END = "2015-06-01", "2025-09-30"
NEWS_FIRST, NEWS_LAST = "2016-10", "2025-09"
MIN_PRICE, MIN_DV20 = 1.0, 1e6              # the looser of the two tracks' filters


def calendar_and_spy():
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    dates = st.weekly_dates(spy.index, FIRST_SIGNAL, LAST_SIGNAL)
    ends = st.horizon_ends(spy.index, dates, LABEL_END)
    dates = [t for t in dates if t in ends["ret21"]]
    spy_r = {}
    for name, m in ends.items():
        r = {}
        for t, end in m.items():
            i = spy.index.get_loc(t)
            x = spy.index.searchsorted(end, side="right") - 1
            r[t] = float(spy["c"].iloc[x] / spy["o"].iloc[i + 1] - 1)
        spy_r[name] = r
    return dates, ends, spy_r


def process(symbols, dates, ends, fetch=bars_batch.fetch_three):
    fr = fetch(symbols, BARS_START, alpaca.sip_end(BARS_END))
    feats, rts = [], []
    for sym in symbols:
        if sym not in fr["all"] or sym not in fr["raw"] or sym not in fr["split"]:
            continue
        d = align(fr["all"][sym], fr["raw"][sym], fr["split"][sym])
        if d.empty:
            continue
        rts.append(split_points(sym, d))
        f = st.features(sym, d, dates, ends)
        if len(f):
            f = f[(f["price"] >= MIN_PRICE) & (f["dv20"] >= MIN_DV20)]
            num = f.select_dtypes("float64").columns
            feats.append(f.astype({c: "float32" for c in num if c not in ("price", "dv20")}))
    f = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    r = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
    return f, r


def bar_batches(symbols, dates, ends, cache_dir, deadline=None, size=500, fetch=bars_batch.fetch_three):
    """Returns (features, split points), or None when the time budget ran out first."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    feats, rts = [], []
    symbols = sorted(set(symbols))
    nb = -(-len(symbols) // size)
    for b in range(0, len(symbols), size):
        ff, fr = cache / f"feat_{b // size:04d}.parquet", cache / f"rtab_{b // size:04d}.parquet"
        if not (ff.exists() and fr.exists()):
            if deadline is not None and time.monotonic() > deadline:
                print(f"[bars] budget reached at batch {b // size + 1}/{nb}")
                return None
            f, r = process(symbols[b:b + size], dates, ends, fetch)
            for df, path in ((f, ff), (r, fr)):
                tmp = path.with_suffix(".tmp")
                df.to_parquet(tmp)
                tmp.replace(path)
            print(f"[bars] batch {b // size + 1}/{nb}: {len(f)} rows")
        feats.append(pd.read_parquet(ff))
        rts.append(pd.read_parquet(fr))
    return (pd.concat([f for f in feats if len(f)], ignore_index=True),
            pd.concat([r for r in rts if len(r)], ignore_index=True))


def news_events(news_dir):
    """(all single-symbol headline days, M6 analyst events), read month by month."""
    days, acts = [], []
    for per in pd.period_range(NEWS_FIRST, NEWS_LAST, freq="M"):
        m = pd.read_parquet(Path(news_dir) / f"news_{per}.parquet", columns=["created_at", "headline", "symbols"])
        s = m["symbols"].fillna("")
        m = m[(s != "") & ~s.str.contains(",")]
        ts = pd.to_datetime(m["created_at"], utc=True).dt.tz_convert("America/New_York")
        days.append(pd.DataFrame({"symbol": m["symbols"].to_numpy(),
                                  "day": ts.dt.tz_localize(None).dt.normalize().to_numpy()}))
        cls = m["headline"].map(analyst.classify)
        acts.append(m[cls != 0])
    return pd.concat(days, ignore_index=True), analyst.events(pd.concat(acts, ignore_index=True))


def add_news(panel, ev_days, ev):
    k = panel[["symbol", "date"]]
    c = st.window_counts(k, ev_days, (7, 30))
    up = st.window_counts(k, ev[ev["direction"] > 0][["symbol", "day"]], (analyst.WINDOW_DAYS,))[analyst.WINDOW_DAYS]
    dn = st.window_counts(k, ev[ev["direction"] < 0][["symbol", "day"]], (analyst.WINDOW_DAYS,))[analyst.WINDOW_DAYS]
    out = panel.copy()
    out["news7"], out["news30"], out["an_up"], out["an_down"] = c[7], c[30], up, dn
    tot = up + dn
    out["an_net"] = np.where(tot > 0, (up - dn) / np.sqrt(np.maximum(tot, 1)), 0.0).astype("float32")
    return out


def build(cache, deadline=None):
    """Panel and split points (cached at cache/boom/panel.parquet), funnel, dates, ends and SPY returns;
    None when the bar download is not finished."""
    cache = Path(cache)
    box = cache / "boom"
    dates, ends, spy_r = calendar_and_spy()
    print(f"signal dates {len(dates)} ({dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}), 5-day labels {len(ends['ret5'])}")
    pf, rf = box / "panel.parquet", box / "rtab.parquet"
    if pf.exists() and rf.exists():
        panel, rtab = pd.read_parquet(pf), pd.read_parquet(rf)
        funnel = dict(pd.read_json(box / "funnel.json", typ="series"))
        return panel, rtab, funnel, dates, ends, spy_r
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST_SIGNAL]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    funnel = {"alpaca_assets": int(len(assets)), "alpaca_common": int(common["base_symbol"].nunique()),
              "common_named_in_form345": len(syms), "signal_dates": len(dates)}
    got = bar_batches(syms, dates, ends, box / "bars", deadline)
    if got is None:
        return None
    panel, rtab = got
    funnel["rows_price1_dv1m"] = int(len(panel))
    panel["cik"] = sec_data.map_ciks(panel[["symbol", "date"]], tmap)
    funnel["unmapped_rows"] = int(panel["cik"].isna().sum())
    panel = panel.dropna(subset=["cik"]).astype({"cik": "int64"}).reset_index(drop=True)
    # The shared cache can come back without the headline months (another screen saved it last), so
    # fetch whatever is missing; months already cached are skipped.
    if not news_history.fetch_range(NEWS_FIRST, NEWS_LAST, cache / "news", deadline):
        return None
    ev_days, ev = news_events(cache / "news")
    funnel["single_symbol_headlines"] = int(len(ev_days))
    funnel["analyst_events"] = int(len(ev))
    panel = add_news(panel, ev_days, ev)
    for df, path in ((panel, pf), (rtab, rf)):
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
    pd.Series(funnel).to_json(box / "funnel.json")
    return panel, rtab, funnel, dates, ends, spy_r
