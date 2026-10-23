"""Week-end stock panel and weekly test helpers shared by R1 (research/prereg/R1.md) and I1 (research/prereg/I1.md).

Bars (all, raw and split adjusted, Alpaca SIP, delisted kept) are fetched in ~500-symbol batches and
reduced at once to one row per symbol and week-end: universe fields, the 5-day return, the 200-day test
and forward returns. Rows are cached per batch, so a rerun resumes after the last finished batch.
"""
import re

import numpy as np
import pandas as pd

from core import alpaca, sec, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import align, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals, eligible
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

BARS_START, BARS_END = "2016-01-01", "2025-09-30"
LABEL_END = pd.Timestamp("2025-10-01")
HOLDS = (5, 10, 20)                 # trading days from entry (entry day = day 1)
MIN_PRICE, MIN_DV20 = 1.0, 5e5      # rows below both penny floors can't enter either universe
LAGS, BLOCK, N_BOOT = 4, 4, 10_000   # Newey-West lags and bootstrap blocks for weekly series
FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"


def week_ends(cal):
    """Last trading day of each calendar week (Monday-Sunday)."""
    s = pd.Series(cal, index=cal)
    return list(s.groupby(s.index.to_period("W")).max())


def hold_ends(cal, dates, hold):
    """{t: exit date} with exit = the hold-th trading day counted from the day after t, for every t whose
    exit is before LABEL_END."""
    cal = pd.DatetimeIndex(cal)
    out = {}
    for t in dates:
        k = cal.searchsorted(t)
        if k + hold < len(cal) and cal[k + hold] < LABEL_END:
            out[t] = cal[k + hold]
    return out


def forward(d, seg, i, end):
    """Adjusted open of bar i+1 to the adjusted close of the last bar on or before `end` in the same
    series; NaN when there is no next bar in the series."""
    if i + 1 >= len(d) or seg[i + 1] != seg[i]:
        return np.nan
    x = int(d.index.searchsorted(end, side="right")) - 1
    while x > i + 1 and seg[x] != seg[i]:
        x -= 1
    return float(d["c"].iloc[x] / d["o"].iloc[i + 1] - 1)


def symbol_rows(sym, d, dates, ends):
    """Week-end rows for one symbol. d: align() frame (all-adjusted o/c, raw rc/v, split sc).
    ends: {hold: {t: exit date}}."""
    if d.empty:
        return []
    seg = segment_ids(d.index)
    nb = pd.Series(seg, index=d.index).groupby(seg).cumcount().to_numpy() + 1
    c = d["c"].to_numpy()
    dv = d["rc"].to_numpy() * d["v"].to_numpy()
    pos = {t: k for k, t in enumerate(d.index)}
    out = []
    for t in dates:
        i = pos.get(t)
        if i is None or nb[i] < 21:
            continue
        dv20 = float(np.median(dv[i - 19:i + 1]))
        price = float(d["rc"].iloc[i])
        if price < MIN_PRICE or dv20 < MIN_DV20:
            continue
        row = {"symbol": sym, "date": t, "price": price, "dv20": dv20, "nb": int(nb[i]),
               "ret5": float(c[i] / c[i - 5] - 1) if nb[i] > 5 else np.nan,
               "above200": bool(c[i] > c[i - 199:i + 1].mean()) if nb[i] >= 200 else False}
        for h in HOLDS:
            row[f"fwd{h}"] = forward(d, seg, i, ends[h][t]) if t in ends[h] else np.nan
        out.append(row)
    return out


def parse_form_idx(text, forms):
    """EDGAR quarterly form.idx -> frame (form, cik, filed) for `forms`."""
    rows = []
    for ln in text.splitlines():
        m = re.match(r"^(\S+(?: \S+)*?)\s{2,}.*?\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2}|\d{8})\s+\S+\s*$", ln)
        if m and m.group(1) in forms:
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=["form", "cik", "filed"])
    df["cik"] = df["cik"].astype("int64")
    df["filed"] = pd.to_datetime(df["filed"].str.replace("-", ""), format="%Y%m%d")
    return df


def filings(forms, cache_dir, first_year=2016, last=LABEL_END, get=sec.get):
    """Filings of `forms` from EDGAR's quarterly indexes, cached per quarter."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in pd.period_range(f"{first_year}Q1", pd.Period(last, "Q"), freq="Q"):
        f = cache_dir / f"idx_{p}.parquet"
        if not f.exists():
            r = get(FULL_INDEX.format(y=p.year, q=p.quarter))
            df = parse_form_idx(r.text if r is not None else "", forms)
            tmp = f.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(f)
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def build(cache, tag="wk"):
    """Panel of week-end rows (all common stock) with a CIK where Form 3/4/5 data maps one, and market
    cap for rows that could enter M4's universe. Returns (rows, calendar, week-ends, funnel)."""
    cache.mkdir(parents=True, exist_ok=True)
    h = alpaca_headers()
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all", headers=h))["SPY"]
    cal = list(spy.index)
    dates = [t for t in week_ends(cal) if t >= pd.Timestamp(BARS_START)]
    ends = {k: hold_ends(cal, dates, k) for k in HOLDS}
    dates = [t for t in dates if t in ends[5]]
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]))
    funnel = {"common_symbols": len(syms), "week_ends": len(dates)}
    bdir = cache / "bars"
    bdir.mkdir(exist_ok=True)
    nb = -(-len(syms) // 500)
    for b in range(0, len(syms), 500):
        f, rp = bdir / f"{tag}_{b // 500:04d}.parquet", bdir / f"{tag}rtab_{b // 500:04d}.parquet"
        if f.exists() and rp.exists():
            continue
        fr = bars_batch.fetch_three(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in syms[b:b + 500]:
            if all(s in fr[a] for a in ("all", "raw", "split")):
                d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
                if len(d):
                    rows += symbol_rows(s, d, dates, ends)
                    rts.append(split_points(s, d))
        rt = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
        for df, path in ((rt, rp), (pd.DataFrame(rows), f)):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
        print(f"[panel] batch {b // 500 + 1}/{nb}: {len(rows)} rows", flush=True)
    rows = pd.concat([pd.read_parquet(p) for p in sorted(bdir.glob(f"{tag}_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted(bdir.glob(f"{tag}rtab_*.parquet"))], ignore_index=True)
    funnel["rows"] = int(len(rows))
    tmap = sec_data.dera_ticker_map(2015, pd.Period(dates[-1], "Q"), cache / "dera")
    rows["cik"] = sec_data.map_ciks(rows[["symbol", "date"]], tmap)
    funnel["rows_with_cik"] = int(rows["cik"].notna().sum())
    big = rows["cik"].notna() & (rows["price"] >= 5) & (rows["dv20"] >= 2e6) & (rows["nb"] >= 253)
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    cand = rows[big].astype({"cik": "int64"})
    facts = sec_data.facts_for(zp, cand["cik"].unique(), cache / "facts")
    main = eligible(attach_fundamentals(cand, facts, rtab))
    rows["main"] = False
    rows.loc[rows.set_index(["symbol", "date"]).index.isin(main.set_index(["symbol", "date"]).index), "main"] = True
    rows = rows.merge(main[["symbol", "date", "mcap"]], on=["symbol", "date"], how="left")
    rows["penny"] = (rows["price"] >= 1) & (rows["price"] < 5) & (rows["dv20"] >= 5e5) & (rows["nb"] >= 60)
    funnel.update({"main_rows": int(rows["main"].sum()), "penny_rows": int(rows["penny"].sum())})
    print("FUNNEL", funnel, flush=True)
    return rows, cal, dates, funnel, spy


def weekly(sel, cost, col="fwd5"):
    """Per week: mean net return of picks (role "pick") and controls (names with an entry bar) and the
    difference."""
    s = sel.dropna(subset=[col]).assign(net=lambda d: d[col] - cost)
    t = s.pivot_table(index="date", columns="role", values="net", aggfunc="mean")
    t["n_picks"] = s[s["role"] == "pick"].groupby("date").size()
    t["diff"] = t["pick"] - t["control"]
    return t.dropna(subset=["diff"])


def summary(t, seed):
    m, se, tt = nw_tstat(t["diff"], LAGS)
    lo, hi = block_bootstrap_ci(t["diff"], BLOCK, N_BOOT, seed)
    return {"weeks": int(len(t)), "mean": m, "se": se, "t_hac": tt, "ci95": [lo, hi],
            "mean_n_picks": float(t["n_picks"].mean())}


def alpha(t, fac, cal, hold=5):
    """Weekly picks-minus-control on the 6 daily factors summed over each holding window."""
    cal = pd.DatetimeIndex(cal)
    wins = [cal[cal.searchsorted(d) + 1:cal.searchsorted(d) + 1 + hold] for d in t.index]
    out = stats.ols_hac(t["diff"].to_numpy(), stats.window_factors(fac, wins), LAGS)
    out["factors"] = list(fac.columns) + ["(summed over the holding window)"]
    return out


def factors():
    """Daily FF5 + momentum and the two files' sha256."""
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    return ffd.join(umdd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]], sha5, shau
