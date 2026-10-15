"""M3 historical screen (research/prereg/M3.md): residual momentum, top 25 vs the universe, 1 month.

Bars are fetched in ~500-symbol batches and reduced at once to month-end rows (score inputs, universe
fields, forward returns), cached per batch. Fama-French 3-factor monthly data come from Kenneth French's
library (sha256 saved with the results).

Usage: python -m tracks.main.m3_screen --cache <dir> --out <dir>
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import align, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals, eligible
from tracks.main.longterm.screen import month_ends
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FIRST_SIGNAL, LAST_SIGNAL = pd.Timestamp("2016-01-29"), pd.Timestamp("2025-08-29")
LABEL_END = pd.Timestamp("2025-10-01")
BARS_START, BARS_END = "2012-11-01", "2025-09-30"
TOP_N, COST, SEED = 25, 0.002, 20261015
REG_MONTHS, MIN_REG, MOM_MONTHS = 36, 24, 11
MIN_SERIES_BARS = 253


def parse_ff(raw_zip):
    """Monthly Mkt-RF, SMB, HML, RF as decimals, indexed by month period."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for ln in text.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 5 and len(p[0]) == 6 and p[0].isdigit():
            rows.append([pd.Period(f"{p[0][:4]}-{p[0][4:]}", "M")] + [float(x) / 100 for x in p[1:]])
        if rows and ln.strip().lower().startswith("annual"):
            break
    return pd.DataFrame(rows, columns=["m", "mkt", "smb", "hml", "rf"]).drop_duplicates("m").set_index("m")


def monthly_returns(d, seg):
    """Month-end to month-end total return per month (adjusted closes); NaN across a series break or
    after a month with no bar."""
    me = pd.Series(np.arange(len(d)), index=d.index).groupby(d.index.to_period("M")).max()
    c, s = d["c"].to_numpy(), np.asarray(seg)
    i = me.to_numpy()
    consecutive = np.array([(b - a).n == 1 for a, b in zip(me.index[:-1], me.index[1:])], dtype=bool)
    r = np.full(len(i), np.nan)
    r[1:] = np.where((s[i[1:]] == s[i[:-1]]) & consecutive, c[i[1:]] / c[i[:-1]] - 1, np.nan)
    return pd.Series(r, index=me.index)


def resid_score(y, X):
    """y: 36 monthly excess returns ending at t (NaN allowed); X: matching factor rows. Residual momentum
    over months t-11..t-1 (all 11 must be present), or NaN."""
    ok = np.isfinite(y)
    if ok.sum() < MIN_REG:
        return np.nan
    A = np.c_[np.ones(ok.sum()), X[ok]]
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    e = y - np.c_[np.ones(len(y)), X] @ beta
    w = e[-1 - MOM_MONTHS:-1]
    if not np.isfinite(w).all():
        return np.nan
    sd = w.std(ddof=1)
    return float(w.sum() / sd) if sd > 0 else np.nan


def forward(d, seg, i, end):
    """Open of bar i+1 to the close of the last bar <= end in the same series."""
    if i + 1 >= len(d) or seg[i + 1] != seg[i]:
        return np.nan
    x = int(d.index.searchsorted(end, side="right")) - 1
    while x > i + 1 and seg[x] != seg[i]:
        x -= 1
    return d["c"].iloc[x] / d["o"].iloc[i + 1] - 1


def symbol_rows(sym, d, dates, ends1, ends3, ff):
    if len(d) < MIN_SERIES_BARS:
        return []
    seg = segment_ids(d.index)
    nb = pd.Series(seg, index=d.index).groupby(seg).cumcount().to_numpy() + 1
    dv = d["rc"].to_numpy() * d["v"].to_numpy()
    mr = monthly_returns(d, seg)
    pos = {t: k for k, t in enumerate(d.index)}
    out = []
    for t in dates:
        i = pos.get(t)
        if i is None or nb[i] < MIN_SERIES_BARS:
            continue
        months = pd.period_range(end=t.to_period("M"), periods=REG_MONTHS, freq="M")
        r = mr.reindex(months).to_numpy()
        f = ff.reindex(months)
        y = r - f["rf"].to_numpy()
        row = {"symbol": sym, "date": t, "price": d["rc"].iloc[i], "dv20": float(np.median(dv[i - 19:i + 1])),
               "score": resid_score(y, f[["mkt", "smb", "hml"]].to_numpy()),
               "score_capm": resid_score(y, f[["mkt"]].to_numpy()),
               "mom": float(np.prod(1 + r[-1 - MOM_MONTHS:-1]) - 1) if np.isfinite(r[-1 - MOM_MONTHS:-1]).all() else np.nan,
               "ret1": forward(d, seg, i, ends1[t]) if t in ends1 else np.nan,
               "ret3": forward(d, seg, i, ends3[t]) if t in ends3 else np.nan}
        out.append(row)
    return out


def excess(df, col, score, n, rng_seed=None):
    """Per date: mean `col` of the top n by `score` (or n random names) minus the universe mean, minus
    COST. Picks are chosen before looking at returns; a pick with no return (no next bar) is dropped, not
    replaced, and counted."""
    rows = []
    for t, g in df.groupby("date", sort=True):
        if rng_seed is None:
            top = g[g[score].notna()].nlargest(n, score)
        else:
            rng = np.random.default_rng(rng_seed + t.toordinal())
            top = g.iloc[rng.choice(len(g), size=min(n, len(g)), replace=False)]
        if not top[col].notna().any():
            continue
        rows.append({"date": t, "excess": top[col].mean() - g[col].mean() - COST, "n": int(top[col].notna().sum()),
                     "dropped": int(top[col].isna().sum())})
    return pd.DataFrame(rows)


def summary(ex, lags):
    x = ex["excess"].to_numpy()
    m, se, t = nw_tstat(x, lags)
    lo, hi = block_bootstrap_ci(x, 6, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_dates": int(np.isfinite(x).sum()),
            "hit_rate": float((ex["excess"] > 0).mean())}


def prepare(cache):
    """Shared by M3 and M1 (same 116 dates): month-end rows of the eligible universe with scores and
    forward returns, plus the factor file, SPY, the horizon ends and the funnel."""
    cache.mkdir(parents=True, exist_ok=True)
    raw = requests.get(FF_URL, timeout=120).content
    ff = parse_ff(raw)
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    cal = spy.index
    dates = month_ends(cal, FIRST_SIGNAL, LAST_SIGNAL)
    me_all = month_ends(cal, FIRST_SIGNAL, pd.Timestamp(BARS_END))
    nxt = {t: me_all[k + 1] for k, t in enumerate(me_all[:-1])}
    ends1 = {t: nxt[t] for t in dates if nxt[t] < LABEL_END}
    ends3 = {t: me_all[me_all.index(t) + 3] for t in dates if me_all.index(t) + 3 < len(me_all)
             and me_all[me_all.index(t) + 3] < LABEL_END}
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST_SIGNAL]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    funnel = {"common_named_in_form345": len(syms)}
    (cache / "bars").mkdir(exist_ok=True)
    for b in range(0, len(syms), 500):
        f, fr_path = cache / "bars" / f"m3_{b // 500:04d}.parquet", cache / "bars" / f"m3rtab_{b // 500:04d}.parquet"
        if f.exists() and fr_path.exists():
            continue
        fr = bars_batch.fetch_three(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in syms[b:b + 500]:
            if all(s in fr[a] for a in ("all", "raw", "split")):
                d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
                if len(d):
                    rows += symbol_rows(s, d, dates, ends1, ends3, ff)
                    rts.append(split_points(s, d))
        rt = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
        for df, path in ((rt, fr_path), (pd.DataFrame(rows), f)):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
        print(f"[bars] batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(rows)} rows")
    feat = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("m3_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("m3rtab_*.parquet"))], ignore_index=True)
    funnel["symbol_dates"] = int(len(feat))
    feat["cik"] = sec_data.map_ciks(feat[["symbol", "date"]], tmap)
    feat = feat.dropna(subset=["cik"]).astype({"cik": "int64"})
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, feat["cik"].unique(), cache / "facts")
    uni = eligible(attach_fundamentals(feat, facts, rtab))
    first_reg = pd.Period(FIRST_SIGNAL, "M") - (REG_MONTHS - 1)
    ff_ok = bool(ff.index.min() <= first_reg and ff.index.max() >= pd.Period(LAST_SIGNAL, "M"))
    funnel.update({"eligible_stock_months": int(len(uni)), "with_score": int(uni["score"].notna().sum()),
                   "dates": int(uni["date"].nunique())})
    print("FUNNEL", json.dumps(funnel))
    return {"uni": uni, "ff": ff, "ff_raw": raw, "ff_ok": ff_ok, "spy": spy, "ends1": ends1, "ends3": ends3,
            "funnel": funnel}


def run(cache, out):
    p = prepare(cache)
    uni, ff, raw, ff_ok, ends1, ends3, funnel = (p[k] for k in ("uni", "ff", "ff_raw", "ff_ok", "ends1", "ends3",
                                                                  "funnel"))
    s1 = uni[uni["date"].isin(ends1.keys())]
    prim = summary(excess(s1, "ret1", "score", TOP_N), 3)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= 2.0 and prim["ci95"][0] > 0)
    res = {"idea": "M3", "prereg": "research/prereg/M3.md", "phase2_count": 10, "main_count": 7,
           "phase1_plus_phase2": 25, "ff_sha256": hashlib.sha256(raw).hexdigest(),
           "ff_months": [str(ff.index.min()), str(ff.index.max())], "ff_covers_all_windows": ff_ok, "funnel": funnel, "primary": prim,
           "control": summary(excess(s1, "ret1", "score", TOP_N, rng_seed=SEED), 3)}
    s3 = uni[uni["date"].isin(ends3.keys())]
    dec = s1.dropna(subset=["score", "ret1"]).copy()
    dec["decile"] = dec.groupby("date")["score"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    ex = excess(s1, "ret1", "score", TOP_N).assign(year=lambda e: e["date"].dt.year)
    res["exploratory"] = {
        "capm_residuals": summary(excess(s1, "ret1", "score_capm", TOP_N), 3),
        "plain_12_1": summary(excess(s1, "ret1", "mom", TOP_N), 3),
        "top_50": summary(excess(s1, "ret1", "score", 50), 3),
        "hold_3m": summary(excess(s3, "ret3", "score", TOP_N), 5),
        "per_year": ex.groupby("year")["excess"].mean().to_dict(),
        "deciles_mean_ret1": dec.groupby("decile")["ret1"].mean().to_dict(),
        "score_pct_by_year": uni.groupby(uni["date"].dt.year)["score"].quantile([.05, .5, .95]).unstack().to_dict("index")}
    out.mkdir(parents=True, exist_ok=True)
    top = s1[s1["score"].notna()].sort_values("score", ascending=False).groupby("date").head(TOP_N)
    res["qa_largest_pick_returns"] = top.nlargest(20, "ret1")[["date", "symbol", "ret1"]].to_dict("records")
    (out / "M3_screen.json").write_text(json.dumps(res, indent=1, default=str))
    top[["date", "symbol", "score", "ret1", "price", "mcap"]].to_csv(out / "M3_screen_picks.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
