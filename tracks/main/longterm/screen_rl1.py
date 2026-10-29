"""RL1 historical screen (research/prereg/RL1.md): bell-curve rank label vs raw-return label, same XGBoost and
M4 inputs, monthly long-short decile. Runs on GitHub Actions through study.yml (needs Alpaca and SEC access).

The month-end panel is M4's (price, market cap and liquidity filters, at least 4 of the 6 families, one row per
company) on M3's month-end dates, with the next month's return (next open to the next month-end close). Bars
are fetched in 500-symbol batches cached under <cache>/bars/rl1_*.parquet, so a rerun resumes.

Usage: python -m tracks.main.longterm.screen_rl1 --cache <dir> [--out research/results/RL1]
"""
import argparse
import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main import m3_screen as m3
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import align, features, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals, composite, eligible
from tracks.main.longterm.screen import month_ends
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

ROOT = Path(__file__).resolve().parents[3]
FIRST_SIGNAL, LAST_SIGNAL, LABEL_END = m3.FIRST_SIGNAL, m3.LAST_SIGNAL, m3.LABEL_END
BARS_START, BARS_END = m3.BARS_START, m3.BARS_END
TEST_YEARS = range(2019, 2026)
COST, BORROW = 0.002, 0.03               # config/risk.yaml main: round trip per side, borrow a year
SEED = 20261015
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=20, tree_method="hist", random_state=SEED)
RAW_INPUTS = ["mom", "vol", "gpa", "roa", "ep", "bm", "issuance", "assetgrowth"]
INPUTS = [f"r_{k}" for k in RAW_INPUTS]
LAGS, T_LABEL, OVERFIT, REGISTERED_TRIALS = 3, 2.0, 2.0, 66


# ---------- pure helpers (tested) ----------

def symbol_rows(sym, d, dates, ends1):
    """M4's price features (12-1 momentum, 252-day volatility, price, dv20) at each month-end plus the next
    month's return from the next open to the close on the next month-end, inside one series."""
    f = features(sym, d, dates)
    if f.empty:
        return []
    seg = segment_ids(d.index)
    pos = {t: k for k, t in enumerate(d.index)}
    f = f[["symbol", "date", "mom", "vol", "price", "dv20"]].copy()
    f["ret1"] = [m3.forward(d, seg, pos[t], ends1[t]) if t in ends1 else np.nan for t in f["date"]]
    return f.to_dict("records")


def add_inputs(uni):
    """Each M4 input as a within-date percentile rank (NaN stays NaN), and the rank label: Phi^-1((rank - 0.5)
    / n) of ret1 within its date (rank 1 = lowest)."""
    g = uni.groupby("date")
    out = uni.assign(**{f"r_{k}": g[k].rank(pct=True) for k in RAW_INPUTS})
    r = g["ret1"].rank(method="first")
    n = g["ret1"].transform("count")
    out["y_rank"] = [NormalDist().inv_cdf((a - 0.5) / b) if np.isfinite(a) else np.nan for a, b in zip(r, n)]
    return out


def fit(X, y):
    from xgboost import XGBRegressor
    return XGBRegressor(**XGB).fit(X, y)


def long_short(df, col, seed=None):
    """Per date: equal-weight top decile by `col` minus bottom decile, minus costs (0.2% round trip on each
    leg, a month of borrow on the short leg). With `seed`, both deciles are drawn at random (the control)."""
    rows = []
    for t, g in df[df["ret1"].notna()].groupby("date", sort=True):
        k = len(g) // 10
        if k < 1:
            continue
        if seed is None:
            o = g.sort_values(col, kind="stable")
            lo, hi = o.head(k), o.tail(k)
        else:
            idx = np.random.default_rng(seed + t.toordinal()).permutation(len(g))
            lo, hi = g.iloc[idx[:k]], g.iloc[idx[k:2 * k]]
        rows.append({"date": t, "ls": hi["ret1"].mean() - lo["ret1"].mean() - 2 * COST - BORROW / 12,
                     "n_leg": k})
    return pd.DataFrame(rows, columns=["date", "ls", "n_leg"])


def ic(df, col):
    """Mean over dates of the Spearman correlation of `col` with ret1."""
    v = [g[col].corr(g["ret1"], method="spearman") for _, g in df[df["ret1"].notna()].groupby("date")]
    return float(np.nanmean(v)) if v else float("nan")


def walk_forward(uni, ends1, years=TEST_YEARS):
    """Per test year: both models fit on every month-end whose label closed before the year's first day,
    scored on that year's month-ends. Returns test predictions and each model's in-sample long-short mean
    and IC (rule 21)."""
    preds, ins = [], {"raw": [], "rank": []}
    lab_end = uni["date"].map(ends1)
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        tr = uni[(lab_end < start) & uni["ret1"].notna()]
        te = uni[(uni["date"] >= start) & (uni["date"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        Xtr, Xte = tr[INPUTS].to_numpy(float), te[INPUTS].to_numpy(float)
        p = te[["symbol", "date", "ret1"]].copy()
        for name, y in (("raw", "ret1"), ("rank", "y_rank")):
            m = fit(Xtr, tr[y].to_numpy(float))
            p[f"p_{name}"] = m.predict(Xte)
            s = tr[["date", "ret1"]].assign(p=m.predict(Xtr))
            ls = long_short(s, "p")
            ins[name].append({"year": yr, "ls_mean": float(ls["ls"].mean()), "months": int(len(ls)), "ic": ic(s, "p")})
        preds.append(p)
    return pd.concat(preds, ignore_index=True), ins


def summary(x):
    x = np.asarray(x, float)
    m, se, t = nw_tstat(x, LAGS)
    lo, hi = block_bootstrap_ci(x, 6, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_months": int(np.isfinite(x).sum())}


def verdict(pred, ins, ff5=None, umd=None, trials=REGISTERED_TRIALS):
    raw, rank = long_short(pred, "p_raw"), long_short(pred, "p_rank")
    both = raw.merge(rank, on="date", suffixes=("_raw", "_rank"))
    res = {"raw": summary(raw["ls"]), "rank": summary(rank["ls"]),
           "rank_minus_raw": summary(both["ls_rank"] - both["ls_raw"]),
           "control_random": summary(long_short(pred, "p_raw", seed=SEED)["ls"]),
           "ic": {"raw": ic(pred, "p_raw"), "rank": ic(pred, "p_rank")}}
    over = {}
    for name in ("raw", "rank"):
        w = [x["months"] for x in ins[name]]
        i_ls = float(np.average([x["ls_mean"] for x in ins[name]], weights=w)) if w else float("nan")
        res[f"insample_{name}"] = {"ls_mean": i_ls, "ic": float(np.average([x["ic"] for x in ins[name]], weights=w)),
                                   "per_year": ins[name]}
        over[name] = bool(i_ls > OVERFIT * max(res[name]["mean"], 0.0))
    res["overfit_rule21"] = over
    d = res["rank_minus_raw"]
    res["label_claim_passes"] = bool(d["mean"] > 0 and d["t_hac"] >= T_LABEL and not over["rank"] and not over["raw"])
    if ff5 is not None:
        alpha = stats.factor_alpha(rank["ls"], stats.month_after(rank["date"]), ff5, umd, LAGS)
        res["rank"]["factor_alpha"] = alpha
        res["rank"]["deflated_sharpe"] = stats.deflated_sharpe(rank["ls"], trials)
        res["rank_trading_passes"] = bool(stats.screen_pass(res["rank"], alpha) and not over["rank"])
    res["per_year"] = {str(y): {"raw": float(g["ls_raw"].mean()), "rank": float(g["ls_rank"].mean())}
                       for y, g in both.groupby(both["date"].dt.year)}
    return res, raw, rank


# ---------- data (network / disk) ----------

def prepare(cache, deadline):
    cache.mkdir(parents=True, exist_ok=True)
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    dates = month_ends(spy.index, FIRST_SIGNAL, LAST_SIGNAL)
    me_all = month_ends(spy.index, FIRST_SIGNAL, pd.Timestamp(BARS_END))
    nxt = {t: me_all[k + 1] for k, t in enumerate(me_all[:-1])}
    ends1 = {t: nxt[t] for t in dates if t in nxt and nxt[t] < LABEL_END}
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST_SIGNAL]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    funnel = {"common_named_in_form345": len(syms), "signal_dates": len(dates), "dates_with_label": len(ends1)}
    (cache / "bars").mkdir(exist_ok=True)
    for b in range(0, len(syms), 500):
        f, fr_path = cache / "bars" / f"rl1_{b // 500:04d}.parquet", cache / "bars" / f"rl1rtab_{b // 500:04d}.parquet"
        if f.exists() and fr_path.exists():
            continue
        if time.monotonic() > deadline:
            print("PANEL INCOMPLETE: budget reached; rerun to continue")
            return None
        fr = bars_batch.fetch_three(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in syms[b:b + 500]:
            if all(s in fr[a] for a in ("all", "raw", "split")):
                d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
                if len(d):
                    rows += symbol_rows(s, d, dates, ends1)
                    rts.append(split_points(s, d))
        rt = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
        for df, path in ((rt, fr_path), (pd.DataFrame(rows), f)):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
        print(f"[bars] batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(rows)} rows")
    feat = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("rl1_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("rl1rtab_*.parquet"))], ignore_index=True)
    funnel["symbol_dates"] = int(len(feat))
    feat["cik"] = sec_data.map_ciks(feat[["symbol", "date"]], tmap)
    feat = feat.dropna(subset=["cik"]).astype({"cik": "int64"})
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, feat["cik"].unique(), cache / "facts")
    uni = composite(eligible(attach_fundamentals(feat, facts, rtab)))
    uni = add_inputs(uni.drop(columns=[c for c in ("rank",) if c in uni.columns]))
    funnel.update({"eligible_stock_months": int(len(uni)), "with_label": int(uni["ret1"].notna().sum()),
                   "dates": int(uni["date"].nunique()), "median_per_date": float(uni.groupby("date").size().median())})
    print("FUNNEL", json.dumps(funnel))
    return uni, ends1, funnel


def run(cache, out, budget_min=290):
    cache, out = Path(cache), Path(out)
    p = prepare(cache, time.monotonic() + budget_min * 60)
    if p is None:
        return None
    uni, ends1, funnel = p
    pred, ins = walk_forward(uni, ends1)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts(ROOT / "research" / "IDEAS.md")
    res, raw, rank = verdict(pred, ins, ff5, umd)
    res["deflated_sharpe_current_count"] = stats.deflated_sharpe(rank["ls"], both)
    res = {"idea": "RL1", "prereg": "research/prereg/RL1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_sha256": sha5, "umd_sha256": shau, "funnel": funnel, "xgb_params": XGB} | res
    out.mkdir(parents=True, exist_ok=True)
    (out / "RL1_screen.json").write_text(json.dumps(res, indent=1, default=str))
    raw.merge(rank, on="date", suffixes=("_raw", "_rank")).to_csv(out / "RL1_monthly.csv", index=False)
    print(json.dumps({k: res[k] for k in ("raw", "rank", "rank_minus_raw", "control_random", "ic", "overfit_rule21",
                                          "label_claim_passes", "rank_trading_passes")}, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/RL1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
