"""M11 historical screen (research/prereg/M11.md): M10's boom candidates filtered by quality, value and
buybacks. Reuses M10's cached weekly panel and refits M10's walk-forward XGBoost (same settings, same
seed). Usage: python -m tracks.main.shortterm.screen_m11 --cache <dir> --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import shortterm as st
from tracks.main.longterm import sec_data
from tracks.main.longterm.score import MIN_DV20, MIN_PRICE
from tracks.main.longterm.screen import SEED, excess_series, summarize
from tracks.main.shortterm import evaluate, panel as pn, screen_m10

TOP_N, POOL = 20, 100
COUNTS = {"phase2_count": 8, "main_count": 7, "phase1_plus_phase2": 23}


def fundamentals_score(uni):
    """quality, value and buyback percentiles within each date, and their mean (>= 2 present)."""
    g = uni.groupby("date")
    r = lambda c, hi=True: g[c].rank(pct=True) if hi else (-uni[c]).groupby(uni["date"]).rank(pct=True)
    out = pd.DataFrame(index=uni.index)
    out["f_quality"] = pd.concat([r("gpa"), r("roa")], axis=1).mean(axis=1)
    out["f_value"] = pd.concat([r("ep"), r("bm")], axis=1).mean(axis=1)
    out["f_buyback"] = r("issuance", False)
    fam = out[["f_quality", "f_value", "f_buyback"]]
    out["fscore"] = fam.mean(axis=1).where(fam.notna().sum(axis=1) >= 2)
    return out


def pick_rank(df, pool, col):
    """Rank 1..: within each date's top `pool` by M10 probability, highest `col` first (ties by
    probability); rows outside the pool or without `col` get NaN."""
    d = df.copy()
    d["prank"] = d.groupby("date")["p_xgb"].rank(ascending=False, method="first")
    inpool = (d["prank"] <= pool) & d[col].notna()
    d["_k"] = np.where(inpool, 1, 0)
    d = d.sort_values(["date", "_k", col, "p_xgb"], ascending=[True, False, False, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    d.loc[~inpool, "rank"] = np.nan
    return d.drop(columns="_k")


def tails(df, n=TOP_N):
    top = df[(df["rank"] <= n) & df["ret21"].notna()]
    return {"share_down_30pct": float((top["ret21"] <= -0.3).mean()), "share_up_30pct": float((top["ret21"] >= 0.3).mean()),
            "median": float(top["ret21"].median()), "n": int(len(top))}


def pool_control(df, pool, cost):
    rows = []
    for t, g in df.groupby("date", sort=True):
        cand = g[g["prank"] <= pool]
        rng = np.random.default_rng(SEED + 1 + t.toordinal())
        top = cand.iloc[rng.choice(len(cand), size=min(TOP_N, len(cand)), replace=False)]
        rows.append({"date": t, "pick_mean": top["ret21"].mean(), "uni_mean": g["ret21"].mean(),
                     "excess": top["ret21"].mean() - g["ret21"].mean() - cost})
    return pd.DataFrame(rows)


def run(cache, out, budget_min=300):
    cache, out = Path(cache), Path(out)
    got = pn.build(cache, time.monotonic() + budget_min * 60)
    if got is None:
        print("PANEL INCOMPLETE: rerun this workflow to continue (cached batches are kept)")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    rows = panel[(panel["price"] >= MIN_PRICE) & (panel["dv20"] >= MIN_DV20)]
    if not (cache / "companyfacts.zip").exists():
        sec_data.download_companyfacts(cache / "companyfacts.zip")
    facts = sec_data.facts_for(cache / "companyfacts.zip", rows["cik"].unique(), cache / "boom" / "facts_m10")
    uni = screen_m10.universe(panel, rtab, facts).reset_index(drop=True)
    uni = pd.concat([uni, fundamentals_score(uni)], axis=1)
    mp = evaluate.model_panel(uni, screen_m10.FEATURES)
    mp = pd.concat([mp, uni[["f_quality", "f_value", "f_buyback", "fscore"]]], axis=1)
    wf = st.walk_forward(mp, ends["ret21"], {"xgb": screen_m10.FEATURES}, evaluate.YEARS)
    cost, lags, block = screen_m10.COST, evaluate.LAGS, evaluate.BLOCK
    summ = lambda ex: summarize(ex, lags, block)
    res = {"idea": "M11", "prereg": "research/prereg/M11.md", **COUNTS,
           "funnel": {"weeks": int(wf["date"].nunique()), "rows": int(len(wf)),
                      "fscore_coverage": float(wf["fscore"].notna().mean()),
                      "fscore_coverage_in_pool": float(pick_rank(wf, POOL, "fscore").query("prank <= @POOL")["fscore"].notna().mean())}}
    d = pick_rank(wf, POOL, "fscore")
    ex = excess_series(d, "ret21", TOP_N, cost)
    prim = summ(ex)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= evaluate.T_BAR and prim["ci95"][0] > 0)
    prim["picks_short_of_20"] = int((TOP_N - ex["n_picks"]).sum())
    prim.update(tails(d))
    m10 = wf.assign(rank=wf.groupby("date")["p_xgb"].rank(ascending=False, method="first"))
    ex10 = excess_series(m10, "ret21", TOP_N, cost)
    res["primary"] = prim
    res["m10_same_weeks"] = {**summ(ex10), **tails(m10)}
    res["control_universe"] = summ(excess_series(d, "ret21", TOP_N, cost, rng_seed=SEED))
    res["control_pool"] = summ(pool_control(d, POOL, cost))
    print("PRIMARY", json.dumps(prim, indent=1))
    print("M10", json.dumps(res["m10_same_weeks"], indent=1))
    exp = {}
    for pool in (50, 200):
        dp = pick_rank(wf, pool, "fscore")
        exp[f"pool_{pool}"] = {**summ(excess_series(dp, "ret21", TOP_N, cost)), **tails(dp)}
    for f in ("f_quality", "f_value", "f_buyback"):
        dp = pick_rank(wf, POOL, f)
        exp[f"{f}_only"] = {**summ(excess_series(dp, "ret21", TOP_N, cost)), **tails(dp)}
    paired = ex[["date", "excess"]].merge(ex10[["date", "excess"]], on="date", suffixes=("", "_m10"))
    exp["minus_m10"] = summ(paired.assign(excess=paired["excess"] - paired["excess_m10"], pick_mean=np.nan, uni_mean=np.nan))
    exp["per_year"] = {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}
    exp["vs_spy"] = summ(ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret21"]) - cost))
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "M11_screen.json").write_text(json.dumps(res, indent=1, default=str))
    ex.to_csv(out / "M11_primary_series.csv", index=False)
    d[d["rank"] <= TOP_N][["date", "rank", "symbol", "price", "p_xgb", "fscore", "f_quality", "f_value",
                            "f_buyback", "ret21", "y"]].to_csv(out / "M11_screen_picks.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
