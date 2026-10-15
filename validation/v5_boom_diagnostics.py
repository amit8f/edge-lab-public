"""V5 (research/prereg/V5.md): what the boom models M10 and P3 learned. Pick characteristics, exact tree
SHAP attribution by feature group, a shuffled-label placebo and a stale-signal placebo, on the same
cached weekly panel and walk-forward as the screens.

Usage: python -m validation.v5_boom_diagnostics --cache <dir> [--out <dir>] [--budget-min N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import shortterm as st
from tracks.main.longterm import sec_data
from tracks.main.longterm.screen import excess_series
from tracks.main.shortterm import evaluate, panel as pn, screen_m10
from tracks.penny.shortterm import screen_p3

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015
TOP_N = 20
GROUPS = {"volatility": ["vol21", "vol63", "volratio", "maxret21", "gap5"],
          "trend": ["r1", "r5", "r21", "r63", "r126", "mom", "hi52", "lo52", "up21", "rsi14"],
          "size_liquidity": ["ldv20", "dvr", "lprice", "lmcap"],
          "news": ["news7", "news30", "an_up", "an_down", "an_net"]}
CHARS = ["vol21", "vol63", "maxret21", "lprice", "lmcap"]


def walk_forward_shap(panel, end21, feats, years):
    """Walk-forward XGBoost exactly as the screens (st.fit_predict per year), keeping each year's model
    to compute exact tree SHAP values on that year's test rows. Returns test rows with p_xgb and one
    shap_<feature> column per feature."""
    import xgboost as xgb
    labelled = panel[panel["y"].notna()]
    wend = labelled["date"].map(end21)
    parts = []
    for y in years:
        test = panel[(panel["date"].dt.year == y) & panel["date"].isin(list(end21))].copy()
        if test.empty:
            continue
        train = labelled[wend < test["date"].min()]
        p, model = st.fit_predict("xgb", train, test, feats)
        test["p_xgb"] = p
        contrib = model.get_booster().predict(xgb.DMatrix(test[feats].to_numpy(dtype="float32")), pred_contribs=True)
        for k, f in enumerate(feats):
            test[f"shap_{f}"] = contrib[:, k]
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def ranked(df, col):
    return df.assign(rank=df.groupby("date")[col].rank(ascending=False, method="first"))


def shap_shares(df, feats):
    a = df[[f"shap_{f}" for f in feats]].abs().mean()
    tot = a.sum()
    return {g: float(sum(a.get(f"shap_{f}", 0.0) for f in fs if f in feats) / tot) for g, fs in GROUPS.items()}


def pick_percentiles(uni, picks):
    """Median within-date percentile of each characteristic among the picks."""
    pct = uni[["symbol", "date"]].copy()
    for c in CHARS:
        if c in uni:
            pct[c] = uni.groupby("date")[c].rank(pct=True)
    m = picks[["symbol", "date"]].merge(pct, on=["symbol", "date"], how="left")
    return {c: float(m[c].median()) for c in CHARS if c in m}


def vol_baseline(uni, panel, end21, years):
    """Top-2% hit rate and excess of the 20 highest-vol63 stocks each week (amendment after the
    placebo flag): what picking extremes alone does, with no model."""
    p = panel[(panel["date"].dt.year.isin(years)) & panel["date"].isin(list(end21)) & panel["y"].notna()]
    v = uni[["symbol", "date", "vol63"]]
    p = p.drop(columns=[c for c in ("vol63",) if c in p]).merge(v, on=["symbol", "date"], how="left")
    r = ranked(p, "vol63")
    top = r[r["rank"] <= TOP_N]
    return {"hit_rate_top2pct": float(top["y"].mean()), "n_picks": int(len(top))}


def placebo_shuffled(panel, end21, feats, years, cost, uni=None):
    """Labels permuted within each date before fitting; picks scored on the real returns."""
    p = panel.copy()
    out = []
    for t, g in p.groupby("date", sort=True):
        rng = np.random.default_rng(SEED + t.toordinal())
        out.append(pd.Series(rng.permutation(g["y"].to_numpy()), index=g.index))
    p["y_real"] = p["y"]
    p["y"] = pd.concat(out)
    wf = st.walk_forward(p, end21, {"xgb": feats}, years, log=lambda s: None)
    r = ranked(wf, "p_xgb")
    ex = excess_series(r, "ret21", TOP_N, cost)
    top = r[(r["rank"] <= TOP_N) & r["y_real"].notna()]
    res = {**evaluate._summ(ex), "hit_rate_top2pct": float(top["y_real"].mean())}
    if uni is not None:
        res["pick_percentiles"] = pick_percentiles(uni, top)
    return res


def placebo_stale(wf, cost):
    """Week t's picks held over the window that starts one signal date later."""
    dates = sorted(wf["date"].unique())
    nxt = dict(zip(dates[:-1], dates[1:]))
    r = ranked(wf, "p_xgb")
    top = r[r["rank"] <= TOP_N][["symbol", "date"]].assign(date=lambda x: x["date"].map(nxt)).dropna()
    later = wf[["symbol", "date", "ret21"]]
    rows = []
    for t, g in later.groupby("date", sort=True):
        picks = top[top["date"] == t].merge(g, on=["symbol", "date"])
        if len(picks):
            rows.append({"date": t, "pick_mean": picks["ret21"].mean(), "uni_mean": g["ret21"].mean(),
                         "excess": picks["ret21"].mean() - g["ret21"].mean() - cost, "n_picks": len(picks)})
    return evaluate._summ(pd.DataFrame(rows))


def diagnose(uni, feats, end21, cost, years):
    panel = evaluate.model_panel(uni, feats)
    wf = walk_forward_shap(panel, end21, feats, years)
    r = ranked(wf, "p_xgb")
    picks = r[r["rank"] <= TOP_N]
    real = evaluate._summ(excess_series(r, "ret21", TOP_N, cost))
    return {"real": real,
            "pick_percentiles": pick_percentiles(uni, picks),
            "shap_share_all_rows": shap_shares(wf, feats),
            "shap_share_picks": shap_shares(picks, feats),
            "placebo_shuffled_labels": placebo_shuffled(panel, end21, feats, years, cost, uni),
            "baseline_top20_vol63": vol_baseline(uni, panel, end21, years),
            "placebo_stale_signal": placebo_stale(wf, cost)}


def verdicts(res):
    m, p = res["M10"], res["P3"]
    high = lambda d: all(d["pick_percentiles"].get(c, 0) >= 0.9 for c in ("vol21", "maxret21"))
    return {"a_picks_top_decile_vol_and_max": bool(high(m) and high(p)),
            "b_volatility_features_lead": bool(all(d["shap_share_picks"]["volatility"] ==
                                                   max(d["shap_share_picks"].values()) for d in (m, p))),
            "c_factor_adjusted": "see V2",
            "d_p3_worse_than_m10": bool(p["real"]["mean"] < m["real"]["mean"]),
            "leakage_flag": bool(any(d["placebo_shuffled_labels"]["mean"] > 0.01 or
                                     d["placebo_shuffled_labels"]["hit_rate_top2pct"] > 0.04 for d in (m, p)))}


def run(cache, out, budget_min=290, years=None):
    cache, out = Path(cache), Path(out)
    years = years or evaluate.YEARS
    got = pn.build(cache, time.monotonic() + budget_min * 60)
    if got is None:
        print("PANEL INCOMPLETE: rerun to continue")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    rows = panel[(panel["price"] >= screen_m10.MIN_PRICE) & (panel["dv20"] >= screen_m10.MIN_DV20)]
    facts = sec_data.facts_for(zp, rows["cik"].unique(), cache / "boom" / "facts_m10")
    res = {"study": "V5", "prereg": "research/prereg/V5.md"}
    res["M10"] = diagnose(screen_m10.universe(panel, rtab, facts), screen_m10.FEATURES, ends["ret21"],
                          screen_m10.COST, years)
    res["P3"] = diagnose(screen_p3.universe(panel), screen_p3.FEATURES, ends["ret21"], screen_p3.COST, years)
    res["predictions"] = verdicts(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "V5_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "V5"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
