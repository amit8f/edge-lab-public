"""Screen runner for the short-term boom models (M10 main, P3 penny). Each track passes its own universe
rows, feature list, costs and output folder; models are fitted here per call, so nothing fitted crosses
tracks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import shortterm as st
from tracks.main.longterm.screen import SEED, excess_series, summarize

TOP_N = 20
LAGS, BLOCK = 6, 8
YEARS = range(2019, 2026)
T_BAR = 3.0


def model_panel(uni, feats):
    p = uni[["symbol", "date", "cik", "price", "ret21", "ret21_trunc", "ret5"]].reset_index(drop=True)
    p = pd.concat([p, st.rank_within_date(uni.reset_index(drop=True), feats)], axis=1)
    p["y"] = st.top_label(p, "ret21")
    p["ret21_rank"] = p.groupby("date")["ret21"].rank(pct=True)
    return p


def _summ(ex):
    return summarize(ex, LAGS, BLOCK)


def _with_rank(df, col, ascending=False):
    return df.assign(rank=df.groupby("date")[col].rank(ascending=ascending, method="first"))


def hit_rate(df, col, n=TOP_N):
    top = _with_rank(df, col)
    top = top[(top["rank"] <= n) & top["y"].notna()]
    return float(top["y"].mean())


def run(uni, feats, cost, borrow_year, idea, counts, end21, spy_r, out, short_ok=None, fat_tails=False,
        log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"idea": idea, "prereg": f"research/prereg/{idea}.md", **counts, "features": feats,
           "xgb_params": {k: v for k, v in st.XGB_PARAMS.items() if k != "n_jobs"}}
    panel = model_panel(uni, feats)
    per_date = panel.groupby("date").size()
    res["funnel_universe"] = {"rows": int(len(panel)), "dates": int(per_date.size), "per_date_min": int(per_date.min()),
                              "per_date_median": float(per_date.median()), "per_date_max": int(per_date.max()),
                              "label_rate": float(panel["y"].mean()),
                              "feature_coverage": {f: float(uni[f].notna().mean()) for f in feats}}
    log("UNIVERSE " + json.dumps(res["funnel_universe"], indent=1))

    news = [f for f in feats if f not in st.NEWS_FEATURES]
    models = {"xgb": feats, "lgbm": feats, "logit": feats, "xgb_nonews": news, "xgb_reg": feats}
    wf = st.walk_forward(panel, end21, models, YEARS, log)
    wf["p_blend"] = sum(wf.groupby("date")[f"p_{m}"].rank(pct=True) for m in ("xgb", "lgbm", "logit")) / 3

    # ---- primary (pre-registered) ----
    prim_df = _with_rank(wf, "p_xgb")
    ex = excess_series(prim_df, "ret21", TOP_N, cost)
    prim = _summ(ex)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= T_BAR and prim["ci95"][0] > 0)
    prim["picks_without_return"] = int((TOP_N - ex["n_picks"]).sum())
    prim["hit_rate_top2pct"] = hit_rate(wf, "p_xgb")
    prim["median_excess"] = float(ex["excess"].median())
    res["primary"] = prim
    res["control"] = _summ(excess_series(prim_df, "ret21", TOP_N, cost, rng_seed=SEED))
    log("PRIMARY " + json.dumps(prim, indent=1))
    log("CONTROL " + json.dumps(res["control"], indent=1))
    if fat_tails:
        top = prim_df[prim_df["rank"] <= TOP_N].dropna(subset=["ret21"])
        g = top.groupby("date")["ret21"]
        uni_m = prim_df.groupby("date")["ret21"].mean()
        wo = ((g.sum() - g.max()) / (g.count() - 1)) - uni_m.loc[g.sum().index] - cost
        res["primary"]["mean_without_best_pick"] = float(wo.mean())

    # ---- QA ----
    r = panel["ret21"].dropna()
    res["qa"] = {"ret21_percentiles": {str(q): float(r.quantile(q)) for q in (0, .001, .01, .5, .98, .99, .999, 1)},
                 "truncated_share": float(panel.loc[panel["ret21"].notna(), "ret21_trunc"].mean()),
                 "largest_ret21": panel.nlargest(15, "ret21")[["symbol", "date", "price", "ret21", "ret21_trunc"]]
                 .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
                 "universe_mean_ret21": float(ex["uni_mean"].mean()),
                 "spy_mean_ret21": float(np.mean([spy_r["ret21"][t] for t in ex["date"]]))}
    log("QA " + json.dumps(res["qa"], indent=1, default=str))

    # ---- exploratory (listed in the pre-registration) ----
    exp = {}
    for m in ("lgbm", "logit", "blend", "xgb_nonews", "xgb_reg"):
        exp[m] = _summ(excess_series(_with_rank(wf, f"p_{m}"), "ret21", TOP_N, cost))
        exp[m]["hit_rate_top2pct"] = hit_rate(wf, f"p_{m}")
    exp["top10"] = _summ(excess_series(prim_df, "ret21", 10, cost))
    exp["top50"] = _summ(excess_series(prim_df, "ret21", 50, cost))
    exp["vs_spy"] = _summ(ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret21"]) - cost))
    exp["per_year"] = {str(y): {"mean": float(g["excess"].mean()), "dates": int(len(g))}
                       for y, g in ex.groupby(ex["date"].dt.year)}
    sh = wf if short_ok is None else wf[wf["symbol"].isin(short_ok)]
    exs = excess_series(_with_rank(sh, "p_xgb", ascending=True), "ret21", TOP_N, 0.0)
    exs["excess"] = exs["uni_mean"] - exs["pick_mean"] - cost - borrow_year * 21 / 252
    exp["short_bottom20"] = _summ(exs)
    if short_ok is not None:
        exp["short_bottom20"]["note"] = "easy-to-borrow by today's Alpaca list, so optimistic"
    ex5 = excess_series(prim_df[prim_df["ret5"].notna()], "ret5", TOP_N, cost)
    exp["horizon_5d_same_model"] = summarize(ex5, 1, 4)
    res["exploratory"] = exp
    log("EXPLORATORY " + json.dumps(exp, indent=1))

    # ---- calibration for the dashboard (out-of-sample top 20) ----
    top = prim_df[(prim_df["rank"] <= TOP_N) & prim_df["ret21"].notna()]
    res["calibration"] = {"n": int(len(top)), "median": float(top["ret21"].median()),
                          "p25": float(top["ret21"].quantile(.25)), "p75": float(top["ret21"].quantile(.75)),
                          "share_top2pct": float(top["y"].mean()), "share_up_30pct": float((top["ret21"] >= 0.3).mean()),
                          "share_down_30pct": float((top["ret21"] <= -0.3).mean()),
                          "beat_spy": float((top["ret21"] > top["date"].map(spy_r["ret21"])).mean())}

    # ---- final model: refit on every label closed before the holdout ----
    lab = panel[panel["y"].notna()]
    _, final = st.fit_predict("xgb", lab, lab.head(1), feats)
    final.save_model(out / f"{idea}_xgb.json")
    gain = final.get_booster().get_score(importance_type="gain")
    res["feature_gain"] = {feats[int(k[1:])]: float(v) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    last = prim_df[prim_df["date"] == prim_df["date"].max()].nsmallest(TOP_N, "rank")
    res["hand_check_last_date"] = last[["symbol", "date", "price", "p_xgb", "ret21", "y"]].assign(
        date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records")

    (out / f"{idea}_screen.json").write_text(json.dumps(res, indent=1, default=str))
    ex.to_csv(out / f"{idea}_primary_series.csv", index=False)
    prim_df[prim_df["rank"] <= TOP_N][["date", "rank", "symbol", "price", "p_xgb", "ret21", "y"]].to_csv(
        out / f"{idea}_screen_picks.csv", index=False)
    return res
