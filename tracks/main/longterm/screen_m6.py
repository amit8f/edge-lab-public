"""M6 historical screen (research/prereg/M6.md): analyst actions from Benzinga headlines.

The headline download is large, so it is resumable: each run fetches missing months until its time
budget ends; the screen itself runs only once every month is cached.
Usage: python -m tracks.main.longterm.screen_m6 --cache <dir> --out <dir> [--budget-min 280]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main.longterm import analyst, news_history
from tracks.main.longterm.screen import COST, SEED, calibration, excess_series, prepare, summarize
from tracks.main.longterm.score import TOP_N

FIRST_MONTH, LAST_MONTH = "2016-10", "2025-09"


def classified_rows(cache_dir, ratings_only=False):
    """Single-symbol headlines that classify as an action, read month by month (never the full feed),
    and the total article count."""
    keep, n = [], 0
    for per in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"):
        m = pd.read_parquet(Path(cache_dir) / f"news_{per}.parquet")
        n += len(m)
        m = m[~m["symbols"].fillna("").str.contains(",") & (m["symbols"].fillna("") != "")]
        d = m["headline"].map(lambda h: analyst.classify(h, ratings_only))
        keep.append(m[d != 0])
    return pd.concat(keep, ignore_index=True), n


def run(cache, out, budget_min=280):
    cache, out = Path(cache), Path(out)
    deadline = time.monotonic() + budget_min * 60
    if not news_history.fetch_range(FIRST_MONTH, LAST_MONTH, cache / "news", deadline):
        print("NEWS DOWNLOAD INCOMPLETE: rerun this workflow to continue (cached months are kept)")
        return None
    out.mkdir(parents=True, exist_ok=True)
    res = {"idea": "M6", "prereg": "research/prereg/M6.md", "phase2_count": 3, "main_count": 3,
           "phase1_plus_phase2": 18}
    rows, n_articles = classified_rows(cache / "news")
    ev = analyst.events(rows)
    uni, ends, spy_r, funnel = prepare(cache)
    scored = analyst.score_m6(uni, ev, SEED)
    funnel.update({"articles": int(n_articles), "classified_single_symbol": int(len(rows)),
                   "events_after_dedupe": int(len(ev)), "upgrades_or_raises": int((ev["direction"] > 0).sum()),
                   "downgrades_or_cuts": int((ev["direction"] < 0).sum()),
                   "universe_share_with_action": float((scored["an_total"] > 0).mean())})
    res["funnel"] = funnel
    print("FUNNEL", json.dumps(funnel, indent=1))
    sample = ev.sample(min(50, len(ev)), random_state=SEED)
    res["classification_sample"] = sample.assign(day=sample["day"].dt.strftime("%Y-%m-%d")).to_dict("records")
    print("SAMPLE", json.dumps(res["classification_sample"], indent=1))

    s6 = scored[scored["date"].isin(ends["ret6"].keys())]
    ex = excess_series(s6, "ret6", TOP_N, COST)
    prim = summarize(ex, 5)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= 2.0 and prim["ci95"][0] > 0)
    prim["picks_without_return"] = int((TOP_N - ex["n_picks"]).sum())
    res["primary"] = prim
    res["control"] = summarize(excess_series(s6, "ret6", TOP_N, COST, rng_seed=SEED), 5)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))

    exp = {}
    ev_r = analyst.events(classified_rows(cache / "news", ratings_only=True)[0], ratings_only=True)
    exp["ratings_only"] = summarize(excess_series(analyst.score_m6(uni, ev_r, SEED)
                                                  .pipe(lambda x: x[x["date"].isin(ends["ret6"].keys())]),
                                                  "ret6", TOP_N, COST), 5)
    s30 = analyst.score_m6(uni, ev, SEED, window_days=30)
    exp["window_30d"] = summarize(excess_series(s30[s30["date"].isin(ends["ret6"].keys())], "ret6", TOP_N, COST), 5)
    s12 = scored[scored["date"].isin(ends["ret12"].keys())]
    exp["horizon_12m"] = summarize(excess_series(s12, "ret12", TOP_N, COST), 11)
    exp["top50"] = summarize(excess_series(s6, "ret6", 50, COST), 5)
    exp["vs_spy"] = summarize(ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret6"]) - COST), 5)
    exp["per_year"] = {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1))

    cal = calibration(s6, spy_r["ret6"])
    cal.update({"source": "M6 screen", "primary_passes": prim["passes"], "primary_mean_excess": prim["mean"],
                "primary_t": prim["t_hac"]})
    res["calibration"] = cal
    (out / "M6_screen.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "M6_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
    ex.to_csv(out / "M6_primary_series.csv", index=False)
    s6[s6["rank"] <= TOP_N][["date", "rank", "symbol", "score", "an_up", "an_down", "ret6"]].to_csv(
        out / "M6_screen_picks.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=280)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
