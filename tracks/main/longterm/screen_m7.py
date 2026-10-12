"""M7 historical screen (research/prereg/M7.md): standardized earnings surprise from SEC filings.
Usage: python -m tracks.main.longterm.screen_m7 --cache <dir> --out <dir>"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main.longterm import earnings, sec_data
from tracks.main.longterm.screen import COST, LAST_SIGNAL, SEED, calibration, excess_series, prepare, summarize
from tracks.main.longterm.score import TOP_N


def run(cache, out):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"idea": "M7", "prereg": "research/prereg/M7.md", "phase2_count": 4, "main_count": 4,
           "phase1_plus_phase2": 19}
    uni, ends, spy_r, funnel = prepare(cache)
    facts = sec_data.facts_for(cache / "companyfacts.zip", uni["cik"].unique(), cache / "facts")
    q = earnings.quarterly_ni(facts)
    q = q[q["avail"] <= LAST_SIGNAL]
    sue = earnings.sue_table(q)
    scored = earnings.score_m7(uni, sue, SEED)
    late = int((scored["sue_avail"] > scored["date"]).sum())
    per_date = scored.groupby("date").size()
    funnel.update({"companies_with_quarters": int(q["cik"].nunique()), "quarters": int(len(q)),
                   "quarters_with_sue": int(sue["sue"].notna().sum()), "with_fresh_sue": int(len(scored)),
                   "universe_share_with_sue": float(len(scored) / max(len(uni), 1)),
                   "sue_filed_after_signal_date": late,
                   "universe_per_date": {"min": int(per_date.min()), "median": float(per_date.median()),
                                         "max": int(per_date.max())}})
    res["funnel"] = funnel
    print("FUNNEL", json.dumps(funnel, indent=1))
    if late:
        raise RuntimeError(f"{late} SUE values filed after their signal date: look-ahead bug")

    s6 = scored[scored["date"].isin(ends["ret6"].keys())]
    ex = excess_series(s6, "ret6", TOP_N, COST)
    prim = summarize(ex, 5)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= 2.0 and prim["ci95"][0] > 0)
    prim["picks_without_return"] = int((TOP_N - ex["n_picks"]).sum())
    res["primary"] = prim
    res["control"] = summarize(excess_series(s6, "ret6", TOP_N, COST, rng_seed=SEED), 5)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))

    big = sue.dropna(subset=["sue"]).reindex(sue["sue"].abs().sort_values(ascending=False).index).head(20)
    res["qa"] = {"largest_abs_sue": big.assign(end=big["end"].dt.strftime("%Y-%m-%d"),
                                               avail=big["avail"].dt.strftime("%Y-%m-%d"))
                 [["cik", "end", "avail", "ni", "change", "sue"]].to_dict("records"),
                 "sue_percentiles": {str(p): float(np.nanpercentile(scored["sue"], p)) for p in (1, 5, 50, 95, 99)}}
    print("QA", json.dumps(res["qa"], indent=1, default=str))

    exp = {}
    dec = s6.assign(decile=np.ceil(s6.groupby("date")["score"].rank(pct=True) * 10).clip(1, 10))
    exp["deciles"] = {str(int(k)): float((g["ret6"] - g["date"].map(s6.groupby("date")["ret6"].mean())).mean())
                      for k, g in dec.groupby("decile")}
    s12 = scored[scored["date"].isin(ends["ret12"].keys())]
    exp["horizon_12m"] = summarize(excess_series(s12, "ret12", TOP_N, COST), 11)
    exp["top50"] = summarize(excess_series(s6, "ret6", 50, COST), 5)
    exp["vs_spy"] = summarize(ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret6"]) - COST), 5)
    exp["per_year"] = {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}
    exp["filing_reaction"] = "not computed: daily bars around filing dates are not cached by this pipeline"
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1))

    cal = calibration(s6, spy_r["ret6"])
    cal.update({"source": "M7 screen", "primary_passes": prim["passes"], "primary_mean_excess": prim["mean"],
                "primary_t": prim["t_hac"]})
    res["calibration"] = cal
    (out / "M7_screen.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "M7_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
    ex.to_csv(out / "M7_primary_series.csv", index=False)
    s6[s6["rank"] <= TOP_N][["date", "rank", "symbol", "cik", "score", "q_end", "ret6"]].to_csv(
        out / "M7_screen_picks.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
