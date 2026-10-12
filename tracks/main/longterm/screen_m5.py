"""M5 historical screen (research/prereg/M5.md): insider buying plus buybacks. Reuses M4's data
preparation and statistics. Usage: python -m tracks.main.longterm.screen_m5 --cache <dir> --out <dir>"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tracks.main.longterm import sec_data
from tracks.main.longterm.insider import score_m5
from tracks.main.longterm.screen import COST, LAST_SIGNAL, SEED, calibration, excess_series, prepare, summarize
from tracks.main.longterm.score import TOP_N


def run(cache, out):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"idea": "M5", "prereg": "research/prereg/M5.md", "phase2_count": 2, "main_count": 2,
           "phase1_plus_phase2": 17}
    uni, ends, spy_r, funnel = prepare(cache)
    buys = sec_data.dera_insider_buys(2015, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    buys = buys[buys["filed"] <= LAST_SIGNAL]
    scored = score_m5(uni, buys)
    per_date = scored.groupby("date").size()
    funnel.update({"purchase_rows": int(len(buys)), "purchase_issuers": int(buys["cik"].nunique()),
                   "with_issuance": int(len(scored)),
                   "share_with_any_buy": float((scored["insider_value"] > 0).mean()),
                   "picks_with_any_buy": float((scored.loc[scored["rank"] <= TOP_N, "insider_value"] > 0).mean()),
                   "universe_per_date": {"min": int(per_date.min()), "median": float(per_date.median()),
                                         "max": int(per_date.max())}})
    res["funnel"] = funnel
    print("FUNNEL", json.dumps(funnel, indent=1))

    s6 = scored[scored["date"].isin(ends["ret6"].keys())]
    ex = excess_series(s6, "ret6", TOP_N, COST)
    prim = summarize(ex, 5)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= 2.0 and prim["ci95"][0] > 0)
    prim["picks_without_return"] = int((TOP_N - ex["n_picks"]).sum())
    res["primary"] = prim
    res["control"] = summarize(excess_series(s6, "ret6", TOP_N, COST, rng_seed=SEED), 5)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))

    top_buys = buys.nlargest(20, "value").assign(filed=lambda x: x["filed"].dt.strftime("%Y-%m-%d"))
    res["qa"] = {"largest_buys": top_buys[["cik", "filed", "accn", "shares", "price", "value"]].to_dict("records"),
                 "largest_ratio": scored.nlargest(10, "insider_ratio")[["symbol", "date", "insider_value", "mcap", "insider_ratio"]]
                 .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records")}
    print("QA", json.dumps(res["qa"], indent=1, default=str))

    exp = {}
    alone = s6.assign(rank=s6.groupby("date")["insider_ratio"].rank(ascending=False, method="first"))
    exp["insider_alone"] = summarize(excess_series(alone, "ret6", TOP_N, COST), 5)
    nb = s6.copy()
    nb["score"] = (nb.groupby("date")["insider_buyers"].rank(pct=True) + nb["f_issuance"]) / 2
    nb["rank"] = nb.groupby("date")["score"].rank(ascending=False, method="first")
    exp["n_buyers_version"] = summarize(excess_series(nb, "ret6", TOP_N, COST), 5)
    s12 = scored[scored["date"].isin(ends["ret12"].keys())]
    exp["horizon_12m"] = summarize(excess_series(s12, "ret12", TOP_N, COST), 11)
    exp["top50"] = summarize(excess_series(s6, "ret6", 50, COST), 5)
    exp["vs_spy"] = summarize(ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret6"]) - COST), 5)
    exp["per_year"] = {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1))

    cal = calibration(s6, spy_r["ret6"])
    cal.update({"source": "M5 screen", "primary_passes": prim["passes"], "primary_mean_excess": prim["mean"],
                "primary_t": prim["t_hac"]})
    res["calibration"] = cal
    (out / "M5_screen.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "M5_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
    ex.to_csv(out / "M5_primary_series.csv", index=False)
    s6[s6["rank"] <= TOP_N][["date", "rank", "symbol", "cik", "score", "insider_value", "insider_buyers",
                             "issuance", "ret6"]].to_csv(out / "M5_screen_picks.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
