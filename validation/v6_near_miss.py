"""V6 (post-hoc, not pre-registered; Amit asked 2026-10-29 to diagnose the near-misses M6 and A1): where
each screen's excess return came from, from the committed result files, plus a raw-bar look at A1's
largest trades. Changes no verdict and adds no idea; a fix found here is a new, forward-tested idea.

Usage: python -m validation.v6_near_miss --cache <dir> [--out <dir>] [--budget-min N]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm.stats import nw_tstat

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "research" / "results"
LAGS = 3          # A1's monthly cohorts overlap by up to 3 months (63-day hold), as in a1_screen
DROP = (1, 2, 3, 5)


def a1_months(picks):
    """Per signal month: mean flag short return minus mean random-control short return (gross)."""
    m = picks.pivot_table(index="date", columns="role", values="gross", aggfunc="mean").dropna()
    return (m["flag"] - m["control"]).sort_index()


def without_best(diff, k):
    """Mean and HAC t of the monthly differences after dropping the k best months."""
    d = diff.drop(diff.sort_values().index[-k:]) if k else diff
    m, _, t = nw_tstat(d.to_numpy(), LAGS)
    return {"dropped": int(k), "months": int(len(d)), "mean": float(m), "t_hac": float(t)}


def a1_diagnosis(picks):
    diff = a1_months(picks)
    role = picks.groupby("role").agg(trades=("gross", "size"), mean_gross=("gross", "mean"),
                                     median_gross=("gross", "median"), stopped=("stopped", "mean"),
                                     won=("gross", lambda x: float((x > 0).mean())))
    worst = picks.reindex(picks["gross"].sort_values().index[:8])
    flags = picks[picks["role"] == "flag"]
    bins = pd.cut(flags["m"], [-np.inf, 0, 2, 10, np.inf], labels=["-1.78..0", "0..2", "2..10", ">10"])
    by_m = flags.groupby(bins, observed=False)["gross"].agg(["size", "mean", "median"])
    contrib = diff.sort_values(ascending=False)
    return {
        "months": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "share_from_best_2_months": float(contrib.iloc[:2].sum() / diff.sum()),
        "drop_best": [without_best(diff, k) for k in (0, *DROP)],
        "best_months": {str(k): float(v) for k, v in contrib.iloc[:5].items()},
        "by_role": {r: {k: float(v) for k, v in row.items()} for r, row in role.iterrows()},
        "worst_trades": worst[["date", "symbol", "role", "gross", "held", "stopped"]].to_dict("records"),
        "flags_by_mscore": {str(k): {c: float(v) for c, v in row.items()} for k, row in by_m.iterrows()},
    }


def raw_check(trades, headers):
    """Raw and adjusted daily closes from entry to exit for A1's largest losses, to tell a real gap from a
    split or symbol-reuse artifact."""
    out = []
    for t in trades:
        start = pd.Timestamp(t["date"]) - pd.Timedelta(days=5)
        end = (pd.Timestamp(t["date"]) + pd.Timedelta(days=int(t["held"]) * 2 + 10)).strftime("%Y-%m-%d")
        start = start.strftime("%Y-%m-%d")
        rows = {}
        for adj in ("raw", "all"):
            f = to_frames(fetch_daily_bars([t["symbol"]], start, end, adjustment=adj, headers=headers))
            b = f.get(t["symbol"])
            rows[adj] = [] if b is None else [[f"{i:%Y-%m-%d}", float(r["o"]), float(r["c"]), float(r["v"])]
                                              for i, r in b.iloc[:int(t["held"]) + 7].iterrows()]
        out.append({**{k: t[k] for k in ("date", "symbol", "role", "gross", "held")}, "bars": rows})
    return out


def m6_diagnosis():
    """M6's factor loadings as V2 computed them (FF5 + momentum on the 6-month excess), and who the picks
    were."""
    v2 = json.loads((RES / "V2" / "V2_results.json").read_text())["ideas"]["M6"]
    picks = pd.read_csv(RES / "M6" / "M6_screen_picks.csv")
    top = picks["symbol"].value_counts()
    return {"raw_excess_6m": v2["raw_mean"], "alpha_6m": v2["alpha"], "alpha_t": v2["alpha_t"], "r2": v2["r2"],
            "loadings": v2["loadings"], "pick_slots": int(len(picks)), "distinct_symbols": int(top.size),
            "most_picked": {k: int(v) for k, v in top.head(12).items()},
            "top12_share_of_slots": float(top.head(12).sum() / len(picks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(RES / "V6"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = pd.read_csv(RES / "A1" / "A1_picks.csv")
    res = {"study": "V6", "post_hoc": True, "A1": a1_diagnosis(picks), "M6": m6_diagnosis()}
    res["A1"]["raw_check"] = raw_check(res["A1"]["worst_trades"][:4], alpaca_headers())
    tmp = out / "V6_results.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=str))
    tmp.replace(out / "V6_results.json")
    print(json.dumps({k: res["A1"][k] for k in ("mean_diff", "share_from_best_2_months", "drop_best")}, indent=1))


if __name__ == "__main__":
    main()
