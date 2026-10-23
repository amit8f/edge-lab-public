"""R1 historical screen (research/prereg/R1.md): each week buy the bottom-decile 5-day losers among
$300M+ stocks that are still above their 200-day average, hold 5 days, against a same-week random
control.

Usage: python -m tracks.main.r1_screen --cache <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main import weekly_panel

COST, PENNY_COST, SEED = 0.002, 0.02, 20261023
MAX_PICKS = 20


def pick(rows, universe, dates):
    """Per week: losers (bottom decile of the universe by ret5, ties by symbol, above the 200-day average,
    at most MAX_PICKS lowest) and a control of as many names from the rest, default_rng(SEED + week
    index). Picks are made before looking at returns."""
    out = []
    u = rows[rows[universe] & rows["ret5"].notna()]
    for w, t in enumerate(dates):
        g = u[u["date"] == t].sort_values(["ret5", "symbol"]).reset_index(drop=True)
        n = len(g) // 10
        if n < 1:
            continue
        low = g.iloc[:n]
        los = low[low["above200"]].head(MAX_PICKS)
        if los.empty:
            continue
        rest = g[~g["symbol"].isin(los["symbol"])].sort_values("symbol").reset_index(drop=True)
        k = np.random.default_rng(SEED + w).choice(len(rest), size=min(len(los), len(rest)), replace=False)
        out.append(los.assign(role="pick", week=w))
        out.append(rest.iloc[np.sort(k)].assign(role="control", week=w))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def eightk_split(sel, eightk, cost):
    """Descriptive: losers with an 8-K filed in the 5 calendar days up to t vs without."""
    s = sel[(sel["role"] == "pick")].dropna(subset=["fwd5", "cik"]).copy()
    if s.empty or eightk.empty:
        return {}
    e = eightk[["cik", "filed"]].sort_values("filed")
    s["cik"] = s["cik"].astype("int64")
    s = s.sort_values("date")
    m = pd.merge_asof(s, e.rename(columns={"filed": "f"}), left_on="date", right_on="f", by="cik",
                      direction="backward", tolerance=pd.Timedelta(days=5))
    m["with_8k"] = m["f"].notna()
    return {str(k): {"n": int(len(g)), "mean_net": float((g["fwd5"] - cost).mean())} for k, g in m.groupby("with_8k")}


def run(cache, out):
    rows, cal, dates, funnel, _ = weekly_panel.build(cache)
    fac, sha5, shau = weekly_panel.factors()
    p2, both = stats.idea_counts()
    sel = pick(rows, "main", dates)
    t = weekly_panel.weekly(sel, COST)
    prim = weekly_panel.summary(t, SEED)
    a = weekly_panel.alpha(t, fac, cal)
    prim["factor_alpha"] = a
    prim["deflated_sharpe"] = stats.deflated_sharpe(t["diff"].to_numpy(), both)
    prim["passes"] = stats.screen_pass(prim, a)
    prim["first_week"], prim["last_week"] = str(t.index.min().date()), str(t.index.max().date())
    prim["weeks_without_losers"] = int(len(dates) - sel["date"].nunique()) if len(sel) else len(dates)
    los = sel[sel["role"] == "pick"]
    small = sel.assign(small=sel["mcap"] < 2e9)
    eightk = weekly_panel.filings({"8-K", "8-K/A"}, cache / "edgar")
    res = {"idea": "R1", "prereg": "research/prereg/R1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_daily_sha256": sha5, "umd_daily_sha256": shau, "cost_round_trip": COST, "funnel": funnel,
           "PRIMARY": prim,
           "alongside": {
               "before_costs": weekly_panel.summary(weekly_panel.weekly(sel, 0.0), SEED),
               "losers_mcap_300m_2b": weekly_panel.summary(
                   weekly_panel.weekly(small[(small["role"] == "control") | small["small"]], COST), SEED),
               "losers_mcap_2b_plus": weekly_panel.summary(
                   weekly_panel.weekly(small[(small["role"] == "control") | ~small["small"]], COST), SEED),
               "losers_by_8k_in_5_days": eightk_split(sel, eightk, COST),
               "penny_universe_2pct": weekly_panel.summary(
                   weekly_panel.weekly(pick(rows, "penny", dates), PENNY_COST), SEED),
               "per_year": t.groupby(t.index.year)["diff"].mean().to_dict(),
               "losers_median_ret5": float(los["ret5"].median()),
               "quote_costs": "not in this run: measured on 300 random picks with core/costs.py before the report"}}
    out.mkdir(parents=True, exist_ok=True)
    chk = los.dropna(subset=["fwd5"]).sample(n=min(20, len(los)), random_state=SEED)
    chk[["date", "symbol", "price", "ret5", "above200", "fwd5"]].assign(checked_by="", ok="", note="").to_csv(
        out / "R1_rows_check.csv", index=False)
    sel[["date", "week", "symbol", "role", "ret5", "mcap", "fwd5"]].to_csv(out / "R1_picks.csv", index=False)
    (out / "R1_result.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
