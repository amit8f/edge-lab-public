"""V3 (research/prereg/V3.md): can M8's forward test reach a verdict?

Part A: power of M8's amended test, from M6's published 6-month series (no data download).
Part B: M6's exact ranking on the weekly panel at 21 days (2017-2025, before the holdout), measured three
ways (M8's top-20 statistic, whole-universe rank IC, net-direction spread), with the forward years each
needs for 80% power.

Usage: python -m validation.v3_power --cache <dir> [--out <dir>] [--budget-min N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tracks.main.longterm import analyst, news_history, sec_data
from tracks.main.longterm.screen_m6 import FIRST_MONTH, LAST_MONTH, classified_rows
from tracks.main.longterm.stats import nw_tstat
from tracks.main.shortterm import panel as pn, screen_m10
from validation.stats import period_sr

ROOT = Path(__file__).resolve().parents[1]
COST, TOP_N, LAGS, SEED = 0.002, 20, 6, 20261012
Z80 = norm.ppf(0.8)
T_BAR = 2.0
DAYS = 21


def years_for_power(ir_annual, bar=T_BAR, power_z=Z80):
    """Years of data for P(t >= bar) = 80% when t grows as IR x sqrt(years)."""
    return float(((bar + power_z) / ir_annual) ** 2) if ir_annual > 0 else float("inf")


def part_a(m6_series):
    """M6's 6-month effect scaled to 21 days; expected t and power at 26 and 52 weekly logs."""
    x = pd.read_csv(m6_series)["excess"].to_numpy()
    sr6, _, _, _ = period_sr(x, 6)
    subs = [x[o::6] for o in range(6)]
    mean6 = float(np.mean([s.mean() for s in subs]))
    sd6 = float(np.mean([s.std(ddof=1) for s in subs]))
    k = 126 / DAYS
    mean21, sd21 = mean6 / k, sd6 / np.sqrt(k)
    ir_annual = mean21 / sd21 * np.sqrt(252 / DAYS)
    out = {"mean_6m": mean6, "sd_6m": sd6, "mean_21d": mean21, "sd_21d": sd21, "ir_annual": float(ir_annual),
           "years_for_80pct_power": years_for_power(ir_annual)}
    for weeks in (26, 52, 104):
        n_eff = weeks * 5 / DAYS                     # independent 21-day windows in `weeks` of weekly logs
        et = mean21 / sd21 * np.sqrt(n_eff)
        out[f"weeks_{weeks}"] = {"expected_t": float(et), "p_t_ge_2": float(norm.sf(T_BAR - et))}
    return out


def weekly_stats(scored, ret="ret21"):
    """Per week: M8's top-20 excess, a same-day random 20, rank IC across the universe, and the
    net-direction spread (net score > 0 minus < 0), costs charged where a trade is implied."""
    rows = []
    for t, g in scored.groupby("date", sort=True):
        g = g[g[ret].notna()]
        if len(g) < TOP_N:
            continue
        uni = g[ret].mean()
        top = g.nsmallest(TOP_N, "rank")
        rnd = g.iloc[np.random.default_rng(SEED + t.toordinal()).choice(len(g), TOP_N, replace=False)]
        pos, neg = g[g["score"] > 0], g[g["score"] < 0]
        rows.append({"date": t, "top20": top[ret].mean() - uni - COST, "random20": rnd[ret].mean() - uni - COST,
                     "ic": g["score"].rank().corr(g[ret].rank()),
                     "net_spread": (pos[ret].mean() - neg[ret].mean() - COST) if len(pos) and len(neg) else np.nan,
                     "n_pos": len(pos), "n_neg": len(neg), "n": len(g)})
    return pd.DataFrame(rows)


def summarize_stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 12:
        return {"n_weeks": int(len(x)), "ir_annual": float("nan"), "years_for_80pct_power": float("inf")}
    m, se, t = nw_tstat(x, LAGS)
    sr, sk, ku, T = period_sr(x, 4)
    ir = sr * np.sqrt(52 / 4)
    return {"mean": m, "t_hac": t, "n_weeks": int(len(x)), "ir_annual": float(ir),
            "years_for_80pct_power": years_for_power(ir), "skew": sk}


def part_b(cache, deadline):
    cache = Path(cache)
    if not news_history.fetch_range(FIRST_MONTH, LAST_MONTH, cache / "news", deadline):
        print("NEWS DOWNLOAD INCOMPLETE: rerun to continue")
        return None
    got = pn.build(cache, deadline)
    if got is None:
        print("PANEL INCOMPLETE: rerun to continue")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    rows = panel[(panel["price"] >= screen_m10.MIN_PRICE) & (panel["dv20"] >= screen_m10.MIN_DV20)]
    facts = sec_data.facts_for(zp, rows["cik"].unique(), cache / "boom" / "facts_m10")
    uni = screen_m10.universe(panel, rtab, facts)
    uni = uni[uni["date"].isin(list(ends["ret21"]))]
    ev = analyst.events(classified_rows(cache / "news")[0])
    scored = analyst.score_m6(uni, ev, SEED)
    w = weekly_stats(scored)
    res = {k: summarize_stat(w[k]) for k in ("top20", "random20", "ic", "net_spread")}
    res["coverage"] = {"weeks": int(len(w)), "median_universe": float(w["n"].median()),
                       "median_net_pos": float(w["n_pos"].median()), "median_net_neg": float(w["n_neg"].median())}
    res["msprt_tau"] = abs(res["top20"]["mean"])
    return res, w


def run(cache, out, budget_min=290, m6_series=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m6_series = m6_series or ROOT / "research" / "results" / "M6" / "M6_primary_series.csv"
    res = {"study": "V3", "prereg": "research/prereg/V3.md", "part_a": part_a(m6_series)}
    print("PART A", json.dumps(res["part_a"], indent=1))
    got = part_b(cache, time.monotonic() + budget_min * 60)
    if got is None:
        (out / "V3_part_a.json").write_text(json.dumps(res, indent=1))
        return None
    res["part_b"], w = got
    best = min(res["part_b"][k]["years_for_80pct_power"] for k in ("ic", "net_spread"))
    res["decision"] = ("register a whole-universe forward statistic" if best <= 2.0 else
                       "no statistic reaches 80% power within 2 years; M8 stays a record")
    print("PART B", json.dumps(res["part_b"], indent=1))
    print("DECISION", res["decision"])
    w.to_csv(out / "V3_weekly.csv", index=False)
    (out / "V3_results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "V3"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
