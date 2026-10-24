"""I1 historical screen (research/prereg/I1.md): when an industry's big stocks beat SPY by 2%+ in a week
(top 2 of French's 12), buy its penny stocks that haven't caught up, hold 5 days, against a random penny
control.

Usage: python -m tracks.penny.i1_screen --cache <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry, stats
from tracks.main import weekly_panel

COST, SEED = 0.02, 20261024
MAX_PICKS, TOP_IND, MIN_MOVE, MIN_BIG = 20, 2, 0.02, 10


def spy_ret5(spy, dates):
    c = spy["c"]
    k = {t: i for i, t in enumerate(c.index)}
    return {t: float(c.iloc[k[t]] / c.iloc[k[t] - 5] - 1) for t in dates if t in k and k[t] >= 5}


def moves(rows, spy5):
    """Per (date, industry): equal-weighted ret5 of the industry's big stocks minus SPY's, with >= MIN_BIG
    of them; industry 12 (Other) left out."""
    b = rows[rows["main"] & rows["ind"].notna() & (rows["ind"] != 12) & rows["ret5"].notna()]
    g = b.groupby(["date", "ind"])["ret5"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= MIN_BIG]
    g["move"] = g["mean"] - g["date"].map(spy5)
    return g.dropna(subset=["move"])


def pick(rows, mv, dates, top=True):
    """Per week: picks among penny stocks in the top (or bottom) TOP_IND industries by move, if the move
    is >= +MIN_MOVE (or <= -MIN_MOVE); a pick's own ret5 is below (above) the move, at most MAX_PICKS by
    the largest gap. Control: as many names from the whole penny universe, default_rng(SEED + week)."""
    pen = rows[rows["penny"]]
    out = []
    for w, t in enumerate(dates):
        m = mv[mv["date"] == t].sort_values(["move", "ind"], ascending=not top).head(TOP_IND)
        m = m[m["move"] >= MIN_MOVE] if top else m[m["move"] <= -MIN_MOVE]
        if m.empty:
            continue
        p = pen[(pen["date"] == t)]
        c = p[p["ind"].isin(m["ind"]) & p["ret5"].notna()].copy()
        c["gap"] = (c["ind"].map(m.set_index("ind")["move"]) - c["ret5"]) * (1 if top else -1)
        c = c[c["gap"] > 0].sort_values(["gap", "symbol"], ascending=[False, True]).head(MAX_PICKS)
        if c.empty:
            continue
        rest = p[~p["symbol"].isin(c["symbol"])].sort_values("symbol").reset_index(drop=True)
        k = np.random.default_rng(SEED + w).choice(len(rest), size=min(len(c), len(rest)), replace=False)
        out.append(c.assign(role="pick", week=w))
        out.append(rest.iloc[np.sort(k)].assign(role="control", week=w))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(cache, out):
    rows, cal, dates, funnel, spy = weekly_panel.build(cache)
    ciks = rows.loc[(rows["main"] | rows["penny"]) & rows["cik"].notna(), "cik"].astype("int64").unique()
    ind = industry.industries(ciks, cache / "sic")
    rows["ind"] = rows["cik"].map(lambda c: ind.get(int(c)) if pd.notna(c) else np.nan)
    funnel["penny_rows_with_industry"] = int((rows["penny"] & rows["ind"].notna() & (rows["ind"] != 12)).sum())
    mv = moves(rows, spy_ret5(spy, dates))
    fac, sha5, shau = weekly_panel.factors()
    p2, both = stats.idea_counts()
    sel = pick(rows, mv, dates)
    t = weekly_panel.weekly(sel, COST)
    prim = weekly_panel.summary(t, SEED)
    a = weekly_panel.alpha(t, fac, cal)
    prim["factor_alpha"] = a
    prim["deflated_sharpe"] = stats.deflated_sharpe(t["diff"].to_numpy(), both)
    prim["passes"] = stats.screen_pass(prim, a)
    prim["signal_weeks_share"] = float(sel["date"].nunique() / len(dates)) if len(sel) else 0.0
    picks = sel[sel["role"] == "pick"]
    bottom = pick(rows, mv, dates, top=False)
    res = {"idea": "I1", "prereg": "research/prereg/I1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_daily_sha256": sha5, "umd_daily_sha256": shau, "cost_round_trip": COST, "funnel": funnel,
           "industry_note": "current EDGAR SIC per company, not point-in-time (stated in the prereg)",
           "PRIMARY": prim,
           "alongside": {
               "before_costs": weekly_panel.summary(weekly_panel.weekly(sel, 0.0), SEED),
               "bottom_2_industries_long": weekly_panel.summary(weekly_panel.weekly(bottom, COST), SEED)
               if len(bottom) else {},
               "hold_10": weekly_panel.summary(weekly_panel.weekly(sel, COST, "fwd10"), SEED),
               "hold_20": weekly_panel.summary(weekly_panel.weekly(sel, COST, "fwd20"), SEED),
               "per_year": t.groupby(t.index.year)["diff"].mean().to_dict(),
               "picks_median_dv20": float(picks["dv20"].median())}}
    out.mkdir(parents=True, exist_ok=True)
    chk = picks.dropna(subset=["fwd5"]).sample(n=min(20, len(picks)), random_state=SEED)
    chk = chk.merge(mv[["date", "ind", "move"]], on=["date", "ind"], how="left")
    chk[["date", "symbol", "ind", "move", "ret5", "price", "fwd5"]].assign(checked_by="", ok="", note="").to_csv(
        out / "I1_rows_check.csv", index=False)
    sel[["date", "week", "symbol", "role", "ind", "ret5", "fwd5"]].to_csv(out / "I1_picks.csv", index=False)
    mv.to_csv(out / "I1_moves.csv", index=False)
    big = rows[rows["main"] & rows["ind"].notna() & (rows["ind"] != 12)]
    big.reindex(big["ret5"].abs().sort_values(ascending=False).index).head(50)[
        ["date", "symbol", "ind", "price", "ret5", "fwd5"]].to_csv(out / "I1_big_ret5_extremes.csv", index=False)
    (out / "I1_result.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
