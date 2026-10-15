"""P3 historical screen (research/prereg/P3.md): short-term boom model, penny track. Uses the weekly
panel M10 builds (shared data, tracks/main/shortterm/panel.py) and fits its own models.

Usage: python -m tracks.penny.shortterm.screen_p3 --cache <dir> --out <dir>
"""
import argparse
import time
from pathlib import Path

from core import shortterm as st
from tracks.main.longterm import bars_batch
from tracks.main.shortterm import evaluate, panel as pn

MIN_PRICE, MAX_PRICE, MIN_DV20 = 1.0, 5.0, 1e6
COST, BORROW = 0.02, 0.20           # config/risk.yaml tracks.penny
FEATURES = st.PRICE_FEATURES + st.NEWS_FEATURES
COUNTS = {"phase2_count": 7, "penny_count": 1, "phase1_plus_phase2": 22}


def universe(panel):
    m = (panel["price"] >= MIN_PRICE) & (panel["price"] < MAX_PRICE) & (panel["dv20"] >= MIN_DV20)
    d = panel[m].sort_values("dv20", ascending=False)
    return d.drop_duplicates(["date", "cik"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def run(cache, out, budget_min=300, short_ok=None):
    got = pn.build(Path(cache), time.monotonic() + budget_min * 60)
    if got is None:
        print("PANEL INCOMPLETE: rerun this workflow to continue (cached batches are kept)")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    print("FUNNEL", funnel)
    if short_ok is None:
        a = bars_batch.assets_all()
        short_ok = set(a.loc[a.get("easy_to_borrow", False) == True, "base_symbol"])  # noqa: E712
    counts = {**COUNTS, "panel_funnel": funnel}
    return evaluate.run(universe(panel), FEATURES, COST, BORROW, "P3", counts, ends["ret21"], spy_r, out,
                        short_ok=short_ok, fat_tails=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
