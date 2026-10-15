"""M10 historical screen (research/prereg/M10.md): short-term boom model, main track.

Resumable: the weekly bar panel is built in cached batches until the time budget ends; the screen runs
once the panel is complete. Usage: python -m tracks.main.shortterm.screen_m10 --cache <dir> --out <dir>
"""
import argparse
import time
from pathlib import Path

import numpy as np

from core import shortterm as st
from tracks.main.longterm import sec_data
from tracks.main.longterm.score import MIN_DV20, MIN_PRICE, attach_fundamentals, eligible
from tracks.main.shortterm import evaluate, panel as pn

COST, BORROW = 0.002, 0.03          # config/risk.yaml tracks.main
FEATURES = st.PRICE_FEATURES + ["lmcap"] + st.NEWS_FEATURES
COUNTS = {"phase2_count": 6, "main_count": 6, "phase1_plus_phase2": 21}


def universe(panel, rtab, facts):
    rows = panel[(panel["price"] >= MIN_PRICE) & (panel["dv20"] >= MIN_DV20)]
    uni = eligible(attach_fundamentals(rows, facts, rtab))
    uni["lmcap"] = np.log(uni["mcap"]).astype("float32")
    return uni


def run(cache, out, budget_min=300):
    cache = Path(cache)
    got = pn.build(cache, time.monotonic() + budget_min * 60)
    if got is None:
        print("PANEL INCOMPLETE: rerun this workflow to continue (cached batches are kept)")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    print("FUNNEL", funnel)
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    rows = panel[(panel["price"] >= MIN_PRICE) & (panel["dv20"] >= MIN_DV20)]
    facts = sec_data.facts_for(zp, rows["cik"].unique(), cache / "boom" / "facts_m10")
    uni = universe(panel, rtab, facts)
    counts = {**COUNTS, "panel_funnel": funnel}
    return evaluate.run(uni, FEATURES, COST, BORROW, "M10", counts, ends["ret21"], spy_r, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
