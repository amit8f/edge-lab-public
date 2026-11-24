"""DSR1 (research/prereg/DSR1.md): deflate the reported t-statistics of the NX3 family's history passes for the number
of tries, and the live books' day-t for the number of live books. Descriptive; summary numbers only.

    python -m tracks.tools.deflate --out research/results/DSR1
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

GAMMA = 0.5772156649
N_GRID = (190, 40, 10)
FLAG = 0.80

# (name, reported daily NW t, test days when the registration states them, source line in research/IDEAS.md)
HISTORY = [
    ("NX3 (XGBoost arm of NXM)", 4.04, 1638), ("NXM-LGBM", 5.96, 1638), ("NXM-ENS", 6.32, 1638),
    ("NXM-MLP", 4.42, 1638), ("NXM-LIN", 3.18, 1638), ("NXR-XGBR", 4.60, 1638), ("NXR-RFR", 4.69, 1638),
    ("NXK-KRR", 3.72, 1638), ("NXK2-CAT", 5.30, 1638), ("NXK2-ET", 3.88, 1638),
    ("NX3-ALL", 4.87, 1686), ("NX3-HI", 3.88, 1533), ("NX3-ALL-HI", 5.08, 1533), ("NX3-BARS", 2.85, 1423),
    ("NXE-DAY", 5.20, 1638), ("NXE-EXA1", 4.37, 1638), ("NXF-NOV", 4.72, 1638), ("STOP8", 3.87, 1638),
    ("NXT-TR", 3.71, 1638), ("COST1 (gain vs all trades)", 5.08, 421), ("NXM-VOTE (vs random)", 3.29, 421),
]
# live books, 2026-11-24: day t of the capped $500 books and days settled (fluke check, state-nx3 ledger)
LIVE = [("NX3-HI", 2.60, 13), ("NXM-LGBM", 2.28, 17), ("NXM-ENS", 2.01, 17), ("NXM-XGBR", 1.62, 17),
        ("NXM-RFR", 2.03, 17)]
N_LIVE = 25


def expected_max_z(n):
    """Expected largest of n independent standard normals (Bailey and Lopez de Prado's approximation)."""
    if n <= 1:
        return 0.0
    return float((1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * np.e)))


def deflated(t, n):
    """Probability that a t-statistic of t exceeds the best of n null tries (normal approximation)."""
    return float(norm.cdf(t - expected_max_z(n)))


def annual_sharpe(t, days):
    return float(t * np.sqrt(252 / days))


def table():
    rows = []
    for name, t, days in HISTORY:
        r = {"name": name, "t": t, "days": days, "sharpe_annual": annual_sharpe(t, days)}
        for n in N_GRID:
            r[f"dsr_n{n}"] = deflated(t, n)
        r["flag"] = r["dsr_n40"] < FLAG
        rows.append(r)
    live = []
    for name, t, days in LIVE:
        live.append({"name": name, "t": t, "days": days, "sharpe_annual": annual_sharpe(t, days),
                     "dsr_live": deflated(t, N_LIVE)})
    return {"expected_max_t": {str(n): expected_max_z(n) for n in N_GRID + (N_LIVE,)},
            "history": rows, "live": live, "n_live": N_LIVE, "flag_below": FLAG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/results/DSR1")
    a = ap.parse_args()
    rep = table()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DSR1_result.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
