"""Writes the NXK-KRR news model (research/results/NX3/nxk_v1_KRR.pkl) as a small JSON file the dashboard can run
in the browser (dashboard/site/nxk_krr.json). The model: fill missing inputs with the training median (plus a 0/1
"was missing" column for inputs that had gaps), standardise, then score = b + sum_j w_j * exp(-gamma * |z - L_j|^2)
over 300 reference events L_j (Nystroem landmarks); w = normalization_.T @ ridge.coef_. `score` is the same sum the
page runs; tests/test_dashboard.py checks it against the scikit-learn pipeline.

Usage: python -m dashboard.export_krr [--out dashboard/site/nxk_krr.json]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

RES = Path("research/results/NX3")
NEWS_INPUTS = __import__("dashboard.export_trees", fromlist=["NEWS_INPUTS"]).NEWS_INPUTS


def compact(pipe, inputs, cut):
    imp, sc, inner = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    ny, rg = inner.steps[0][1], inner.steps[1][1]
    w = ny.normalization_.T @ rg.coef_
    ind = [int(i) for i in imp.indicator_.features_] if imp.indicator_ is not None else []
    r = lambda a: [float(f"{v:.7g}") for v in np.ravel(a)]      # noqa: E731
    return {"model": "NXK-KRR v1 news model", "inputs": list(inputs), "no_news": {k: 0.0 for k in NEWS_INPUTS},
            "cut": float(cut), "median": r(imp.statistics_), "missing_cols": ind, "mean": r(sc.mean_),
            "scale": r(sc.scale_), "gamma": float(ny.gamma), "b": float(rg.intercept_), "w": r(w),
            "L": [r(row) for row in ny.components_]}


def score(m, x):
    """(score, contributions) of one input row (None or NaN = missing), as the page computes it."""
    x = [np.nan if v is None else float(v) for v in x]
    full = [m["median"][i] if np.isnan(v) else v for i, v in enumerate(x)]
    full += [1.0 if np.isnan(x[i]) else 0.0 for i in m["missing_cols"]]
    z = (np.array(full) - np.array(m["mean"])) / np.array(m["scale"])
    k = np.exp(-m["gamma"] * ((np.array(m["L"]) - z) ** 2).sum(1))
    c = np.array(m["w"]) * k
    return float(m["b"] + c.sum()), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/site/nxk_krr.json")
    a = ap.parse_args()
    meta = json.loads((RES / "nxk_v1_meta.json").read_text())
    m = pickle.loads((RES / meta["KRR"]["file"]).read_bytes())
    out = compact(m["news"], meta["inputs"], meta["KRR"]["cut"]) | {"sha256": meta["KRR"]["sha256"]}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {len(out['w'])} reference events, {len(out['inputs'])} inputs")


if __name__ == "__main__":
    main()
