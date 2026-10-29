"""Writes NX3's news model as a compact tree file for the dashboard (dashboard/site/nx3_trees.json): per tree the
arrays l (left child, -1 at a leaf), r, f (input index), c (split value, or the leaf value at a leaf) and d (1 when
a missing input goes left), plus base_score, the input names, the no-news inputs and the fire cut. `walk` is the
same walk the page runs; tests/test_dashboard.py checks it against XGBoost's own predictions.

Usage: python -m dashboard.export_trees [--out dashboard/site/nx3_trees.json]
"""
import argparse
import json
import math
import struct
from pathlib import Path

RES = Path("research/results/NX3")
NEWS_INPUTS = ["n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos", "fb_neu",
               "head_len", "f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger",
               "f_fda", "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive", "time_bucket"]


def compact(model_json, meta):
    m = model_json["learner"]
    trees = []
    for t in m["gradient_booster"]["model"]["trees"]:
        trees.append({"l": t["left_children"], "r": t["right_children"], "f": t["split_indices"],
                      "c": [float(f"{x:.9g}") for x in t["split_conditions"]], "d": t["default_left"]})
    base = float(m["learner_model_param"]["base_score"].strip("[]"))
    return {"model": "NX3 v1 news model", "sha256": meta["sha256"]["news"], "inputs": meta["inputs"],
            "no_news": {k: 0.0 for k in NEWS_INPUTS}, "cut": meta["fire_cut"], "base_score": base, "trees": trees}


def f32(v):
    """XGBoost compares inputs and split values as 32-bit floats; a value on a split must go the same way here."""
    return struct.unpack("f", struct.pack("f", v))[0]


def walk(tree, x):
    """(leaf value, node path) of one input row (list, None or NaN = missing) through one compact tree."""
    n, path = 0, [0]
    while tree["l"][n] != -1:
        v = x[tree["f"][n]]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            n = tree["l"][n] if tree["d"][n] else tree["r"][n]
        else:
            n = tree["l"][n] if f32(v) < f32(tree["c"][n]) else tree["r"][n]
        path.append(n)
    return tree["c"][n], path


def score(model, x):
    return model["base_score"] + sum(walk(t, x)[0] for t in model["trees"])


def effect(model, x):
    x0 = [model["no_news"].get(k, v) for k, v in zip(model["inputs"], x)]
    return score(model, x) - score(model, x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/site/nx3_trees.json")
    a = ap.parse_args()
    meta = json.loads((RES / "nx3_v1_meta.json").read_text())
    out = compact(json.loads((RES / "nx3_v1_news.json").read_text()), meta)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out['trees'])} trees -> {a.out}")


if __name__ == "__main__":
    main()
