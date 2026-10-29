"""TB1 (research/prereg/TB1.md): TabICLv2 as META1's take-or-skip model.

Everything as META1 (tracks/main/meta1_study.py: picks, 18 inputs, label, folds, take rule, primary test)
except the model: TabICLClassifier with default settings, CPU, random_state 20261024, the checkpoint
tabicl-classifier-v2-20260212.ckpt (sha256 logged). META1's LightGBM is rerun on the same folds beside it
(its numbers must equal META1_result.json). Rule 21: in-sample AUC (each fold's model on its own training
rows) beside out-of-sample AUC; a gap > 0.10 fails first. Rule 22: the filter's effect on the mean and on the
top and bottom 5% of pick returns. Reads META1's data cache; no new downloads beyond its stage_data.

Usage: python -m tracks.main.tb1_study --cache cache/meta1 --out research/results/TB1 [--ckpt-dir <dir>]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main import meta1_study as M

CKPT = "tabicl-classifier-v2-20260212.ckpt"
GAP = 0.10


def lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(**M.PARAMS)


def tabicl(ckpt_dir):
    def make():
        from tabicl import TabICLClassifier
        return TabICLClassifier(model_path=Path(ckpt_dir) / CKPT, checkpoint_version=CKPT, device="cpu",
                                random_state=M.SEED)
    return make


def walk_forward(x, make, inputs=M.INPUTS, log=print):
    """META1's walk-forward with any classifier factory; also each fold's in-sample AUC."""
    x = x.dropna(subset=["fwd5"]).copy()
    weeks = np.sort(x["date"].unique())
    x["prob"], x["take"], x["model_year"] = np.nan, np.nan, np.nan
    folds = {}
    for y in M.TEST_YEARS:
        test = x["date"].dt.year == y
        if not test.any():
            continue
        start = x.loc[test, "date"].min()
        cut = weeks[max(0, np.searchsorted(weeks, np.datetime64(start)) - M.EMBARGO_WEEKS)]
        tr = x["date"] < cut
        m = make().fit(x.loc[tr, inputs], x.loc[tr, "label"])
        p_in = m.predict_proba(x.loc[tr, inputs])[:, 1]
        thr = float(np.median(p_in))
        pr = m.predict_proba(x.loc[test, inputs])[:, 1]
        x.loc[test, "prob"], x.loc[test, "take"], x.loc[test, "model_year"] = pr, (pr >= thr).astype(float), y
        folds[str(y)] = {"train_rows": int(tr.sum()), "test_rows": int(test.sum()), "threshold": thr,
                         "auc_in": M.auc(x.loc[tr, "label"], p_in), "auc_oos": M.auc(x.loc[test, "label"], pr)}
        log(f"[tb1] {y}: {folds[str(y)]}", flush=True)
    return x[x["prob"].notna()], folds


def rule21(folds):
    """Fold-weighted (by test rows) in-sample and out-of-sample AUC; fails when the gap exceeds 0.10."""
    w = np.array([f["test_rows"] for f in folds.values()], float)
    ai = float(np.average([f["auc_in"] for f in folds.values()], weights=w))
    ao = float(np.average([f["auc_oos"] for f in folds.values()], weights=w))
    return {"auc_in": ai, "auc_oos": ao, "gap": ai - ao, "fails": bool(ai - ao > GAP)}


def tails(t):
    """Rule 22: taken vs all on the mean and in the top and bottom 5% of pick returns (pooled)."""
    hi, lo = t["fwd5"].quantile(0.95), t["fwd5"].quantile(0.05)
    k = t[t["take"] == 1]
    return {"mean_all": float(t["fwd5"].mean()), "mean_taken": float(k["fwd5"].mean()),
            "taken_share": float(t["take"].mean()),
            "taken_share_top5pct": float(t.loc[t["fwd5"] >= hi, "take"].mean()),
            "taken_share_bottom5pct": float(t.loc[t["fwd5"] <= lo, "take"].mean()),
            "mean_taken_without_top5pct": float(k.loc[k["fwd5"] < hi, "fwd5"].mean()),
            "mean_all_without_top5pct": float(t.loc[t["fwd5"] < hi, "fwd5"].mean())}


def arm(x, make, log=print):
    t, folds = walk_forward(x, make, log=log)
    res = M.evaluate(t, {})
    r21 = rule21(folds)
    res["PRIMARY"]["rule21"] = r21
    res["PRIMARY"]["passes"] = bool(res["PRIMARY"]["passes"] and not r21["fails"])
    res["alongside"].pop("importance", None)
    res["alongside"]["folds"] = folds
    res["alongside"]["rule22_tails"] = tails(t)
    return res, t


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cache, out, ckpt_dir):
    cache, out, ckpt_dir = Path(cache), Path(out), Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    spy, tmap, eightk = M.stage_data(cache)
    inputs = pd.concat([pd.read_parquet(f) for f in sorted((cache / "inputs").glob("in_*.parquet"))],
                       ignore_index=True)
    x = M.assemble(M.picks(), inputs, spy, tmap, eightk)
    ref, _ = arm(x, lgbm)
    meta1 = json.loads(Path("research/results/META1/META1_result.json").read_text())["PRIMARY"]
    same = abs(ref["PRIMARY"]["mean"] - meta1["mean"]) < 1e-9 and abs(ref["PRIMARY"]["t_nw"] - meta1["t_nw"]) < 1e-6
    print("[tb1] LightGBM reproduces META1:", same, flush=True)
    import tabicl
    res, t = arm(x, tabicl(ckpt_dir))
    ck = ckpt_dir / CKPT
    out_res = {"idea": "TB1", "prereg": "research/prereg/TB1.md", "tabicl_version": tabicl.__version__,
               "checkpoint": CKPT, "checkpoint_sha256": sha256(ck) if ck.exists() else None,
               "lightgbm_reproduces_meta1": bool(same)}
    out_res.update(res)
    out_res["META1_lightgbm_same_folds"] = {"PRIMARY": ref["PRIMARY"],
                                            "rule22_tails": ref["alongside"]["rule22_tails"],
                                            "folds": ref["alongside"]["folds"]}
    (out / "TB1_result.json").write_text(json.dumps(out_res, indent=1, default=str) + "\n")
    t[["date", "symbol", "fwd5", "label", "prob", "take", "model_year"]].to_csv(out / "TB1_oos.csv", index=False)
    print("PRIMARY", json.dumps(res["PRIMARY"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/TB1")
    ap.add_argument("--ckpt-dir", default="cache/tabicl")
    a = ap.parse_args()
    run(a.cache, a.out, a.ckpt_dir)


if __name__ == "__main__":
    main()
