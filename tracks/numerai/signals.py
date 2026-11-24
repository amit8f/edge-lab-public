"""NUM1: Numerai Signals ranker (research/prereg/NUM1.md). Unstaked: nothing is bought, no money moves.

    python -m tracks.numerai.signals --stage train|submit --cache cache/numerai --out research/results/NUM1

train: LightGBM on Numerai's own Signals features (train.parquet, 2003-2018) for `target`, scored on validation
(2019 on, rows with a target) by the per-date Spearman correlation, beside Numerai's example predictions; the model
goes to the cache. submit: today's live.parquet -> ranks in (0, 1) -> upload to model edgelab_signals.
Summaries only are committed; Numerai's data stays in the cache.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v3.1"
MODEL_NAME = "edgelab_signals"
TARGET = "target"
LGB = dict(n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=31, colsample_bytree=0.1,
           random_state=7, verbose=-1)


def api():
    import numerapi
    return numerapi.SignalsAPI(os.environ["NUMERAI_PUBLIC_ID"], os.environ["NUMERAI_SECRET_KEY"])


def features(cols):
    return [c for c in cols if c.startswith("feature_") and c != "feature_country"]


def rank01(x):
    """Ranks in the open interval (0, 1), ties averaged, as Signals asks."""
    r = pd.Series(x).rank(method="average").to_numpy(float)
    return (r - 0.5) / len(r)


def date_corr(df, pred, target=TARGET):
    """Per date: Spearman correlation of a prediction column with the target."""
    out = {}
    for d, g in df.groupby("date"):
        g = g[[pred, target]].dropna()
        if len(g) > 50:
            out[d] = g[pred].rank().corr(g[target].rank())
    return pd.Series(out).sort_index()


def summary(s):
    s = s.dropna()
    return {"dates": int(len(s)), "mean": float(s.mean()), "sd": float(s.std()),
            "sharpe": float(s.mean() / s.std()) if s.std() > 0 else None,
            "share_positive": float((s > 0).mean()), "by_year": {str(k): float(v) for k, v in s.groupby(s.index.str[:4]).mean().items()}}


def get(a, cache, name):
    f = Path(cache) / name
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        a.download_dataset(f"signals/{VERSION}/{name}", str(f))
    return f


def stage_train(cache, out):
    import lightgbm as lgb
    a = api()
    tr = pd.read_parquet(get(a, cache, "train.parquet"))
    feats = features(tr.columns)
    tr = tr[tr[TARGET].notna()]
    m = lgb.LGBMRegressor(**LGB).fit(tr[feats].astype(np.float32), tr[TARGET])
    m.booster_.save_model(str(Path(cache) / "model.txt"))
    (Path(cache) / "model.json").write_text(json.dumps({"features": feats, "version": VERSION}))
    del tr
    va = pd.read_parquet(get(a, cache, "validation.parquet"))
    va = va[(va["data_type"] == "validation") & va[TARGET].notna()].copy()
    va["date"] = va["date"].astype(str)
    va["pred"] = m.predict(va[feats].astype(np.float32))
    rep = {"model": "LightGBM on Numerai Signals features", "features": len(feats), "params": LGB,
           "validation": summary(date_corr(va, "pred"))}
    va["us"] = va["numerai_ticker"].astype(str).str.endswith(" US")
    rep["validation_us_only"] = summary(date_corr(va[va["us"]], "pred"))
    try:
        ex = pd.read_parquet(get(a, cache, "validation_example_preds.parquet"))
        col = [c for c in ex.columns if c not in ("numerai_ticker", "date")][0]
        ex["date"] = ex["date"].astype(str)
        both = va.merge(ex.rename(columns={col: "example"}), on=["numerai_ticker", "date"], how="inner")
        rep["numerai_example_same_rows"] = summary(date_corr(both, "example"))
        rep["ours_same_rows"] = summary(date_corr(both, "pred"))
    except Exception as e:      # noqa: BLE001 (reported only)
        rep["numerai_example_same_rows"] = f"not read: {type(e).__name__}"
    for name in ("train.parquet", "validation.parquet", "validation_example_preds.parquet"):
        (Path(cache) / name).unlink(missing_ok=True)      # only the model is cached (Actions has a 10 GB limit)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "NUM1_validation.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def stage_submit(cache, out):
    import lightgbm as lgb
    a = api()
    meta = json.loads((Path(cache) / "model.json").read_text())
    b = lgb.Booster(model_file=str(Path(cache) / "model.txt"))
    live_f = Path(cache) / "live.parquet"
    if live_f.exists():
        live_f.unlink()
    live = pd.read_parquet(get(a, cache, "live.parquet"))
    sub = pd.DataFrame({"numerai_ticker": live["numerai_ticker"],
                        "signal": rank01(b.predict(live[meta["features"]].astype(np.float32)))})
    sub = sub.drop_duplicates("numerai_ticker")
    model_id = a.get_models()[MODEL_NAME]
    sid = a.upload_predictions(df=sub, model_id=model_id)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    rec = {"date": str(live["date"].iloc[0]), "round": a.get_current_round(), "tickers": int(len(sub)),
           "submission_id": sid}
    with open(o / "NUM1_submissions.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    live_f.unlink(missing_ok=True)
    print(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "submit"])
    ap.add_argument("--cache", default="cache/numerai")
    ap.add_argument("--out", default="research/results/NUM1")
    x = ap.parse_args()
    (stage_train if x.stage == "train" else stage_submit)(x.cache, x.out)


if __name__ == "__main__":
    main()
