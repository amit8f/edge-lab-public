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
           random_state=7, verbose=-1, n_jobs=3)   # one core left for the runner


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


TRAIN_EVERY = 12                # every 12th weekly date: the 20-day targets overlap (NUM1.md Amendments 1 and 3)
MAX_FEATURES = 500              # a seeded random subset of the features (NUM1.md Amendment 3)
FEATURE_SEED = 7


def pick_features(feats, limit=MAX_FEATURES, seed=FEATURE_SEED):
    """A seeded random subset of at most `limit` features, in file order; all of them if there are fewer."""
    if limit is None or len(feats) <= limit:
        return list(feats)
    keep = set(np.random.default_rng(seed).choice(len(feats), size=limit, replace=False).tolist())
    return [c for i, c in enumerate(feats) if i in keep]


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # kB on Linux


def read(f, keep=(), every=1, rows=None, use=None):
    """Only the feature columns (as float32) and `keep` columns of a Numerai parquet, read in batches; with every > 1
    only every n-th date is kept; `rows(df)` (a boolean mask) filters each batch as it is read. The full train file
    does not fit in a runner's 16 GB as float64."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    names = pf.schema_arrow.names
    feats = list(use) if use is not None else pick_features(features(names))
    dates = None
    if every > 1:
        d = np.sort(pq.read_table(f, columns=["date"]).column("date").to_pandas().unique())
        dates = set(d[::every])
    parts = []
    for b in pf.iter_batches(batch_size=200_000, columns=[c for c in names if c in feats or c in keep]):
        df = b.to_pandas()
        if dates is not None:
            df = df[df["date"].isin(dates)]
        if rows is not None:
            df = df[rows(df)]
        parts.append(df.astype({c: np.float32 for c in feats}))
    return pd.concat(parts, ignore_index=True), feats


def thin_dates(df, every=TRAIN_EVERY):
    d = np.sort(df["date"].unique())
    return df[df["date"].isin(d[::every])]


def get(a, cache, name):
    f = Path(cache) / name
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        a.download_dataset(f"signals/{VERSION}/{name}", str(f))
    return f


def stage_data(cache):
    """Downloads the three training files into the cache and nothing else, logging time and size per file, so a lost
    runner can be told apart from a lost download (NUM1.md Amendment 4)."""
    import time
    a = api()
    for name in ("validation_example_preds.parquet", "validation.parquet", "train.parquet"):
        t0 = time.monotonic()
        f = get(a, cache, name)
        print(f"[num1] {name}: {f.stat().st_size / 1e9:.2f} GB in {time.monotonic() - t0:.0f} s, "
              f"peak memory {peak_rss_gb():.1f} GB", flush=True)


def stage_train(cache, out):
    import lightgbm as lgb
    a = api()
    tr, feats = read(get(a, cache, "train.parquet"), keep=("date", TARGET), every=TRAIN_EVERY,
                     rows=lambda df: df[TARGET].notna().to_numpy())
    print(f"[num1] train rows {len(tr)}, dates {tr['date'].nunique()}, features {len(feats)}, "
          f"peak memory {peak_rss_gb():.1f} GB", flush=True)
    m = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr[TARGET])
    print(f"[num1] fitted, peak memory {peak_rss_gb():.1f} GB", flush=True)
    m.booster_.save_model(str(Path(cache) / "model.txt"))
    (Path(cache) / "model.json").write_text(json.dumps({"features": feats, "version": VERSION}))
    del tr
    va, _ = read(get(a, cache, "validation.parquet"), keep=("date", TARGET, "data_type", "numerai_ticker"), use=feats,
                 rows=lambda df: (df["data_type"] == "validation").to_numpy() & df[TARGET].notna().to_numpy())
    va["date"] = va["date"].astype(str)
    va["pred"] = np.concatenate([m.predict(va[feats].iloc[i:i + 200_000]) for i in range(0, len(va), 200_000)])
    va = va[["date", TARGET, "numerai_ticker", "pred"]].copy()      # drop the 500 feature columns before the correlations
    print(f"[num1] validation rows {len(va)}, dates {va['date'].nunique()}, peak memory {peak_rss_gb():.1f} GB", flush=True)
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


MODEL_DIR = Path("models/num1")    # committed copy: the daily job on main cannot read a branch's Actions cache


def model_dir(cache):
    return MODEL_DIR if (MODEL_DIR / "model.txt").exists() else Path(cache)


def stage_export(cache, out):
    """Copies the trained model from the cache to models/num1 so it can be committed."""
    import shutil
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.txt", "model.json"):
        shutil.copyfile(Path(cache) / name, MODEL_DIR / name)
    print(f"[num1] model exported to {MODEL_DIR}")


def stage_submit(cache, out):
    import lightgbm as lgb
    a = api()
    md = model_dir(cache)
    meta = json.loads((md / "model.json").read_text())
    b = lgb.Booster(model_file=str(md / "model.txt"))
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
    ap.add_argument("--stage", required=True, choices=["data", "train", "submit", "export"])
    ap.add_argument("--cache", default="cache/numerai")
    ap.add_argument("--out", default="research/results/NUM1")
    x = ap.parse_args()
    if x.stage == "data":
        stage_data(x.cache)
        return
    {"train": stage_train, "submit": stage_submit, "export": stage_export}[x.stage](x.cache, x.out)


if __name__ == "__main__":
    main()
