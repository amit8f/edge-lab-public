"""Numerai Signals probe (Amit 2026-11-24: unstaked Signals submissions). Checks the API keys work and prints what
the live universe and the training data look like, so the submitter can be built against the real format.
Prints names, counts and column lists only; never the keys."""
import os
import tempfile

import pandas as pd


def main():
    import numerapi
    pid, key = os.environ.get("NUMERAI_PUBLIC_ID"), os.environ.get("NUMERAI_SECRET_KEY")
    print(f"keys present: public {bool(pid)}, secret {bool(key)}")
    api = numerapi.SignalsAPI(pid, key)
    print("models:", list(api.get_models().keys()) if pid and key else "(no keys)")
    print("current round:", api.get_current_round())
    ds = api.list_datasets()
    print("datasets:", ds)
    tmp = tempfile.mkdtemp()
    ver = sorted({d.split("/")[1] for d in ds if d.startswith("signals/")})[-1]
    for name in ("live.parquet", "validation.parquet", "train.parquet"):
        f = f"signals/{ver}/{name}"
        if f not in ds:
            continue
        out = os.path.join(tmp, name)
        api.download_dataset(f, out)
        x = pd.read_parquet(out)
        print(f"{f}: {len(x)} rows, columns {list(x.columns)[:40]}")
        tk = next((c for c in ("numerai_ticker", "bloomberg_ticker", "ticker") if c in x.columns), None)
        if tk:
            us = x[tk].astype(str).str.endswith(" US").mean()
            print(f"  ticker column {tk}: {x[tk].nunique()} names, {us:.0%} end in ' US', e.g. {list(x[tk].astype(str).head(5))}")
        for c in ("date", "friday_date", "era", "data_type"):
            if c in x.columns:
                print(f"  {c}: {x[c].min()} .. {x[c].max()}, {x[c].nunique()} values")
        tg = [c for c in x.columns if c.startswith("target")]
        if tg:
            print(f"  targets: {tg}")
        os.remove(out)


if __name__ == "__main__":
    main()
