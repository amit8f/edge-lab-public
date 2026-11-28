"""AUTO2-AUTO4 harness (research/prereg/AUTO2.md). Frozen at registration and read-only to the agents. Each loop
lives in auto/loops/<name>/ (start.py = the model's original recipe, candidate.py = the file its agent edits)
and on its own git branch <name>; all three agents read auto/program2.md.

A candidate defines fit(train) -> model and score(model, frame) -> one number per event in units of its buy/sell
bar: the book buys the day's first 3 events (by time) with score >= 1 and sells short the first 3 with score <= -1,
as NX3-HI does. `train` holds real and placebo events (column `kind`) with the label column `mret`; `frame` holds
only real events and no outcome columns.

    python -m auto.harness2 --stage prepare --cache cache/nx1 --events cache/auto2/events.parquet
    python -m auto.harness2 --stage tune  --loop auto2 --candidate auto/loops/auto2/candidate.py
    python -m auto.harness2 --stage check --loop auto2      (2023, once: best vs start.py, deflated Sharpe)
    python -m auto.harness2 --stage test  --loop auto2      (2024-01..2025-09, once)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from auto.harness import (
    FEATURES,
    LAST_EXIT,
    PURGE_DAYS,
    check_source,
    deflated_sharpe,
    load_candidate,
    sharpe,
)

TUNE_YEARS, CHECK_YEARS, TEST_YEARS = (2020, 2021, 2022), (2023,), (2024, 2025)
CAP, KEEP_MARGIN, LEAK_CUTS = 3, 0.05, 6
EXTRA = ["entry", "prev_ac", "prev_c", "dv20"]
OUTCOME = ["net_long", "net_short", "mret", "ret", "next", "days", "spy_ret", "kind"]
LOOPS = ("auto2", "auto3", "auto4")


def book(frame, score, cap=CAP):
    """Daily 3+3 book: first `cap` events by time with score >= 1 bought, first `cap` with score <= -1 sold short;
    equal weight across the day's trades; 0 on a day with none. Returns (daily net, trades)."""
    f = frame.assign(_s=np.asarray(score, float)).sort_values(["d", "t"], kind="stable")
    lo = f[f["_s"] >= 1].groupby("d").head(cap).assign(_n=lambda x: x["net_long"])
    sh = f[f["_s"] <= -1].groupby("d").head(cap).assign(_n=lambda x: x["net_short"])
    days = pd.DatetimeIndex(sorted(f["d"].unique()))
    take = pd.concat([lo, sh])
    return take.groupby("d")["_n"].mean().reindex(days, fill_value=0.0), len(take)


def leak_check(cand, model, frame, scores, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(frame["t"].unique())
    for cut in rng.choice(ts, size=min(LEAK_CUTS, len(ts)), replace=False):
        keep = (frame["t"] <= cut).to_numpy()
        part = np.asarray(cand.score(model, frame[keep].reset_index(drop=True)), float)
        if part.shape != (keep.sum(),) or not np.allclose(part, scores[keep], rtol=1e-6, atol=1e-9, equal_nan=True):
            return pd.Timestamp(cut)
    return None


def walk_forward(events, cand, years):
    daily, trades, per = [], 0, {}
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        tr = events[events["next"] < start - pd.Timedelta(days=PURGE_DAYS)]
        te = events[(events["d"].dt.year == y) & (events["kind"] == "real")].reset_index(drop=True)
        if te.empty:
            continue
        m = cand.fit(tr.drop(columns=["net_long", "net_short"]).reset_index(drop=True))
        blind = te.drop(columns=[c for c in OUTCOME if c in te.columns])
        s = np.asarray(cand.score(m, blind), float)
        if s.shape != (len(te),):
            raise ValueError(f"score() returned shape {s.shape}, need ({len(te)},)")
        bad = leak_check(cand, m, blind, s, seed=y)
        if bad is not None:
            raise ValueError(f"score() uses rows after a row's own time (cut {bad}); rejected")
        dly, n = book(te, s)
        daily.append(dly)
        trades += n
        per[str(y)] = {"sharpe": sharpe(dly), "mean": float(dly.mean()), "trades": n}
    d = pd.concat(daily) if daily else pd.Series(dtype=float)
    return {"sharpe": sharpe(d), "mean_daily": float(d.mean()) if len(d) else 0.0, "days": len(d),
            "trades": trades, "per_year": per}, d


def stage_prepare(cache, out):
    """Real and placebo NX3 events 2016-10..2025-09 with the 37 inputs, NX3's label input mret, and each event's
    long and short net after flat IBKR costs (borrow on shorts)."""
    from tracks.news import nx1_study as N
    from tracks.news import nx3_study as X
    cols = ["symbol", "d", "t", "kind", "next", "price", "days", "ret", "mret", "spy_ret", *FEATURES, *EXTRA]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= LAST_EXIT)]
    ev = ev.drop_duplicates(["symbol", "d", "t", "kind"]).reset_index(drop=True)
    ev["net_long"] = ev["ret"] - N.cost_frac(ev["price"], np.zeros(len(ev), bool), ev["days"])
    ev["net_short"] = -ev["ret"] - N.cost_frac(ev["price"], np.ones(len(ev), bool), ev["days"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    N.atomic_parquet(ev, Path(out))
    print(f"[auto2 prepare] {len(ev)} events ({(ev['kind'] == 'real').sum()} real), "
          f"{ev['d'].min().date()}..{ev['d'].max().date()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "tune", "check", "test"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--events", default="cache/auto2/events.parquet")
    ap.add_argument("--loop", choices=LOOPS)
    ap.add_argument("--candidate")
    a = ap.parse_args()
    if a.stage == "prepare":
        return stage_prepare(Path(a.cache), a.events)
    d = Path("auto/loops") / a.loop
    events = pd.read_parquet(a.events)
    if a.stage == "tune":
        res, _ = walk_forward(events[events["d"] < pd.Timestamp("2023-01-01")], load_candidate(a.candidate), TUNE_YEARS)
        print(json.dumps(res))
        return
    years = CHECK_YEARS if a.stage == "check" else TEST_YEARS
    res, db = walk_forward(events, load_candidate(d / "candidate.py"), years)
    base, ds = walk_forward(events, load_candidate(d / "start.py"), years)
    diff = (db - ds.reindex(db.index, fill_value=0.0)).to_numpy(float)
    from scipy.stats import kurtosis, skew
    trials = [json.loads(x) for x in (Path("research/results") / a.loop.upper() / "trials.jsonl").read_text().splitlines() if x.strip()]
    srs = [t["score"] / math.sqrt(252) for t in trials if t.get("score") is not None]
    x = db.to_numpy(float)
    res |= {"start": base, "trials": len(trials),
            "deflated_sharpe_p": deflated_sharpe(res["sharpe"] / math.sqrt(252), srs, len(x), float(skew(x)),
                                                 float(kurtosis(x, fisher=False))),
            "gain_vs_start_mean": float(diff.mean())}
    print(json.dumps(res, default=float))


assert check_source  # re-exported for the loop's safety check
if __name__ == "__main__":
    main()
