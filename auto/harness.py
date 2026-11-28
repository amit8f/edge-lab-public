"""AUTO1 harness (research/prereg/AUTO1.md). Frozen at registration and read-only to the agent: a change here is a
new idea. The agent edits only auto/candidate.py, which must define fit(train) -> model and score(model, frame) ->
one number per row (higher = better buy; a row is bought only if its score is > 0).

    python -m auto.harness --stage prepare --cache cache/nx1 --events cache/auto1/events.parquet
    python -m auto.harness --stage tune  --events cache/auto1/events.parquet --candidate auto/candidate.py
    python -m auto.harness --stage check --events ... --candidate ... --trials research/results/AUTO1/trials.jsonl
    python -m auto.harness --stage test  --events ... --candidate ...

tune scores 2020-2022 walk-forward and prints only that. check (2023) and test (2024-01..2025-09) are run once each,
by hand, as the registration says; the loop never calls them.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos", "fb_neu",
            "head_len", "f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger", "f_fda",
            "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive", "time_bucket", "novelty", "r1", "r5",
            "r20", "vol20", "log_dv", "log_px", "spy_r1", "spy_r5", "mins_open", "r_open", "r_react", "rv30", "spy_open"]
TUNE_YEARS, CHECK_YEARS, TEST_YEARS = (2020, 2021, 2022), (2023,), (2024, 2025)
CAP, KEEP_MARGIN, PURGE_DAYS = 3, 0.05, 14
LAST_EXIT = pd.Timestamp("2025-09-30")
ALLOWED = {"numpy", "pandas", "sklearn", "xgboost", "lightgbm", "math", "scipy"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                "delattr", "input", "breakpoint"}


# ---------- safety ----------

def check_source(src):
    """Reasons the candidate may not run (empty list = fine): imports outside ALLOWED, banned builtins, dunder access,
    and no fit/score."""
    bad = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad += [f"import {a.name}" for a in n.names if a.name.split(".")[0] not in ALLOWED]
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED or n.level:
                bad.append(f"from {n.module} import")
        elif isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            bad.append(f"name {n.id}")
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            bad.append(f"attribute {n.attr}")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    bad += [f"missing {f}()" for f in ("fit", "score") if f not in names]
    return bad


def load_candidate(path):
    src = Path(path).read_text()
    bad = check_source(src)
    if bad:
        raise ValueError("candidate rejected: " + "; ".join(bad))
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- scoring (pure, tested) ----------

def book(frame, score, cap=CAP):
    """Daily long-only book: rows with score > 0, first `cap` of each day by time; equal weight; 0 on a day with
    none. Returns (daily net Series over every day in frame, number of trades)."""
    f = frame.assign(_s=np.asarray(score, float))
    days = pd.DatetimeIndex(sorted(f["d"].unique()))
    take = f[np.isfinite(f["_s"]) & (f["_s"] > 0)].sort_values(["d", "t"], kind="stable").groupby("d").head(cap)
    daily = take.groupby("d")["net"].mean().reindex(days, fill_value=0.0)
    return daily, len(take)


def sharpe(daily):
    x = np.asarray(daily, float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0


def deflated_sharpe(sr_daily, trial_srs_daily, n_obs, skew, kurt):
    """Bailey and Lopez de Prado (2014): probability that the true Sharpe is above the best of N null trials.
    All Sharpes per day (not annualised); kurt is the plain (not excess) kurtosis."""
    from scipy.stats import norm
    n = max(len(trial_srs_daily), 2)
    v = float(np.var(trial_srs_daily, ddof=1)) if len(trial_srs_daily) > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))
    den = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2, 1e-12))
    return float(norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / den))


def walk_forward(events, cand, years):
    daily, trades, per = [], 0, {}
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        tr = events[events["next"] < start - pd.Timedelta(days=PURGE_DAYS)]
        te = events[events["d"].dt.year == y]
        if te.empty:
            continue
        m = cand.fit(tr.reset_index(drop=True))
        s = np.asarray(cand.score(m, te.reset_index(drop=True)), float)
        if s.shape != (len(te),):
            raise ValueError(f"score() returned shape {s.shape}, need ({len(te)},)")
        dly, n = book(te.reset_index(drop=True), s)
        daily.append(dly)
        trades += n
        per[str(y)] = {"sharpe": sharpe(dly), "mean": float(dly.mean()), "trades": n}
    d = pd.concat(daily) if daily else pd.Series(dtype=float)
    return {"sharpe": sharpe(d), "mean_daily": float(d.mean()) if len(d) else 0.0, "days": len(d),
            "trades": trades, "per_year": per}, d


# ---------- stages ----------

def stage_prepare(cache, out):
    """Real NX3 events 2016-10..2025-09 with NX3's 37 inputs and each event's long net after flat IBKR costs."""
    from tracks.news import nx1_study as N
    from tracks.news import nx3_study as X
    cols = ["symbol", "d", "t", "kind", "next", "price", "days", "ret", *FEATURES]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True)
    ev = ev[(ev["kind"] == "real") & np.isfinite(ev["ret"]) & (ev["next"] <= LAST_EXIT)]
    ev = ev.drop_duplicates(["symbol", "d", "t"]).drop(columns="kind").reset_index(drop=True)
    ev["net"] = ev["ret"] - N.cost_frac(ev["price"], np.zeros(len(ev), bool), ev["days"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    N.atomic_parquet(ev, Path(out))
    print(f"[auto1 prepare] {len(ev)} events, {ev['d'].min().date()}..{ev['d'].max().date()}")


def baseline(cache, years):
    """Plain NX3-HI buys (frozen walk-forward models5 picks, first 3 longs a day) with the same flat costs as the
    candidate, as a daily book over the same days. Only the check and test stages read it."""
    from tracks.news import long_study as L
    from tracks.news import models5_study as M
    from tracks.news import nx1_study as N
    p = pd.read_parquet(M.mdir(cache) / "picks.parquet")
    lo = L.cap_longs(p)
    lo = lo[lo["d"].dt.year.isin(years)]
    return lo.assign(net=lo["ret"] - N.cost_frac(lo["price"], np.zeros(len(lo), bool), lo["days"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "tune", "check", "test"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--events", default="cache/auto1/events.parquet")
    ap.add_argument("--candidate", default="auto/candidate.py")
    ap.add_argument("--trials", default="research/results/AUTO1/trials.jsonl")
    a = ap.parse_args()
    if a.stage == "prepare":
        return stage_prepare(Path(a.cache), a.events)
    events = pd.read_parquet(a.events)
    cand = load_candidate(a.candidate)
    if a.stage == "tune":
        res, _ = walk_forward(events[events["d"] < pd.Timestamp("2023-01-01")], cand, TUNE_YEARS)
        print(json.dumps(res))
        return
    years = CHECK_YEARS if a.stage == "check" else TEST_YEARS
    res, d = walk_forward(events, cand, years)
    b = baseline(Path(a.cache), years)
    bd = b.groupby("d")["net"].mean().reindex(d.index, fill_value=0.0)
    res["nx3hi_sharpe"] = sharpe(bd)
    if a.stage == "check":
        trials = [json.loads(x) for x in Path(a.trials).read_text().splitlines() if x.strip()]
        srs = [t["score"] / math.sqrt(252) for t in trials if t.get("score") is not None]
        x = d.to_numpy(float)
        from scipy.stats import kurtosis, skew
        res["deflated_sharpe_p"] = deflated_sharpe(res["sharpe"] / math.sqrt(252), srs, len(x), float(skew(x)),
                                                   float(kurtosis(x, fisher=False)))
        res["trials"] = len(trials)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
