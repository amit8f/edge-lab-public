"""NXR: NX3's design and data with tamer trees (research/prereg/NXR.md).

NXM's machinery (tracks/news/nxm_study.py) with two arms, XGBR and RFR, beside NX3's XGBoost as the reference.
Each finished test year is saved under --state, so a rerun resumes. Summary numbers only ->
research/results/NXR/ (and, for an arm that passes, its version 1 models).

Usage: python -m tracks.news.nxr_study --cache cache/nx1 --state cache/nxr --out research/results/NXR
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X3
from tracks.news import nxm_study as M

SEED = X3.SEED
ARMS = ("XGB", "XGBR", "RFR")
NEW_ARMS = ARMS[1:]
XGBR = dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5,
            min_child_weight=500, reg_lambda=10.0, objective="reg:pseudohubererror", huber_slope=1.0,
            tree_method="hist", random_state=SEED)
RFR = dict(n_estimators=300, max_depth=6, min_samples_leaf=1000, max_samples=0.2, max_features=0.33, n_jobs=-1,
           random_state=SEED)


def learner(arm):
    if arm == "XGB":
        return X3.fit

    def fit(X, y):
        if arm == "XGBR":
            from xgboost import XGBRegressor
            return XGBRegressor(**XGBR).fit(X, y)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**RFR).fit(X, y).set_params(n_jobs=1)
    return fit


def year(ev, cal, yr):
    caldates = pd.DatetimeIndex(cal["date"])
    start = pd.Timestamp(f"{yr}-01-01")
    cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
    tr = ev[ev["d"] <= cut]
    te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
    if tr.empty or te.empty:
        return None
    trr = M.train_real(tr)
    summ, days, picks = {}, [], []
    for a in ARMS:
        m = X3.Model(tr, learner(a))
        n, pk, real, pe = X3.evaluate(te, m, SEED + yr)
        ins = M.insample(trr, m, SEED + yr)
        if len(pk):
            pk = pk.assign(net=N.net(pk))
            n = n.merge(pk.groupby("d")["net"].mean().rename("pick_net").reset_index(), on="d", how="left")
            picks.append(pk[["d", "symbol", "side", "net"]].assign(arm=a))
        days.append(n.assign(arm=a))
        summ[a] = {"fire_cut": m.cut, "real_fire_rate": float(real["fired"].mean()),
                   "placebo_fire_rate": float(m.fires(pe).mean()) if len(pe) else None,
                   "insample_mean": float(ins.mean()) if len(ins) else None, "insample_days": int(len(ins))}
    return summ, pd.concat(days, ignore_index=True), (pd.concat(picks, ignore_index=True) if picks else
                                                       pd.DataFrame(columns=["d", "symbol", "side", "net", "arm"]))


def report(summ, days, picks):
    out = {"arms": {}, "per_year": summ}
    xgb = days[days["arm"] == "XGB"].set_index("d")["pick_net"]
    xk = picks[picks["arm"] == "XGB"][["d", "symbol", "side"]].assign(x=1)
    h2h = {}
    for a in ARMS:
        d = days[days["arm"] == a]
        pooled = {k: X3.tstat(d[k].to_numpy(float)) for k in ("pick_minus_control", "pick_minus_shuffled")}
        gate = all(v["mean"] > 0 and v["t_nw"] >= X3.GATE_T for v in pooled.values())
        ins = [(v[a]["insample_mean"], v[a]["insample_days"]) for v in summ.values() if v[a]["insample_mean"] is not None]
        ins_mean = float(np.average([m for m, _ in ins], weights=[k for _, k in ins])) if ins else float("nan")
        oos = pooled["pick_minus_control"]["mean"]
        overfit = bool(ins_mean > M.OVERFIT * max(oos, 0.0))
        pa = picks[picks["arm"] == a].merge(xk, on=["d", "symbol", "side"], how="left")
        out["arms"][a] = {"insample_mean": ins_mean, "outsample_mean": oos,
                          "ratio": ins_mean / oos if oos > 0 else None, "overfit_rule21": overfit,
                          "pooled": pooled, "gate": gate,
                          "verdict": "overfit (rule 21)" if overfit else ("passed" if gate else "failed"),
                          "pick_overlap_with_xgb": float(pa["x"].notna().mean()) if len(pa) else None}
        if a != "XGB":
            both = pd.concat([d.set_index("d")["pick_net"], xgb], axis=1, keys=["a", "x"]).dropna()
            h2h[a] = X3.tstat((both["a"] - both["x"]).to_numpy(float))
    for (a, v), p in zip(h2h.items(), M.holm([M.pvalue(v["t_nw"]) for v in h2h.values()])):
        out["arms"][a]["vs_xgb"] = v | {"holm_p": float(p)}
    return out


def save_v1(ev, arms, out):
    meta = {}
    for a in arms:
        m = X3.Model(ev, learner(a))
        f = out / f"nxr_v1_{a}.pkl"
        f.write_bytes(pickle.dumps({"base": m.base, "news": m.news, "cut": m.cut}))
        meta[a] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    import sklearn
    import xgboost
    meta |= {"version": 1, "inputs": X3.INPUTS, "base_inputs": X3.BASE_INPUTS, "train_first": str(ev["d"].min().date()),
             "train_last": str(ev["d"].max().date()), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    (out / "nxr_v1_meta.json").write_text(json.dumps(meta, indent=1, default=str))


def run(ev, cal, state, out, years=N.TEST_YEARS, deadline=float("inf"), models=True):
    state, out = Path(state), Path(out)
    state.mkdir(parents=True, exist_ok=True)
    for yr in years:
        if (state / f"y{yr}.json").exists():
            continue
        if time.monotonic() > deadline:
            return None
        t0 = time.monotonic()
        res = year(ev, cal, yr)
        if res is None:
            continue
        summ, days, picks = res
        N.atomic_parquet(days, state / f"y{yr}_days.parquet")
        N.atomic_parquet(picks, state / f"y{yr}_picks.parquet")
        (state / f"y{yr}.json").write_text(json.dumps(summ, default=float))
        print(f"[nxr] {yr} done in {time.monotonic() - t0:.0f}s", flush=True)
    yrs = [yr for yr in years if (state / f"y{yr}.json").exists()]
    summ = {str(yr): json.loads((state / f"y{yr}.json").read_text()) for yr in yrs}
    rd = lambda k: pd.concat([pd.read_parquet(state / f"y{yr}_{k}.parquet") for yr in yrs], ignore_index=True)
    rep = report(summ, rd("days"), rd("picks"))
    out.mkdir(parents=True, exist_ok=True)
    rep["passing_arms"] = [a for a in NEW_ARMS if rep["arms"][a]["verdict"] == "passed"]
    if rep["passing_arms"] and models:
        save_v1(ev, rep["passing_arms"], out)
    (out / "NXR_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({a: {k: v[k] for k in ("verdict", "insample_mean", "outsample_mean", "ratio")} |
                      {k: (round(v["pooled"][k]["mean"], 5), round(v["pooled"][k]["t_nw"], 2)) for k in v["pooled"]}
                      for a, v in rep["arms"].items()}, indent=1, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--state", default="cache/nxr")
    ap.add_argument("--out", default="research/results/NXR")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    d3 = X3.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X3.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    if run(ev, X3.calendar(a.cache), a.state, a.out, deadline=deadline) is None:
        print("[nxr] time budget used; rerun to continue")


if __name__ == "__main__":
    main()
