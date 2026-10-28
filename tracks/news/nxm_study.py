"""NXM: NX3's design and data with other model types in place of XGBoost (research/prereg/NXM.md).

Reads the trades NX3's train stage read (cache/nx1/nx3/trades_v2) and runs NX3's walk-forward, picks, control,
shuffled-news test, costs and gate once per arm. Each finished test year is saved under --state, so a rerun
resumes. Summary numbers only -> research/results/NXM/ (and, for an arm that passes, its version 1 models).

Usage: python -m tracks.news.nxm_study --cache cache/nx1 --state cache/nxm --out research/results/NXM
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

SEED = X3.SEED
BASE_ARMS = ("XGB", "LGBM", "LIN", "RF", "MLP")
ARMS = BASE_ARMS + ("ENS",)
NEW_ARMS = ARMS[1:]                    # XGB is NX3 itself, rerun as a check
LGBM = dict(n_estimators=400, num_leaves=15, max_depth=4, learning_rate=0.05, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=50, objective="huber", alpha=1.0, random_state=SEED,
            deterministic=True, force_col_wise=True, verbose=-1)
LIN = dict(epsilon=1.35, alpha=1e-4, max_iter=200)
RF = dict(n_estimators=200, max_features=0.33, min_samples_leaf=200, max_samples=0.3, n_jobs=-1, random_state=SEED)
MLP = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3, max_iter=60,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)
OVERFIT = 2.0                          # PROTOCOL rule 21: in-sample more than twice out-of-sample
HOLM_P = 0.05
TONE = "fb_net"
KEEP = ["symbol", "d", "lm_net", *[f"f_{k}" for k in N.FLAGS]]


def learner(arm):
    """(X, y) -> fitted model with .predict, with the settings fixed in NXM.md."""
    if arm == "XGB":
        return X3.fit

    def fit(X, y):
        if arm == "LGBM":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**LGBM).fit(X, y)
        if arm == "RF":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(**RF).fit(X, y)
            return m.set_params(n_jobs=1)          # scoring a day's few hundred events is faster on one core
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if arm == "LIN":
            from sklearn.linear_model import HuberRegressor
            est = HuberRegressor(**LIN)
        else:
            from sklearn.neural_network import MLPRegressor
            est = MLPRegressor(**MLP)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est).fit(X, y)
    return fit


class Ensemble:
    """ENS: the mean of the arms' effects, each over its own fire cut, fired at the training years' 90th
    percentile of that mean (X = the training real events)."""

    def __init__(self, models, X):
        self.models = models
        self.cut = float(np.quantile(np.abs(self.effect(X)), X3.FIRE_Q))

    def effect(self, X):
        return np.mean([m.effect(X) / m.cut for m in self.models], axis=0)

    def fires(self, e):
        return (np.abs(e) >= self.cut) & (np.abs(e) > 0)


def train_real(tr):
    tr = tr.assign(z=X3.zlabel(tr))
    return tr[np.isfinite(tr["z"]) & (tr["kind"] == "real")]


def fit_arms(tr, arms=BASE_ARMS):
    models = {a: X3.Model(tr, learner(a)) for a in arms}
    models["ENS"] = Ensemble(list(models.values()), train_real(tr)[X3.INPUTS].to_numpy(float))
    return models


def insample(trr, model, seed):
    """Rule 21: the model scored on its own training events with NX3's picks and same-day control."""
    e = model.effect(trr[X3.INPUTS].to_numpy(float))
    real = trr.assign(e=e, fired=model.fires(e))
    rng = np.random.default_rng(seed)
    out = []
    for _, day in real.groupby("d", sort=True):
        p, c = X3.picks(day, rng)
        if len(p) and len(c):
            out.append(N.net(p).mean() - N.net(c).mean())
    return np.array(out, float)


def year(ev, cal, yr):
    """One walk-forward test year for every arm: per-arm summary, daily rows, picks and test effects."""
    caldates = pd.DatetimeIndex(cal["date"])
    start = pd.Timestamp(f"{yr}-01-01")
    cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
    tr = ev[ev["d"] <= cut]
    te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
    if tr.empty or te.empty:
        return None
    models = fit_arms(tr)
    trr = train_real(tr)
    summ, days, picks = {}, [], []
    eff = te[te["kind"] == "real"][KEEP].assign(**{TONE: (te["fb_pos"] - te["fb_neg"])[te["kind"] == "real"]})
    for a, m in models.items():
        n, pk, real, pe = X3.evaluate(te, m, SEED + yr)
        ins = insample(trr, m, SEED + yr)
        eff[f"e_{a}"] = (real["e"] / m.cut).astype("float32").to_numpy()
        eff[f"fired_{a}"] = real["fired"].to_numpy()
        if len(pk):
            pk = pk.assign(net=N.net(pk))
            n = n.merge(pk.groupby("d")["net"].mean().rename("pick_net").reset_index(), on="d", how="left")
            picks.append(pk[["d", "symbol", "side", "net"]].assign(arm=a))
        days.append(n.assign(arm=a))
        summ[a] = {"fire_cut": m.cut, "real_fire_rate": float(real["fired"].mean()),
                   "placebo_fire_rate": float(m.fires(pe).mean()) if len(pe) else None,
                   "insample_mean": float(ins.mean()) if len(ins) else None, "insample_days": int(len(ins)),
                   "picks_median_vol20_pct": float(pk["vpct"].median()) if len(pk) else None,
                   "picks_share_before_open": float((pk["mins_open"] <= 0).mean()) if len(pk) else None}
    return summ, pd.concat(days, ignore_index=True), (pd.concat(picks, ignore_index=True) if picks else
                                                       pd.DataFrame(columns=["d", "symbol", "side", "net", "arm"])), eff


def holm(p):
    """Holm step-down adjusted p-values (same order as p)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def pvalue(t):
    from math import erf, sqrt
    return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))) if np.isfinite(t) else 1.0


def report(summ, days, picks, eff):
    """Pooled verdicts per arm (rule 21 first, then NX3's gate), head-to-head against XGBoost with Holm, and
    the descriptive 'how they react' tables."""
    out = {"arms": {}, "per_year": summ}
    xgb = days[days["arm"] == "XGB"].set_index("d")["pick_net"]
    h2h = {}
    for a in ARMS:
        d = days[days["arm"] == a]
        pooled = {k: X3.tstat(d[k].to_numpy(float)) for k in ("pick_minus_control", "pick_minus_shuffled")}
        gate = all(v["mean"] > 0 and v["t_nw"] >= X3.GATE_T for v in pooled.values())
        ins = [(v[a]["insample_mean"], v[a]["insample_days"]) for v in summ.values() if v[a]["insample_mean"] is not None]
        ins_mean = float(np.average([m for m, _ in ins], weights=[k for _, k in ins])) if ins else float("nan")
        oos = pooled["pick_minus_control"]["mean"]
        overfit = bool(ins_mean > OVERFIT * max(oos, 0.0))
        verdict = "overfit (rule 21)" if overfit else ("passed" if gate else "failed")
        out["arms"][a] = {"insample_mean": ins_mean, "outsample_mean": oos, "overfit_rule21": overfit,
                          "pooled": pooled, "gate": gate, "verdict": verdict}
        if a != "XGB":
            both = pd.concat([d.set_index("d")["pick_net"], xgb], axis=1, keys=["a", "x"]).dropna()
            h2h[a] = X3.tstat((both["a"] - both["x"]).to_numpy(float))
    adj = holm([pvalue(v["t_nw"]) for v in h2h.values()])
    for (a, v), p in zip(h2h.items(), adj):
        out["arms"][a]["vs_xgb"] = v | {"holm_p": float(p), "beats_xgb": bool(v["mean"] > 0 and p < HOLM_P)}
    out["react"] = react(eff, picks)
    return out


def react(eff, picks):
    """How each arm reacts: e/cut agreement with XGBoost, pick overlap, and e/cut and fire shares by tone
    decile and by news-type flag (test events, all years)."""
    r = {"spearman_with_xgb": {}, "pick_overlap_with_xgb": {}, "by_tone_decile": {}, "by_flag": {}}
    xk = picks[picks["arm"] == "XGB"][["d", "symbol", "side"]].assign(x=1)
    for a in ARMS:
        r["spearman_with_xgb"][a] = float(eff[f"e_{a}"].rank().corr(eff["e_XGB"].rank()))
        pa = picks[picks["arm"] == a].merge(xk, on=["d", "symbol", "side"], how="left")
        r["pick_overlap_with_xgb"][a] = float(pa["x"].notna().mean()) if len(pa) else None
    dec = pd.qcut(eff[TONE].rank(method="first"), 10, labels=False)
    for a in ARMS:
        g = eff.assign(dec=dec, long=eff[f"fired_{a}"] & (eff[f"e_{a}"] > 0),
                       short=eff[f"fired_{a}"] & (eff[f"e_{a}"] < 0)).groupby("dec")
        r["by_tone_decile"][a] = {"mean_e": g[f"e_{a}"].mean().round(4).tolist(),
                                  "long_share": g["long"].mean().round(4).tolist(),
                                  "short_share": g["short"].mean().round(4).tolist()}
        r["by_flag"][a] = {k: {"n": int(eff[f"f_{k}"].gt(0).sum()),
                               "mean_e": float(eff.loc[eff[f"f_{k}"] > 0, f"e_{a}"].mean())} for k in N.FLAGS}
    r["tone_decile_edges"] = eff.groupby(dec)[TONE].median().round(4).tolist()
    return r


def save_v1(ev, gate_arms, out):
    """Version 1 of each passing arm, trained on all events as NX3's was (for its shadow book)."""
    models = fit_arms(ev)
    meta = {}
    for a in gate_arms:
        f = out / f"nxm_v1_{a}.pkl"
        m = models[a]
        if a == "ENS":
            m = {"members": list(BASE_ARMS), "member_cuts": [x.cut for x in m.models], "cut": m.cut}
        f.write_bytes(pickle.dumps(m if a == "ENS" else {"base": m.base, "news": m.news, "cut": m.cut}))
        meta[a] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    if "ENS" in gate_arms:   # ENS needs its members too
        for a in BASE_ARMS:
            f = out / f"nxm_v1_member_{a}.pkl"
            f.write_bytes(pickle.dumps({"base": models[a].base, "news": models[a].news, "cut": models[a].cut}))
            meta[f"member_{a}"] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    import lightgbm
    import sklearn
    import xgboost
    meta |= {"version": 1, "inputs": X3.INPUTS, "base_inputs": X3.BASE_INPUTS, "train_first": str(ev["d"].min().date()),
             "train_last": str(ev["d"].max().date()), "lightgbm": lightgbm.__version__,
             "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    (out / "nxm_v1_meta.json").write_text(json.dumps(meta, indent=1, default=str))


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
        summ, days, picks, eff = res
        N.atomic_parquet(days, state / f"y{yr}_days.parquet")
        N.atomic_parquet(picks, state / f"y{yr}_picks.parquet")
        N.atomic_parquet(eff, state / f"y{yr}_eff.parquet")
        (state / f"y{yr}.json").write_text(json.dumps(summ, default=float))
        print(f"[nxm] {yr} done in {time.monotonic() - t0:.0f}s", flush=True)
    yrs = [yr for yr in years if (state / f"y{yr}.json").exists()]
    summ = {str(yr): json.loads((state / f"y{yr}.json").read_text()) for yr in yrs}
    rd = lambda k: pd.concat([pd.read_parquet(state / f"y{yr}_{k}.parquet") for yr in yrs], ignore_index=True)
    rep = report(summ, rd("days"), rd("picks"), rd("eff"))
    out.mkdir(parents=True, exist_ok=True)
    passing = [a for a in NEW_ARMS if rep["arms"][a]["verdict"] == "passed"]
    rep["passing_arms"] = passing
    if passing and models:
        save_v1(ev, passing, out)
    (out / "NXM_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({a: {k: v[k] for k in ("verdict", "insample_mean", "outsample_mean")} |
                      {k: (round(v["pooled"][k]["mean"], 5), round(v["pooled"][k]["t_nw"], 2)) for k in v["pooled"]}
                      for a, v in rep["arms"].items()}, indent=1, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--state", default="cache/nxm")
    ap.add_argument("--out", default="research/results/NXM")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    d3 = X3.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X3.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    if run(ev, X3.calendar(a.cache), a.state, a.out, deadline=deadline) is None:
        print("[nxm] time budget used; rerun to continue")


if __name__ == "__main__":
    main()
