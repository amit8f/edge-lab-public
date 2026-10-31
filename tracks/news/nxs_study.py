"""NXS-ML: a trained sizer for NX3-ALL's trades (research/prereg/NXS.md).

NX3-ALL's walk-forward picks and controls are rebuilt exactly (same models, seeds and random draws). For each
test year from 2021, a small XGBoost regressor fitted on the earlier test years' out-of-sample picks predicts
each pick's net return; its percentile among the training predictions sets the size between 5% and 10% of
the book. Compared with NX3-ALL's confidence formula on the same picks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X3

FIRST_YEAR = 2021                     # sizer test years; 2019-2020 only train it
SIZER = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=100, objective="reg:pseudohubererror", huber_slope=1.0, tree_method="hist",
             random_state=X3.SEED)
FEATURES = [*X3.INPUTS, "conf", "side", "order"]


def pick_rows(ev, cal, years=N.TEST_YEARS):
    """NX3-ALL's picks per test year with their paired control's net: same draws as X3.evaluate(cap=None)."""
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = ev[ev["d"] <= cut]
        te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = X3.Model(tr)
        real = te[te["kind"] == "real"]
        e = model.effect(real[X3.INPUTS].to_numpy(float))
        real = real.assign(e=e, fired=model.fires(e))
        rng = np.random.default_rng(X3.SEED + yr)
        for d, day in real.groupby("d", sort=True):
            p, c = X3.picks(day, rng, cap=None)
            if p.empty or c.empty:
                continue
            X3.picks(X3.shuffled(day, model, rng), rng, cap=None)     # keeps the draws identical
            out.append(p.assign(year=yr, conf=np.abs(p["e"].to_numpy(float)) / model.cut,
                                order=p["t"].rank(method="first").to_numpy() - 1,
                                net_p=N.net(p).to_numpy(float), net_c=N.net(c).to_numpy(float)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rank_sizes(pred, train_pred):
    """5% at the bottom of the training predictions, 10% at the top, straight line on the percentile."""
    q = np.searchsorted(np.sort(train_pred), pred, side="right") / max(len(train_pred), 1)
    return X3.ALL_MIN + (X3.ALL_MAX - X3.ALL_MIN) * np.clip(q, 0.0, 1.0)


def fit_ml(rows):
    from xgboost import XGBRegressor
    return XGBRegressor(**SIZER).fit(rows[FEATURES].to_numpy(float), rows["net_p"].to_numpy(float))


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)).rank(), pd.Series(np.asarray(b, float)).rank()
    return float(a.corr(b)) if len(a) > 2 else float("nan")


def scaled(w, day):
    """NX3-ALL's rule: a day whose sizes add up to more than the book is scaled down together."""
    s = pd.Series(w).groupby(day.to_numpy()).transform("sum").to_numpy()
    return w * np.minimum(1.0, 1.0 / s)


def measure(rows, fit=fit_ml, first=FIRST_YEAR):
    daily, ins, outs = [], [], []
    for yr in sorted(rows["year"].unique()):
        if yr < first:
            continue
        trn, tst = rows[rows["year"] < yr], rows[rows["year"] == yr].copy()
        if len(trn) < 500 or tst.empty:
            continue
        m = fit(trn)
        p_tr = m.predict(trn[FEATURES].to_numpy(float))
        p_te = m.predict(tst[FEATURES].to_numpy(float))
        ins.append(spearman(p_tr, trn["net_p"])), outs.append(spearman(p_te, tst["net_p"]))
        tst["w_ml"] = scaled(rank_sizes(p_te, p_tr), tst["d"])
        tst["w_f"] = scaled(X3.ALL_MIN + (X3.ALL_MAX - X3.ALL_MIN) * np.clip(tst["conf"] - 1, 0, 1), tst["d"])
        tst["diff"] = tst["net_p"] - tst["net_c"]
        g = tst.assign(ml=tst["w_ml"] * tst["diff"], f=tst["w_f"] * tst["diff"]).groupby("d")
        daily.append(pd.DataFrame({"ml": g["ml"].sum(), "formula": g["f"].sum(), "equal": g["diff"].mean(),
                                   "year": yr}))
    if not daily:
        return {"verdict": "failed", "reason": "no test year"}
    dd = pd.concat(daily)
    t = lambda x: X3.tstat(np.asarray(x, float))
    i_m, o_m = float(np.nanmean(ins)), float(np.nanmean(outs))
    overfit = not (o_m > 0) or i_m > 2 * o_m
    rep = {"ml_sized_minus_control": t(dd["ml"]), "ml_minus_formula": t(dd["ml"] - dd["formula"]),
           "formula_sized_minus_control": t(dd["formula"]), "equal_minus_control": t(dd["equal"]),
           "rule21_spearman_insample": i_m, "rule21_spearman_outsample": o_m, "rule21_overfit": overfit,
           "by_year": {str(y): {"ml": float(g["ml"].mean()), "formula": float(g["formula"].mean()), "days": len(g)}
                       for y, g in dd.groupby("year")}}
    a, b = rep["ml_sized_minus_control"], rep["ml_minus_formula"]
    ok = a["mean"] > 0 and a["t_nw"] >= X3.ALL_PASS_T and b["mean"] > 0 and b["t_nw"] >= X3.ALL_PASS_T
    rep["verdict"] = "passed" if ok and not overfit else "failed"
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/NXS")
    a = ap.parse_args()
    d3 = X3.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X3.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    rep = measure(pick_rows(ev, X3.calendar(a.cache)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NXS_result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
