"""NX2 (NEWS-XGB version 2, the news-only effect) walk-forward, gate and model (research/prereg/NX2.md).

Reads NX1's cached event files (real and placebo events with their 32 inputs, label and costs; built by
tracks/news/nx1_study.py) and adds no data. Writes summary numbers and, when the gate passes, the version 1
models to research/results/NX2/.

Usage: python -m tracks.news.nx2_study --cache <dir> --out <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main.longterm.stats import nw_tstat
from tracks.news import nx1_study as N

SEED = 20261023
Z_CLIP = 6.0
FIRE_Q = 0.90
GATE_T = 2.0
BASE_INPUTS = N.STOCK_INPUTS + N.MARKET_INPUTS
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=50, objective="reg:pseudohubererror", huber_slope=1.0, tree_method="hist",
           random_state=SEED)
GROUPS = {"news": N.NEWS_INPUTS, "novelty": ["novelty"], "stock": N.STOCK_INPUTS, "market": N.MARKET_INPUTS}


def zlabel(ev):
    """3-day market-adjusted return in units of the stock's own 3-day swing, clipped; NaN without vol20."""
    v = ev["vol20"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ev["mret"].to_numpy(float) / (v * np.sqrt(N.HOLD))
    z[~(v > 0)] = np.nan
    return np.clip(z, -Z_CLIP, Z_CLIP)


def fit(X, y):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB)
    m.fit(X, y)
    return m


def no_news(X):
    """X with the 23 news inputs at NX1's no-news values (novelty, stock and market inputs kept)."""
    X0 = X.copy()
    nv = N.placebo_news()
    for i, k in enumerate(N.INPUTS):
        if k in nv:
            X0[:, i] = nv[k]
    return X0


class Model:
    """Baseline B on placebo events, news model N on real events' z - B; effect e = N(x) - N(x0)."""

    def __init__(self, tr):
        tr = tr.assign(z=zlabel(tr))
        tr = tr[np.isfinite(tr["z"])]
        pl, re = tr[tr["kind"] == "placebo"], tr[tr["kind"] == "real"]
        self.base = fit(pl[BASE_INPUTS].to_numpy(float), pl["z"].to_numpy(float))
        X = re[N.INPUTS].to_numpy(float)
        self.news = fit(X, re["z"].to_numpy(float) - self.base.predict(re[BASE_INPUTS].to_numpy(float)))
        self.cut = float(np.quantile(np.abs(self.effect(X)), FIRE_Q))
        self.n_base, self.n_news = int(len(pl)), int(len(re))

    def effect(self, X):
        return self.news.predict(X) - self.news.predict(no_news(X))

    def fires(self, e):
        return (np.abs(e) >= self.cut) & (np.abs(e) > 0)


def picks(night, rng, col="e"):
    """Top PICKS fired events by effect long, bottom PICKS short; control as NX1 (random, same sides)."""
    f = night[night["fired"]]
    lo = f[f[col] > 0].nlargest(N.PICKS, col).assign(side=1)
    sh = f[f[col] < 0].nsmallest(N.PICKS, col).assign(side=-1)
    p = pd.concat([lo, sh])
    if p.empty or len(night) < len(p):
        return p, p.iloc[0:0]
    c = night.iloc[rng.choice(len(night), size=len(p), replace=False)].assign(side=p["side"].to_numpy())
    return p, c


def shuffled(night, model, rng):
    """The night's real events with their news inputs permuted among them, re-scored by the same model."""
    X = night[N.INPUTS].to_numpy(float)
    idx = [N.INPUTS.index(k) for k in N.NEWS_INPUTS]
    X[:, idx] = X[rng.permutation(len(X))][:, idx]
    e = model.effect(X)
    return night.assign(e=e, fired=model.fires(e))


def evaluate(te, model, seed):
    """Per night: picks minus control, and picks minus shuffled-news picks (net, NX1 costs)."""
    real = te[te["kind"] == "real"]
    X = real[N.INPUTS].to_numpy(float)
    e = model.effect(X)
    real = real.assign(e=e, fired=model.fires(e))
    plac = te[te["kind"] == "placebo"]
    pe = model.effect(plac[N.INPUTS].to_numpy(float)) if len(plac) else np.zeros(0)
    rng = np.random.default_rng(seed)
    rows, pk = [], []
    for d, night in real.groupby("d", sort=True):
        p, c = picks(night, rng)
        if p.empty or c.empty:
            continue
        s, _ = picks(shuffled(night, model, rng), rng)
        pn = N.net(p).mean()
        rows.append({"d": d, "pick_minus_control": pn - N.net(c).mean(),
                     "pick_minus_shuffled": pn - N.net(s).mean() if len(s) else np.nan})
        pct = night["vol20"].rank(pct=True)
        pk.append(pct.loc[p.index])
    return pd.DataFrame(rows), (pd.concat(pk) if pk else pd.Series(dtype=float)), real, pe


def tstat(x):
    m, se, t = nw_tstat(x, 5) if np.isfinite(x).sum() > 2 else (float("nan"),) * 3
    return {"mean": m, "se_nw": se, "t_nw": t, "nights": int(np.isfinite(x).sum())}


def walk_forward(ev, cal, years=N.TEST_YEARS):
    caldates = pd.DatetimeIndex(cal["date"])
    out, nights = {}, []
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = ev[ev["d"] <= cut]
        te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = Model(tr)
        n, pct, real, pe = evaluate(te, model, SEED + yr)
        nights.append(n)
        rho = pd.Series(np.abs(real["e"].to_numpy())).corr(pd.Series(real["vol20"].to_numpy()), method="spearman")
        out[str(yr)] = {"train_placebo": model.n_base, "train_real": model.n_news, "fire_cut": model.cut,
                        "real_fire_rate": float(real["fired"].mean()),
                        "placebo_fire_rate": float(model.fires(pe).mean()) if len(pe) else None,
                        "pick_minus_control": tstat(n["pick_minus_control"].to_numpy(float)) if len(n) else None,
                        "pick_minus_shuffled": tstat(n["pick_minus_shuffled"].to_numpy(float)) if len(n) else None,
                        "picks_median_vol20_pct": float(pct.median()) if len(pct) else None,
                        "spearman_abs_effect_vol20": float(rho), "importance_gain_by_group": importance(model)}
    alln = pd.concat(nights, ignore_index=True) if nights else pd.DataFrame(
        columns=["pick_minus_control", "pick_minus_shuffled"])
    pooled = {k: tstat(alln[k].to_numpy(float)) for k in ("pick_minus_control", "pick_minus_shuffled")}
    gate = {k: bool(v["mean"] > 0 and v["t_nw"] >= GATE_T) for k, v in pooled.items()}
    return out, pooled, gate


def importance(model):
    gain = model.news.get_booster().get_score(importance_type="gain")
    by = {N.INPUTS[int(k[1:])]: v for k, v in gain.items()}
    return {g: float(sum(by.get(k, 0.0) for k in ks)) for g, ks in GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/NX2")
    a = ap.parse_args()
    root = N.nx1dir(a.cache)
    cols = sorted(set(N.INPUTS) | {"kind", "d", "exit", "ret", "mret", "price", "days", "symbol"})
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((root / "events").glob("e_*.parquet"))],
                   ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["d"] <= N.LAST_DAY) & (ev["exit"] <= N.LAST_EXIT)].reset_index(drop=True)
    cal = pd.read_parquet(root / "cal.parquet")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wf, pooled, gate = walk_forward(ev, cal)
    rep = {"walk_forward": wf, "pooled": pooled, "gate": gate, "gate_passed": all(gate.values()),
           "events": {"real": int((ev["kind"] == "real").sum()), "placebo": int((ev["kind"] == "placebo").sum())}}
    if rep["gate_passed"]:
        import xgboost
        m = Model(ev)
        files = {"base": out / "nx2_v1_base.json", "news": out / "nx2_v1_news.json"}
        m.base.get_booster().save_model(files["base"])
        m.news.get_booster().save_model(files["news"])
        meta = {"version": 1, "inputs": N.INPUTS, "base_inputs": BASE_INPUTS, "fire_cut": m.cut, "params": XGB,
                "train_first": str(ev["d"].min().date()), "train_last": str(ev["d"].max().date()),
                "train_placebo": m.n_base, "train_real": m.n_news, "xgboost": xgboost.__version__,
                "importance_gain_by_group": importance(m),
                "sha256": {k: hashlib.sha256(f.read_bytes()).hexdigest() for k, f in files.items()}}
        (out / "nx2_v1_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "NX2_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"pooled": pooled, "gate": gate}, indent=1, default=float))


if __name__ == "__main__":
    main()
