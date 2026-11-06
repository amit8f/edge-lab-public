"""CHART1: a chart-and-volume-only daily stock picker, several model families (research/prereg/CHART1.md).

No news inputs. Each session's close: score every liquid stock from its daily bars (core/chart_features.py),
buy the 3 highest and short the 3 lowest at the next session's open, sell at that session's close. Same
walk-forward years, purge, Newey-West t and costs as NX3. Reads the chart files the nx3-study `chart_bars`
stage writes (cache/nx1/nx3/chart). Writes research/results/CHART1/CHART1_walkforward.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import chart_features as C
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

SEED = 20261106
MIN_PX, MIN_DV = 5.0, 5e6
FIRST = pd.Timestamp("2016-01-01")
TRAIN_ROWS = 400_000
K = 3
Z_CLIP = 6.0
ARMS = ("CAT", "LGBM", "ET", "RIDGE", "SEQ", "REV")
INPUTS = C.FEATURES + C.MARKET
SEQ_INPUTS = C.SEQ_COLS + C.MARKET


def label(df):
    v = df["vol20"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = df["mret"].to_numpy(float) / v
    z[~(v > 0)] = np.nan
    return np.clip(z, -Z_CLIP, Z_CLIP)


def learner(arm):
    """The model families: CatBoost and LightGBM (boosted trees), Extra Trees (random-split forest), Ridge
    (linear), SEQ (a small neural net on the last 10 days' bar shape only). REV is the simple-rule baseline."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if arm == "CAT":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                 loss_function="Huber:delta=1.0", random_seed=SEED, verbose=0, thread_count=-1,
                                 allow_writing_files=False)
    if arm == "LGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.03, min_child_samples=500,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="huber", alpha=1.0,
                             random_state=SEED, verbose=-1)
    if arm == "ET":
        from sklearn.ensemble import ExtraTreesRegressor
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=200, max_features=0.33, min_samples_leaf=300, max_samples=0.3, bootstrap=True, n_jobs=-1,
            random_state=SEED))
    if arm == "RIDGE":
        from sklearn.linear_model import Ridge
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
    if arm == "SEQ":
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3, max_iter=30,
            early_stopping=True, n_iter_no_change=4, random_state=SEED))
    raise ValueError(arm)


def arm_inputs(arm):
    return SEQ_INPUTS if arm == "SEQ" else INPUTS


def score(arm, model, df):
    if arm == "REV":       # buy the biggest 5-day losers, short the biggest winners (short-term reversal)
        return -(df["r5"] / df["vol20"]).to_numpy(float)
    X = df[arm_inputs(arm)].to_numpy(np.float32)
    return model.predict(X)


def net(rows, side):
    """Net return of one-day open-to-close trades after NX3's costs (entry price = raw next open)."""
    r = rows["ret"].to_numpy(float)
    return side * r - N.cost_frac(rows["o1"].to_numpy(float), np.full(len(rows), side < 0), np.ones(len(rows)))


def day_picks(day, s, rng, k=K, side=None):
    """Top k by score long, bottom k short; control: as many random stocks of the same day, same sides."""
    o = np.argsort(s, kind="stable")
    lo, sh = o[::-1][:k], o[:k]
    if side == 1:
        sh = sh[:0]
    if side == -1:
        lo = lo[:0]
    if len(day) < 2 * k:
        return None
    c = rng.permutation(len(day))
    cl, cs = c[:len(lo)], c[len(lo):len(lo) + len(sh)]
    p = np.r_[net(day.iloc[lo], 1), net(day.iloc[sh], -1)]
    q = np.r_[net(day.iloc[cl], 1), net(day.iloc[cs], -1)]
    return float(np.nanmean(p)), float(np.nanmean(q)), (float(np.nanmean(net(day.iloc[lo], 1))) if len(lo) else np.nan,
                                                         float(np.nanmean(net(day.iloc[sh], -1))) if len(sh) else np.nan)


def load(cache):
    d = X.nx3dir(cache) / X.CHART
    spy = pd.read_parquet(d / "spy.parquet")
    spy = spy.assign(spy_ret=spy["ac1"] / spy["ao1"] - 1)[["date", *C.MARKET, "spy_ret"]]
    cols = ["symbol", "date", "c", "dv20", "o1", "ao1", "ac1", "nxt", *C.FEATURES]
    parts = []
    for f in sorted(d.glob("c_*.parquet")):
        x = pd.read_parquet(f, columns=cols)
        x = x[(x["date"] >= FIRST) & (x["c"] >= MIN_PX) & (x["dv20"] >= MIN_DV) & x["vol20"].notna()
              & x["nxt"].notna() & (x["nxt"] <= N.LAST_EXIT)]
        parts.append(x)
    df = pd.concat(parts, ignore_index=True).merge(spy, on="date", how="inner")
    df["ret"] = df["ac1"] / df["ao1"] - 1
    df["mret"] = df["ret"] - df["spy_ret"]
    df = df[np.isfinite(df["mret"]) & (df["o1"] > 0)].reset_index(drop=True)
    df["z"] = label(df)
    return df.rename(columns={"date": "d"})


def evaluate(te, s, seed, k=K, side=None):
    rng = np.random.default_rng(seed)
    te = te.assign(s=s)
    rows = []
    for d, day in te.groupby("d", sort=True):
        r = day_picks(day.reset_index(drop=True), day["s"].to_numpy(float), rng, k, side)
        if r:
            rows.append({"d": d, "pick": r[0], "control": r[1], "long": r[2][0], "short": r[2][1]})
    x = pd.DataFrame(rows, columns=["d", "pick", "control", "long", "short"])
    x["d"] = pd.to_datetime(x["d"])
    x["pick_minus_control"] = x["pick"] - x["control"]
    return x


def summarise(x):
    t = X.tstat
    return {"pick_minus_control": t(x["pick_minus_control"].to_numpy(float)), "net_per_day": t(x["pick"].to_numpy(float)),
            "long_net": t(x["long"].to_numpy(float)), "short_net": t(x["short"].to_numpy(float)),
            "control_net": t(x["control"].to_numpy(float)),
            "per_year": {int(y): {"pick_minus_control": float(g["pick_minus_control"].mean()), "net": float(g["pick"].mean())}
                         for y, g in x.groupby(x["d"].dt.year)}}


def run(cache, out, arms=ARMS):
    df = load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    print(f"[chart1] {len(df)} stock-days, {df['symbol'].nunique()} symbols, {df['d'].nunique()} days")
    res = {a: [] for a in arms}
    expl = {f"{a}_{tag}": [] for a in arms for tag in ("top10", "long_only")}
    imp = {}
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = df[(df["d"] <= cut_d) & np.isfinite(df["z"])]
        te = df[(df["d"] >= start) & (df["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        if len(tr) > TRAIN_ROWS:
            tr = tr.iloc[np.sort(np.random.default_rng(SEED + yr).choice(len(tr), TRAIN_ROWS, replace=False))]
        for a in arms:
            m = None
            if a != "REV":
                m = learner(a).fit(tr[arm_inputs(a)].to_numpy(np.float32), tr["z"].to_numpy(float))
                if a == "CAT":
                    imp[yr] = dict(zip(INPUTS, map(float, m.get_feature_importance())))
            s = score(a, m, te)
            res[a].append(evaluate(te, s, SEED + yr))
            expl[f"{a}_top10"].append(evaluate(te, s, SEED + yr, k=10))
            expl[f"{a}_long_only"].append(evaluate(te, s, SEED + yr, side=1))
            print(f"[chart1] {yr} {a}: pick-minus-control {res[a][-1]['pick_minus_control'].mean():+.5f}, "
                  f"net {res[a][-1]['pick'].mean():+.5f}")
    rep = {"universe": {"min_price": MIN_PX, "min_dv20": MIN_DV, "stock_days": int(len(df)),
                        "symbols": int(df["symbol"].nunique())}, "gate_t": X.GATE_T, "arms": {}, "exploratory": {}}
    cat = {a: pd.concat(res[a], ignore_index=True) for a in arms}
    for a in arms:
        r = summarise(cat[a])
        pmc, nd = r["pick_minus_control"], r["net_per_day"]
        r["gate"] = bool(pmc["mean"] > 0 and pmc["t_nw"] >= X.GATE_T and nd["mean"] > 0)
        if a != "REV":
            m = cat[a].merge(cat["REV"], on="d", suffixes=("", "_rev"))
            r["minus_rev"] = X.tstat((m["pick"] - m["pick_rev"]).to_numpy(float))
        rep["arms"][a] = r
    for k, v in expl.items():
        rep["exploratory"][k] = summarise(pd.concat(v, ignore_index=True))
    if imp:
        mean_imp = pd.DataFrame(imp).mean(axis=1).sort_values(ascending=False)
        rep["cat_importance_top15"] = {k: float(v) for k, v in mean_imp.head(15).items()}
    models = [a for a in arms if a != "REV"]
    rep["verdict"] = "passed" if any(rep["arms"][a]["gate"] for a in models) else "failed"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CHART1_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep
