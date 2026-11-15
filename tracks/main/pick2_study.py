"""PICK2-MAIN: the same-day picker trained on what a trade earns (research/prereg/PICK2.md, Amendment 1).

Stage fit (2019-2023 only, PICK's sampled joined rows): per input set and learner one Huber regressor of
net / swing30. The 2019-2022 model is the validation model, the 2019-2023 one the refit. Bars are top shares of
each model's own training predictions. No 2023 result is computed here.

Stage scan (resumable, month by month; 2022-12 is a swing warm-up only): every 2023 main decision row is scored by
the validation models; per day, model and bar the first rows to clear are kept and their trades simulated. When the
last month is done the 2023 book of every (input set, learner, bar) is built and the choice and the gate are
written to <out>/PICK2_choose.json.

Stage oof (PICK2-OOF, prereg PICK2.md): bars and tops from out-of-fold predictions, one fold per year, to
<out>/PICK2_oof.json; stage scan-oof is stage scan with those bars (its own --state), choice to PICK2_choose_oof.json.

Usage: python -m tracks.main.pick2_study --stage fit|scan|oof|scan-oof --w1 cache/w1 --nx1 cache/nx1 --fitstate cache/pickfit \
           --pick research/results/PICK --out research/results/PICK2 --state cache/pick2 --budget-min 320
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main import pick_data as P
from tracks.main import pick_study as S
from tracks.main import pick_test as T
from tracks.main.longterm.stats import nw_tstat
from tracks.news import w1_data as W

TRACK = "main"
CLIP = 10.0                                   # the target is clipped to +-10 swings
DELTA = 1.0                                   # Huber: squared inside one swing, linear outside
LEARNERS = ["XGBoost", "CatBoost", "LightGBM"]
BAR_Q = (0.001, 0.0005, 0.0002, 0.0001)       # top shares of the training predictions
MAX_BUYS = 2
MIN_TRADES = 100
KEEP_N = 10                                   # first stocks kept per day, model and bar (the book takes 2)
TOP_Q = 0.99
WARM, FIRST, LAST = "2022-12", "2023-01", "2023-12"
WIDTHS = P.TRACKS[TRACK]["widths"]
NET_COLS = [f"net_{w}" for w in WIDTHS]


# ---------- pure helpers (tested) ----------

def frozen(pick_fit):
    """What PICK2 takes from PICK's committed fit: market parameters and cut, tier cuts, widths per tier."""
    t = pick_fit["tracks"][TRACK]
    return {"mkt": {k: tuple(v) for k, v in t["market"]["parts"].items()}, "cut": float(t["market"]["cut_20pct"]),
            "tier_cuts": [float(c) for c in t["tier_cuts_swing30"]],
            "wpt": {v: {int(k): float(w) for k, w in t["versions"][v]["widths_per_tier"].items()}
                    for v in S.VERSIONS}}


def row_net(D, tier, wpt):
    """Each row's net return at the stop width its tier has."""
    return S.row_scores({w: D[f"net_{w}"].to_numpy(np.float32) for w in set(wpt.values())}, np.asarray(tier), wpt)


def target(net, swing):
    """Net return in units of the stock's usual 30-minute swing, clipped; NaN where undefined."""
    net, swing = np.asarray(net, float), np.asarray(swing, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(swing > 0, net / swing, np.nan)
    return np.clip(y, -CLIP, CLIP).astype(np.float32)


def make(name):
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:pseudohubererror", huber_slope=DELTA, n_estimators=300, max_depth=4,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                            tree_method="hist", random_state=S.SEED, n_jobs=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function=f"Huber:delta={DELTA}", iterations=300, depth=6, learning_rate=0.08,
                                 random_seed=S.SEED, verbose=False, thread_count=-1, allow_writing_files=False)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="huber", alpha=DELTA, n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=S.SEED, verbose=-1, n_jobs=-1)
    raise ValueError(name)


def load_model(path):
    p = str(path)
    if p.endswith(".json"):
        from xgboost import XGBRegressor
        m = XGBRegressor()
        m.load_model(p)
        return m
    if p.endswith(".cbm"):
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(p)
        return m
    import lightgbm
    return lightgbm.Booster(model_file=p)


def predict(m, X):
    out = [np.asarray(m.predict(X[i:i + S.TREE_CHUNK]), float) for i in range(0, len(X), S.TREE_CHUNK)]
    return np.concatenate(out) if out else np.empty(0)


def oof_predict(L, X, y, year, folds):
    """Out-of-fold predictions: each year in `folds` predicted by learner L fitted on the other years' rows."""
    out = np.full(len(y), np.nan, np.float32)
    for yr in folds:
        te = year == yr
        m = make(L)
        m.fit(X[~te], y[~te])
        out[te] = predict(m, X[te])
    return out


def bars_of(pred_net):
    """{top share: the predicted net return above which that share of the rows lies}."""
    return {str(q): S.top_threshold(pred_net, q) for q in BAR_Q}


def tops_of(pred_net, bars):
    """Per bar: the 99th percentile of the predictions at or above it (the sizing's top, PICK's rule)."""
    out = {}
    for q, b in bars.items():
        above = pred_net[pred_net >= b]
        out[q] = float(np.quantile(above, TOP_Q)) if len(above) else float(b)
    return out


def first_clear(g, col, thr, n=None):
    """One day's rows -> the stocks whose prediction clears the bar, each at its first clearing time, in buying
    order (earlier first; at the same time the higher prediction first)."""
    c = g[g[col].to_numpy(float) >= thr].sort_values(["t", col], ascending=[True, False], kind="stable")
    c = c.drop_duplicates("symbol", keep="first")
    return c if n is None else c.head(n)


def keep_first(g, bars):
    """Rows of one day any (model, bar) book could buy: the first KEEP_N stocks of each. bars: {column: {q: thr}}."""
    idx = set()
    for col, by_q in bars.items():
        for thr in by_q.values():
            idx.update(first_clear(g, col, thr, KEEP_N).index)
    return g.loc[sorted(idx)]


def book(R, col, thr, wpt):
    """Scan rows -> the trades of one (model, bar): per day the first MAX_BUYS stocks to clear, one row a stock."""
    out = []
    for _, g in R.groupby("d", sort=True):
        c = first_clear(g, col, thr)
        if not len(c):
            continue
        w = c["tier"].map(wpt).to_numpy(float)
        recs = c.to_dict("records")
        c = c.assign(net=[r[f"net_{x}"] for r, x in zip(recs, w)], exit=[r[f"exit_{x}"] for r, x in zip(recs, w)],
                     pred=c[col].to_numpy(float))
        out.append(c[np.isfinite(c["net"].to_numpy(float))].head(MAX_BUYS))
    cols = ["symbol", "d", "t", "tier", "bad", "pred", "net", "exit"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def summary(tk):
    n = len(tk)
    if not n:
        return {"trades": 0, "mean_net": float("nan")}
    net = tk["net"].to_numpy(float)
    day = pd.Series(net).groupby(tk["d"].to_numpy()).mean().to_numpy()
    t = nw_tstat(day, 5)[2] if len(day) >= 3 else float("nan")
    return {"trades": n, "mean_net": float(net.mean()), "median_net": float(np.median(net)),
            "share3": float((net >= S.HOLD_RET).mean()), "days": len(day), "t_nw_daily": float(t),
            "exits": {str(k): float(v) for k, v in tk["exit"].value_counts(normalize=True).items()},
            "bad_vs_other": {("bad" if k else "other"): {"n": len(g), "mean_net": float(g["net"].mean())}
                             for k, g in tk.groupby(tk["bad"].astype(bool))}}


def choose(table):
    """table: {learner: {q: summary}} -> the (learner, q) with the best mean net among those with MIN_TRADES."""
    ok = [(L, q, s) for L, by_q in table.items() for q, s in by_q.items() if s["trades"] >= MIN_TRADES]
    if not ok:
        return None
    L, q, s = max(ok, key=lambda x: x[2]["mean_net"])
    return {"learner": L, "q": q, "trades": s["trades"], "mean_net": s["mean_net"]}


def col_of(version, learner):
    return f"p_{version}_{learner}"


# ---------- fit ----------

def load(state):
    cols = ["track", "d", *S.FIT_NUM, *NET_COLS]
    parts = []
    for y in range(S.FIT_YEARS[0], S.FIT_YEARS[1] + 1):
        for m in pd.period_range(f"{y}-01", f"{y}-12", freq="M"):
            df = S.read_joined(state, str(m), cols)
            if len(df):
                parts.append(df[df["track"] == TRACK])
    D = pd.concat(parts, ignore_index=True)
    D["year"] = pd.to_datetime(D["d"]).dt.year.astype(np.int16)
    for c in S.FIT_NUM + NET_COLS:
        D[c] = D[c].astype(np.float32)
    return D


def stage_fit(state, pick_dir, out):
    out = Path(out)
    for d in ("models_val", "models"):
        (out / d).mkdir(parents=True, exist_ok=True)
    F = frozen(json.loads((Path(pick_dir) / "PICK_fit.json").read_text()))
    D = load(state)
    assert int(D["year"].max()) <= S.FIT_YEARS[1], "2024+ rows reached the fit stage"
    D["market_score"] = S.market_score(D, F["mkt"])
    D = D[D["market_score"].notna()].reset_index(drop=True)
    year = D["year"].to_numpy()
    sw = D["swing30"].to_numpy(float)
    tier = S.tier_of(sw, F["tier_cuts"])
    res = {"target": "net / swing30, clipped to +-10", "huber_delta": DELTA, "bar_shares": list(BAR_Q),
           "train": "2019-01..2022-12", "refit": "2019-01..2023-12", "frozen_from_PICK": {
               "market": F["mkt"], "cut_20pct": F["cut"], "tier_cuts_swing30": F["tier_cuts"],
               "widths_per_tier": {v: {str(k): w for k, w in d.items()} for v, d in F["wpt"].items()}},
           "versions": {}}
    for V, feats in S.VERSIONS.items():
        y = target(row_net(D, tier, F["wpt"][V]), sw)
        k = np.isfinite(y)
        tr = k & (year <= S.TRAIN_YEARS[1])
        X = D[feats].to_numpy(np.float32)
        res["versions"][V] = {"rows": {"train": int(tr.sum()), "refit": int(k.sum())}, "learners": {}}
        for L in LEARNERS:
            m = make(L)
            m.fit(X[tr], y[tr])
            p_val = S.save_model(L, m, str(out / "models_val" / f"{V}_{L}"))
            b_val = bars_of(predict(m, X[tr]) * sw[tr])
            m = make(L)
            m.fit(X[k], y[k])
            p_fit = S.save_model(L, m, str(out / "models" / f"{V}_{L}"))
            pn = predict(m, X[k]) * sw[k]
            b_fit = bars_of(pn)
            res["versions"][V]["learners"][L] = {"model_val": p_val, "bars_val": b_val, "model": p_fit,
                                                 "bars": b_fit, "tops": tops_of(pn, b_fit)}
            print(f"[pick2 fit] {V} {L}: fitted", flush=True)
            del m
        del X
        S.atomic_json(res, out / "PICK2_fit.json")
    return res


def stage_oof(state, pick_dir, out):
    """PICK2-OOF: bars and tops from out-of-fold predictions (the models themselves are PICK2_fit.json's)."""
    out = Path(out)
    F = frozen(json.loads((Path(pick_dir) / "PICK_fit.json").read_text()))
    D = load(state)
    assert int(D["year"].max()) <= S.FIT_YEARS[1], "2024+ rows reached the oof stage"
    D["market_score"] = S.market_score(D, F["mkt"])
    D = D[D["market_score"].notna()].reset_index(drop=True)
    year = D["year"].to_numpy()
    sw = D["swing30"].to_numpy(float)
    tier = S.tier_of(sw, F["tier_cuts"])
    val_years = list(range(S.TRAIN_YEARS[0], S.TRAIN_YEARS[1] + 1))
    fit_years = list(range(S.FIT_YEARS[0], S.FIT_YEARS[1] + 1))
    res = {"bars": "top shares of out-of-fold predicted net returns, one fold per year",
           "val_folds": val_years, "refit_folds": fit_years, "versions": {}}
    for V, feats in S.VERSIONS.items():
        y = target(row_net(D, tier, F["wpt"][V]), sw)
        k = np.isfinite(y)
        tr = k & (year <= S.TRAIN_YEARS[1])
        X = D[feats].to_numpy(np.float32)
        res["versions"][V] = {"learners": {}}
        for L in LEARNERS:
            pv = oof_predict(L, X[tr], y[tr], year[tr], val_years) * sw[tr]
            pf = oof_predict(L, X[k], y[k], year[k], fit_years) * sw[k]
            bf = bars_of(pf)
            res["versions"][V]["learners"][L] = {"bars_val": bars_of(pv), "bars": bf, "tops": tops_of(pf, bf)}
            print(f"[pick2 oof] {V} {L}: bars {res['versions'][V]['learners'][L]['bars_val']}", flush=True)
        del X
        S.atomic_json(res, out / "PICK2_oof.json")
    return res


def with_oof(fit, oof):
    """PICK2_fit.json with its bars and tops replaced by PICK2_oof.json's (models unchanged)."""
    out = json.loads(json.dumps(fit))
    for V, vv in oof["versions"].items():
        for L, ll in vv["learners"].items():
            out["versions"][V]["learners"][L].update(ll)
    return out


def read_fit(out, oof):
    fit = json.loads((Path(out) / "PICK2_fit.json").read_text())
    return with_oof(fit, json.loads((Path(out) / "PICK2_oof.json").read_text())) if oof else fit


def stage_diag(fitstate, w1, nx1, pick_dir, out, cal, days=("2023-03-01", "2023-06-01", "2023-09-01")):
    """Data check (2026-11-15, after PICK2-OOF made 0 trades): (1) the validation models' predicted net by year on
    the sampled 2019-2023 training rows, and the share of rows at or above each bar; (2) parity on a few 2023 days:
    the scan path's inputs and predictions against the training rows of the same (symbol, minute)."""
    fit = read_fit(out, False)
    oof = json.loads((Path(out) / "PICK2_oof.json").read_text())
    F = frozen(json.loads((Path(pick_dir) / "PICK_fit.json").read_text()))
    val = Val(fit)
    cols = ["symbol", "t", "track", "d", *S.FIT_NUM]
    parts = []
    for m in pd.period_range("2019-01", "2023-12", freq="M"):
        df = S.read_joined(fitstate, str(m), cols)
        if len(df):
            parts.append(df[df["track"] == TRACK])
    D = pd.concat(parts, ignore_index=True)
    D["d"] = pd.to_datetime(D["d"])
    D["market_score"] = S.market_score(D, F["mkt"])
    D = D[D["market_score"].notna()].reset_index(drop=True)
    yr = D["d"].dt.year.to_numpy()
    val.score(D)
    res = {"by_year": {}, "parity": {}}
    for V, vv in fit["versions"].items():
        for L in vv["learners"]:
            c = col_of(V, L)
            p = D[c].to_numpy(float)
            b_in = float(vv["learners"][L]["bars_val"]["0.001"])
            b_oof = float(oof["versions"][V]["learners"][L]["bars_val"]["0.001"])
            res["by_year"][c] = {str(y): {"n": int((yr == y).sum()),
                                          "q50": float(np.nanquantile(p[yr == y], 0.5)),
                                          "q99": float(np.nanquantile(p[yr == y], 0.99)),
                                          "q999": float(np.nanquantile(p[yr == y], 0.999)),
                                          "max": float(np.nanmax(p[yr == y])),
                                          "share_ge_bar_in": float(np.mean(p[yr == y] >= b_in)),
                                          "share_ge_bar_oof": float(np.mean(p[yr == y] >= b_oof))}
                                 for y in range(2019, 2024)}
            print(f"[pick2 diag] {c}: " + ", ".join(f"{y} q999 {v['q999']:.4f}" for y, v in res["by_year"][c].items()),
                  flush=True)
    spy = pd.read_parquet(Path(nx1) / "nx1" / "spy.parquet")
    sess = np.unique(S.to_ns(spy["date"]))
    sp = S.spy_parts(spy)
    daily = W.read_rows(w1, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", pd.Timestamp("2022-12-01")), ("date", "<=", pd.Timestamp("2023-12-31")),
                         ("c", ">=", 1.0), ("dv20", ">=", 1e6)])
    by_day = {d: g for d, g in daily.groupby("date")}
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    h = alpaca_headers()
    feats = sorted({f for v in S.VERSIONS.values() for f in v})
    for ds in days:
        day = dates[dates >= pd.Timestamp(ds)][0]
        i = dates.get_loc(day)
        swings = {}
        for wd in dates[i - P.SWING_N:i]:
            j = dates.get_loc(wd)
            for sym, b in P.bars_5min(sorted(P.universe(by_day.get(dates[j - 1], daily.iloc[0:0]))) + ["SPY"], wd, h).items():
                s_ = P.swing30(b)
                if np.isfinite(s_):
                    swings[sym] = (swings.get(sym, []) + [s_])[-P.SWING_N:]
        uni = P.universe(by_day.get(dates[i - 1], daily.iloc[0:0]))
        bars = P.bars_5min(sorted(uni) + ["SPY"], day, h)
        g = T.day_inputs(day, uni, bars, swings)
        g = g[g["track"] == TRACK].reset_index(drop=True)
        tone = pd.concat([S.read_tone(nx1, str(x)) for x in (pd.Period(day, "M") - 1, pd.Period(day, "M"))],
                         ignore_index=True)
        g = S.join_rows(g, tone, sess, sp)
        g["market_score"] = S.market_score(g, val.mkt)
        val.score(g)
        tr = D[D["d"] == day]
        m = tr.merge(g, on=["symbol", "t"], suffixes=("_train", "_scan"))
        out_d = {"scan_rows": len(g), "train_rows": len(tr), "matched": len(m), "max_abs_diff": {}}
        for c in feats + [col_of(V, L) for V in S.VERSIONS for L in LEARNERS]:
            a, b = m[f"{c}_train"].to_numpy(float), m[f"{c}_scan"].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            out_d["max_abs_diff"][c] = {"max": float(np.max(np.abs(a[both] - b[both]))) if both.any() else None,
                                        "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum())}
        out_d["scan_q999"] = {col_of(V, L): float(np.nanquantile(g[col_of(V, L)], 0.999))
                              for V in S.VERSIONS for L in LEARNERS}
        res["parity"][str(day.date())] = out_d
        bad = {c: v for c, v in out_d["max_abs_diff"].items() if (v["max"] or 0) > 1e-4 or v["nan_mismatch"]}
        print(f"[pick2 diag] {day.date()}: {len(m)} matched of {len(tr)}; differing: {bad}", flush=True)
    S.atomic_json(res, Path(out) / "PICK2_diag.json")
    return res


# ---------- scan and choice ----------

class Val:
    """The validation models (2019-2022) and their bars."""

    def __init__(self, fit, root="."):
        f = fit["frozen_from_PICK"]
        self.mkt = {k: tuple(v) for k, v in f["market"].items()}
        self.cut, self.tier_cuts = float(f["cut_20pct"]), list(f["tier_cuts_swing30"])
        self.wpt = {v: {int(k): float(w) for k, w in d.items()} for v, d in f["widths_per_tier"].items()}
        self.models, self.bars = {}, {}
        for V, vv in fit["versions"].items():
            for L, ll in vv["learners"].items():
                self.models[(V, L)] = load_model(Path(root) / ll["model_val"])
                self.bars[col_of(V, L)] = {q: float(b) for q, b in ll["bars_val"].items()}

    def score(self, g):
        sw = g["swing30"].to_numpy(float)
        for V, feats in S.VERSIONS.items():
            X = g[feats].to_numpy(np.float32)
            for L in LEARNERS:
                g[col_of(V, L)] = predict(self.models[(V, L)], X) * sw
        return g


def scan_day(D, tone, sess, parts, val, bars):
    g = S.join_rows(D[D["track"] == TRACK].reset_index(drop=True), tone, sess, parts)
    if not len(g):
        return g
    g["market_score"] = S.market_score(g, val.mkt)
    g["bad"] = g["market_score"].to_numpy() < val.cut
    g["tier"] = S.tier_of(g["swing30"].to_numpy(float), val.tier_cuts)
    K = keep_first(val.score(g), val.bars).reset_index(drop=True)
    if not len(K):
        return K
    K = T.add_trades(K, bars, TRACK)
    keep = ["symbol", "d", "t", "tier", "bad", "market_score", "swing30", *val.bars]
    return K[keep + [c for c in K.columns if c.startswith(("net_", "exit_"))]]


def stage_scan(w1, nx1, out, state, deadline, cal, oof=False):
    val = Val(read_fit(out, oof))
    spy = pd.read_parquet(Path(nx1) / "nx1" / "spy.parquet")
    sess = np.unique(S.to_ns(spy["date"]))
    parts = S.spy_parts(spy)
    daily = W.read_rows(w1, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", pd.Timestamp(f"{WARM}-01") - pd.Timedelta(days=10)),
                         ("date", "<=", pd.Period(LAST, "M").end_time), ("c", ">=", 1.0), ("dv20", ">=", 1e6)])
    by_day = {d: g for d, g in daily.groupby("date")}
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    sf = Path(state) / "swings.json"
    swings = json.loads(sf.read_text()) if sf.exists() else {}
    h = alpaca_headers()
    for m in pd.period_range(WARM, LAST, freq="M"):
        f = Path(state) / "rows" / f"{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        tone = None if str(m) == WARM else pd.concat([S.read_tone(nx1, str(x)) for x in (m - 1, m)], ignore_index=True)
        res, n_all = [], 0
        for day in dates[(dates >= m.start_time) & (dates <= m.end_time)]:
            i = dates.get_loc(day)
            if i == 0:
                continue
            uni = P.universe(by_day.get(dates[i - 1], daily.iloc[0:0]))
            bars = P.bars_5min(sorted(uni) + ["SPY"], day, h)
            if tone is not None:
                D = T.day_inputs(day, uni, bars, swings)
                n_all += len(D)
                if len(D):
                    K = scan_day(D, tone, sess, parts, val, bars)
                    if len(K):
                        res.append(K)
            for sym, b in bars.items():
                s = P.swing30(b)
                if np.isfinite(s):
                    swings[sym] = (swings.get(sym, []) + [s])[-P.SWING_N:]
        S.atomic(pd.concat(res, ignore_index=True) if res else pd.DataFrame(), f)
        tmp = sf.with_suffix(".tmp")
        tmp.write_text(json.dumps(swings))
        tmp.replace(sf)
        print(f"[pick2 scan] {m}: {n_all} decision rows, {sum(len(x) for x in res)} kept", flush=True)
    return True


def stage_choose(out, state, oof=False):
    fit = read_fit(out, oof)
    files = [Path(state) / "rows" / f"{m}.parquet" for m in pd.period_range(FIRST, LAST, freq="M")]
    R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    R["d"] = pd.to_datetime(R["d"])
    res = {"validate": f"{FIRST}..{LAST} (every decision row, validation models)", "rows_kept": len(R),
           "min_trades": MIN_TRADES, "max_buys_a_day": MAX_BUYS, "versions": {}}
    for V, vv in fit["versions"].items():
        wpt = {int(k): float(w) for k, w in fit["frozen_from_PICK"]["widths_per_tier"][V].items()}
        table, books = {}, {}
        for L, ll in vv["learners"].items():
            table[L] = {}
            for q, thr in ll["bars_val"].items():
                books[(L, q)] = book(R, col_of(V, L), float(thr), wpt)
                table[L][q] = summary(books[(L, q)])
        ch = choose(table)
        res["versions"][V] = {"table_2023": table, "chosen": ch}
        if ch:
            tk = books[(ch["learner"], ch["q"])]
            ll = vv["learners"][ch["learner"]]
            res["versions"][V]["forward"] = {"model": ll["model"], "bar": ll["bars"][ch["q"]],
                                             "top": ll["tops"][ch["q"]]}
            res["versions"][V]["trades_2023"] = [
                {"d": str(r.d.date()), "t": f"{pd.Timestamp(r.t):%H:%M}", "symbol": r.symbol, "net": float(r.net),
                 "exit": r.exit} for r in tk.itertuples(index=False)]
        print(f"[pick2 choose] {V}: {ch}", flush=True)
    c = res["versions"]["COMBINED"]["chosen"]
    res["gate"] = {"decides": "COMBINED", "rule": "2023 book mean net per trade > 0 with at least 100 trades",
                   "verdict": "start the forward book" if c and c["mean_net"] > 0 else "failed at validation"}
    if oof:
        res["bars"] = "out-of-fold (PICK2-OOF)"
    S.atomic_json(res, Path(out) / ("PICK2_choose_oof.json" if oof else "PICK2_choose.json"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "scan", "oof", "scan-oof", "diag"])
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--nx1", default="cache/nx1")
    ap.add_argument("--fitstate", default="cache/pickfit")
    ap.add_argument("--pick", default="research/results/PICK")
    ap.add_argument("--out", default="research/results/PICK2")
    ap.add_argument("--state", default="cache/pick2")
    ap.add_argument("--budget-min", type=float, default=320)
    a = ap.parse_args()
    if a.stage == "fit":
        stage_fit(a.fitstate, a.pick, a.out)
        print("[pick2 fit] done", flush=True)
        return
    if a.stage == "oof":
        stage_oof(a.fitstate, a.pick, a.out)
        print("[pick2 oof] done", flush=True)
        return
    oof = a.stage == "scan-oof"
    cal = alpaca.trading_calendar(pd.Timestamp(f"{WARM}-01") - pd.Timedelta(days=10), pd.Timestamp("2024-01-01"))
    if a.stage == "diag":
        stage_diag(a.fitstate, a.w1, a.nx1, a.pick, a.out, cal)
        print("[pick2 diag] done", flush=True)
        return
    if stage_scan(a.w1, a.nx1, a.out, a.state, time.monotonic() + 60 * a.budget_min, cal, oof):
        stage_choose(a.out, a.state, oof)
        print("[pick2 scan] done", flush=True)
    else:
        print("[pick2 scan] time budget used; rerun to continue", flush=True)


if __name__ == "__main__":
    main()
