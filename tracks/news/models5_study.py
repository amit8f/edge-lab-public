"""VOL1 and REGIME1 (research/prereg/VOL1.md, REGIME1.md): NX3-ALL-HI's history re-sized by a GARCH volatility
forecast, cut in a hidden-Markov "storm" regime, SURV1's survival-model exits (research/prereg/SURV1.md) and QRANGE1's
quantile-model take profit and stop (research/prereg/QRANGE1.md; qpaths fetches each pick's minute path first).

    python -m tracks.news.models5_study --stage picks|vol1|regime1|surv1|qpaths|qrange1 --cache cache/nx1 --out research/results/MODELS5

picks: NX3-ALL-HI's walk-forward picks (same models and seeds as NX3-ALL, confidence >= 1.5, uncapped, sized by
confidence) with each trade's net after costs -> cache nx3/models5/picks.parquet. Choose years 2019-2023, one
look 2024-01..2025-09. Writes summaries only.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

HI = 1.5
CHOOSE = (2019, 2020, 2021, 2022, 2023)
TEST = (2024, 2025)
GARCH_N, GARCH_MIN = 500, 250
CLIP = (0.25, 2.0)
HMM_FIRST, HMM_LAST, HMM_SEED = "2016-01-01", "2022-12-31", 7
STORM_P, STORM_GROSS, SIDE_CAP = 0.5, 0.5, 0.6
BOOT, SEED = 2000, 20261124


def mdir(cache):
    return X.nx3dir(cache) / "models5"


# ---------- pure helpers (tested) ----------

def day_return(w, net):
    """Book return of one day: sum of size x net (sizes are fractions of the book)."""
    w, net = np.asarray(w, float), np.asarray(net, float)
    return float(w @ net) if len(w) else 0.0


def vol_weights(base, s, target, gross=None):
    """VOL1: base size x clip(target / s); rescaled so the day's gross equals the base gross. A missing s keeps
    the base size."""
    base, s = np.asarray(base, float), np.asarray(s, float)
    k = np.where(np.isfinite(s) & (s > 0), np.clip(target / np.where(s > 0, s, 1.0), *CLIP), 1.0)
    w = base * k
    g = base.sum() if gross is None else gross
    return w * (g / w.sum()) if w.sum() > 0 else w


def storm_weights(w, side, conf, storm):
    """REGIME1: in a storm, keep the heavier side's highest-confidence trades until it is <= SIDE_CAP of the kept
    gross, then scale everything to STORM_GROSS of the day's plain gross."""
    w, side, conf = np.asarray(w, float).copy(), np.asarray(side, int), np.asarray(conf, float)
    if not storm or w.sum() <= 0:
        return w
    gross = w.sum()
    heavy = 1 if w[side == 1].sum() >= w[side == -1].sum() else -1
    light = w[side != heavy].sum()
    idx = np.where(side == heavy)[0]
    idx = idx[np.argsort(-conf[idx], kind="stable")]
    cap = SIDE_CAP / (1 - SIDE_CAP) * light if light > 0 else None
    kept = 0.0
    for i in idx:
        if cap is not None and kept + w[i] > cap + 1e-12:
            w[i] = max(cap - kept, 0.0)
        kept += w[i]
    return w * (STORM_GROSS * gross / w.sum()) if w.sum() > 0 else w


def sharpe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def boot_sharpe_diff(a, b, month, n=BOOT, seed=SEED):
    """Month-block bootstrap of sharpe(a) - sharpe(b) on paired daily series. Returns (diff, lo, hi)."""
    a, b, month = np.asarray(a, float), np.asarray(b, float), np.asarray(month)
    ms = np.unique(month)
    by = {m: np.where(month == m)[0] for m in ms}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = np.concatenate([by[m] for m in rng.choice(ms, size=len(ms), replace=True)])
        d.append(sharpe(a[idx]) - sharpe(b[idx]))
    d = np.asarray(d)
    return sharpe(a) - sharpe(b), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def forward_filter(model, Xs):
    """Filtered (not smoothed) state probabilities of a fitted hmmlearn GaussianHMM: row i uses rows <= i only."""
    from scipy.stats import multivariate_normal
    k = model.n_components
    lik = np.column_stack([multivariate_normal(model.means_[j], model.covars_[j], allow_singular=True).pdf(Xs)
                           for j in range(k)])
    out = np.zeros((len(Xs), k))
    p = model.startprob_.copy()
    for i in range(len(Xs)):
        prior = p if i == 0 else out[i - 1] @ model.transmat_
        post = prior * lik[i]
        out[i] = post / post.sum() if post.sum() > 0 else prior
    return out


def garch_sigma(r):
    """One-day-ahead volatility from GARCH(1,1) with Student-t errors on daily log returns r (fractions)."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(np.asarray(r, float) * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        if res.convergence_flag != 0:
            return float("nan")
        return float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / 100)


def tstat(x):
    return X.tstat(np.asarray(x, float))


SURV_TARGET, SURV_MAX, SURV_GRID = 0.03, 5, (0.3, 0.4, 0.5)
SURV_LGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=7, verbose=-1)
SURV_INPUTS = ["j", "r_so_far", "conf", "side", "penny", "atr", "spy_so_far", "dow"]


def surv_path(ret1, ac_after):
    """Signed-direction-agnostic close path of a trade: close 1 is NX3's exit (return ret1, gross, unsigned);
    later closes scale it by the all-adjusted close ratio. ac_after: closes at sessions 1..n (session 1 = next)."""
    ac = np.asarray(ac_after, float)
    return (1 + ret1) * ac / ac[0] - 1


def surv_exit(sig, probs, pstar, target=SURV_TARGET):
    """Close index (0-based) at which a trade exits. sig: signed returns at closes 0..n-1; probs[k]: model's
    chance of reaching the target at a later close, known at close k. Exit at the first close where the target
    is reached, or where probs[k] < pstar, or at the last close."""
    n = len(sig)
    for k in range(n):
        if sig[k] >= target or k == n - 1 or not probs[k] >= pstar:
            return k
    return n - 1


# ---------- stages ----------

def stage_picks(cache):
    d3 = X.nx3dir(cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    rows = []
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr, te = ev[ev["d"] <= cut_d], ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = X.Model(tr)
        real = te[te["kind"] == "real"]
        e = model.effect(real[X.INPUTS].to_numpy(float))
        real = real.assign(e=e, fired=model.fires(e))
        rng = np.random.default_rng(X.SEED + yr)
        for d, day in real.groupby("d", sort=True):
            p, _ = X.picks(day, rng, cap=None, hi=HI, cut=model.cut)
            if p.empty:
                continue
            conf = np.abs(p["e"].to_numpy(float)) / model.cut
            rows.append(p[["symbol", "d", "t", "next", "side", "price", "days", "ret"]].assign(
                conf=conf, w=X.conf_sizes(conf), net=N.net(p).to_numpy(float), year=yr))
    out = pd.concat(rows, ignore_index=True)
    N.atomic_parquet(out, mdir(cache) / "picks.parquet")
    print(f"[picks] {len(out)} NX3-ALL-HI trades on {out['d'].nunique()} days, {out['year'].min()}-{out['year'].max()}")


def chart_panel(cache, symbols):
    d = X.nx3dir(cache) / X.CHART
    parts = []
    for f in sorted(d.glob("c_*.parquet")):
        x = pd.read_parquet(f, columns=["symbol", "date", "seg", "ac", "atr14"])
        parts.append(x[x["symbol"].isin(symbols)])
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"])


def risk_inputs(picks, panel):
    """Per pick: ATR(14)/price and GARCH sigma from the stock's own daily closes before the entry day (latest
    segment only, last GARCH_N returns, at least GARCH_MIN)."""
    out = np.full((len(picks), 2), np.nan)
    by = {s: g for s, g in panel.groupby("symbol")}
    for i, (s, d) in enumerate(zip(picks["symbol"], picks["d"])):
        g = by.get(s)
        if g is None:
            continue
        g = g[g["date"] < d]
        if g.empty:
            continue
        out[i, 0] = g["atr14"].iloc[-1]
        g = g[g["seg"] == g["seg"].iloc[-1]]
        r = np.diff(np.log(g["ac"].to_numpy(float)))[-GARCH_N:]
        r = r[np.isfinite(r)]
        if len(r) >= GARCH_MIN:
            out[i, 1] = garch_sigma(r)
    return out[:, 0], out[:, 1]


def daily_books(p, cols):
    rows = []
    for d, g in p.groupby("d", sort=True):
        rows.append({"d": d, **{k: day_return(g[k], g["net"]) for k in cols}})
    df = pd.DataFrame(rows)
    df["month"] = df["d"].dt.to_period("M").astype(str)
    df["year"] = df["d"].dt.year
    return df


def stage_vol1(cache, out):
    p = pd.read_parquet(mdir(cache) / "picks.parquet")
    f = mdir(cache) / "vol1_inputs.parquet"
    if f.exists():
        p = p.merge(pd.read_parquet(f), on=["symbol", "d", "t"], how="left")
    else:
        atr, sg = risk_inputs(p, chart_panel(cache, set(p["symbol"])))
        p = p.assign(atr=atr, garch=sg)
        N.atomic_parquet(p[["symbol", "d", "t", "atr", "garch"]], f)
    choose = p["year"].isin(CHOOSE)
    p["s_garch"] = p["garch"].where(np.isfinite(p["garch"]), p["atr"])
    tg, ta = float(p.loc[choose, "s_garch"].median()), float(p.loc[choose, "atr"].median())
    p["w_vol"], p["w_atr"] = np.nan, np.nan
    for d, g in p.groupby("d"):
        p.loc[g.index, "w_vol"] = vol_weights(g["w"], g["s_garch"], tg)
        p.loc[g.index, "w_atr"] = vol_weights(g["w"], g["atr"], ta)
    days = daily_books(p, ["w", "w_vol", "w_atr"])
    test = days[days["year"].isin(TEST)]
    gross = p.groupby("d")["w"].sum().reindex(test["d"]).to_numpy(float)
    vb = boot_sharpe_diff(test["w_vol"], test["w"], test["month"])
    va = boot_sharpe_diff(test["w_vol"], test["w_atr"], test["month"], seed=SEED + 1)
    own = tstat(test["w_vol"] / gross)
    rep = {"rule": "VOL1 passes if Sharpe(VOL1)-Sharpe(plain) > 0 with month-bootstrap 95% CI > 0, Sharpe(VOL1)-Sharpe(ATR) > 0, "
                   "and VOL1's own mean daily net per dollar of gross > 0 (2024-01..2025-09)",
           "trades": int(len(p)), "test_trades": int(p["year"].isin(TEST).sum()), "test_days": int(len(test)),
           "garch_missing_share": float((~np.isfinite(p["garch"])).mean()),
           "clip_bind_share": float(((p["w_vol"] / p["w"]).isin([CLIP[0], CLIP[1]])).mean()),
           "targets": {"garch": tg, "atr": ta},
           "test": {"sharpe": {k: sharpe(test[k]) for k in ("w", "w_vol", "w_atr")},
                    "vol_minus_plain": dict(zip(("diff", "lo", "hi"), vb)),
                    "vol_minus_atr": dict(zip(("diff", "lo", "hi"), va)),
                    "own_net_per_gross": own,
                    "worst_1pct_day": {k: float(np.percentile(test[k], 1)) for k in ("w", "w_vol", "w_atr")}},
           "per_year_sharpe": {int(y): {k: sharpe(g[k]) for k in ("w", "w_vol", "w_atr")} for y, g in days.groupby("year")},
           "per_side_test_net": {int(s): float((g["w_vol"] * g["net"]).sum() / g["w_vol"].sum())
                                 for s, g in p[p["year"].isin(TEST)].groupby("side")}}
    rep["verdict"] = "passed" if (vb[0] > 0 and vb[1] > 0 and va[0] > 0 and own["mean"] > 0) else "failed"
    write(out, "VOL1_result.json", rep)


def spy_states(cache):
    spy = pd.read_parquet(X.nx3dir(cache) / X.CHART / "spy.parquet")[["date", "ac"]].sort_values("date")
    spy["r"] = np.log(spy["ac"]).diff()
    spy["rv20"] = spy["r"].rolling(20).std() * np.sqrt(252)
    spy = spy.dropna().reset_index(drop=True)
    return spy


def stage_regime1(cache, out):
    from hmmlearn.hmm import GaussianHMM
    p = pd.read_parquet(mdir(cache) / "picks.parquet")
    spy = spy_states(cache)
    fit = spy[(spy["date"] >= HMM_FIRST) & (spy["date"] <= HMM_LAST)]
    Xf = fit[["r", "rv20"]].to_numpy(float)
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=HMM_SEED).fit(Xf)
    storm_state = int(np.argmax(hmm.means_[:, 1]))
    probs = forward_filter(hmm, spy[["r", "rv20"]].to_numpy(float))[:, storm_state]
    # the decision on day d uses the state filtered through the PREVIOUS close
    spy["p_storm_next"] = probs
    thr = float(fit["rv20"].quantile(0.8))
    spy["vol_storm_next"] = spy["rv20"] > thr
    dates = spy["date"].to_numpy()
    prev = {d: np.searchsorted(dates, np.datetime64(d)) - 1 for d in p["d"].unique()}
    p["storm"] = [bool(spy["p_storm_next"].iloc[prev[d]] > STORM_P) if prev[d] >= 0 else False for d in p["d"]]
    p["vstorm"] = [bool(spy["vol_storm_next"].iloc[prev[d]]) if prev[d] >= 0 else False for d in p["d"]]
    p["w_reg"], p["w_vsw"] = np.nan, np.nan
    for d, g in p.groupby("d"):
        p.loc[g.index, "w_reg"] = storm_weights(g["w"], g["side"], g["conf"], bool(g["storm"].iloc[0]))
        p.loc[g.index, "w_vsw"] = storm_weights(g["w"], g["side"], g["conf"], bool(g["vstorm"].iloc[0]))
    days = daily_books(p, ["w", "w_reg", "w_vsw"])
    test = days[days["year"].isin(TEST)]
    gross = p.groupby("d")["w"].sum().reindex(test["d"]).to_numpy(float)
    diff, diff_v = (test["w_reg"] - test["w"]) / gross, (test["w_reg"] - test["w_vsw"]) / gross
    w1 = {k: float(np.percentile(test[k] / gross, 1)) for k in ("w", "w_reg", "w_vsw")}
    gd, gv = tstat(diff), tstat(diff_v)
    sd = p.drop_duplicates("d").set_index("d")
    rep = {"rule": "REGIME1 passes if its worst-1% day (per dollar of plain gross) is smaller than plain NX3-ALL-HI's AND the "
                   "paired daily gain is >= 0 with NW t >= 1.5 AND the gain vs a plain vol-percentile switch is >= 0 (2024-01..2025-09)",
           "hmm": {"means": hmm.means_.tolist(), "storm_state": storm_state, "transmat": hmm.transmat_.tolist(),
                   "fit_days": int(len(fit))},
           "storm_share": {"hmm_test": float(sd.loc[sd["year"].isin(TEST), "storm"].mean()),
                           "volswitch_test": float(sd.loc[sd["year"].isin(TEST), "vstorm"].mean()),
                           "hmm_all": float(sd["storm"].mean())},
           "test": {"worst_1pct_day": w1, "gain_vs_plain": gd, "gain_vs_volswitch": gv,
                    "own_net_per_gross": {k: tstat(test[k] / gross) for k in ("w", "w_reg", "w_vsw")}},
           "per_year_gain": {int(y): tstat((g["w_reg"] - g["w"]).to_numpy(float)) for y, g in days.groupby("year")}}
    rep["verdict"] = ("passed" if w1["w_reg"] > w1["w"] and gd["mean"] >= 0 and gd["t_nw"] >= 1.5 and gv["mean"] >= 0
                      else "failed")
    write(out, "REGIME1_result.json", rep)


def surv_rows(p, panel, spy):
    """Per pick: its closes for sessions 1..SURV_MAX after entry (session 1 = NX3's exit close), from the
    all-adjusted panel; trades with a gap in those sessions are dropped."""
    by = {s: g.set_index("date") for s, g in panel.groupby("symbol")}
    sp = spy.set_index("date")["ac"]
    cal = sp.index
    paths, spaths, keep = [], [], []
    for i, r in enumerate(p.itertuples(index=False)):
        g = by.get(r.symbol)
        k = cal.searchsorted(r.next)
        if g is None or k + SURV_MAX > len(cal) or cal[k] != r.next:
            continue
        ds = cal[k:k + SURV_MAX]
        if not ds.isin(g.index).all() or ds[-1] > N.LAST_EXIT:
            continue
        ac = g.loc[ds, "ac"].to_numpy(float)
        if not np.isfinite(ac).all() or g.loc[ds, "seg"].nunique() != 1:
            continue
        paths.append(r.side * surv_path(r.ret, ac))
        spaths.append(sp.loc[ds].to_numpy(float) / sp.iloc[max(k - 1, 0)] - 1)
        keep.append(i)
    return np.array(keep), np.vstack(paths), np.vstack(spaths), [cal[cal.searchsorted(n_):][:SURV_MAX] for n_ in p["next"].iloc[keep]]


def hazard_table(p, paths, spaths):
    """Discrete-time hazard rows: at close j (1..SURV_MAX-1, 1-based) for trades not yet at the target, label = the
    target is reached at close j+1."""
    rows = []
    for i in range(len(p)):
        r = p.iloc[i]
        for j in range(1, SURV_MAX):
            if paths[i, j - 1] >= SURV_TARGET:
                break
            rows.append({"i": i, "j": j, "r_so_far": paths[i, j - 1], "conf": r.conf, "side": r.side,
                         "penny": float(r.price < 5), "atr": r.atr, "spy_so_far": r.side * spaths[i, j - 1],
                         "dow": float(pd.Timestamp(r.d).dayofweek), "y": int(paths[i, j] >= SURV_TARGET)})
    return pd.DataFrame(rows)


def surv_probs(model, p, paths, spaths):
    """probs[i, k]: chance of reaching the target at a later close, known at close k (0-based), with the hazard
    model's inputs held at close k for the remaining days (only j changes)."""
    out = np.zeros((len(p), SURV_MAX))
    for k in range(SURV_MAX - 1):
        surv = np.ones(len(p))
        for j in range(k + 1, SURV_MAX):
            Xk = pd.DataFrame({"j": float(j), "r_so_far": paths[:, k], "conf": p["conf"].to_numpy(float),
                               "side": p["side"].to_numpy(float), "penny": (p["price"] < 5).to_numpy(float),
                               "atr": p["atr"].to_numpy(float), "spy_so_far": p["side"].to_numpy(float) * spaths[:, k],
                               "dow": p["d"].dt.dayofweek.to_numpy(float)})[SURV_INPUTS]
            surv *= 1 - model.predict_proba(Xk)[:, 1]
        out[:, k] = 1 - surv
    return out


def surv_net(p, paths, dates, probs, pstar):
    net = np.zeros(len(p))
    held = np.zeros(len(p), int)
    for i in range(len(p)):
        k = surv_exit(paths[i], probs[i], pstar)
        days = (dates[i][k] - p["d"].iloc[i]).days
        net[i] = paths[i, k] - N.cost_frac([p["price"].iloc[i]], [p["side"].iloc[i] < 0], [days])[0]
        held[i] = k + 1
    return net, held


def stage_surv1(cache, out):
    import lightgbm as lgb
    p = pd.read_parquet(mdir(cache) / "picks.parquet")
    f = mdir(cache) / "vol1_inputs.parquet"
    p = p.merge(pd.read_parquet(f)[["symbol", "d", "t", "atr"]], on=["symbol", "d", "t"], how="left")
    panel = chart_panel(cache, set(p["symbol"]))
    spy = pd.read_parquet(X.nx3dir(cache) / X.CHART / "spy.parquet")[["date", "ac"]].sort_values("date")
    keep, paths, spaths, dates = surv_rows(p, panel, spy)
    p = p.iloc[keep].reset_index(drop=True)
    tr = p["year"] <= 2022
    ht = hazard_table(p[tr].reset_index(drop=True), paths[tr.to_numpy()], spaths[tr.to_numpy()])
    model = lgb.LGBMClassifier(**SURV_LGB).fit(ht[SURV_INPUTS], ht["y"])
    probs = surv_probs(model, p, paths, spaths)
    base = paths[:, 0] - N.cost_frac(p["price"], p["side"] < 0, p["days"])
    sel = (p["year"] == 2023).to_numpy()
    choose = {}
    for ps in SURV_GRID:
        n, h = surv_net(p[sel].reset_index(drop=True), paths[sel], [dates[i] for i in np.where(sel)[0]], probs[sel], ps)
        choose[str(ps)] = {"net_per_trade": float(n.mean()), "held_past_day1": float((h > 1).mean())}
    pstar = float(max(SURV_GRID, key=lambda ps: choose[str(ps)]["net_per_trade"]))
    te = p["year"].isin(TEST).to_numpy()
    n, h = surv_net(p[te].reset_index(drop=True), paths[te], [dates[i] for i in np.where(te)[0]], probs[te], pstar)
    diff = pd.DataFrame({"d": p.loc[te, "d"].to_numpy(), "x": n - base[te]}).groupby("d")["x"].mean()
    own = pd.DataFrame({"d": p.loc[te, "d"].to_numpy(), "x": n}).groupby("d")["x"].mean()
    gd, go = tstat(diff.to_numpy()), tstat(own.to_numpy())
    rep = {"rule": "SURV1 passes if net per trade minus next-close NX3-ALL-HI on the same trades is > 0 with day-clustered t >= 2 "
                   "and 300+ trades, and its own net per trade > 0 (2024-01..2025-09)",
           "trades_with_5_day_path": int(len(p)), "hazard_rows_train": int(len(ht)), "hazard_rate_train": float(ht["y"].mean()),
           "choose_2023": choose, "pstar": pstar,
           "test": {"trades": int(te.sum()), "gain_vs_next_close_by_day": gd, "own_net_by_day": go,
                    "mean_net_per_trade": float(n.mean()), "next_close_net_per_trade": float(base[te].mean()),
                    "held_past_day1": float((h > 1).mean()), "mean_days_held": float(h.mean()),
                    "per_side": {int(s): float(n[(p.loc[te, "side"] == s).to_numpy()].mean()) for s in (1, -1)}},
           "km_reach_target_by_close": [float((paths[:, :k + 1].max(1) >= SURV_TARGET).mean()) for k in range(SURV_MAX)]}
    rep["verdict"] = "passed" if (gd["mean"] > 0 and gd["t_nw"] >= 2 and te.sum() >= 300 and n.mean() > 0) else "failed"
    write(out, "SURV1_result.json", rep)


def stage_surv1_export(cache, out_dir="models/surv1"):
    """SURV1-LONG (research/prereg/SURV1LONG.md): the SURV1 hazard model refit exactly as in stage_surv1 (trades
    with a 5-close path, 2019-2022) and saved as LightGBM text with its inputs and p* = 0.3."""
    import lightgbm as lgb
    p = pd.read_parquet(mdir(cache) / "picks.parquet")
    p = p.merge(pd.read_parquet(mdir(cache) / "vol1_inputs.parquet")[["symbol", "d", "t", "atr"]],
                on=["symbol", "d", "t"], how="left")
    panel = chart_panel(cache, set(p["symbol"]))
    spy = pd.read_parquet(X.nx3dir(cache) / X.CHART / "spy.parquet")[["date", "ac"]].sort_values("date")
    keep, paths, spaths, _ = surv_rows(p, panel, spy)
    p = p.iloc[keep].reset_index(drop=True)
    tr = p["year"] <= 2022
    ht = hazard_table(p[tr].reset_index(drop=True), paths[tr.to_numpy()], spaths[tr.to_numpy()])
    model = lgb.LGBMClassifier(**SURV_LGB).fit(ht[SURV_INPUTS], ht["y"])
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(o / "hazard.txt"))
    (o / "meta.json").write_text(json.dumps({"inputs": SURV_INPUTS, "pstar": 0.3, "target": SURV_TARGET,
                                             "max_closes": SURV_MAX, "train_rows": int(len(ht)),
                                             "train_rate": float(ht["y"].mean())}, indent=1))
    print(f"[surv1 export] {len(ht)} hazard rows -> {o}")


# ---------- QRANGE1 (research/prereg/QRANGE1.md) ----------

Q_TP_GRID, Q_SL = (0.5, 0.7), 0.9
Q_LGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=7, verbose=-1)
Q_INPUTS = ["atr", "vol20", "penny", "side", "conf", "gap", "spy_open"]
Q_FLOOR = 0.001          # a predicted target or stop below 0.1% is set to 0.1%
QPATHS = "qpaths"


def q_side_path(o, h, l, entry, side):
    """Minute bars -> per bar the open's signed return (fo), the best signed excursion (fh) and the worst
    adverse excursion as a positive number (fa), all from the entry price in the trade's direction."""
    o, h, l = (np.asarray(x, float) / entry - 1 for x in (o, h, l))
    if side > 0:
        return o, h, -l
    return -o, -l, h


def q_exit(fo, fh, fa, tp, sl, ret_end):
    """Signed return and bar index of a trade with take profit tp and stop sl (both > 0). A bar that opens past
    either level fills at its open (a gap); a bar touching both counts as the stop; else the logged next-close
    return (index -1)."""
    for k in range(len(fo)):
        if -fo[k] >= sl or fo[k] >= tp:
            return float(fo[k]), k
        if fa[k] >= sl:
            return -sl, k
        if fh[k] >= tp:
            return tp, k
    return float(ret_end), -1


def minute_hl(symbols, open_t, close_t, headers):
    """{symbol: 1-minute SIP bars (t, o, h, l, c), adjustment=raw} for one session; zero-volume bars dropped."""
    s = pd.Timestamp(open_t, tz=X.ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = pd.Timestamp(close_t, tz=X.ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for i in range(0, len(symbols), X.SYM_CHUNK):
        tok = None
        while True:
            q = {"symbols": ",".join(symbols[i:i + X.SYM_CHUNK]), "timeframe": "1Min", "start": s, "end": e,
                 "feed": "sip", "adjustment": "raw", "limit": 10000}
            if tok:
                q["page_token"] = tok
            j = X.alpaca.get(X.BARS_URL, q, headers=headers)
            for sym, bs in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bs if b["v"] > 0)
            tok = j.get("next_page_token")
            if not tok:
                break
    res = {}
    for sym, rows in out.items():
        d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(X.ET).dt.tz_localize(None)
        res[sym] = d.sort_values("t").reset_index(drop=True)
    return res


def q_trades(cache):
    """picks joined with their logged entry (price, time, gap, SPY move at entry, vol20)."""
    d3 = X.nx3dir(cache)
    p = pd.read_parquet(mdir(cache) / "picks.parquet")
    cols = ["symbol", "d", "t", "entry", "entry_t", "prev_ac", "exit_ac", "spy_open", "vol20"]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((d3 / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True).drop_duplicates(["symbol", "d", "t"])
    p = p.merge(ev, on=["symbol", "d", "t"], how="left")
    p["gap"] = p["entry"] / p["prev_ac"] - 1
    return p


def stage_qpaths(cache, deadline):
    """Per pick, the minute path from entry to the next close (day d after the entry, then the next session in
    day d's raw prices), one file per month of entries."""
    import time
    p = q_trades(cache)
    ses = X.sessions(X.calendar(cache)).set_index("date")
    h = X.alpaca_headers()
    out = mdir(cache) / QPATHS
    for m, gm in p.groupby(p["d"].dt.to_period("M")):
        f = out / f"q_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        syms = sorted(set(gm["symbol"]))
        dc = X.daily_closes(syms, gm["d"].min(), gm["next"].max(), h)
        rows = []
        for d, g in gm.groupby("d"):
            s0, s1 = ses.loc[d], ses.loc[g["next"].iloc[0]]
            b0 = minute_hl(sorted(set(g["symbol"])), s0["open_t"], s0["close_t"], h)
            b1 = minute_hl(sorted(set(g["symbol"])), s1["open_t"], s1["close_t"], h)
            for r in g.itertuples(index=False):
                x0, x1 = b0.get(r.symbol), b1.get(r.symbol)
                try:
                    raw_d, all_d = dc[(r.symbol, d)]
                    raw_n, all_n = dc[(r.symbol, r.next)]
                except KeyError:
                    continue
                if x0 is None or x1 is None or not (raw_d > 0 and raw_n > 0 and all_d > 0) or not r.entry > 0:
                    continue
                x0 = x0[x0["t"] >= r.entry_t]
                k = (all_n / raw_n) / (all_d / raw_d)
                o = np.r_[x0["o"], x1["o"] * k]
                hh = np.r_[x0["h"], x1["h"] * k]
                ll = np.r_[x0["l"], x1["l"] * k]
                if not len(x0) or not np.isfinite(o).all():
                    continue
                fo, fh, fa = q_side_path(o, hh, ll, r.entry, r.side)
                rows.append({"symbol": r.symbol, "d": d, "t": r.t, "n_day0": len(x0),
                             "fo": fo.astype("float32"), "fh": fh.astype("float32"), "fa": fa.astype("float32"),
                             "end": r.side * (x1["c"].iloc[-1] * k / r.entry - 1)})
        N.atomic_parquet(pd.DataFrame(rows), f)
        print(f"[qpaths] {m}: {len(gm)} picks, {len(rows)} paths")
    return True


def q_net(p, tp, sl):
    """Net per trade under the bracket (tp, sl arrays), and which exit fired (1 target, -1 stop, 0 next close)."""
    net, hit = np.zeros(len(p)), np.zeros(len(p), int)
    for i, r in enumerate(p.itertuples(index=False)):
        g, k = q_exit(r.fo, r.fh, r.fa, tp[i], sl[i], r.side * r.ret)
        days = 1 if 0 <= k < r.n_day0 else r.days
        net[i] = g - N.cost_frac([r.price], [r.side < 0], [days])[0]
        hit[i] = 0 if k < 0 else (1 if g > 0 else -1)
    return net, hit


def stage_qrange1(cache, out):
    import lightgbm as lgb
    p = q_trades(cache)
    p = p.merge(pd.read_parquet(mdir(cache) / "vol1_inputs.parquet")[["symbol", "d", "t", "atr"]],
                on=["symbol", "d", "t"], how="left")
    files = sorted((mdir(cache) / QPATHS).glob("q_*.parquet"))
    missing = set(p["d"].dt.to_period("M").astype(str)) - {f.stem[2:] for f in files}
    if missing:
        raise SystemExit(f"qpaths unfinished: {len(missing)} months missing; run --stage qpaths again")
    qp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_picks = len(p)
    p = p.merge(qp, on=["symbol", "d", "t"], how="inner").reset_index(drop=True)
    p["penny"] = (p["price"] < 5).astype(float)
    p["F"] = [float(np.max(x)) for x in p["fh"]]
    p["A"] = [float(np.max(x)) for x in p["fa"]]
    end_err = (p["end"] - p["side"] * p["ret"]).abs()
    tr = (p["year"] <= 2022).to_numpy()
    Xs = p[Q_INPUTS].astype(float)
    pred = {}
    for name, col, q in [("F0.5", "F", 0.5), ("F0.7", "F", 0.7), ("A0.9", "A", Q_SL)]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **Q_LGB).fit(Xs[tr], p.loc[tr, col])
        pred[name] = np.maximum(m.predict(Xs), Q_FLOOR)
    sl = pred["A0.9"]
    sel = (p["year"] == 2023).to_numpy()
    choose, calib = {}, {"stop_A0.9": float((p.loc[sel, "A"] > sl[sel]).mean())}
    for q in Q_TP_GRID:
        n, h = q_net(p[sel], pred[f"F{q}"][sel], sl[sel])
        choose[str(q)] = {"net_per_trade": float(n.mean()), "target_hit": float((h == 1).mean()),
                          "stop_hit": float((h == -1).mean())}
        calib[f"target_F{q}"] = float((p.loc[sel, "F"] > pred[f"F{q}"][sel]).mean())
    qtp = float(max(Q_TP_GRID, key=lambda q: choose[str(q)]["net_per_trade"]))
    te = p["year"].isin(TEST).to_numpy()
    pt = p[te].reset_index(drop=True)
    n, h = q_net(pt, pred[f"F{qtp}"][te], sl[te])
    base = pt["net"].to_numpy(float)
    diff = pd.DataFrame({"d": pt["d"], "x": n - base}).groupby("d")["x"].mean()
    own = pd.DataFrame({"d": pt["d"], "x": n}).groupby("d")["x"].mean()
    gd, go = tstat(diff.to_numpy()), tstat(own.to_numpy())
    rep = {"rule": "QRANGE1 passes if net per trade minus next-close NX3-ALL-HI on the same trades is > 0 with day-clustered "
                   "t >= 2 and 300+ trades, and its own net per trade > 0 (2024-01..2025-09)",
           "picks": int(n_picks), "with_minute_path": int(len(p)),
           "qa_path_end_vs_logged_ret": {"median_abs": float(end_err.median()), "share_over_1pct": float((end_err > 0.01).mean())},
           "choose_2023": choose, "q_tp": qtp, "calibration_2023_share_beyond": calib,
           "test": {"trades": int(te.sum()), "gain_vs_next_close_by_day": gd, "own_net_by_day": go,
                    "mean_net_per_trade": float(n.mean()), "next_close_net_per_trade": float(base.mean()),
                    "target_hit": float((h == 1).mean()), "stop_hit": float((h == -1).mean()),
                    "median_target": float(np.median(pred[f"F{qtp}"][te])), "median_stop": float(np.median(sl[te])),
                    "per_side": {int(s_): float(n[(pt["side"] == s_).to_numpy()].mean()) for s_ in (1, -1)}},
           "ngboost": "not run (exploratory; would add a dependency)"}
    rep["verdict"] = "passed" if (gd["mean"] > 0 and gd["t_nw"] >= 2 and te.sum() >= 300 and n.mean() > 0) else "failed"
    write(out, "QRANGE1_result.json", rep)


def write(out, name, rep):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["picks", "vol1", "regime1", "surv1", "qpaths", "qrange1", "surv1_export"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/MODELS5")
    ap.add_argument("--minutes", type=float, default=300)
    a = ap.parse_args()
    cache = Path(a.cache)
    {"picks": lambda: stage_picks(cache), "vol1": lambda: stage_vol1(cache, a.out),
     "regime1": lambda: stage_regime1(cache, a.out), "surv1": lambda: stage_surv1(cache, a.out),
     "qpaths": lambda: stage_qpaths(cache, __import__("time").monotonic() + 60 * a.minutes),
     "qrange1": lambda: stage_qrange1(cache, a.out), "surv1_export": lambda: stage_surv1_export(cache)}[a.stage]()


if __name__ == "__main__":
    main()
