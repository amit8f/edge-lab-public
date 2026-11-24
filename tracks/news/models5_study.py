"""VOL1 and REGIME1 (research/prereg/VOL1.md, REGIME1.md): NX3-ALL-HI's history re-sized by a GARCH volatility
forecast, and cut in a hidden-Markov "storm" regime. SURV1, QRANGE1 and ANOM1 come in a later stage.

    python -m tracks.news.models5_study --stage picks|vol1|regime1 --cache cache/nx1 --out research/results/MODELS5

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


def write(out, name, rep):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["picks", "vol1", "regime1"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/MODELS5")
    a = ap.parse_args()
    cache = Path(a.cache)
    {"picks": lambda: stage_picks(cache), "vol1": lambda: stage_vol1(cache, a.out),
     "regime1": lambda: stage_regime1(cache, a.out)}[a.stage]()


if __name__ == "__main__":
    main()
