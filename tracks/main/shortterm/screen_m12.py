"""M12 historical screen (research/prereg/M12.md): M10's boom model with volatility, size and industry
regressed out of every feature and out of the label, traded as a capped, inverse-volatility-weighted
portfolio and judged on its Fama-French 5 + momentum adjusted excess.

Usage: python -m tracks.main.shortterm.screen_m12 --cache <dir> --out <dir> [--budget-min N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry
from core import shortterm as st
from tracks.main.longterm import sec_data
from tracks.main.longterm.stats import block_bootstrap_ci
from tracks.main.shortterm import evaluate, panel as pn, screen_m10
from validation import v2_rejudge as v2
from validation.stats import deflated_sr, expected_max_sr, ols_hac, period_sr

ROOT = Path(__file__).resolve().parents[3]
TOP_N, IND_CAP, W_CAP = 20, 4, 0.10
COST_FLAT, LAGS, BLOCK, T_BAR, SEED = 0.002, 6, 8, 3.0, 20261012
N_TRIALS = 31
COUNTS = {"phase2_count": 16, "main_count": 10, "phase1_plus_phase2": 31}
CONTROLS = ["vol63", "lmcap"]
NEUTRAL_FEATURES = [f for f in screen_m10.FEATURES if f not in CONTROLS]


def neutralize(df, cols, ind_col="ind"):
    """Per date: OLS residual of each column on the vol63 rank, the lmcap rank and industry dummies
    (rows missing a regressor get NaN; a column's own NaNs stay NaN)."""
    out = np.full((len(df), len(cols)), np.nan)
    ctrl = df[[f"rk_{c}" for c in CONTROLS]].to_numpy(float)
    vals = df[cols].to_numpy(float)
    ind = df[ind_col].to_numpy()
    for idx in df.groupby("date", sort=False).indices.values():
        levels = np.unique(ind[idx])
        dummies = (ind[idx][:, None] == levels[None, 1:]).astype(float)
        X = np.c_[np.ones(len(idx)), ctrl[idx], dummies]
        okx = np.isfinite(X).all(axis=1)
        for j in range(len(cols)):
            y = vals[idx, j]
            ok = okx & np.isfinite(y)
            if ok.sum() <= X.shape[1] + 1:
                continue
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            out[idx[ok], j] = y[ok] - X[ok] @ beta
    return pd.DataFrame(out, index=df.index, columns=cols)


def neutral_panel(uni):
    """Ranked features, residualised features and label, plus the raw columns the portfolio needs."""
    p = uni[["symbol", "date", "cik", "price", "ret21", "ret21_trunc", "ret5", "vol63", "ind"]].reset_index(drop=True)
    rk = st.rank_within_date(uni.reset_index(drop=True), screen_m10.FEATURES)
    for c in CONTROLS:
        p[f"rk_{c}"] = rk[c]
    p = pd.concat([p, rk[NEUTRAL_FEATURES]], axis=1)
    p["ret21_res"] = np.nan
    res = neutralize(p.assign(ret21_res=p["ret21"]), NEUTRAL_FEATURES + ["ret21_res"])
    p[NEUTRAL_FEATURES] = res[NEUTRAL_FEATURES].astype("float32")
    p["ret21_res"] = res["ret21_res"]
    p["y"] = st.top_label(p, "ret21_res")
    p["ret21_rank"] = p.groupby("date")["ret21_res"].rank(pct=True)
    return p


def capped_weights(vol, cap=W_CAP):
    """Inverse-volatility weights summing to 1, each <= cap, the excess spread pro rata over the rest."""
    w = 1 / np.asarray(vol, float)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w[np.isfinite(w)]) if np.isfinite(w).any() else 1.0)
    w = w / w.sum()
    if cap * len(w) < 1:
        return np.full(len(w), 1 / len(w))
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        w[~over] += extra * w[~over] / w[~over].sum()
    return w


def select(g, order, cap=IND_CAP, n=TOP_N):
    """First n rows of g in `order`, skipping a row whose industry already has `cap` picks."""
    taken, counts = [], {}
    for i in order:
        k = g["ind"].iloc[i]
        if counts.get(k, 0) >= cap:
            continue
        counts[k] = counts.get(k, 0) + 1
        taken.append(i)
        if len(taken) == n:
            break
    return g.iloc[taken]


def portfolio_series(wf, score, cost, weights="invvol", cap=IND_CAP, random_seed=None):
    rows = []
    for t, g in wf.groupby("date", sort=True):
        g = g[g["ret21"].notna()]
        if len(g) < TOP_N:
            continue
        if random_seed is not None:
            order = np.random.default_rng(random_seed + t.toordinal()).permutation(len(g))
        else:
            order = np.argsort(-g[score].to_numpy(), kind="stable")
        picks = select(g, order, cap=cap if cap else TOP_N)
        w = capped_weights(picks["vol63"]) if weights == "invvol" else np.full(len(picks), 1 / len(picks))
        pm = float(np.sum(w * picks["ret21"].to_numpy()))
        um = float(g["ret21"].mean())
        rows.append({"date": t, "pick_mean": pm, "uni_mean": um, "excess": pm - um - cost, "n_picks": len(picks),
                     "max_per_industry": int(picks["ind"].value_counts().max())})
    return pd.DataFrame(rows)


def factor_test(ex, f):
    w = v2.window_sums(f, ex["date"], 21)
    ok = w[v2.FACTORS].notna().all(axis=1).to_numpy()
    y = ex["excess"].to_numpy()[ok]
    beta, se, t, r2, e = ols_hac(y, w[v2.FACTORS].to_numpy()[ok], LAGS)
    lo, hi = block_bootstrap_ci(beta[0] + e, BLOCK, 10_000, SEED)
    sr, sk, ku, T = period_sr(ex["excess"].to_numpy(), 4)
    return {"raw_mean": float(np.mean(y)), "alpha": float(beta[0]), "alpha_t": float(t[0]), "ci95": [lo, hi],
            "r2": r2, "n_weeks": int(ok.sum()),
            "loadings": {k: float(b) for k, b in zip(v2.FACTORS, beta[1:])},
            "sharpe_per_period": sr, "skew": sk, "kurtosis": ku, "T": T}


def cost_used(results=None):
    p = Path(results or ROOT / "research" / "results") / "V4" / "V4_results.json"
    if p.exists():
        m = json.loads(p.read_text()).get("costs", {}).get("M10", {}).get("measured_mean")
        if m is not None and m > COST_FLAT:
            return float(m), "V4 measured mean for M10 picks"
    return COST_FLAT, "flat main-track cost (config/risk.yaml)"


def run(cache, out, budget_min=290, years=None, industries=None, factors=None):
    cache, out = Path(cache), Path(out)
    years = years or evaluate.YEARS
    got = pn.build(cache, time.monotonic() + budget_min * 60)
    if got is None:
        print("PANEL INCOMPLETE: rerun to continue")
        return None
    panel, rtab, funnel, dates, ends, spy_r = got
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    rows = panel[(panel["price"] >= screen_m10.MIN_PRICE) & (panel["dv20"] >= screen_m10.MIN_DV20)]
    facts = sec_data.facts_for(zp, rows["cik"].unique(), cache / "boom" / "facts_m10")
    uni = screen_m10.universe(panel, rtab, facts).reset_index(drop=True)
    ind = industries if industries is not None else industry.industries(uni["cik"].unique(), cache / "industry")
    uni["ind"] = uni["cik"].map(ind).fillna(12).astype(int)
    cost, cost_src = cost_used()
    f, hashes = factors if factors is not None else v2.daily_factors(cache)
    res = {"idea": "M12", "prereg": "research/prereg/M12.md", **COUNTS, "cost": cost, "cost_source": cost_src,
           "french_sha256": hashes, "industry_share": uni["ind"].value_counts(normalize=True).sort_index().to_dict()}
    p = neutral_panel(uni)
    wf = st.walk_forward(p, ends["ret21"], {"xgb": NEUTRAL_FEATURES}, years)
    ex = portfolio_series(wf, "p_xgb", cost)
    prim = factor_test(ex, f)
    prim["passes"] = bool(prim["alpha"] > 0 and prim["alpha_t"] >= T_BAR and prim["ci95"][0] > 0)
    v2r = Path(ROOT / "research" / "results" / "V2" / "V2_results.json")
    if v2r.exists():
        var_sr = json.loads(v2r.read_text())["dsr"]["var_annual_sr"]
        sr0 = expected_max_sr(var_sr, N_TRIALS) / np.sqrt(13)
        prim["dsr"] = deflated_sr(prim["sharpe_per_period"], sr0, prim["T"], prim["skew"], prim["kurtosis"])
    res["primary"] = prim
    res["control"] = factor_test(portfolio_series(wf, "p_xgb", cost, random_seed=SEED), f)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))
    raw = evaluate.model_panel(uni, screen_m10.FEATURES).assign(vol63=uni["vol63"].to_numpy(), ind=uni["ind"].to_numpy())
    wf_raw = st.walk_forward(raw, ends["ret21"], {"xgb": screen_m10.FEATURES}, years)
    exp = {"not_neutralised_same_portfolio": factor_test(portfolio_series(wf_raw, "p_xgb", cost), f),
           "equal_weight": factor_test(portfolio_series(wf, "p_xgb", cost, weights="equal"), f),
           "no_industry_cap": factor_test(portfolio_series(wf, "p_xgb", cost, cap=None), f),
           "per_year": {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}}
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1, default=str))
    out.mkdir(parents=True, exist_ok=True)
    ex.to_csv(out / "M12_primary_series.csv", index=False)
    (out / "M12_screen.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "M12"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
