"""V2 (research/prereg/V2.md): every finished screen re-judged against the Fama-French 5 factors plus
momentum, and against the number of ideas tried.

Usage: python -m validation.v2_rejudge --cache <dir> [--results <dir>] [--out <dir>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation import french
from validation.stats import (deflated_sr, expected_max_sr, holm, ols_hac, p_one_sided, period_sr,
                              romano_wolf)

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "mom"]
# idea: (holding days h, Newey-West lags, the idea's own t bar, non-overlap step, periods per year)
SPECS = {
    "M4": (126, 5, 2.0, 6, 2), "M5": (126, 5, 2.0, 6, 2), "M6": (126, 5, 2.0, 6, 2), "M7": (126, 5, 2.0, 6, 2),
    "M1": (21, 3, 2.0, 1, 12), "M3": (21, 3, 2.0, 1, 12),
    "M10": (21, 6, 3.0, 4, 13), "P3": (21, 6, 3.0, 4, 13), "M11": (21, 6, 3.0, 4, 13),
}
N_TRIALS = 28                      # phase 1 + phase 2 ideas at registration
RW_LAGS, RW_BLOCK, RW_BOOT, SEED = 5, 6, 10_000, 20261015
PLACEBO_SHIFT = pd.Timedelta(days=364)


def daily_factors(cache):
    ff5, h5 = french.fetch("ff5_daily", cache)
    mom, hm = french.fetch("mom_daily", cache)
    f = french.parse(ff5, "D").join(french.parse(mom, "D"), how="inner")
    f = f.rename(columns={c: "mom" for c in f.columns if c.startswith("mom")})
    return f[FACTORS + ["rf"]], {"ff5_daily": h5, "mom_daily": hm}


def window_sums(f, dates, h, shift=None):
    """Sum of each daily factor (and rf) over each window: the h trading days after t, entry day first.
    Rows whose window runs past the factor data are NaN."""
    cal = f.index
    cs = np.vstack([np.zeros(f.shape[1]), np.cumsum(f.to_numpy(), axis=0)])
    out = np.full((len(dates), f.shape[1]), np.nan)
    for i, t in enumerate(pd.DatetimeIndex(dates)):
        t = t + shift if shift is not None else t
        a = int(cal.searchsorted(t, side="right"))            # entry day index
        b = a + h                                              # one past the exit day
        if b <= len(cal):
            out[i] = cs[b] - cs[a]
    return pd.DataFrame(out, columns=f.columns, index=pd.DatetimeIndex(dates))


def load_series(results):
    out = {}
    for idea in SPECS:
        p = Path(results) / idea / f"{idea}_primary_series.csv"
        if p.exists():
            s = pd.read_csv(p, parse_dates=["date"])
            out[idea] = s[s["excess"].notna()].reset_index(drop=True)
    return out


def regress(s, w, lags):
    ok = w[FACTORS].notna().all(axis=1).to_numpy()
    y = s["excess"].to_numpy()[ok]
    beta, se, t, r2, e = ols_hac(y, w[FACTORS].to_numpy()[ok], lags)
    res = {"n": int(ok.sum()), "alpha": float(beta[0]), "alpha_se": float(se[0]), "alpha_t": float(t[0]),
           "p_one_sided": p_one_sided(t[0]), "r2": r2,
           "loadings": {f: {"beta": float(b), "t": float(tt)} for f, b, tt in zip(FACTORS, beta[1:], t[1:])},
           "raw_mean": float(y.mean())}
    adjusted = pd.Series(beta[0] + e, index=s["date"][ok].to_numpy())
    return res, adjusted


def run(cache, results, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f, hashes = daily_factors(cache)
    series = load_series(results)
    print(f"ideas with a primary series: {sorted(series)}")
    res = {"study": "V2", "prereg": "research/prereg/V2.md", "french_sha256": hashes, "n_trials": N_TRIALS,
           "factor_days": [str(f.index.min().date()), str(f.index.max().date())], "ideas": {}}
    adjusted, rows = {}, []
    for idea, s in series.items():
        h, lags, bar, step, ppy = SPECS[idea]
        w = window_sums(f, s["date"], h)
        r, adj = regress(s, w, lags)
        r["bar"] = bar
        r["passes_factor_bar"] = bool(r["alpha"] > 0 and r["alpha_t"] >= bar)
        if "pick_mean" in s:
            own = s.assign(excess=s["pick_mean"] - w["rf"].to_numpy())
            r["picks_own"], _ = regress(own, w, lags)
            r["picks_own"].pop("raw_mean")
        pw = window_sums(f, s["date"], h, shift=PLACEBO_SHIFT)
        r["placebo_shift_52w"], _ = regress(s, pw, lags)
        sr, sk, ku, T = period_sr(s["excess"].to_numpy(), step)
        r["sharpe"] = {"per_period": sr, "annual": sr * np.sqrt(ppy), "skew": sk, "kurtosis": ku, "T": T}
        res["ideas"][idea] = r
        adjusted[idea] = adj
        w.assign(excess=s["excess"].to_numpy()).rename_axis("date").to_csv(out / f"windows_{idea}.csv")
        print(f"{idea}: raw {r['raw_mean']:+.4f}  factor-adjusted {r['alpha']:+.4f} (t {r['alpha_t']:.2f})  "
              f"R2 {r['r2']:.2f}  placebo R2 {r['placebo_shift_52w']['r2']:.2f}")

    ideas = list(res["ideas"])
    # Holm on the per-idea one-sided p-values
    for idea, p in zip(ideas, holm([res["ideas"][i]["p_one_sided"] for i in ideas])):
        res["ideas"][idea]["p_holm"] = float(p)
    # Romano-Wolf on calendar-month averages over the common months
    monthly = pd.DataFrame({i: a.groupby(a.index.to_period("M")).mean() for i, a in adjusted.items()}).dropna()
    t_rw, p_rw = romano_wolf(monthly.to_numpy(), RW_LAGS, RW_BLOCK, RW_BOOT, SEED)
    res["romano_wolf"] = {"months": [str(monthly.index.min()), str(monthly.index.max())], "n_months": len(monthly)}
    for idea, t, p in zip(monthly.columns, t_rw, p_rw):
        res["ideas"][idea]["rw_t_monthly"] = float(t)
        res["ideas"][idea]["p_romano_wolf"] = float(p)
    # Deflated Sharpe ratio with the phase-1 + phase-2 trial count
    annual = np.array([res["ideas"][i]["sharpe"]["annual"] for i in ideas])
    var_sr = float(np.var(annual, ddof=1)) if len(annual) > 1 else np.nan
    sr0_annual = expected_max_sr(var_sr, N_TRIALS)
    res["dsr"] = {"var_annual_sr": var_sr, "sr0_annual": sr0_annual}
    for idea in ideas:
        sh = res["ideas"][idea]["sharpe"]
        ppy = SPECS[idea][4]
        res["ideas"][idea]["dsr"] = deflated_sr(sh["per_period"], sr0_annual / np.sqrt(ppy), sh["T"],
                                                sh["skew"], sh["kurtosis"])
    for idea in ideas:
        r = res["ideas"][idea]
        r["survives"] = bool(r["passes_factor_bar"] and r.get("p_romano_wolf", 1.0) < 0.05)
        rows.append({"idea": idea, "raw_mean": r["raw_mean"], "alpha": r["alpha"], "alpha_t": r["alpha_t"],
                     "r2": r["r2"], **{f"b_{k}": v["beta"] for k, v in r["loadings"].items()},
                     "p_holm": r["p_holm"], "p_rw": r.get("p_romano_wolf"), "sharpe_annual": r["sharpe"]["annual"],
                     "dsr": r["dsr"], "survives": r["survives"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(out / "V2_table.csv", index=False)
    monthly.to_csv(out / "V2_monthly_adjusted.csv")
    (out / "V2_results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", default=str(ROOT / "research" / "results"))
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "V2"))
    ap.add_argument("--budget-min", type=float, default=0)
    a = ap.parse_args()
    run(a.cache, a.results, a.out)


if __name__ == "__main__":
    main()
