"""Shared research statistics (PROTOCOL.md rules 11-12): factor alpha, deflated Sharpe ratio, event-study
tests, Fama-MacBeth, and trading-cost estimators. numpy and the standard library only."""
import io
import math
import zipfile
from statistics import NormalDist

import numpy as np
import pandas as pd

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FF5 = "F-F_Research_Data_5_Factors_2x3"
UMD = "F-F_Momentum_Factor"
_N = NormalDist()


def nw_se(e, X, lags):
    """Newey-West (Bartlett) covariance of OLS coefficients, given residuals e and design X."""
    n = len(e)
    xe = X * e[:, None]
    s = xe.T @ xe / n
    for k in range(1, min(lags, n - 1) + 1):
        g = xe[k:].T @ xe[:-k] / n
        s += (1 - k / (lags + 1)) * (g + g.T)
    inv = np.linalg.inv(X.T @ X / n)
    return inv @ s @ inv / n


def ols_hac(y, F, lags):
    """y = a + F b + e with Newey-West errors. Returns dict(alpha, se, t, betas, r2, n) on rows where y
    and every factor are finite."""
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(F).all(axis=1)
    y, F = y[ok], F[ok]
    X = np.c_[np.ones(len(y)), F]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(nw_se(e, X, lags)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())) if len(y) > 1 else float("nan")
    return {"alpha": float(b[0]), "se": float(se[0]), "t": float(b[0] / se[0]) if se[0] > 0 else float("nan"),
            "betas": [float(v) for v in b[1:]], "r2": float(r2), "n": int(len(y))}


def parse_french(raw_zip):
    """Monthly table of a Ken French CSV zip as decimals, indexed by month period, columns as in the file
    header (e.g. Mkt-RF, SMB, HML, RMW, CMA, RF; or Mom). Stops at the annual section."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    cols, rows = None, []
    for ln in text.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if rows and ("annual" in ln.lower() or not ln.strip()):
            break
        if cols is None and len(p) > 1 and p[0] == "" and all(p[1:]):
            cols = p[1:]
        elif cols and len(p) == len(cols) + 1 and len(p[0]) in (6, 8) and p[0].isdigit():
            k = pd.Period(f"{p[0][:4]}-{p[0][4:]}", "M") if len(p[0]) == 6 else pd.Timestamp(p[0])
            rows.append([k] + [float(x) / 100 for x in p[1:]])
    return pd.DataFrame(rows, columns=["m"] + cols).drop_duplicates("m").set_index("m")


def fetch_french(name):
    """(table, sha256) for one Ken French file, e.g. FF5, UMD, or their '_daily' versions."""
    import hashlib
    import requests
    raw = requests.get(FRENCH.format(name=name), timeout=120).content
    return parse_french(raw), hashlib.sha256(raw).hexdigest()


def idea_counts(ideas_md="research/IDEAS.md"):
    """(phase-2 count, phase 1 + phase 2) from the IDEAS.md header, the reference count."""
    import re
    text = open(ideas_md, encoding="utf-8").read()
    p2 = int(re.search(r"\*\*Phase 2 count: (\d+)\*\*", text).group(1))
    both = int(re.search(r"phase 1 \+ phase 2 = (\d+)", text).group(1))
    return p2, both


def factor_alpha(ret, months, ff5, umd, lags):
    """Monthly alpha of `ret` (already net of costs; an excess or long-short return) on the Fama-French 5
    factors plus momentum. `months` are the return months (periods); factor rows are matched by month."""
    f = ff5.join(umd, how="inner").reindex(pd.PeriodIndex(months, freq="M"))
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", umd.columns[0]]
    out = ols_hac(np.asarray(ret, dtype=float), f[cols].to_numpy(), lags)
    out["factors"] = cols
    return out


def window_factors(daily, days):
    """Sum of daily factor returns over each window of trading days (list of DatetimeIndex slices):
    a close-to-close approximation of the factor exposure of a multi-day trade."""
    return np.array([daily.reindex(d).sum(min_count=1).to_numpy() for d in days])


def deflated_sharpe(x, n_trials, sr_var_trials=None):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of a return series x: the probability that its
    true Sharpe ratio is above the best one expected from `n_trials` tries at zero skill. Per-period units.
    Without the trials' Sharpe variance, the null sampling variance 1/(T-1) is used."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    sd = x.std(ddof=1)
    sr = x.mean() / sd
    z = (x - x.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    v = sr_var_trials if sr_var_trials is not None else 1 / (t - 1)
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials) + g * _N.inv_cdf(1 - 1 / (n_trials * math.e))) \
        if n_trials > 1 else 0.0
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return {"sr": float(sr), "sr0": float(sr0), "n_trials": int(n_trials), "t_obs": t,
            "dsr": float(_N.cdf((sr - sr0) * math.sqrt(t - 1) / denom))}


def market_model_car(r, m, est, win):
    """Abnormal returns for one event from a market model fitted on `est` (index slice of r and m) and
    applied on `win`. Returns (CAR, standardized CAR, residuals on est). r, m: aligned numpy arrays."""
    re, me = r[est], m[est]
    ok = np.isfinite(re) & np.isfinite(me)
    X = np.c_[np.ones(ok.sum()), me[ok]]
    b, *_ = np.linalg.lstsq(X, re[ok], rcond=None)
    res = re[ok] - X @ b
    s = res.std(ddof=2)
    ar = r[win] - (b[0] + b[1] * m[win])
    car = float(np.nansum(ar))
    return car, car / (s * math.sqrt(np.isfinite(ar).sum())), res


def bmp_kp(scar, rbar):
    """Boehmer-Musumeci-Poulsen t of standardized CARs, adjusted for cross-sectional correlation of the
    events' residuals (Kolari & Pynnonen 2010): t * sqrt((1 - rbar) / (1 + (n - 1) rbar))."""
    s = np.asarray(scar, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    t = s.mean() * math.sqrt(n) / s.std(ddof=1)
    adj = t * math.sqrt((1 - rbar) / (1 + (n - 1) * rbar))
    return {"n": n, "t_bmp": float(t), "rbar": float(rbar), "t_kp": float(adj)}


def mean_residual_corr(resids):
    """Average pairwise correlation of equal-length residual series (rows = events)."""
    R = np.corrcoef(np.asarray(resids, dtype=float))
    n = len(R)
    return float((R.sum() - n) / (n * (n - 1))) if n > 1 else 0.0


def fama_macbeth(df, y, xs, lags, date="date"):
    """Per-date cross-sectional OLS of y on xs; time-series means of the coefficients with Newey-West t."""
    coefs = []
    for _, g in df.groupby(date, sort=True):
        g = g[[y] + xs].dropna()
        if len(g) > len(xs) + 1:
            X = np.c_[np.ones(len(g)), g[xs].to_numpy(dtype=float)]
            coefs.append(np.linalg.lstsq(X, g[y].to_numpy(dtype=float), rcond=None)[0])
    C = np.array(coefs)
    out = {}
    for j, name in enumerate(["const"] + xs):
        r = ols_hac(C[:, j], np.empty((len(C), 0)), lags)
        out[name] = {"mean": r["alpha"], "t": r["t"]}
    out["n_dates"] = len(C)
    return out


def corwin_schultz(high, low):
    """Corwin & Schultz (2012) bid-ask spread estimate from consecutive daily highs and lows (arrays of
    raw prices). One estimate per day pair, negative estimates set to 0; returns the array."""
    h, lo = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    beta = np.log(h[:-1] / lo[:-1]) ** 2 + np.log(h[1:] / lo[1:]) ** 2
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
    k = 3 - 2 * math.sqrt(2)
    a = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(a) - 1) / (1 + np.exp(a))
    return np.maximum(s, 0.0)


def amihud(ret, dollar_volume):
    """Amihud (2002) illiquidity: mean |return| per dollar traded, on days with volume."""
    r, dv = np.asarray(ret, dtype=float), np.asarray(dollar_volume, dtype=float)
    ok = np.isfinite(r) & (dv > 0)
    return float(np.mean(np.abs(r[ok]) / dv[ok]))


def effective_spread(price, mid):
    """Effective spread of trades, as a fraction of the midquote: 2|p - m| / m."""
    p, m = np.asarray(price, dtype=float), np.asarray(mid, dtype=float)
    return 2 * np.abs(p - m) / m


def implementation_shortfall(decision_price, fill_price, side):
    """Cost of a fill vs the price when the decision was made, as a fraction; side +1 buy, -1 sell/short.
    Positive = paid more (or received less) than the decision price."""
    return side * (fill_price - decision_price) / decision_price


T_PASS, ALPHA_T_PASS = 3.0, 2.0


def screen_pass(prim, alpha):
    """PROTOCOL rules 11-12: mean > 0, HAC t >= 3.0, bootstrap CI above 0, and factor alpha > 0 with
    t >= 2.0."""
    return bool(prim["mean"] > 0 and prim["t_hac"] >= T_PASS and prim["ci95"][0] > 0
                and alpha["alpha"] > 0 and alpha["t"] >= ALPHA_T_PASS)


def month_after(dates):
    """Return month of a position opened after month-end t: the month after t."""
    return [pd.Period(pd.Timestamp(t), "M") + 1 for t in dates]
