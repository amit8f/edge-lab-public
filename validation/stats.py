"""Statistics for re-judging results: OLS with Newey-West errors, Romano-Wolf stepdown, Holm, the
deflated Sharpe ratio (Bailey and Lopez de Prado, 2014), and information ratios from non-overlapping
subsamples of overlapping-window series."""
import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649


def ols_hac(y, X, lags):
    """OLS of y on [1, X] with Newey-West (Bartlett) covariance. Returns coef, se, t (intercept first),
    R^2 and the residuals."""
    y = np.asarray(y, float)
    A = np.c_[np.ones(len(y)), np.asarray(X, float)]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    n = len(y)
    bread = np.linalg.pinv(A.T @ A)
    u = A * e[:, None]
    S = u.T @ u
    for k in range(1, min(lags, n - 1) + 1):
        G = u[k:].T @ u[:-k]
        S += (1 - k / (lags + 1)) * (G + G.T)
    cov = bread @ S @ bread
    se = np.sqrt(np.diag(cov))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / tss if tss > 0 else np.nan
    return beta, se, np.where(se > 0, beta / se, np.nan), float(r2), e


def p_one_sided(t):
    """P(Z >= t) under the normal approximation."""
    return float(norm.sf(t))


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def _nw_se(x, lags):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return np.sqrt(var / n)


def romano_wolf(Z, lags, block, n_boot, seed):
    """One-sided Romano-Wolf stepdown on the columns of Z (rows = aligned periods, H0: mean <= 0).
    Studentized by each column's Newey-West SE; joint moving-block bootstrap of rows. Returns t-stats
    and adjusted p-values in column order."""
    Z = np.asarray(Z, float)
    n, m = Z.shape
    mu = Z.mean(axis=0)
    se = np.array([_nw_se(Z[:, j], lags) for j in range(m)])
    t = mu / se
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    tb = np.empty((n_boot, m))
    for b in range(n_boot):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        tb[b] = (Z[idx].mean(axis=0) - mu) / se
    order = np.argsort(-t)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        rest = order[i:]
        p = float((tb[:, rest].max(axis=1) >= t[j]).mean())
        run = max(run, p)
        adj[j] = run
    return t, adj


def nonoverlap(x, step):
    """The `step` interleaved non-overlapping subsamples of a series of overlapping windows."""
    x = np.asarray(x, float)
    return [x[o::step] for o in range(step)]


def period_sr(x, step):
    """Per-period Sharpe (mean / sd) of non-overlapping returns, averaged across the `step` offsets, with
    the pooled skew, kurtosis (non-excess) and the number of non-overlapping periods."""
    subs = [s[np.isfinite(s)] for s in nonoverlap(x, step)]
    subs = [s for s in subs if len(s) > 2]
    sr = float(np.mean([s.mean() / s.std(ddof=1) for s in subs]))
    pooled = np.concatenate([s - s.mean() for s in subs])
    return sr, float(skew(pooled)), float(kurtosis(pooled, fisher=False)), int(np.mean([len(s) for s in subs]))


def expected_max_sr(var_sr, n_trials):
    """Expected maximum Sharpe of n_trials zero-skill trials whose Sharpes have variance var_sr."""
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                    + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sr(sr, sr0, T, sk, ku):
    """Probability that the true per-period Sharpe exceeds sr0, given an estimate `sr` from T periods
    with skew sk and (non-excess) kurtosis ku."""
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))
