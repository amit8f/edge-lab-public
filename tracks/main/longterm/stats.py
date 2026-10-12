"""Statistics for overlapping monthly series (M4 primary test)."""
import numpy as np


def nw_tstat(x, lags):
    """Mean, Newey-West (Bartlett) standard error and t-stat of a series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    m = x.mean()
    e = x - m
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    return float(m), float(se), float(m / se) if se > 0 else float("nan")


def block_bootstrap_ci(x, block, n_boot, seed, alpha=0.05):
    """Moving-block bootstrap percentile CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
