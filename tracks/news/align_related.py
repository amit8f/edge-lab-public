"""ALIGN: related stocks for each symbol and year (research/prereg/ALIGN.md).

related() returns, for one year, each symbol's same-SIC-3 peers plus its k most correlated stocks over the
`window` trading days before January 1 of that year. Pure function, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pairwise_corr(x: np.ndarray, min_shared: int) -> np.ndarray:
    """Pearson correlation on pairwise-complete days, as matrix products (fast for thousands of columns).
    x: days x symbols with NaN for missing. Pairs with fewer than min_shared shared days get NaN."""
    m = (~np.isnan(x)).astype(float)
    z = np.where(m > 0, x, 0.0)
    n = m.T @ m                      # shared days per pair
    sx = z.T @ m                     # sum of i's returns on days both have data
    sxx = (z * z).T @ m
    sxy = z.T @ z
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        vi = sxx - sx * sx / n
        corr = cov / np.sqrt(vi * vi.T)
    corr[(n < min_shared) | ~np.isfinite(corr)] = np.nan
    return corr


def related(daily_ret: pd.DataFrame, sic3: dict, year: int, k: int = 10, window: int = 250,
            min_shared: int = 200) -> dict:
    """symbol -> sorted list of related symbols for `year` (same SIC-3 members plus top-k correlated).
    Uses only returns dated before January 1 of `year`."""
    cols = sorted(daily_ret.columns)
    past = daily_ret.loc[pd.DatetimeIndex(daily_ret.index) < pd.Timestamp(year, 1, 1), cols].sort_index()
    past = past.iloc[-window:]
    by_sic: dict = {}
    for s in cols:
        if sic3.get(s):
            by_sic.setdefault(sic3[s], set()).add(s)
    out = {s: set(by_sic.get(sic3.get(s), set())) - {s} for s in cols}
    if len(past) and k > 0:
        corr = _pairwise_corr(past.to_numpy(dtype=float), min_shared)
        np.fill_diagonal(corr, np.nan)
        names = np.array(cols)
        for i, s in enumerate(cols):
            ok = np.flatnonzero(~np.isnan(corr[i]))
            if not len(ok):
                continue
            # highest correlation first; ties broken by symbol name (cols is sorted, so index order = name order)
            order = ok[np.lexsort((ok, -corr[i, ok]))][:k]
            out[s] |= set(names[order])
    return {s: sorted(v) for s, v in out.items()}
