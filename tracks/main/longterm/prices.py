"""Per-symbol price features at signal dates (M4). Pure functions on bar frames, no network.

Each symbol arrives as three frames from core.bars.to_frames (adjustment all, raw and split), indexed by
date. A >15-calendar-day gap ends a series; features and forward returns never cross it.
"""
import numpy as np
import pandas as pd

GAP_DAYS = 15
LOOKBACK = 252        # momentum and volatility window, trading days
SKIP = 21             # momentum skips the last month
MIN_BARS = LOOKBACK + 1
HORIZONS = {"ret6": 126, "ret12": 252}


def align(f_all, f_raw, f_split):
    """One frame with all-adjusted o/c, raw c/v and split-adjusted c on the dates all three share."""
    d = pd.DataFrame({"o": f_all["o"], "c": f_all["c"]})
    d = d.join(f_raw[["c", "v"]].rename(columns={"c": "rc", "v": "v"}), how="inner")
    d = d.join(f_split[["c"]].rename(columns={"c": "sc"}), how="inner")
    return d[(d["v"] > 0) & (d["c"] > 0) & (d["rc"] > 0) & (d["sc"] > 0)].sort_index()


def segment_ids(index):
    """Series number per bar: a new series starts after any gap longer than GAP_DAYS calendar days."""
    gaps = pd.Series(index, index=index).diff().dt.days.fillna(0).to_numpy()
    return np.cumsum(gaps > GAP_DAYS)


def split_points(sym, d):
    """Rows (symbol, date, R) where R = raw close / split-adjusted close, at the first bar and wherever R
    changes by more than 1% (a split). R(t0) / R(t) is the split factor from t0 to t."""
    r = (d["rc"] / d["sc"]).to_numpy()
    keep = np.r_[True, np.abs(r[1:] / r[:-1] - 1) > 0.01]
    return pd.DataFrame({"symbol": sym, "date": d.index[keep], "R": r[keep]})


def horizon_ends(calendar, dates, label_end=None):
    """{horizon: {t: calendar date h trading days after t}}, leaving out any t whose window would reach
    `label_end` or run past the calendar (holdout guard; a window cut short by the data end is not a
    delisting)."""
    cal = pd.DatetimeIndex(sorted(calendar))
    out = {}
    for name, h in HORIZONS.items():
        ends = {}
        for t in dates:
            k = cal.searchsorted(pd.Timestamp(t))
            if k + h < len(cal) and (label_end is None or cal[k + h] < pd.Timestamp(label_end)):
                ends[pd.Timestamp(t)] = cal[k + h]
        out[name] = ends
    return out


def features(sym, d, dates, ends=None):
    """Feature rows for `sym` at each signal date in `dates` on which it traded with a full lookback.

    Forward returns (only for t in `ends[horizon]`, see horizon_ends) enter at the next bar's open and exit
    at the close on the calendar date `h` trading days after t (or the bar before it), or at the series' last close when it ends first (`*_trunc` = True).
    No next bar in the series means no entry: the return stays NaN.
    """
    if len(d) < MIN_BARS:
        return pd.DataFrame()
    ends = ends or {}
    seg = segment_ids(d.index)
    pos = {t: i for i, t in enumerate(d.index)}
    c, o, rc, v = d["c"].to_numpy(), d["o"].to_numpy(), d["rc"].to_numpy(), d["v"].to_numpy()
    logret = np.r_[np.nan, np.diff(np.log(c))]
    rows = []
    for t in dates:
        i = pos.get(pd.Timestamp(t))
        if i is None:
            continue
        s = np.searchsorted(seg, seg[i])                      # first bar of this series
        if i - s < LOOKBACK:
            continue
        e = np.searchsorted(seg, seg[i], side="right") - 1    # last bar of this series
        row = {"symbol": sym, "date": d.index[i],
               "mom": c[i - SKIP] / c[i - LOOKBACK] - 1,
               "vol": float(np.std(logret[i - LOOKBACK + 1:i + 1], ddof=1) * np.sqrt(252)),
               "price": rc[i],
               "dv20": float(np.median(rc[i - 19:i + 1] * v[i - 19:i + 1]))}
        for name, h in HORIZONS.items():
            row[name], row[name + "_trunc"] = np.nan, False
            if d.index[i] not in ends.get(name, {}) or i + 1 > e:
                continue
            end = ends[name][d.index[i]]
            x = min(int(d.index.searchsorted(end, side="right")) - 1, e)
            row[name] = c[x] / o[i + 1] - 1
            row[name + "_trunc"] = bool(d.index[x] < end)
        rows.append(row)
    return pd.DataFrame(rows)
