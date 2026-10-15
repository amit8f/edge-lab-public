"""Shared machinery for the short-term boom models (research/prereg/M10.md, P3.md).

Pure functions: weekly signal dates, per-symbol price/volume features and 21/5-day forward returns,
trailing headline counts, within-date ranks, top-2% labels, the fixed model settings and the yearly
walk-forward. Each track calls these on its own rows and fits its own models; nothing fitted is shared.
"""
import numpy as np
import pandas as pd

GAP_DAYS = 15
LOOKBACK = 252
HORIZONS = {"ret21": 21, "ret5": 5}
TOP_Q = 0.98
SEED = 20261015

PRICE_FEATURES = ["r1", "r5", "r21", "r63", "r126", "mom", "hi52", "lo52", "vol21", "vol63", "volratio",
                  "maxret21", "gap5", "up21", "rsi14", "dvr", "ldv20", "lprice"]
NEWS_FEATURES = ["news7", "news30", "an_up", "an_down", "an_net"]

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, tree_method="hist", random_state=SEED, n_jobs=-1)
LGBM_PARAMS = dict(n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.05, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)


def weekly_dates(calendar, first, last):
    """Last trading day of each week (Monday-Sunday) in [first, last]."""
    s = pd.Series(pd.DatetimeIndex(sorted(calendar)))
    wk = s.groupby(s.dt.to_period("W-SUN")).max()
    return [pd.Timestamp(d) for d in wk if first <= d <= last]


def horizon_ends(calendar, dates, label_end):
    """{name: {t: calendar date h trading days after t}}, leaving out windows reaching label_end."""
    cal = pd.DatetimeIndex(sorted(calendar))
    out = {}
    for name, h in HORIZONS.items():
        m = {}
        for t in dates:
            k = cal.searchsorted(pd.Timestamp(t))
            if k + h < len(cal) and cal[k + h] < pd.Timestamp(label_end):
                m[pd.Timestamp(t)] = cal[k + h]
        out[name] = m
    return out


def _segment(s):
    """Features for one gap-free series (o, c all-adjusted; rc raw close; v volume)."""
    c, o, rc, v = s["c"], s["o"], s["rc"], s["v"]
    lr = np.log(c).diff()
    r1 = c.pct_change()
    f = pd.DataFrame(index=s.index)
    f["r1"] = r1
    for n in (5, 21, 63, 126):
        f[f"r{n}"] = c / c.shift(n) - 1
    f["mom"] = c.shift(21) / c.shift(252) - 1
    f["hi52"] = c / c.rolling(252).max()
    f["lo52"] = c / c.rolling(252).min()
    f["vol21"] = lr.rolling(21).std() * np.sqrt(252)
    f["vol63"] = lr.rolling(63).std() * np.sqrt(252)
    f["volratio"] = f["vol21"] / f["vol63"]
    f["maxret21"] = r1.rolling(21).max()
    f["gap5"] = (o / c.shift(1) - 1).abs().rolling(5).max()
    f["up21"] = (lr > 0).astype(float).rolling(21).mean()
    gain, loss = lr.clip(lower=0).rolling(14).mean(), (-lr).clip(lower=0).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f.loc[(loss == 0) & (gain > 0), "rsi14"] = 100.0
    dv = rc * v
    f["dvr"] = dv.rolling(5).mean() / dv.rolling(63).mean()
    f["dv20"] = dv.rolling(20).median()
    f["ldv20"] = np.log(f["dv20"])
    f["price"] = rc
    f["lprice"] = np.log(rc)
    f["n_in_seg"] = np.arange(len(s))
    return f


def features(sym, d, dates, ends):
    """Rows for `sym` at each signal date it traded with >= LOOKBACK prior bars in the same series, with
    forward returns (next open to the close h trading days later, or the series' last close) where
    `ends[name]` has that date. `d` is prices.align output (o, c, rc, v, sc)."""
    if len(d) <= LOOKBACK:
        return pd.DataFrame()
    gaps = pd.Series(d.index, index=d.index).diff().dt.days.fillna(0).to_numpy()
    seg = np.cumsum(gaps > GAP_DAYS)
    want = pd.DatetimeIndex(dates)
    out = []
    for k in np.unique(seg):
        s = d[seg == k]
        if len(s) <= LOOKBACK or not want.isin(s.index).any():
            continue
        f = _segment(s)
        idx = np.flatnonzero(s.index.isin(want) & (f["n_in_seg"].to_numpy() >= LOOKBACK))
        if not len(idx):
            continue
        rows = f.iloc[idx].drop(columns="n_in_seg").copy()
        o, c, last = s["o"].to_numpy(), s["c"].to_numpy(), len(s) - 1
        for name in HORIZONS:
            r = np.full(len(idx), np.nan)
            tr = np.zeros(len(idx), dtype=bool)
            m = ends.get(name, {})
            for j, i in enumerate(idx):
                t = s.index[i]
                if t not in m or i + 1 > last:
                    continue
                x = min(int(s.index.searchsorted(m[t], side="right")) - 1, last)
                r[j] = c[x] / o[i + 1] - 1
                tr[j] = s.index[x] < m[t]
            rows[name], rows[name + "_trunc"] = r, tr
        out.append(rows)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out).rename_axis("date").reset_index()
    res.insert(0, "symbol", sym)
    return res


def window_counts(keys, ev_days, windows):
    """Counts of events per (symbol, date) with day in (date - w days, date], for each w in `windows`.
    `ev_days` has columns symbol, day (New York dates)."""
    k = keys.reset_index(drop=True)
    out = {w: np.zeros(len(k), dtype=np.int32) for w in windows}
    days = {s: np.sort(g["day"].to_numpy(dtype="datetime64[ns]")) for s, g in ev_days.groupby("symbol")}
    for s, g in k.groupby("symbol"):
        d = days.get(s)
        if d is None:
            continue
        t = g["date"].to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(d, t, side="right")
        for w in windows:
            lo = np.searchsorted(d, t - np.timedelta64(w, "D"), side="right")
            out[w][g.index.to_numpy()] = hi - lo
    return out


def rank_within_date(df, cols):
    """Percentile rank of each column within its date (NaN stays NaN)."""
    g = df.groupby("date")
    return pd.DataFrame({c: g[c].rank(pct=True) for c in cols}, index=df.index).astype("float32")


def top_label(df, col="ret21", q=TOP_Q):
    """1 when `col` is in the top (1 - q) of its date among rows that have it, 0 otherwise, NaN without."""
    p = df.groupby("date")[col].rank(pct=True)
    return pd.Series(np.where(p.isna(), np.nan, (p > q).astype(float)), index=df.index)


def make_model(name):
    if name in ("xgb", "xgb_nonews"):
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS)
    if name == "xgb_reg":
        from xgboost import XGBRegressor
        return XGBRegressor(**XGB_PARAMS)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**LGBM_PARAMS)
    if name == "logit":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=1000)
    raise ValueError(name)


def _xy(df, feats, name):
    X = df[feats].to_numpy(dtype="float32")
    if name == "logit":
        X = np.where(np.isnan(X), 0.5, X)
    y = df["ret21_rank"].to_numpy() if name == "xgb_reg" else df["y"].to_numpy()
    return X, y


def fit_predict(name, train, test, feats):
    m = make_model(name)
    X, y = _xy(train, feats, name)
    m.fit(X, y)
    Xt, _ = _xy(test, feats, name)
    return m.predict(Xt) if name == "xgb_reg" else m.predict_proba(Xt)[:, 1], m


def walk_forward(panel, end21, models, years, log=print):
    """Yearly walk-forward. `panel` holds ranked features, y, ret21_rank and date; `end21` maps each
    labelled date to its window's end. `models` maps a model name to its feature list. Each year's
    models train only on dates whose window ended before that year's first signal date. Returns the
    test rows with one prediction column per model."""
    labelled = panel[panel["y"].notna()]
    wend = labelled["date"].map(end21)
    parts = []
    for y in years:
        test = panel[(panel["date"].dt.year == y) & panel["date"].isin(list(end21))].copy()
        if test.empty:
            continue
        first = test["date"].min()
        train = labelled[wend < first]
        for name, feats in models.items():
            test["p_" + name], _ = fit_predict(name, train, test, feats)
        log(f"[wf] {y}: train {len(train)} rows / {train['date'].nunique()} dates "
            f"(to {train['date'].max():%Y-%m-%d}), test {len(test)} rows / {test['date'].nunique()} dates")
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def ranks_from(df, col):
    """Rank 1 = highest `col` within each date (ties by row order)."""
    return df.groupby("date")[col].rank(ascending=False, method="first")
