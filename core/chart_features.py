"""Chart and volume indicators from daily SIP bars (NX3TA and CHART1: research/prereg/NX3TA.md, CHART1.md).

Every value on date d uses bars up to and including d's close, nothing later. Ratios come from all-adjusted
bars (Alpaca adjusts prices and volume for splits and dividends; a ratio inside a window does not change when
the whole series is rescaled), price level and dollar volume from raw bars. A >15-calendar-day gap ends a
series and nothing is computed across it. Zero-volume bars are dropped before this runs.
"""
import numpy as np
import pandas as pd

GAP_DAYS = 15
SEQ = 10        # days of raw bar shape for the sequence inputs

CLASSIC = ["r1", "r5", "r20", "r60", "r120", "mom12_1", "vol20", "vol60", "vol_ratio", "rsi14", "rsi2", "macd_h",
           "bb_pctb", "bb_width", "atr14", "d_sma20", "d_sma50", "d_sma200", "hi52", "lo52", "gap", "oc", "clv",
           "rng_atr", "vrel20", "vrel5_20", "obv20", "maxret20", "minret20", "streak", "log_dv", "log_px"]
SEQ_COLS = ([f"rz_{i}" for i in range(SEQ)] + [f"rg_{i}" for i in range(SEQ)] + [f"vr_{i}" for i in range(SEQ)])
MARKET = ["spy_r1", "spy_r5", "spy_d200", "spy_vol20"]
FEATURES = CLASSIC + SEQ_COLS


def segments(dates):
    """Segment id per row: a new segment after a gap of more than GAP_DAYS calendar days."""
    d = pd.DatetimeIndex(dates)
    gap = np.r_[False, (d[1:] - d[:-1]).days > GAP_DAYS]
    return np.cumsum(gap)


def _rsi(r, n):
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    a = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    b = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + a / b)


def _one(g):
    """g: one segment's rows indexed by date with ao, ah, al, ac, av (all-adjusted) and c, v (raw)."""
    ac, ao, ah, al, av = g["ac"], g["ao"], g["ah"], g["al"], g["av"]
    f = pd.DataFrame(index=g.index)
    r = ac.pct_change()
    lr = np.log(ac).diff()
    for k in (1, 5, 20, 60, 120):
        f[f"r{k}"] = ac.pct_change(k)
    f["mom12_1"] = ac.shift(21) / ac.shift(252) - 1
    f["vol20"] = lr.rolling(20).std()
    f["vol60"] = lr.rolling(60).std()
    f["vol_ratio"] = f["vol20"] / f["vol60"]
    f["rsi14"] = _rsi(r, 14)
    f["rsi2"] = _rsi(r, 2)
    e12, e26 = ac.ewm(span=12, adjust=False, min_periods=26).mean(), ac.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = e12 - e26
    f["macd_h"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / ac
    m20, s20 = ac.rolling(20).mean(), ac.rolling(20).std()
    f["bb_pctb"] = (ac - m20) / (2 * s20)
    f["bb_width"] = 4 * s20 / m20
    tr = pd.concat([ah - al, (ah - ac.shift()).abs(), (al - ac.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr14"] = atr / ac
    for k in (20, 50, 200):
        f[f"d_sma{k}"] = ac / ac.rolling(k).mean() - 1
    f["hi52"] = ac / ah.rolling(252, min_periods=120).max() - 1
    f["lo52"] = ac / al.rolling(252, min_periods=120).min() - 1
    f["gap"] = ao / ac.shift() - 1
    f["oc"] = ac / ao - 1
    rng = ah - al
    f["clv"] = ((ac - al) / rng).where(rng > 0, 0.5)
    f["rng_atr"] = rng / atr
    v20 = av.rolling(20).mean()
    f["vrel20"] = av / v20
    f["vrel5_20"] = av.rolling(5).mean() / v20
    f["obv20"] = (np.sign(r) * av).rolling(20).sum() / av.rolling(20).sum()
    f["maxret20"] = r.rolling(20).max()
    f["minret20"] = r.rolling(20).min()
    s = np.sign(r).fillna(0).to_numpy()
    st = np.zeros(len(s))
    for i in range(1, len(s)):
        st[i] = st[i - 1] + s[i] if s[i] != 0 and np.sign(st[i - 1]) == s[i] else s[i]
    f["streak"] = st
    dv20 = (g["c"] * g["v"]).rolling(20).median()
    f["log_dv"] = np.log(dv20)
    f["log_px"] = np.log(g["c"])
    rz = r / f["vol20"]
    rg = rng / ac
    for i in range(SEQ):
        f[f"rz_{i}"] = rz.shift(i)
        f[f"rg_{i}"] = rg.shift(i)
        f[f"vr_{i}"] = f["vrel20"].shift(i)
    f["dv20"] = dv20
    f["c"] = g["c"]
    return f


def chart_features(rows):
    """rows: one symbol's daily bars with date, ao, ah, al, ac, av, c, v (any order). Returns one row per date
    with FEATURES plus dv20 and raw close c, and the segment id."""
    rows = rows[rows["v"] > 0].drop_duplicates("date").sort_values("date")
    if rows.empty:
        return pd.DataFrame(columns=["date", "seg", *FEATURES, "dv20", "c"])
    rows = rows.assign(seg=segments(rows["date"]))
    out = []
    for sg, g in rows.groupby("seg", sort=True):
        f = _one(g.set_index("date"))
        out.append(f.assign(seg=sg))
    f = pd.concat(out).rename_axis("date").reset_index()
    num = f.columns.difference(["date", "seg"])
    f[num] = f[num].replace([np.inf, -np.inf], np.nan).astype("float32")
    return f


def market_features(spy):
    """SPY rows (date, ac) -> date, spy_r1, spy_r5, spy_d200, spy_vol20 at that close."""
    s = spy.drop_duplicates("date").sort_values("date").set_index("date")["ac"]
    lr = np.log(s).diff()
    return pd.DataFrame({"date": s.index, "spy_r1": s.pct_change().to_numpy(), "spy_r5": s.pct_change(5).to_numpy(),
                         "spy_d200": (s / s.rolling(200).mean() - 1).to_numpy(),
                         "spy_vol20": lr.rolling(20).std().to_numpy()})
