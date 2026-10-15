import io
import zipfile

import numpy as np
import pandas as pd

from tracks.main import m3_screen as m3
from tracks.main.longterm.prices import segment_ids


def _zip(text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("F-F_Research_Data_Factors.CSV", text)
    return b.getvalue()


def test_parse_ff_monthly_only():
    text = ("This file was created ...\n\n,Mkt-RF,SMB,HML,RF\n192607,    2.96,   -2.56,   -2.43,    0.22\n"
            "192608,    2.64,   -1.17,    3.82,    0.25\n\n Annual Factors: January-December \n"
            ",Mkt-RF,SMB,HML,RF\n  1927,   29.47,   -2.46,   -3.75,    3.12\n")
    ff = m3.parse_ff(_zip(text))
    assert list(ff.index.astype(str)) == ["1926-07", "1926-08"]
    assert np.isclose(ff.loc[pd.Period("1926-07", "M"), "mkt"], 0.0296)
    assert np.isclose(ff.loc[pd.Period("1926-08", "M"), "rf"], 0.0025)


def _frame(idx, c):
    return pd.DataFrame({"o": c, "c": c}, index=pd.DatetimeIndex(idx))


def test_monthly_returns_break_and_missing_month():
    idx = pd.bdate_range("2020-01-02", "2020-08-31")
    idx = idx[(idx.month != 3) & ~((idx >= "2020-06-20") & (idx <= "2020-07-10"))]  # March gone; a 20-day gap
    d = _frame(idx, np.ones(len(idx)))
    me = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).max()
    c = d["c"].to_numpy()
    for k, v in zip(me.to_numpy(), [10.0, 11.0, 12.0, 13.2, 14.52, 20.0, 22.0]):
        c[k] = v
    d["c"] = c
    r = m3.monthly_returns(d, segment_ids(d.index))
    assert np.isclose(r[pd.Period("2020-02", "M")], 0.1)
    assert np.isnan(r[pd.Period("2020-04", "M")])          # March has no bar (and a >15-day gap)
    assert np.isclose(r[pd.Period("2020-05", "M")], 0.1)
    assert np.isclose(r[pd.Period("2020-06", "M")], 0.1)
    assert np.isnan(r[pd.Period("2020-07", "M")])          # series break inside July
    assert np.isclose(r[pd.Period("2020-08", "M")], 0.1)


def test_resid_score_recovers_residual_and_needs_window():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.04, (36, 3))
    e = rng.normal(0, 0.02, 36)
    y = 0.001 + X @ np.array([1.2, 0.3, -0.2]) + e
    s = m3.resid_score(y, X)
    A = np.c_[np.ones(36), X]
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    w = (y - A @ b)[-12:-1]
    assert np.isclose(s, w.sum() / w.std(ddof=1))
    y2 = y.copy()
    y2[:13] = np.nan                                       # 23 months left
    assert np.isnan(m3.resid_score(y2, X))
    y3 = y.copy()
    y3[-5] = np.nan                                        # a hole inside t-11..t-1
    assert np.isnan(m3.resid_score(y3, X))
    y4 = y.copy()
    y4[-1] = np.nan                                        # month t itself may be missing
    assert np.isfinite(m3.resid_score(y4, X))


def test_forward_entry_next_open_exit_and_series_end():
    idx = pd.bdate_range("2021-01-25", periods=10)
    d = pd.DataFrame({"o": np.arange(10.0) + 10, "c": np.arange(10.0) + 10.5}, index=idx)
    seg = np.zeros(10, dtype=int)
    assert np.isclose(m3.forward(d, seg, 2, idx[6]), d["c"].iloc[6] / d["o"].iloc[3] - 1)
    seg2 = np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    assert np.isclose(m3.forward(d, seg2, 2, idx[8]), d["c"].iloc[4] / d["o"].iloc[3] - 1)
    assert np.isnan(m3.forward(d, seg2, 4, idx[8]))       # no next bar in the series


def test_excess_picks_before_returns():
    t = pd.Timestamp("2020-01-31")
    df = pd.DataFrame({"date": t, "symbol": list("abcd"), "score": [4.0, 3.0, 2.0, 1.0],
                       "ret1": [np.nan, 0.10, 0.0, -0.02]})
    ex = m3.excess(df, "ret1", "score", 2)
    # top 2 are a (no return, dropped) and b; universe mean over names with a return
    assert np.isclose(ex["excess"].iloc[0], 0.10 - np.mean([0.10, 0.0, -0.02]) - m3.COST)
    assert ex["n"].iloc[0] == 1 and ex["dropped"].iloc[0] == 1


def test_replication_matches_a_momentum_world():
    rng = np.random.default_rng(7)
    months = pd.date_range("2016-01-31", periods=60, freq="ME")
    f, mk = rng.normal(0, 0.03, 60), rng.normal(0.01, 0.04, 60)
    rows = []
    for k, t in enumerate(months):
        mom = rng.normal(0, 1, 300)
        rows.append(pd.DataFrame({"date": t, "mom": mom, "mcap": rng.lognormal(22, 1, 300),
                                  "ret1": mk[k] + f[k] * np.sign(mom) + rng.normal(0, 0.001, 300)}))
    per = [pd.Period(t, "M") + 1 for t in months]
    ff5 = pd.DataFrame({"Mkt-RF": mk - 0.001, "RF": 0.001, "SMB": 0, "HML": 0, "RMW": 0, "CMA": 0}, index=per)
    umd = pd.DataFrame({"Mom": 2 * f}, index=per)
    rep = m3.replication(pd.concat(rows), ff5, umd)
    assert rep["months"] == 60 and rep["corr_umd"] > 0.99 and rep["corr_mkt"] > 0.95 and rep["passes"]
    rep_bad = m3.replication(pd.concat(rows), ff5, umd.assign(Mom=rng.normal(0, 0.03, 60)))
    assert not rep_bad["passes"]
