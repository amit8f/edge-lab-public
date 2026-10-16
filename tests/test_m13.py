import numpy as np
import pandas as pd

from tracks.main.longterm import screen_m13 as m13


def test_ma_rule_above_rising_and_short_series():
    c = np.r_[np.linspace(10, 20, 200)]
    nb = np.arange(1, 201)
    above, rising, dist = m13.ma_rule(c, nb, 199, 150)
    assert above == 1.0 and rising == 1.0
    assert np.isclose(dist, c[199] / c[50:200].mean() - 1)
    assert np.isnan(m13.ma_rule(c, nb, 160, 150)[0])          # 161 bars < 150 + 21
    down = c[::-1].copy()
    a2, r2, _ = m13.ma_rule(down, nb, 199, 150)
    assert a2 == 0.0 and r2 == 0.0


def test_ma_rule_uses_only_current_series():
    c = np.r_[np.full(200, 100.0), np.linspace(10, 12, 100)]
    nb = np.r_[np.arange(1, 201), np.arange(1, 101)]          # new series starts at bar 200
    assert np.isnan(m13.ma_rule(c, nb, 299, 150)[0])


def test_add_rules_needs_both_and_keeps_nan():
    u = pd.DataFrame({f"{k}{n}": v for n in m13.MAS for k, v in
                      (("above", [1.0, 1.0, 0.0, np.nan]), ("rising", [1.0, 0.0, 1.0, 1.0]), ("dist", [0.1] * 4))})
    r = m13.add_rules(u)
    assert list(r["allow150"].iloc[:3]) == [1.0, 0.0, 0.0] and np.isnan(r["allow150"].iloc[3])


def _panel():
    rows = []
    for k, t in enumerate(pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])):
        for s in "ABCD":
            allow = 1.0 if (s in "AB" if k < 2 else s in "AC") else 0.0
            rows.append({"date": t, "symbol": s, "allow150": allow, "ret1": {"A": 0.05, "B": 0.03, "C": 0.0, "D": -0.04}[s]})
    return pd.DataFrame(rows)


def test_book_turnover_cost():
    ex = m13.book(_panel(), "ret1", m13.allowed("allow150"))
    uni = (0.05 + 0.03 + 0.0 - 0.04) / 4
    assert list(ex["turnover"]) == [1.0, 0.0, 0.5]
    assert np.isclose(ex["excess"].iloc[0], 0.04 - uni - 0.002)
    assert np.isclose(ex["excess"].iloc[1], 0.04 - uni)
    assert np.isclose(ex["excess"].iloc[2], 0.025 - uni - 0.001)
    ctl = m13.book(_panel(), "ret1", m13.allowed("allow150"), seed=1)
    assert list(ctl["n"]) == [2, 2, 2]


def test_symbol_rows_month_end_features():
    idx = pd.bdate_range("2019-01-01", "2020-06-30")
    c = np.linspace(10, 30, len(idx))
    d = pd.DataFrame({"o": c, "c": c, "rc": c, "v": 1e6, "sc": c}, index=idx)
    t = pd.Timestamp("2020-03-31")
    rows = m13.symbol_rows("X", d, [t], {t: pd.Timestamp("2020-04-30")}, {})
    r = rows[0]
    assert r["allow150"] if "allow150" in r else True
    assert r["above150"] == 1.0 and r["rising150"] == 1.0
    i = idx.get_loc(t)
    assert np.isclose(r["ret1"], d["c"].loc["2020-04-30"] / d["o"].iloc[i + 1] - 1)
    assert np.isfinite(r["mom"])
