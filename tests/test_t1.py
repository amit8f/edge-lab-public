"""T1 turn-of-the-year screen on synthetic bars (no network)."""
import numpy as np
import pandas as pd
import pytest

from core import stats
from tracks.penny import t1_screen as T

TDAYS = list(pd.bdate_range("2015-12-01", "2025-01-31"))


def frame(px=3.0, vol=400_000, drop=None):
    d = pd.DataFrame({"o": px, "h": px, "l": px, "c": px, "v": vol}, index=pd.DatetimeIndex(TDAYS))
    return d.drop(drop) if drop is not None else d


def test_year_dates():
    P, F, E, X = T.year_dates(TDAYS, 2020)
    assert P == pd.Timestamp("2019-12-31") and E == pd.Timestamp("2020-12-31")
    assert F == pd.Timestamp("2020-12-25") and X == pd.Timestamp("2021-01-07")   # weekdays only here


def test_symbol_years_ytd_entry_exit_and_gap():
    raw = frame()
    adj = raw.copy()
    adj.loc["2020-01-01":"2020-12-25", "c"] = np.linspace(3.0, 1.5, len(adj.loc["2020-01-01":"2020-12-25"]))
    adj.loc["2020-12-31", "o"] = 1.5
    adj.loc["2021-01-07", "c"] = 1.8
    rows = {r["year"]: r for r in T.symbol_years("AAA", raw, adj, TDAYS)}
    r = rows[2020]
    assert r["group"] == "penny" and r["ytd"] == pytest.approx(1.5 / 3.0 - 1)
    assert r["gross"] == pytest.approx(0.2) and r["exit_reason"] == "exit day"
    # a 20-day gap in March 2019 leaves fewer than 200 bars of 2019 in the series: no 2019 row
    gap = pd.bdate_range("2019-03-01", "2019-03-28")
    assert 2019 not in {r["year"] for r in T.symbol_years("BBB", frame(drop=gap), frame(drop=gap), TDAYS)}
    # stops trading on 2021-01-04: exit at that last close
    stop = pd.bdate_range("2021-01-05", "2025-01-31")
    r = {x["year"]: x for x in T.symbol_years("CCC", frame(drop=stop), frame(drop=stop), TDAYS)}[2020]
    assert r["exit_day"] == "2021-01-04" and r["exit_reason"].startswith("last close")


def test_pick_yearly_primary():
    rows = []
    for y in T.YEARS:
        for i in range(50):
            loser = i < 5
            rows.append({"year": y, "symbol": f"S{i:02d}", "group": "penny", "close": 2.0, "dv20": 1e6,
                         "ytd": -0.9 + i / 100, "gross": 0.10 if loser else 0.01, "exit_day": None, "exit_reason": ""})
    sel = T.pick(pd.DataFrame(rows), "penny")
    assert (sel.groupby(["year", "role"]).size() == 5).all()
    assert set(sel.loc[sel["role"] == "loser", "symbol"]) == {f"S{i:02d}" for i in range(5)}
    t = T.yearly(sel)
    assert t["diff"].tolist() == pytest.approx([0.09] * 9)
    p = T.primary(t)
    assert p["years"] == 9 and p["mean"] == pytest.approx(0.09) and p["ci95"][0] == pytest.approx(0.09)


def test_ols_cluster_recovers_coefficients():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 20)
    F = rng.standard_normal((600, 2))
    y = 0.5 + F @ [1.0, -2.0] + rng.standard_normal(600) * 0.1
    r = stats.ols_cluster(y, F, g)
    assert r["alpha"] == pytest.approx(0.5, abs=0.02) and r["betas"] == pytest.approx([1.0, -2.0], abs=0.02)
    assert r["clusters"] == 30 and r["t"] > 50


def test_alpha_runs_on_window_factors():
    rng = np.random.default_rng(1)
    rows = []
    for y in T.YEARS:
        for i in range(50):
            rows.append({"year": y, "symbol": f"S{i:02d}", "group": "penny", "close": 2.0, "dv20": 1e6,
                         "ytd": -0.9 + i / 100, "gross": 0.05 + rng.normal(0, 0.02), "exit_day": None, "exit_reason": ""})
    sel = T.pick(pd.DataFrame(rows), "penny")
    t = T.yearly(sel)
    cal = pd.DatetimeIndex(TDAYS)
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(cal), 6)), index=cal,
                       columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    a = T.alpha(sel, t, fac, TDAYS)
    assert a["clusters"] == 9 and a["n"] == 45 and np.isfinite(a["alpha"])
