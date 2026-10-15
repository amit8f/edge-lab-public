"""core.stats on synthetic data (no network)."""
import io
import math
import zipfile

import numpy as np
import pandas as pd

from core import stats


def _zip(text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", text)
    return b.getvalue()


FF5_TEXT = ("This file was created using the 202508 CRSP database.\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
            "202001,  -0.11,  -3.11,  -6.25,  -1.18,  -2.42,   0.13\n202002,  -8.13,   0.97,  -3.80,  -1.62,  -2.48,   0.12\n"
            "\n Annual Factors: January-December \n,Mkt-RF,SMB,HML,RMW,CMA,RF\n  2020,  23.66,  12.96, -46.56,   4.46, -11.59,   0.45\n")
UMD_TEXT = "Monthly factors\n\n,Mom   \n202001,   2.01\n202002,   0.07\n\n Annual Factors: \n,Mom\n  2020,  10.0\n"


def test_parse_french_both_files():
    ff = stats.parse_french(_zip(FF5_TEXT))
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] and len(ff) == 2
    assert math.isclose(ff.loc[pd.Period("2020-02", "M"), "Mkt-RF"], -0.0813)
    umd = stats.parse_french(_zip(UMD_TEXT))
    assert list(umd.columns) == ["Mom"] and math.isclose(umd.iloc[0, 0], 0.0201)


def test_ols_hac_recovers_alpha_and_betas():
    rng = np.random.default_rng(1)
    F = rng.normal(0, 0.04, (600, 2))
    y = 0.01 + F @ np.array([1.0, -0.5]) + rng.normal(0, 0.01, 600)
    r = stats.ols_hac(y, F, 3)
    assert abs(r["alpha"] - 0.01) < 0.002 and abs(r["betas"][0] - 1) < 0.05 and r["t"] > 10
    # with no factors, alpha is the mean and the SE equals nw_tstat's
    from tracks.main.longterm.stats import nw_tstat
    m, se, _ = nw_tstat(y, 3)
    r0 = stats.ols_hac(y, np.empty((600, 0)), 3)
    assert math.isclose(r0["alpha"], m) and math.isclose(r0["se"], se, rel_tol=1e-9)


def test_factor_alpha_zero_for_pure_factor_exposure():
    months = pd.period_range("2015-01", periods=120, freq="M")
    rng = np.random.default_rng(2)
    ff5 = pd.DataFrame(rng.normal(0, 0.03, (120, 6)), index=months, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    umd = pd.DataFrame({"Mom": rng.normal(0, 0.04, 120)}, index=months)
    ret = 0.8 * ff5["SMB"] + 1.0 * umd["Mom"] + rng.normal(0, 0.002, 120)
    r = stats.factor_alpha(ret.to_numpy(), months, ff5, umd, 3)
    assert abs(r["t"]) < 3 and abs(r["betas"][1] - 0.8) < 0.05 and abs(r["betas"][5] - 1.0) < 0.05


def test_deflated_sharpe_falls_with_trials():
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.04, 116)
    one, many = stats.deflated_sharpe(x, 1), stats.deflated_sharpe(x, 28)
    assert one["sr0"] == 0 and many["sr0"] > 0 and many["dsr"] < one["dsr"]
    assert 0 <= many["dsr"] <= 1


def test_market_model_car_and_kp_adjustment():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, 260)
    r = 0.0002 + 1.5 * m + rng.normal(0, 0.01, 260)
    r[250:253] += 0.02
    car, scar, res = stats.market_model_car(r, m, slice(0, 240), slice(250, 253))
    assert 0.03 < car < 0.09 and scar > 1.5 and len(res) == 240
    t = stats.bmp_kp([1.0, 2.0, 1.5, 0.5, 1.2], 0.0)
    assert math.isclose(t["t_kp"], t["t_bmp"])
    assert stats.bmp_kp([1.0, 2.0, 1.5, 0.5, 1.2], 0.2)["t_kp"] < t["t_bmp"]
    same = np.tile(rng.normal(0, 1, 50), (3, 1))
    assert math.isclose(stats.mean_residual_corr(same), 1.0)


def test_fama_macbeth_slope():
    rng = np.random.default_rng(5)
    rows = []
    for d in range(60):
        x = rng.normal(0, 1, 200)
        rows.append(pd.DataFrame({"date": d, "x": x, "y": 0.02 * x + rng.normal(0, 0.05, 200)}))
    fm = stats.fama_macbeth(pd.concat(rows), "y", ["x"], 3)
    assert fm["n_dates"] == 60 and abs(fm["x"]["mean"] - 0.02) < 0.003 and fm["x"]["t"] > 10


def test_cost_estimators():
    # a stock whose true range is zero except for the bid-ask bounce: highs at the ask, lows at the bid
    mid, s = 10.0, 0.02
    h, lo = np.full(30, mid * (1 + s / 2)), np.full(30, mid * (1 - s / 2))
    assert np.allclose(stats.corwin_schultz(h, lo), s, rtol=0.05)
    assert math.isclose(stats.amihud([0.01, -0.02, np.nan], [1e6, 2e6, 1e6]), (1e-8 + 1e-8) / 2)
    assert np.allclose(stats.effective_spread([10.05], [10.0]), 0.01)
    assert math.isclose(stats.implementation_shortfall(10.0, 10.1, 1), 0.01)
    assert math.isclose(stats.implementation_shortfall(10.0, 9.9, -1), 0.01)


def test_parse_french_daily_and_counts(tmp_path):
    t = "Daily\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20200102,   0.86,  -0.97,  -0.33,  -0.18,  -0.21,   0.01\n20200103,  -0.67,   0.28,   0.00,  -0.07,  -0.14,   0.01\n"
    d = stats.parse_french(_zip(t))
    assert list(d.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    w = stats.window_factors(d, [pd.DatetimeIndex(["2020-01-02", "2020-01-03"])])
    assert np.isclose(w[0, 0], 0.0086 - 0.0067)
    f = tmp_path / "IDEAS.md"
    f.write_text("**Phase 2 count: 13** (main 9). Phase 1 tested 15 ideas; phase 1 + phase 2 = 28.\n")
    assert stats.idea_counts(f) == (13, 28)
