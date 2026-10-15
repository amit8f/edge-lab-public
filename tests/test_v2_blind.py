"""Synthetic-data tests for validation/v2_blind.py (no network)."""

import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from validation import v2_blind as vb

FF5_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def _daily(start="2015-01-01", end="2027-06-30", seed=1):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(len(days), 6)), index=days,
                      columns=vb.FACTORS)
    df["RF"] = 0.0001
    return df


def _zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


def _fake_zips(daily: pd.DataFrame) -> tuple[bytes, bytes]:
    pct = daily * 100.0
    lines = ["This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202608 CRSP database.",
             "The 1-month TBill return is from Ibbotson and Associates Inc.", "",
             "," + ",".join(FF5_COLS)]
    for d, r in pct.iterrows():
        lines.append(d.strftime("%Y%m%d") + "," + ",".join(f"{r[c]:.6f}" for c in FF5_COLS))
    lines += ["", "Copyright 2026 Eugene F. Fama and Kenneth R. French", ""]
    ff5 = _zip_bytes("F-F_Research_Data_5_Factors_2x3_daily.CSV", "\r\n".join(lines))

    mlines = ["This file was created by CMPT_ME_PRIOR_RETS_DAILY using the 202608 CRSP database.",
              "It contains a momentum factor, constructed from six value-weight portfolios formed using",
              "independent sorts on size and prior return of NYSE, AMEX, and NASDAQ stocks.", "",
              "Missing data are indicated by -99.99.", "", ",Mom   "]
    for d, v in pct["Mom"].items():
        mlines.append(f"{d.strftime('%Y%m%d')},{v:9.6f}")
    mlines += ["", "Copyright 2026 Eugene F. Fama and Kenneth R. French"]
    mom = _zip_bytes("F-F_Momentum_Factor_daily.CSV", "\r\n".join(mlines))
    return ff5, mom


def test_parse_fake_french_zips_roundtrip():
    daily = _daily("2020-01-01", "2020-03-31")
    ff5, mom = _fake_zips(daily)
    got = vb.load_french(ff5, mom)
    assert list(got.columns) == vb.FACTORS + ["RF"]
    assert len(got) == len(daily)
    np.testing.assert_allclose(got.to_numpy(), daily[vb.FACTORS + ["RF"]].to_numpy(), atol=1e-8)


def test_parse_drops_missing_codes_and_stops_at_next_section():
    text = ("header text\n\n,Mom\n20200102, 1.00\n20200103,-99.99\n20200106, 2.00\n\n"
            " Annual Factors: January-December\n,Mom\n2020, 5.0\n")
    df = vb.parse_french_csv_text(text)
    assert len(df) == 3
    assert df["Mom"].isna().sum() == 1
    assert df.loc["2020-01-06", "Mom"] == pytest.approx(0.02)


def test_window_starts_after_signal_and_spans_h_days():
    days = pd.bdate_range("2021-01-01", "2021-03-31")
    daily = pd.DataFrame({"A": np.arange(len(days), dtype=float)}, index=days)
    # signal on a Friday: entry is Monday
    W = vb.window_factor_sums(daily, ["2021-01-08"], h=3)
    assert W["entry"].iloc[0] == pd.Timestamp("2021-01-11")
    assert W["exit"].iloc[0] == pd.Timestamp("2021-01-13")
    p = days.get_loc(pd.Timestamp("2021-01-11"))
    assert W["A"].iloc[0] == pytest.approx(p + (p + 1) + (p + 2))
    # signal on a non-trading day (Saturday) uses the next trading day as entry
    W2 = vb.window_factor_sums(daily, ["2021-01-09"], h=1)
    assert W2["entry"].iloc[0] == pd.Timestamp("2021-01-11")
    # window past the end of the data is NaN
    W3 = vb.window_factor_sums(daily, ["2021-03-30"], h=5)
    assert np.isnan(W3["A"].iloc[0])
    # placebo shift moves the window 364 calendar days later
    W4 = vb.window_factor_sums(daily, ["2020-01-10"], h=1, shift_days=364)
    assert W4["entry"].iloc[0] == pd.Timestamp("2021-01-11")


def test_regression_recovers_known_loadings():
    daily = _daily()
    dates = pd.date_range("2017-01-31", "2024-12-31", freq="BME")
    W = vb.window_factor_sums(daily, dates, 126)
    beta = np.array([1.1, 0.6, -0.4, 0.3, -0.2, 0.5])
    rng = np.random.default_rng(7)
    y = 0.012 + W[vb.FACTORS].to_numpy() @ beta + rng.normal(0, 0.002, len(dates))
    r = vb.factor_regression(pd.Series(y, index=dates), W, lags=5)
    np.testing.assert_allclose([r.loadings[f] for f in vb.FACTORS], beta, atol=0.03)
    assert r.alpha == pytest.approx(0.012, abs=0.002)
    assert r.r2 > 0.95
    assert r.alpha_t > 3
    # placebo calendar: the fit collapses (overlapping windows keep some spurious noise)
    Wp = vb.window_factor_sums(daily, dates, 126, shift_days=364)
    rp = vb.factor_regression(pd.Series(y, index=dates), Wp, lags=5)
    assert abs(rp.loadings["Mkt-RF"]) < 0.6
    assert np.mean(np.abs(list(rp.loadings.values()))) < np.mean(np.abs(beta))
    assert rp.r2 < 0.5 * r.r2


def test_newey_west_zero_lags_is_white_and_lags_match_manual():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(50), rng.normal(size=50)])
    e = rng.normal(size=50)
    XtXi = np.linalg.inv(X.T @ X)
    white = XtXi @ (X.T * e**2) @ X @ XtXi
    np.testing.assert_allclose(vb.newey_west_cov(X, e, 0), white)
    # one-regressor mean case: NW variance of the mean
    z = rng.normal(size=40)
    d = z - z.mean()
    L = 2
    s = d @ d / 40 + sum(2 * (1 - l / (L + 1)) * (d[l:] @ d[:-l]) / 40 for l in range(1, L + 1))
    got = vb.nw_se_of_mean(z[:, None], L)[0]
    assert got == pytest.approx(math.sqrt(s / 40))
    cov = vb.newey_west_cov(np.ones((40, 1)), d, L)[0, 0]
    assert math.sqrt(cov) == pytest.approx(got)


def test_holm_known_example():
    got = vb.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert got == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.5})


def test_romano_wolf_detects_real_effect_only():
    rng = np.random.default_rng(11)
    n = 80
    Z = pd.DataFrame({"real": 0.02 + rng.normal(0, 0.03, n),
                      "null1": rng.normal(0, 0.03, n),
                      "null2": rng.normal(0, 0.03, n) - 0.005},
                     index=pd.period_range("2018-01", periods=n, freq="M"))
    r = vb.romano_wolf(Z, draws=2000)
    p = r["p_adjusted"]
    assert p["real"] < 0.01
    assert p["null1"] > 0.05 and p["null2"] > 0.05
    # stepdown p-values are monotone in t
    order = sorted(p, key=lambda k: -r["t"][k])
    assert all(p[a] <= p[b] for a, b in zip(order, order[1:]))
    # reproducible with the same seed
    assert vb.romano_wolf(Z, draws=2000)["p_adjusted"] == p


def test_deflated_sharpe_formula():
    sr0 = vb.expected_max_sr(0.25, 28)
    z1, z2 = stats.norm.ppf(1 - 1 / 28), stats.norm.ppf(1 - 1 / (28 * math.e))
    g = vb.EULER_GAMMA
    assert sr0 == pytest.approx(0.5 * ((1 - g) * z1 + g * z2))
    d = vb.deflated_sr(0.3, 0.1, 50, -0.5, 4.0)
    expect = stats.norm.cdf(0.2 * 7 / math.sqrt(1 + 0.15 + 0.75 * 0.09))
    assert d == pytest.approx(expect)
    assert vb.deflated_sr(0.5, 0.1, 50, 0, 3) > vb.deflated_sr(0.2, 0.1, 50, 0, 3)


def test_subsample_ir_offsets():
    x = np.arange(12, dtype=float)
    sr, T = vb.subsample_ir(x, 6)
    assert T == 2
    assert sr == pytest.approx(np.mean([np.mean(x[o::6]) / np.std(x[o::6], ddof=1)
                                        for o in range(6)]))


def _write_series(root, idea, dates, excess, cost, rng):
    uni = rng.normal(0.01, 0.03, len(dates))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "pick_mean": uni + excess + cost,
                       "uni_mean": uni, "excess": excess, "n_picks": 20, "n_universe": 1500})
    (root / idea).mkdir(parents=True)
    df.to_csv(root / idea / f"{idea}_primary_series.csv", index=False)


def test_end_to_end_run_offline(tmp_path):
    daily = _daily()
    ff5, mom = _fake_zips(daily)
    cache = tmp_path / "cache"
    (cache / "french_blind").mkdir(parents=True)
    (cache / "french_blind" / vb.FF5_FILE).write_bytes(ff5)
    (cache / "french_blind" / vb.MOM_FILE).write_bytes(mom)

    results = tmp_path / "results"
    rng = np.random.default_rng(5)
    mdates = pd.date_range("2017-01-31", "2024-12-31", freq="BME")
    Wm = vb.window_factor_sums(daily, mdates, 126)
    ex_m = 0.5 * Wm["SMB"].to_numpy() + rng.normal(0, 0.03, len(mdates))
    _write_series(results, "M4", mdates, ex_m, 0.002, rng)
    _write_series(results, "M5", mdates, rng.normal(0, 0.04, len(mdates)), 0.002, rng)
    wdates = pd.date_range("2019-01-04", "2024-12-27", freq="W-FRI")
    Ww = vb.window_factor_sums(daily, wdates, 21)
    ex_w = 0.03 + 1.0 * Ww["Mkt-RF"].to_numpy() + rng.normal(0, 0.01, len(wdates))
    _write_series(results, "M10", wdates, ex_w, 0.002, rng)

    out = tmp_path / "out"
    res = vb.run(cache, results_root=results, out_dir=out, download=False, draws=300)
    assert res["ideas"] == ["M4", "M5", "M10"]
    per = res["per_idea"]
    assert per["M4"]["primary"]["loadings"]["SMB"] == pytest.approx(0.5, abs=0.15)
    assert per["M10"]["primary"]["loadings"]["Mkt-RF"] == pytest.approx(1.0, abs=0.1)
    assert per["M10"]["survives"] and not per["M5"]["survives"]
    assert per["M4"]["dsr_inputs"]["subsample_step"] == 6
    assert per["M10"]["dsr_inputs"]["subsample_step"] == 4
    assert per["M10"]["nw_lags"] == 6 and per["M10"]["t_bar"] == 3.0
    assert per["M4"]["cost_per_window"] == pytest.approx(0.002)
    assert res["romano_wolf"]["n_months"] == 72  # 2019-01 .. 2024-12

    js = json.loads((out / "V2_blind_results.json").read_text())
    assert set(js["french_sha256"]) == {vb.FF5_FILE, vb.MOM_FILE}
    tab = pd.read_csv(out / "V2_blind_table.csv")
    for col in ["raw_mean_excess", "alpha", "alpha_t", "r2", "holm_p", "rw_p", "annual_ir",
                "dsr", "survives"] + [f"beta_{f}" for f in vb.FACTORS]:
        assert col in tab.columns
    assert (out / "V2_blind_placebo.csv").exists()
    assert (out / "V2_blind_window_factors.csv").exists()


def test_unknown_idea_is_an_error():
    daily = _daily("2018-01-01", "2020-12-31")
    s = pd.DataFrame({"excess": [0.0] * 10, "picks_net": [0.0] * 10},
                     index=pd.date_range("2018-01-31", periods=10, freq="BME"))
    with pytest.raises(ValueError):
        vb.run_study(daily, {"Z9": s}, draws=10)
