import numpy as np
import pandas as pd

from core import chart_features as C
from tracks.main import chart1_study as CH
from tracks.news import nx3_study as X


def bars(n=320, seed=0, start="2018-01-02"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h, l = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99
    v = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame({"date": d, "o": o, "h": h, "l": l, "c": c, "v": v, "ao": o, "ah": h, "al": l, "ac": c, "av": v})


def test_no_look_ahead():
    b = bars()
    f1 = C.chart_features(b).set_index("date")
    b2 = b.copy()
    b2.loc[b2.index[-20:], ["c", "ac", "h", "ah", "v", "av"]] *= 3.0
    f2 = C.chart_features(b2).set_index("date")
    d = b["date"].iloc[-21]
    pd.testing.assert_series_equal(f1.loc[d, C.FEATURES], f2.loc[d, C.FEATURES], check_names=False)
    assert f1.loc[b["date"].iloc[-1], C.FEATURES[:20]].notna().all()


def test_gap_ends_series():
    b = bars(120)
    b.loc[60:, "date"] = b.loc[60:, "date"] + pd.Timedelta(days=30)
    f = C.chart_features(b)
    assert f["seg"].nunique() == 2
    assert np.isnan(f.loc[f["date"] == b["date"].iloc[60], "r1"]).all()      # nothing computed across the gap


def test_scale_free_ratios():
    b = bars()
    b2 = b.copy()
    b2[["ao", "ah", "al", "ac"]] *= 0.5                                        # a later split rescales adjusted bars
    f1, f2 = C.chart_features(b), C.chart_features(b2)
    for k in ("rsi14", "macd_h", "bb_pctb", "hi52", "atr14", "rz_3"):
        np.testing.assert_allclose(f1[k], f2[k], rtol=1e-4, equal_nan=True)


def test_chart_next_only_next_session():
    b = bars(10)
    f = C.chart_features(b).merge(b[["date", "o", "ao", "ac"]], on="date")
    cal = pd.DatetimeIndex(b["date"])
    g = X.chart_next(f.drop(index=5).reset_index(drop=True), cal)
    row4 = g[g["date"] == b["date"].iloc[4]].iloc[0]
    assert pd.isna(row4["nxt"]) and np.isnan(row4["ac1"])
    row3 = g[g["date"] == b["date"].iloc[3]].iloc[0]
    assert row3["nxt"] == b["date"].iloc[4] and np.isclose(row3["ac1"], b["ac"].iloc[4], rtol=1e-5)


def test_day_picks_sides_and_control():
    day = pd.DataFrame({"ret": [0.05, 0.04, 0.03, -0.03, -0.04, -0.05, 0.0, 0.0], "o1": [20.0] * 8})
    s = day["ret"].to_numpy()                      # a perfect score
    p, c, (lo, sh) = CH.day_picks(day, s, np.random.default_rng(0))
    assert lo > 0.03 and sh > 0.03 and p > c
    assert CH.day_picks(day.iloc[:4], s[:4], np.random.default_rng(0)) is None


def test_learners_fit():
    rng = np.random.default_rng(1)
    for a in ("RIDGE", "SEQ", "ET", "LGBM", "CAT"):
        n = len(CH.arm_inputs(a))
        Xm = rng.normal(size=(600, n)).astype(np.float32)
        Xm[::7, 0] = np.nan
        y = Xm[:, 1] * 0.5 + rng.normal(size=600)
        y = np.nan_to_num(y)
        m = CH.learner(a).fit(Xm, y)
        assert m.predict(Xm).shape == (600,)


def test_ta_inputs_disjoint_from_nx3():
    ta = X.ta_inputs()
    assert not set(ta) & set(X.INPUTS)
    assert "rsi14" in ta and "hi52" in ta


def test_chart1_run_end_to_end(tmp_path, monkeypatch):
    """Synthetic chart files (one stock trends with its 5-day reversal) through CHART1's walk-forward."""
    cache = tmp_path / "nx1"
    d = X.nx3dir(cache) / X.CHART
    d.mkdir(parents=True)
    days = pd.bdate_range("2016-01-04", "2020-12-31")
    pd.DataFrame({"date": days, "open": "09:30", "close": "16:00"}).to_parquet(N_dir(cache) / "cal.parquet")
    cal = pd.DatetimeIndex(days)
    spy = bars(len(days), 99, "2016-01-04").assign(seg=0)
    spy = X.chart_next(C.market_features(spy).merge(spy, on="date"), cal)
    spy.to_parquet(d / "spy.parquet")
    parts = []
    for i in range(12):
        b = bars(len(days), i, "2016-01-04")
        b[["c", "o"]] *= 5
        b["v"] *= 10
        x = C.chart_features(b).merge(b[["date", "o", "ao", "ac"]], on="date")
        parts.append(X.chart_next(x, cal).assign(symbol=f"S{i}"))
    pd.concat(parts, ignore_index=True).to_parquet(d / "c_0000.parquet")
    monkeypatch.setattr(CH.N, "TEST_YEARS", range(2019, 2021))
    monkeypatch.setattr(CH, "MIN_DV", 1.0)
    rep = CH.run(cache, tmp_path / "out", arms=("RIDGE", "REV"))
    assert set(rep["arms"]) == {"RIDGE", "REV"} and rep["arms"]["REV"]["pick_minus_control"]["days"] > 300
    assert (tmp_path / "out" / "CHART1_walkforward.json").exists()


def N_dir(cache):
    p = cache / "nx1"
    p.mkdir(parents=True, exist_ok=True)
    return p
