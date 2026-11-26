import numpy as np
import pandas as pd
import pytest

from tracks.main import sector4_study as S


def bars(day, pre=None, closes=None):
    """5-minute bars for one ET day: an optional pre-market bar at 09:00 and regular bars from 09:30."""
    rows = []
    if pre is not None:
        rows.append((pd.Timestamp(f"{day} 09:00", tz="America/New_York"), pre, pre))
    t0 = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    prev = closes[0]
    for i, c in enumerate(closes):
        rows.append((t0 + pd.Timedelta(minutes=5 * i), prev, c))
        prev = c
    return pd.DataFrame([(t.tz_convert("UTC"), o, c) for t, o, c in rows], columns=["t", "o", "c"])


def test_daily_points_reads_the_right_bars():
    closes = list(np.linspace(100, 110, 78))           # 09:30 ... 15:55
    p = S.daily_points(bars("2023-03-01", pre=99.0, closes=closes))
    r = p.iloc[0]
    assert r["pre"] == 99.0 and r["open"] == closes[0]
    assert r["p1000"] == pytest.approx(closes[5]) and r["p1530"] == pytest.approx(closes[71])
    assert r["close"] == pytest.approx(closes[77]) and r["rv"] > 0


def test_returns_use_the_previous_close():
    p = pd.DataFrame({"date": pd.to_datetime(["2023-03-01", "2023-03-02"]), "pre": [np.nan, 102.0], "open": [100.0, 103.0],
                      "p1000": [100.0, 104.0], "p1530": [100.0, 105.0], "close": [100.0, 106.0], "rv": [0.01, 0.02]})
    r = S.returns(p).iloc[1]
    assert r["r"] == pytest.approx(0.06) and r["pm"] == pytest.approx(0.02) and r["on"] == pytest.approx(0.03)
    assert r["l30"] == pytest.approx(106 / 105 - 1) and r["rest"] == pytest.approx(106 / 104 - 1)


def test_direction_score_and_cost():
    y = np.array([0.01, -0.01, 0.02, -0.02] * 10)
    s = S.score_direction(y, y)
    assert s["hit_rate"] == 1.0 and s["oos_r2"] == pytest.approx(1.0)
    assert s["net_per_day_cost_0.02pct"] == pytest.approx(0.015 - 0.0002)


def test_ridge_oos_learns_a_real_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = 2 * X[:, 0] + rng.normal(size=600) * 0.1
    p, _ = S.ridge_oos(X[:400], y[:400], X[400:])
    assert np.corrcoef(p, y[400:])[0, 1] > 0.95
