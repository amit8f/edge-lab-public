import numpy as np

from dashboard import drift as D


def test_no_alarm_when_live_matches_history():
    rng = np.random.default_rng(0)
    hist = rng.normal(0.002, 0.03, 5000)
    r = D.drift(rng.normal(0.002, 0.03, 200), hist, sims=400)
    assert r["alarm"] is False and r["h"] > 0


def test_alarm_when_live_runs_well_below():
    rng = np.random.default_rng(1)
    hist = rng.normal(0.002, 0.03, 5000)
    r = D.drift(rng.normal(-0.02, 0.03, 200), hist, sims=400)
    assert r["alarm"] is True and r["live_mean"] < 0


def test_too_little_history():
    assert D.drift([0.01], [0.0] * 10) is None


def test_cusum_path_resets_at_zero():
    s = D.cusum_path([1.0, -1.0, -1.0], 0.0, 1.0, 0.5)
    assert np.allclose(s, [0.0, 0.5, 1.0])
