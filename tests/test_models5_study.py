import numpy as np
import pandas as pd

from tracks.news import models5_study as M


def test_vol_weights_keep_gross_and_tilt_to_calm():
    w = M.vol_weights([0.1, 0.1], [0.01, 0.04], target=0.02)
    assert np.isclose(w.sum(), 0.2) and w[0] > w[1]


def test_vol_weights_clip_and_missing():
    w = M.vol_weights([0.1, 0.1, 0.1], [1e-6, np.nan, 0.02], target=0.02)
    assert np.isclose(w.sum(), 0.3)
    assert w[0] / w[1] == M.CLIP[1]          # calm stock capped at 2x; missing sigma keeps base


def test_storm_weights_cap_side_and_halve_gross():
    w = M.storm_weights([0.1, 0.1, 0.1, 0.05], [-1, -1, -1, 1], [3.0, 2.0, 1.6, 1.5], storm=True)
    assert np.isclose(w.sum(), 0.5 * 0.35)
    short = w[:3].sum() / w.sum()
    assert short <= M.SIDE_CAP + 1e-9
    assert w[0] > 0 and w[2] == 0            # highest confidence kept first


def test_storm_weights_calm_day_unchanged():
    assert np.allclose(M.storm_weights([0.1, 0.2], [1, -1], [2, 2], storm=False), [0.1, 0.2])


def test_forward_filter_uses_no_future_rows():
    from hmmlearn.hmm import GaussianHMM
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 0.5, (200, 2)), rng.normal(3, 0.5, (200, 2))]
    m = GaussianHMM(2, covariance_type="full", n_iter=50, random_state=7).fit(x)
    a = M.forward_filter(m, x)
    x2 = x.copy()
    x2[300:] = -x2[300:]
    b = M.forward_filter(m, x2)
    assert np.allclose(a[:300], b[:300]) and np.allclose(a.sum(1), 1)


def test_garch_sigma_tracks_volatility():
    rng = np.random.default_rng(2)
    calm, wild = M.garch_sigma(rng.normal(0, 0.01, 400)), M.garch_sigma(rng.normal(0, 0.04, 400))
    assert 0.005 < calm < 0.02 and 0.02 < wild < 0.08


def test_boot_sharpe_diff_sign():
    rng = np.random.default_rng(3)
    n = 400
    b = rng.normal(0, 0.01, n)
    a = b + 0.002
    month = np.repeat(np.arange(20), 20)
    d, lo, hi = M.boot_sharpe_diff(a, b, month, n=200)
    assert d > 0 and lo > 0


def test_surv_path_and_exit():
    path = M.surv_path(0.01, [10.0, 10.2, 10.4])
    assert np.isclose(path[0], 0.01) and np.isclose(path[2], 1.01 * 1.04 - 1)
    sig = np.array([0.01, 0.02, 0.035, 0.0, 0.0])
    assert M.surv_exit(sig, np.full(5, 0.9), 0.4) == 2      # target reached at close 3
    assert M.surv_exit(sig, np.full(5, 0.1), 0.4) == 0      # low chance: out at NX3's normal close
    assert M.surv_exit(np.zeros(5), np.full(5, 0.9), 0.4) == 4
