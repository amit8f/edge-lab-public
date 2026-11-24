import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

from tracks.main import anom1_study as A


def test_c_n_known_values():
    assert A.c_n([1])[0] == 0 and A.c_n([2])[0] == 1.0
    assert abs(A.c_n([256])[0] - 10.24) < 0.05


def test_iforest_matches_sklearn_ranking():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(60, 5))
    pts = np.vstack([rng.normal(size=(30, 5)), rng.normal(scale=4, size=(10, 5))])
    ours = [A.iforest_score(train, p, np.random.default_rng(1), trees=400) for p in pts]
    sk = -IsolationForest(n_estimators=400, random_state=1).fit(train).score_samples(pts)
    assert spearmanr(ours, sk).correlation > 0.9
    assert np.mean(ours[30:]) > np.mean(ours[:30])          # far points score higher
    assert abs(np.mean(ours) - np.mean(sk)) < 0.05           # same scale (sklearn's score_samples is -score)


def test_cusum_alarm():
    hist = np.r_[np.zeros(30), np.ones(30)]                  # mean 0.5, sd 0.5
    assert not A.cusum_alarm(hist, [0.5] * 5)
    assert A.cusum_alarm(hist, [2.0, 2.0, 2.0])              # z = 3 a day: 2.5 + 2.5 > 4
    assert A.cusum_alarm(hist, [-1.0, -1.0, -1.0])
    assert not A.cusum_alarm(np.ones(60), [9.0])             # no spread: no alarm


def test_combined_top_reranks():
    g = pd.DataFrame({"symbol": list("abcd"), "score": [4, 3, 2, 1], "anom": [0, 0, 0, 9], "boom": [0, 0, 0, 1]})
    assert list(A.combined_top(g, 0.0, top=2)["symbol"]) == ["a", "b"]
    assert set(A.combined_top(g, 1.0, top=3)["symbol"]) == {"a", "b", "d"}
