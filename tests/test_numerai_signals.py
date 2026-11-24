import numpy as np
import pandas as pd

from tracks.numerai import signals as S


def test_rank01_open_interval_and_ties():
    r = S.rank01([3.0, 1.0, 2.0, 2.0])
    assert r.min() > 0 and r.max() < 1 and r[2] == r[3] and r[0] == r.max()


def test_features_skip_country():
    assert S.features(["numerai_ticker", "feature_country", "feature_a", "target"]) == ["feature_a"]


def test_date_corr_perfect_and_summary():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": np.repeat(["2020-01-03", "2021-01-08"], 100), "target": rng.random(200)})
    df["p"] = df["target"]
    s = S.date_corr(df, "p")
    assert np.allclose(s.values, 1.0)
    by = S.summary(s)["by_year"]
    assert set(by) == {"2020", "2021"} and all(abs(v - 1) < 1e-9 for v in by.values())
