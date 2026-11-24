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


def test_read_keeps_features_as_float32_and_thins_dates(tmp_path):
    df = pd.DataFrame({"numerai_ticker": ["A US"] * 8, "date": np.repeat(["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"], 1),
                       "feature_a": np.arange(8, dtype=float), "feature_country": ["US"] * 8, "target": 0.5,
                       "target_x_60": 0.1})
    f = tmp_path / "t.parquet"
    df.to_parquet(f)
    got, feats = S.read(f, keep=("date", "target"))
    assert feats == ["feature_a"] and list(got.columns) == ["date", "feature_a", "target"]
    assert got["feature_a"].dtype == np.float32
    assert S.thin_dates(got)["date"].tolist() == ["d1", "d5"]
    thin, _ = S.read(f, keep=("date", "target"), every=4)
    assert thin["date"].tolist() == ["d1", "d5"] and thin["feature_a"].tolist() == [0.0, 4.0]
