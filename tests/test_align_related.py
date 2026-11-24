import time

import numpy as np
import pandas as pd

from tracks.news.align_related import related


def make(n_days=300, start="2022-01-03", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    base = rng.normal(0, 0.02, n_days)
    df = pd.DataFrame({
        "AAA": base + rng.normal(0, 0.002, n_days),
        "BBB": base + rng.normal(0, 0.002, n_days),
        "RND": rng.normal(0, 0.02, n_days),
        "SIC": rng.normal(0, 0.02, n_days),
    }, index=idx)
    return df


def test_correlated_pair_related_random_not():
    df = make(600, "2021-06-01")
    out = related(df, {}, 2023, k=1)
    assert out["AAA"] == ["BBB"] and out["BBB"] == ["AAA"]
    assert "RND" not in out["AAA"]


def test_no_look_ahead():
    df = make(600, "2021-06-01")
    a = related(df, {"AAA": "283"}, 2023, k=2)
    df2 = df.copy()
    df2.loc[df2.index >= "2023-01-01"] = np.random.default_rng(9).normal(0, 1, df2.loc[df2.index >= "2023-01-01"].shape)
    assert related(df2, {"AAA": "283"}, 2023, k=2) == a


def test_never_self_and_sic_members_included():
    df = make(600, "2021-06-01")
    out = related(df, {"AAA": "283", "SIC": "283", "RND": None}, 2023, k=1)
    for s, v in out.items():
        assert s not in v
    assert "SIC" in out["AAA"] and "AAA" in out["SIC"]


def test_too_few_shared_days_skipped():
    df = make(600, "2021-06-01")
    df.loc[df.index < "2022-08-01", "BBB"] = np.nan   # only ~110 days before 2023
    out = related(df, {}, 2023, k=1, min_shared=200)
    assert "BBB" not in out["AAA"] and out["BBB"] == []


def test_no_data_gets_only_sic_members():
    df = make(600, "2021-06-01")
    df["NEW"] = np.nan
    out = related(df, {"NEW": "100", "RND": "100"}, 2023, k=3)
    assert out["NEW"] == ["RND"]


def test_ties_broken_by_name():
    idx = pd.bdate_range("2021-01-04", periods=300)
    x = np.random.default_rng(1).normal(0, 0.02, 300)
    df = pd.DataFrame({"ZZZ": x, "MMM": x, "AAA": x, "QQQ": x}, index=idx)
    assert related(df, {}, 2023, k=1)["QQQ"] == ["AAA"]
    assert related(df, {}, 2023, k=2)["AAA"] == ["MMM", "QQQ"]


def test_matches_pandas_corr():
    df = make(600, "2021-06-01")
    df.iloc[::7, 0] = np.nan
    from tracks.news.align_related import _pairwise_corr
    past = df.iloc[:250]
    mine = _pairwise_corr(past.to_numpy(), 50)
    ref = past.corr(min_periods=50).to_numpy()
    assert np.allclose(mine, ref, atol=1e-9)


def test_speed_2500_symbols():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.02, (300, 2500))
    x[rng.random(x.shape) < 0.05] = np.nan
    df = pd.DataFrame(x, index=pd.bdate_range("2021-06-01", periods=300), columns=[f"S{i:04d}" for i in range(2500)])
    t = time.time()
    related(df, {}, 2023)
    assert time.time() - t < 20
