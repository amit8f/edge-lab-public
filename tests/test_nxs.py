"""NXS-ML on synthetic data (no network): its picks are NX3-ALL's, and the sizer stays within 5-10%."""
import numpy as np
import pandas as pd

from tests.test_nx3 import CAL, events
from tracks.news import nx3_study as X
from tracks.news import nxs_study as S


def test_pick_rows_reproduce_nx3_all_sized_days():
    ev = events(1.5, seed=1)
    rows = S.pick_rows(ev, CAL, years=[2019])
    days, pooled, _ = X.walk_forward(ev, CAL, years=[2019], cap=None)
    assert np.isclose(len(rows) / rows["d"].nunique(), days["2019"]["picks_per_day"])
    w = S.scaled(X.ALL_MIN + (X.ALL_MAX - X.ALL_MIN) * np.clip(rows["conf"] - 1, 0, 1), rows["d"])
    sized = (w * (rows["net_p"] - rows["net_c"])).groupby(rows["d"]).sum()
    assert np.isclose(sized.mean(), pooled["sized_pick_minus_control"]["mean"])


def test_rank_sizes_and_the_measure_on_a_known_signal():
    assert np.allclose(S.rank_sizes(np.array([-1.0, 0.5, 9.0]), np.array([0.0, 1.0])), [0.05, 0.075, 0.10])
    rng = np.random.default_rng(3)
    n = 6000
    rows = pd.DataFrame({k: rng.normal(size=n) for k in S.FEATURES})
    rows["year"] = np.repeat([2019, 2020, 2021, 2022], n // 4)
    rows["d"] = pd.Timestamp("2019-01-02") + pd.to_timedelta(np.arange(n) // 10, unit="D")
    rows["conf"] = 1 + rng.uniform(0, 1, n)
    rows["net_c"] = rng.normal(0, 0.01, n)
    rows["net_p"] = 0.006 * rows["rv30"] + rng.normal(0, 0.01, n)
    rep = S.measure(rows, first=2021)
    assert rep["rule21_spearman_outsample"] > 0.2 and rep["ml_minus_formula"]["mean"] > 0
    assert set(rep["by_year"]) == {"2021", "2022"}
