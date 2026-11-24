import numpy as np

from tracks.tools import deflate as D


def test_expected_max_grows_with_tries():
    assert D.expected_max_z(1) == 0.0
    assert D.expected_max_z(10) < D.expected_max_z(40) < D.expected_max_z(190)
    assert 2.6 < D.expected_max_z(190) < 2.9          # about 2.76


def test_deflated_is_half_when_t_equals_the_benchmark():
    m = D.expected_max_z(40)
    assert np.isclose(D.deflated(m, 40), 0.5)
    assert D.deflated(m + 3, 40) > 0.99 and D.deflated(0.0, 40) < 0.02


def test_annual_sharpe_scales_with_days():
    assert np.isclose(D.annual_sharpe(2.0, 252), 2.0)
    assert D.annual_sharpe(2.0, 63) > D.annual_sharpe(2.0, 252)


def test_table_flags_weak_passes():
    t = D.table()
    flagged = {r["name"] for r in t["history"] if r["flag"]}
    assert "NX3-BARS" in flagged and "NXM-LGBM" not in flagged
