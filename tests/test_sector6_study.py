import numpy as np
import pandas as pd
import pytest

from tracks.main import sector6_study as S

ET = "America/New_York"


def test_clock_prices_and_at():
    t = [pd.Timestamp(f"2023-03-10 {h}", tz=ET).tz_convert("UTC") for h in ("08:25", "08:55", "15:55")]
    px = S.clock_prices(pd.DataFrame({"t": t, "c": [100.0, 101.0, 103.0]}))
    d = pd.Timestamp("2023-03-10")
    assert S.at(px, 510)[d] == 100.0 and S.at(px, 540)[d] == 101.0 and S.at(px, 960)[d] == 103.0
    assert S.at(px, 545)[d] == 101.0          # falls back to the last price within 15 minutes


def test_follow_stats_finds_follow_and_fade():
    rng = np.random.default_rng(0)
    s = rng.normal(size=60)
    y = 0.01 * np.sign(s) + rng.normal(scale=0.002, size=60)
    r = S.follow_stats(s[:30], y[:30], s[30:], y[30:], 0.0002)
    assert r["dir"] == "follow" and r["hit_test"] > 0.9
    r = S.follow_stats(s[:30], -y[:30], s[30:], -y[30:], 0.0002)
    assert r["dir"] == "fade" and r["hit_test"] > 0.9


def test_groups_fall_back_to_ff12_for_small_industries():
    date = pd.Series(pd.to_datetime(["2023-01-03"] * 8))
    sic = pd.Series([3571] * 6 + [2834, pd.NA], dtype="Int64")
    ff = pd.Series([6] * 6 + [10, 12])
    g = S.groups(sic, ff, date)
    assert list(g[:6]) == ["s357"] * 6 and g[6] == "f10" and g[7] == "f12"


def test_leaders_leave_the_stock_out():
    d = pd.DataFrame({"date": pd.to_datetime(["2023-01-03"] * 5), "grp": ["a"] * 5,
                      "pdv": [5, 4, 3, 2, 1], "ret": [0.03, 0.01, 0.02, -0.01, 0.0]})
    x = S.leaders(d)
    assert list(x["is_lead"]) == [True, True, True, False, False]
    assert x.loc[3, "lead"] == pytest.approx(0.02) and x.loc[0, "lead"] == pytest.approx(0.015)


def test_forward_stays_in_series():
    d = pd.DataFrame({"symbol": ["A"] * 3 + ["B"] * 2, "seg": [0] * 5, "ac": [10.0, 11.0, 12.1, 5.0, 6.0]})
    f = S.forward(d, 1)
    assert f[0] == pytest.approx(0.1) and np.isnan(f[2]) and f[3] == pytest.approx(0.2)
