import numpy as np
import pandas as pd

from adaptive import costs as AC
from tracks.news import exec1_study as E


def test_bar_open_at_takes_the_first_bar_within_ten_minutes():
    b = pd.DataFrame({"t": pd.to_datetime(["2024-03-01 09:59", "2024-03-01 10:03"]), "o": [10.0, 11.0], "c": [10, 11], "v": [1, 1]})
    assert E.bar_open_at(b, pd.Timestamp("2024-03-01 10:00")) == 11.0
    assert np.isnan(E.bar_open_at(b, pd.Timestamp("2024-03-01 10:20")))
    assert np.isnan(E.bar_open_at(None, pd.Timestamp("2024-03-01 10:00")))


def test_delayed_moves_only_trades_entering_before_t():
    d = pd.to_datetime(["2024-03-01"] * 3)
    et = pd.to_datetime(["2024-03-01 09:31", "2024-03-01 10:15", "2024-03-01 09:40"])
    r2, h2, mv = E.delayed(et, [10.0, 10.0, 10.0], [0.10, 0.10, 0.10], [0.004, 0.004, 0.004], d,
                           [11.0, 12.0, np.nan], [0.001, 0.001, 0.001], "10:00")
    assert list(mv) == [True, False, False]                 # 3rd: no bar at T, keeps its entry
    assert np.isclose(r2[0], 11.0 / 11.0 - 1) and np.isclose(r2[1], 0.10) and np.isclose(r2[2], 0.10)
    assert list(h2) == [0.001, 0.004, 0.004]


def test_net_moc_charges_no_exit_spread():
    n = E.net_moc([1], [0.02], [0.006], [20.0], [0.001])
    assert np.isclose(n[0], 0.02 - (0.006 - AC.SPREAD_ROUND_TRIP["main"] + 0.001))
    s = E.net_moc([-1], [-0.02], [0.006], [20.0], [np.nan])   # no entry quote: flat cost kept
    assert np.isclose(s[0], 0.02 - 0.006)
