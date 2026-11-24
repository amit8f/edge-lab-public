import numpy as np
import pandas as pd

from adaptive import costs as AC
from tracks.news import cost1_study as K


def test_measured_net_swaps_the_flat_spread():
    flat = np.array([0.006, 0.006])
    n = K.measured_net([0.01, 0.01], flat, [20.0, 20.0], [0.001, np.nan], [0.0005, 0.0005])
    assert np.isclose(n[0], 0.01 - (0.006 - AC.SPREAD_ROUND_TRIP["main"] + 0.0015))
    assert np.isclose(n[1], 0.01 - 0.006)                 # missing entry quote: flat cost kept


def test_penny_uses_penny_flat_spread():
    n = K.measured_net([0.0], [0.01], [3.0], [0.002], [0.002])
    assert np.isclose(n[0], -(0.01 - AC.SPREAD_ROUND_TRIP["penny"] + 0.004))


def test_keep_mask():
    assert list(K.keep_mask([0.001, 0.02, np.nan], 0.005)) == [True, False, True]


def test_quote_at_falls_back_to_the_wider_window():
    calls = []

    def fetch(syms, start, end, headers):
        calls.append((end - start).total_seconds())
        t = (end - pd.Timedelta(seconds=1)).isoformat()
        if (end - start).total_seconds() == 2:
            return {"A": [{"t": t, "bp": 10.0, "ap": 10.02}]}
        return {"B": [{"t": t, "bp": 5.0, "ap": 5.05}]}

    got = K.quote_at(["A", "B"], pd.Timestamp("2024-03-01 10:00"), None, fetch=fetch)
    assert np.isclose(got["A"], 0.01 / 10.01) and np.isclose(got["B"], 0.025 / 5.025) and calls == [2, 60]
