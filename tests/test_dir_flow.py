"""DIR-FLOW / DIR-OFI helpers: synthetic data only."""
import numpy as np
import pandas as pd

from tracks.main import dir_flow as F

D = pd.Timestamp("2023-03-01")


def t(sec):
    return D + pd.Timedelta(hours=9) + pd.Timedelta(seconds=sec)


def test_clean_trades_drops_duplicates_and_excluded_conditions():
    tr = pd.DataFrame({"t": [t(1), t(2), t(2), t(3)], "p": [10.0, 10.1, 10.1, 10.2], "s": [100, 200, 200, 50],
                       "x": ["Q", "Q", "Q", "D"], "i": [1, 2, 2, 3], "c": [["@"], ["4"], ["4"], ["I"]]})
    c = F.clean_trades(tr)
    assert list(c["i"]) == [1, 3]


def test_lee_ready_uses_the_quote_one_second_earlier_then_the_tick_rule():
    q = pd.DataFrame({"t": [t(0), t(10)], "bp": [9.9, 10.4], "bs": [1, 1], "ap": [10.1, 10.6], "as": [1, 1]})
    tr = pd.DataFrame({"t": [t(5), t(6), t(10.5), t(12)], "p": [10.05, 9.95, 10.0, 10.5], "s": [1, 1, 1, 1]})
    s = F.lee_ready(tr, q)
    # 10.5s: the 10s quote is under 1s old, so the 0s quote (mid 10.0) applies; at the midpoint the tick rule says up.
    # 12s: the 10s quote applies (mid 10.5); at the midpoint, up from 10.0.
    assert list(s) == [1, -1, 1, 1]


def test_imbalance_and_ofi_signs():
    tr = pd.DataFrame({"s": [300.0, 100.0]})
    assert F.imbalance(tr, np.array([1, -1])) == 0.5
    up = pd.DataFrame({"t": [t(0), t(1), t(2)], "bp": [10.0, 10.1, 10.2], "bs": [5, 5, 5], "ap": [10.2, 10.3, 10.4],
                       "as": [5, 5, 5]})
    assert F.ofi(up) > 0
    dn = up.assign(bp=[10.2, 10.1, 10.0], ap=[10.4, 10.3, 10.2])
    assert F.ofi(dn) < 0 and np.isnan(F.ofi(up.iloc[:1]))


def test_good_quotes_and_pick_summary():
    q = pd.DataFrame({"t": [t(1800 + 60), t(1800 + 120), t(1800 + 180), t(1800 + 900)],
                      "bp": [10.0, 10.1, 10.2, 10.3], "bs": [5, 0, 5, 5], "ap": [10.2, 10.3, 10.2, 10.5], "as": [5, 5, 5, 5]})
    assert len(F.good_quotes(q)) == 2
    tr = pd.DataFrame({"t": [t(100), t(200)], "p": [10.0, 10.1], "s": [100.0, 20000.0], "x": ["Q", "D"], "i": [1, 2],
                       "c": [["T"], ["T", "I"]]})
    r = F.pick_summary(tr, q, D)
    assert r["n_trades"] == 2 and r["d_share"] > 0.99 and r["large_share"] > 0.99 and r["odd_lot_share"] == 0.5
    assert r["n_quotes"] == 2 and np.isfinite(r["ofi"])


def test_balanced_accuracy_and_bootstrap():
    assert F.balanced_accuracy(np.array([1, 1, -1, 0]), np.array([1, -1, -1, 1])) == 0.75
    days = np.repeat(np.arange(30), 5)
    assert F.boot_lower(days, np.ones(150)) == 1.0


def test_frames_read_mixed_timestamp_formats():
    d = F._frames([{"t": "2022-01-03T14:05:55.123456Z"}, {"t": "2022-01-03T14:05:56Z"}], ["t"])
    assert list(d["t"].dt.second) == [55, 56] and str(d["t"].iloc[0].hour) == "9"
