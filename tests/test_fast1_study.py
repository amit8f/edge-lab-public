import numpy as np
import pandas as pd

from tracks.news import fast1_study as F


def bars(o, h=None, l=None, c=None):
    t = pd.date_range("2024-03-01 09:30", periods=len(o), freq="1min")
    return pd.DataFrame({"t": t, "o": o, "h": h or [x * 1.001 for x in o], "l": l or [x * 0.999 for x in o],
                         "c": c or o})


def test_time_exits_and_target():
    b = bars([100.0] * 5 + [101.0, 102.0, 103.5] + [103.0] * 100)
    x = F.exits(b, pd.Timestamp("2024-03-01 09:31"), 1)
    assert x["price"] == 100.0
    assert abs(x["t30"] - 0.03) < 1e-9 and abs(x["t60"] - 0.03) < 1e-9
    assert x["b200"] == 0.02 and x["b300"] == 0.03


def test_stop_gap_fills_at_open_and_both_is_stop():
    o = [100.0, 100.0, 97.0] + [97.0] * 70
    x = F.exits(bars(o), pd.Timestamp("2024-03-01 09:30"), 1)
    assert abs(x["b200"] - (-0.03)) < 1e-9          # gapped past the 1% stop: filled at the open
    h = [100.0, 104.0] + [100.0] * 70
    l = [100.0, 98.0] + [100.0] * 70
    y = F.exits(bars([100.0] * 72, h=h, l=l), pd.Timestamp("2024-03-01 09:30"), 1)
    assert y["b200"] == -0.01 and y["b300"] == -0.015


def test_short_side_and_missing_entry():
    b = bars([100.0] * 3 + [97.0] * 80)
    x = F.exits(b, pd.Timestamp("2024-03-01 09:30"), -1)
    assert x["b300"] == 0.03 and x["t60"] > 0
    assert F.exits(b, pd.Timestamp("2024-03-01 12:00"), 1) is None


def test_choose_and_cluster_t():
    t = pd.DataFrame({k: np.r_[np.full(300, v)] for k, v in
                      (("net_t30", 0.001), ("net_t60", 0.002), ("net_b200", -0.001), ("net_b300", 0.0))})
    best, rep = F.choose(t)
    assert best == "t60" and rep["t60"]["n"] == 300
    assert F.choose(t.head(10))[0] is None
    m, tt = F.cluster_t([0.01, 0.02, 0.0, 0.01], [1, 1, 2, 2])
    assert abs(m - 0.01) < 1e-12 and tt > 0
