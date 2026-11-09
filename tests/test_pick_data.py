"""PICK data rows (synthetic bars only, no network)."""
import numpy as np
import pandas as pd

from tracks.main import pick_data as P

DAY = pd.Timestamp("2024-03-04")


def bars(closes, start="09:30", hi=None, lo=None):
    t = pd.date_range(f"{DAY.date()} {start}", periods=len(closes), freq="5min")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) if hi is None else hi, "l": np.minimum(o, c) if lo is None else lo,
                         "c": c, "v": 1000.0})


def test_universe_splits_main_and_penny_from_the_previous_session():
    d = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "c": [20.0, 3.0, 20.0, 0.9], "dv20": [6e6, 2e6, 4e6, 5e6]})
    u = P.universe(d)
    assert u == {"A": ("main", 20.0, 6e6), "B": ("penny", 3.0, 2e6)}


def test_trade_takes_profit_stops_or_sells_at_1550():
    up = bars([10] + [10.2] * 3 + [11.0] * 74)                     # entry 10.2 (bar 4's open); 11.0 is +7.8%, short of +8%
    net, kind = P.trade(up, 4, 0.08, 0.05, 0.002)
    assert kind == "close" and abs(net - (11.0 / 10.2 - 1 - 0.002)) < 1e-9
    b = bars([10.0] * 10 + [11.0] * 68)                              # entry at 10 (bar 5), jumps to 11 = +10%
    net, kind = P.trade(b, 5, 0.08, 0.05, 0.002)
    assert kind == "tp" and abs(net - (0.08 - 0.002)) < 1e-9          # bar 10 opens at 10, its high 11 passes +8%
    d = bars([10.0] * 10 + [9.0] * 68)                               # drops 10%: the 5% stop fills at the stop
    net, kind = P.trade(d, 5, 0.08, 0.05, 0.002)
    assert kind == "stop" and abs(net - (-0.05 - 0.002)) < 1e-9
    flat = bars([10.0] * 78)
    net, kind = P.trade(flat, 5, 0.08, 0.05, 0.002)
    assert kind == "close" and abs(net + 0.002) < 1e-9


def test_a_bar_that_touches_both_counts_as_the_stop():
    c = [10.0] * 78
    hi, lo = np.array(c), np.array(c)
    hi[6], lo[6] = 11.0, 9.0
    net, kind = P.trade(bars(c, hi=hi, lo=lo), 5, 0.08, 0.05, 0.0)
    assert kind == "stop" and abs(net + 0.05) < 1e-9


def test_inputs_use_only_bars_before_the_decision():
    b = bars(list(np.linspace(10, 12, 78)))
    a = P.inputs(b, 20, 9.8, 1e7, 0.01, None)
    b2 = b.copy()
    b2.loc[20:, ["o", "h", "l", "c"]] *= 3
    assert P.inputs(b2, 20, 9.8, 1e7, 0.01, None) == a
    assert a["mins_open"] == 100 and abs(a["gap"] - (10 / 9.8 - 1)) < 1e-12


def test_sample_is_fixed_and_about_three_percent():
    ks = [P.keep(f"S{i}", DAY, "10:00") for i in range(20000)]
    assert ks == [P.keep(f"S{i}", DAY, "10:00") for i in range(20000)]
    assert 0.025 < np.mean(ks) < 0.035


def test_swing30_is_the_median_30_minute_range():
    b = bars([10.0] * 78, hi=np.full(78, 10.1), lo=np.full(78, 9.9))
    assert abs(P.swing30(b) - 0.02) < 1e-9
