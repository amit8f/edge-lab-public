import numpy as np
import pandas as pd

from tracks.news import long_study as L


def test_cap_longs_takes_first_three_longs_by_time():
    p = pd.DataFrame({"d": pd.to_datetime(["2024-01-02"] * 5), "t": pd.to_datetime(
        ["2024-01-02 09:31", "2024-01-02 09:30", "2024-01-02 10:00", "2024-01-02 09:30", "2024-01-02 11:00"]),
        "side": [1, 1, 1, -1, 1], "symbol": list("ABCDE")})
    assert list(L.cap_longs(p)["symbol"]) == ["B", "A", "C"]


def test_day_book_fills_empty_days_with_cash_and_skips_nan():
    d = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    b = L.day_book(d, [0.02, np.nan, -0.01], pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert list(b) == [0.02, -0.01, 0.0]


def test_confirm_skips_early_longs_below_their_entry():
    d = pd.to_datetime(["2024-03-01"] * 3)
    et = pd.to_datetime(["2024-03-01 09:31", "2024-03-01 09:35", "2024-03-01 10:15"])
    r2, _, skip = L.confirm(et, [10.0, 10.0, 10.0], [0.10, 0.10, 0.10], [0.004] * 3, d,
                             [11.0, 9.5, 12.0], [0.001] * 3, "10:00")
    assert list(skip) == [False, True, False]
    assert np.isclose(r2[0], 0.0) and np.isnan(r2[1]) and np.isclose(r2[2], 0.10)


def test_hold_ret_carries_from_next_close():
    assert np.isclose(L.hold_ret([0.10], [11.0], [12.1])[0], 1.1 * 1.1 - 1)


def test_market_rule_uses_previous_close():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"]).to_numpy()
    on = L.rule_on(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).to_numpy(), dates, np.array([True, False]))
    assert list(on) == [False, True, False]


def test_gate_needs_all_three():
    rng = np.random.default_rng(1)
    own = 0.01 + 0.002 * rng.standard_normal(200)
    assert L.gate(own, own, 0.01)["verdict"] == "passed"
    assert L.gate(own, -own, 0.01)["verdict"] == "failed"
    assert L.gate(own, own, -0.01)["verdict"] == "failed"
