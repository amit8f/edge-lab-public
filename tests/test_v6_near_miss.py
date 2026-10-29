import pandas as pd

from validation import v6_near_miss as v


def picks():
    rows = []
    for i, d in enumerate(pd.date_range("2020-01-31", periods=8, freq="ME")):
        rows += [{"date": f"{d:%Y-%m-%d}", "symbol": f"F{i}", "role": "flag", "m": 1.0, "gross": 0.01 * i,
                  "held": 63, "stopped": False},
                 {"date": f"{d:%Y-%m-%d}", "symbol": f"C{i}", "role": "control", "m": None,
                  "gross": -2.0 if i == 7 else 0.0, "held": 5, "stopped": i == 7}]
    return pd.DataFrame(rows)


def test_one_blown_up_control_carries_the_mean():
    diff = v.a1_months(picks())
    assert len(diff) == 8 and round(diff.iloc[-1], 6) == 2.07
    assert v.without_best(diff, 1)["mean"] < diff.mean() / 5


def test_diagnosis_lists_the_worst_trade_first():
    r = v.a1_diagnosis(picks())
    assert r["worst_trades"][0]["symbol"] == "C7"
    assert r["drop_best"][0]["dropped"] == 0 and r["drop_best"][1]["months"] == 7
