import pandas as pd

from tracks.news import s0_study as S


def bars(prices):
    t = pd.date_range("2023-03-01 09:30", periods=len(prices), freq="1min")
    return pd.DataFrame({"t": t, "o": prices, "h": [p * 1.001 for p in prices], "l": [p * 0.999 for p in prices],
                         "c": prices})


def test_signed_paths_and_first_touch():
    b = bars([100.0] * 5 + [101.0, 102.0, 103.5] + [103.0] * 150)
    x = S.path_stats(b, pd.Timestamp("2023-03-01 09:31"), 1)
    assert x["entry_t"] == pd.Timestamp("2023-03-01 09:31")
    assert abs(x["r5"] - 0.01) < 1e-9 and x["b30"] == 1 and x["b20"] == 1 and x["b50"] == 0
    s = S.path_stats(b, pd.Timestamp("2023-03-01 09:31"), -1)
    assert s["r5"] < 0 and s["b30"] == -1 and s["mae"] < -0.03


def test_delay_and_missing_entry():
    b = bars([100.0] * 40)
    x = S.path_stats(b, pd.Timestamp("2023-03-01 09:31"), 1, delay=15)
    assert x["entry_t"] == pd.Timestamp("2023-03-01 09:46") and x["b30"] == 0
    assert S.path_stats(b, pd.Timestamp("2023-03-01 11:00"), 1) is None


def test_summary_cells():
    b = bars([100.0] * 5 + [101.0] * 150)
    x = S.path_stats(b, pd.Timestamp("2023-03-01 09:31"), 1)
    t = pd.DataFrame([{"d": pd.Timestamp("2023-03-01"), "grp": "event", "delay": 0, "when": "preopen",
                       "side": "long", **x}])
    s = S.summarize(t)["event/0/preopen/long"]
    assert s["n"] == 1 and s["r5"]["pos"] == 1.0 and s["tp0.030_sl0.015"]["neither"] == 1.0
