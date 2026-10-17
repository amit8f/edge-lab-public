import numpy as np
import pandas as pd

from tracks.main import w1c_calls as C

DAYS = pd.bdate_range("2024-02-01", "2024-04-30")


def test_contracts_frame():
    c = C.contracts_frame([{"symbol": "ABC240315C00011000", "type": "call", "strike_price": "11", "expiration_date": "2024-03-15"},
                           {"symbol": "ABC240315C00011000", "type": "call", "strike_price": "11", "expiration_date": "2024-03-15"}])
    assert len(c) == 1 and c["strike"].iloc[0] == 11.0 and c["expiration"].iloc[0] == pd.Timestamp("2024-03-15")


def test_day_agg_filters_expiry_and_strike():
    c = pd.DataFrame({"symbol": ["C1", "C2", "C3", "P1"], "type": ["call", "call", "call", "put"],
                      "strike": [10.6, 10.4, 11.0, 9.0],
                      "expiration": pd.to_datetime(["2024-03-15", "2024-03-15", "2024-06-21", "2024-03-15"])})
    bars = pd.DataFrame({"symbol": ["C1", "C2", "C3", "P1"], "date": pd.Timestamp("2024-03-01"), "v": [100, 50, 999, 30]})
    prior = pd.Series([10.0], index=pd.to_datetime(["2024-03-01"]))
    a = C.day_agg(c, bars, prior)
    r = a.iloc[0]
    assert r["call_otm"] == 100 and r["calls"] == 150 and r["puts"] == 30      # C3 expires too late; C2 not 5% OTM


def test_ratio_and_unusual():
    agg = pd.DataFrame({"date": DAYS[:25], "call_otm": [100] * 24 + [2000], "calls": 0, "puts": [10] * 25})
    a = C.add_ratio(agg, DAYS)
    last = a.iloc[-1]
    assert last["mean20"] == 100 and last["ratio"] == 20
    assert list(C.unusual(a))[-1] and not C.unusual(a).iloc[:-1].any()
    a.loc[a.index[-1], "puts"] = 700                                          # calls < 3x puts
    assert not C.unusual(a).iloc[-1]


def test_one_per_window():
    sig = pd.DataFrame({"symbol": ["A", "A", "A", "B"], "date": [DAYS[0], DAYS[5], DAYS[10], DAYS[5]]})
    k = C.one_per_window(sig, DAYS)
    assert list(zip(k["symbol"], k["date"])) == [("A", DAYS[0]), ("A", DAYS[10]), ("B", DAYS[5])]
