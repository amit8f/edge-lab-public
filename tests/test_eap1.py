import numpy as np
import pandas as pd

from tracks.main import eap1_study as E


def cal(start, end):
    d = pd.bdate_range(start, end)
    return pd.DataFrame({"date": d, "open": "09:30", "close": "16:00"})


def test_build_picks_same_week_last_year_and_controls_avoid_near_reporters():
    c = cal("2017-12-01", "2019-03-01")
    t = pd.Timestamp("2019-01-11")                       # a Friday
    ev = pd.DataFrame({"cik": [1, 2, 3], "accession": ["a", "b", "c"], "accepted_et": pd.NaT, "items": "2.02",
                       "day0": [t - pd.Timedelta(days=360),     # in (t-364, t-357]: expected next week
                                t - pd.Timedelta(days=366),     # just outside: not a pick, but near
                                t - pd.Timedelta(days=200)]})
    uni = pd.DataFrame({"date": t, "cik": [1, 2, 3, 4], "symbol": ["A", "B", "C", "D"], "dv20": 1e7})
    old = E.FIRST_T
    E.FIRST_T = pd.Timestamp("2019-01-01")
    try:
        p, k = E.build(ev, uni, c)
    finally:
        E.FIRST_T = old
    p, k = p[p["t"] == t], k[k["t"] == t]
    assert p["symbol"].tolist() == ["A"] and p["old_accession"].tolist() == ["a"]
    assert len(k) == 1 and k["symbol"].iloc[0] in {"C", "D"}
    assert p["entry"].iloc[0] == pd.Timestamp("2019-01-14") and p["exit5"].iloc[0] == pd.Timestamp("2019-01-18")


def test_weekly_diff_net_of_costs():
    p = pd.DataFrame({"t": pd.to_datetime(["2019-01-11"] * 2), "r5": [0.02, 0.04], "dv20": 1})
    c = pd.DataFrame({"t": pd.to_datetime(["2019-01-11"]), "r5": [0.01]})
    w = E.weekly(p, c)
    assert np.isclose(w["diff"].iloc[0], 0.02) and np.isclose(w["mean"].iloc[0], 0.03 - E.COST)
