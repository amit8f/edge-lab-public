import numpy as np
import pandas as pd

from tracks.main import sp1_study as S


def test_first_10_and_listing_rules():
    fil = pd.DataFrame({"cik": [1, 1, 2, 3], "form": ["10-12B", "10-12B/A", "10-12B", "8-K"],
                        "filed": pd.to_datetime(["2018-01-10", "2018-02-01", "2018-03-01", "2018-01-01"])})
    first, filers = S.first_10(fil)
    assert dict(first) == {1: pd.Timestamp("2018-01-10"), 2: pd.Timestamp("2018-03-01")}
    days = pd.bdate_range("2017-01-02", "2019-12-31")
    rows = pd.concat([
        pd.DataFrame({"cik": 1, "symbol": "NEW", "date": days[days >= "2018-03-01"], "c": 20.0, "dv20": 5e6}),
        pd.DataFrame({"cik": 2, "symbol": "OLD", "date": days[days >= "2017-06-01"], "c": 20.0, "dv20": 5e6}),
    ])
    ev = S.listings(rows, first, days)
    assert ev["cik"].tolist() == [1]                       # cik 2 traded before its Form 10: not a new listing
    r = ev.iloc[0]
    assert r["listing_day"] == pd.Timestamp("2018-03-01")
    assert (days.searchsorted(r["entry"]) - days.searchsorted(r["listing_day"])) == S.WAIT
    assert r["prior_day"] == days[days.searchsorted(r["entry"]) - 1]


def test_controls_skip_recent_form_10_filers_and_the_event_itself():
    ev = pd.DataFrame({"cik": [1], "entry": [pd.Timestamp("2019-05-01")], "prior_day": [pd.Timestamp("2019-04-30")]})
    uni = pd.DataFrame({"date": pd.Timestamp("2019-04-30"), "cik": [1, 2, 3, 4], "symbol": ["A", "B", "C", "D"]})
    filers = pd.Series({2: np.array([np.datetime64("2018-01-01")])})
    c = S.controls(ev, uni, filers)
    assert sorted(c[0]) == ["C", "D"]


def test_cohort_diff():
    ev = pd.DataFrame({"entry": pd.to_datetime(["2019-01-05", "2019-01-20"]), "r250": [0.3, 0.1], "c250": [0.1, 0.1]})
    c = S.cohorts(ev)
    assert len(c) == 1 and np.isclose(c["diff"].iloc[0], 0.1)
