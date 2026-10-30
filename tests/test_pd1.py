import numpy as np
import pandas as pd

from tracks.penny import pd1_study as P


class R:
    def __init__(self, j):
        self.j = j

    def json(self):
        return self.j


def _cal():
    return pd.DataFrame({"date": pd.to_datetime(["2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"]),
                         "open": "09:30", "close": ["16:00", "16:00", "13:00", "16:00"]})


def test_earnings_rows_keep_only_8k_with_202_and_read_the_et_clock():
    recent = {"accessionNumber": ["a1", "a2", "a3", "a4"], "form": ["8-K", "8-K", "8-K/A", "10-Q"],
              "filingDate": ["2019-05-08"] * 4, "items": ["2.02,9.01", "8.01", "2.02", ""],
              "acceptanceDateTime": ["2019-05-08T16:05:12.000Z"] * 4}
    rows = P.earnings_rows(7, get=lambda url: R({"filings": {"recent": recent, "files": []}}))
    assert [r["accession"] for r in rows] == ["a1"]
    assert rows[0]["accepted_et"] == pd.Timestamp("2019-05-08 12:05:12")   # 16:05 UTC is 12:05 EDT


def test_day0_is_the_first_close_after_acceptance():
    acc = pd.to_datetime(["2019-05-08 15:59:00", "2019-05-08 16:00:00", "2019-05-08 16:05:00",
                          "2019-05-09 14:00:00", "2019-05-11 09:00:00"])
    d = P.day0(acc, _cal())
    assert list(d[:4]) == list(pd.to_datetime(["2019-05-08", "2019-05-09", "2019-05-09", "2019-05-10"]))
    assert pd.isna(d[4])


def test_first_per_company_keeps_one_per_30_days():
    ev = pd.DataFrame({"cik": [1, 1, 1, 2], "accession": list("abcd"),
                       "accepted_et": pd.to_datetime(["2019-01-01", "2019-01-20", "2019-02-05", "2019-01-10"])})
    assert P.first_per_company(ev)["accession"].tolist() == ["a", "c", "d"]


def test_events_use_the_penny_universe_on_the_session_before_day0():
    fil = pd.DataFrame({"cik": [1, 2], "accession": ["a", "b"], "form": "8-K", "filed": "2019-05-08",
                        "accepted_et": pd.to_datetime(["2019-05-08 16:05", "2019-05-08 08:00"]), "items": "2.02"})
    pen = pd.DataFrame({"symbol": ["A", "B"], "date": pd.to_datetime(["2019-05-08", "2019-05-08"]),
                        "c": [2.0, 3.0], "dv20": [1e6, 1e6], "cik": [1, 2]})
    ev = P.events(fil, pen, _cal())
    assert ev["symbol"].tolist() == ["A"]
    assert ev.loc[0, "day0"] == pd.Timestamp("2019-05-09")


def test_trade_ret_needs_an_entry_bar_and_stops_at_the_segment_end():
    g = pd.DataFrame({"ao": [10.0, 11.0, 12.0, 20.0], "ac": [10.5, 11.5, 12.5, 21.0], "seg": [0, 0, 0, 1]},
                     index=pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04", "2019-02-01"]))
    assert abs(P.trade_ret(g, pd.Timestamp("2019-01-03"), pd.Timestamp("2019-03-01")) - (12.5 / 11 - 1)) < 1e-12
    assert np.isnan(P.trade_ret(g, pd.Timestamp("2019-01-05"), pd.Timestamp("2019-03-01")))


def test_windows_count_the_entry_day_as_the_first_session():
    cal = pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=80)})
    ev = P.windows(pd.DataFrame({"day0": [cal["date"][3]]}), cal)
    assert ev.loc[0, "entry"] == cal["date"][5]
    assert ev.loc[0, "exit60"] == cal["date"][64]


def test_signal_uses_only_earlier_events():
    d = pd.date_range("2019-01-01", periods=150, freq="D")
    x = np.r_[np.linspace(0, 1, 149), 0.95]
    x[101] = -1.0
    ev = pd.DataFrame({"day0": d, "cik": range(150), "xr": x})
    s = P.signal(ev)
    assert s["side"].iloc[:100].isna().all()
    assert s["side"].iloc[149] == "top"
    assert s["side"].iloc[101] == "bottom"


def test_control_candidates_skip_earnings_filers_and_differ_within_a_day():
    d0, prior = pd.Timestamp("2019-05-09"), pd.Timestamp("2019-05-08")
    picks = pd.DataFrame({"day0": [d0, d0], "prior_day": [prior, prior]}, index=[10, 11])
    pen = pd.DataFrame({"symbol": [f"S{i}" for i in range(20)], "date": prior, "c": 2.0, "dv20": 1e6,
                        "cik": range(20)})
    fil = pd.DataFrame({"cik": [3], "day0": [pd.Timestamp("2019-05-20")]})
    c = P.control_candidates(picks, pen, fil)
    assert len(c[10]) == P.N_CAND and not set(c[10]) & set(c[11])
    assert "S3" not in c[10] + c[11]


def test_cohort_test_subtracts_the_control_and_costs():
    rng = np.random.default_rng(1)
    n = 400
    entry = pd.Timestamp("2018-01-05") + pd.to_timedelta(rng.integers(0, 2000, n), "D")
    tr = pd.DataFrame({"side": "top", "entry": entry, "r60": 0.05 + rng.normal(0, 0.01, n),
                       "c60": rng.normal(0, 0.01, n)})
    months = pd.period_range("2017-12", "2024-12", freq="M")
    ff5 = pd.DataFrame(rng.normal(0, 0.03, (len(months), 5)), index=months,
                       columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    umd = pd.DataFrame({"Mom": rng.normal(0, 0.03, len(months))}, index=months)
    res, c = P.cohort_test(tr, ff5, umd)
    assert abs(res["mean_minus_control"] - 0.05) < 0.005
    assert abs(res["picks_mean_net"] - 0.03) < 0.005
    assert res["t_hac"] > 3
