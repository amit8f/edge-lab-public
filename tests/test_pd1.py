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
    assert rows[0]["accepted_et"] == pd.Timestamp("2019-05-08 16:05:12")


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
