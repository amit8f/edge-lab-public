import io
import zipfile

import pandas as pd

from tracks.main import ins1_study as I


def _tr(rows):
    return pd.DataFrame(rows, columns=["owner_cik", "trade_date"]).assign(trade_date=lambda x: pd.to_datetime(x["trade_date"]))


def test_classify_routine_opportunistic_and_unknown():
    rows = [(1, f"{y}-03-10") for y in (2014, 2015, 2016, 2017)]            # every March: routine
    rows += [(2, "2014-01-05"), (2, "2015-06-05"), (2, "2016-09-05"), (2, "2017-03-05")]   # irregular
    rows += [(3, "2015-03-05"), (3, "2016-03-05"), (3, "2017-03-05")]       # only 2 prior years
    c = I.classify(_tr(rows))
    assert c.iloc[3] == "routine" and c.iloc[7] == "opportunistic" and pd.isna(c.iloc[10])


def test_cluster_keeps_the_first_filing_within_5_sessions():
    days = pd.bdate_range("2024-01-01", periods=30)
    ev = pd.DataFrame({"accession": ["a", "b", "c", "d"], "issuer_cik": [7, 7, 7, 8],
                       "filed": [days[0], days[2], days[9], days[2]], "entry": [days[1], days[3], days[10], days[3]]})
    kept = I.cluster(ev, days)
    assert list(kept["accession"]) == ["a", "d", "c"]


def test_next_session_is_strictly_after_the_filing_date():
    days = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])
    assert I.next_session(days, pd.Timestamp("2024-03-01")) == pd.Timestamp("2024-03-04")
    assert I.next_session(days, pd.Timestamp("2024-03-02")) == pd.Timestamp("2024-03-04")


def test_read_quarter_keeps_form_4_open_market_trades():
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
                                     "x1\t05-MAR-2024\t4\t100\tabc\nx2\t05-MAR-2024\t3\t100\tABC\n")
        f.writestr("REPORTINGOWNER.tsv", "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
                                         "x1\t8\tTenPercentOwner\nx1\t9\tDirector\nx2\t9\tTenPercentOwner\n")
        f.writestr("NONDERIV_TRANS.tsv", "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE"
                                         "\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n"
                                         "x1\t1\tCommon Stock\t01-MAR-2024\tP\t1000\t30\n"
                                         "x1\t2\tCommon Stock\t01-MAR-2024\tA\t10\t0\n"
                                         "x1\t3\t5.95% Preferred Stock\t01-MAR-2024\tP\t1000\t25\n"
                                         "x2\t4\tCommon Stock\t01-MAR-2024\tP\t5\t1\n")
    t = I.read_quarter(z.getvalue())
    assert len(t) == 1 and t["issuer_symbol"].iloc[0] == "ABC" and bool(t["officer_dir"].iloc[0])
    assert t["filed"].iloc[0] == pd.Timestamp("2024-03-05") and t["shares"].iloc[0] * t["price"].iloc[0] == 30000
    assert t["owner_cik"].iloc[0] == 9                                     # one owner per trade, the director


def test_common_stock_titles():
    t = pd.Series(["Common Stock", "Class A Common Stock", "Common Units", "5.95% Preferred Stock",
                   "Series B Preferred", "Warrants", "Senior Notes", None])
    assert I.common_stock(t).tolist() == [True, True, True, False, False, False, False, True]
