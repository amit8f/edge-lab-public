"""L1 first-public-moment study: pure rules on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.news import l1_study as L

CAL = pd.DataFrame({"date": pd.bdate_range("2024-03-04", "2024-03-29"), "open": "09:30", "close": "16:00"})


def minutes(day, px0, step=0.0):
    t = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="min")
    c = px0 + step * np.arange(1, len(t) + 1)
    return pd.DataFrame({"t": t, "o": c - step, "c": c})


def test_category():
    buys = {"0001-24-000001"}
    assert L.category("424B5", [], "x", buys) == "offering"
    assert L.category("S-1/A", [], "x", buys) is None
    assert L.category("8-K", ["5.02", "9.01"], "x", buys) == "officer_departure"
    assert L.category("8-K", ["3.01", "5.02"], "x", buys) == "delisting_notice"
    assert L.category("8-K", ["2.02"], "x", buys) is None
    assert L.category("SC 13D/A", [], "x", buys) is None and L.category("SC 13D", [], "x", buys) == "activist_stake"
    assert L.category("4", [], "0001-24-000001", buys) == "insider_buy" and L.category("4", [], "y", buys) is None


def test_universe_rows_respect_gaps():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    d = pd.DataFrame({"c": 2.0, "v": 400_000}, index=idx).drop(pd.bdate_range("2024-02-01", "2024-02-20"))
    u = L.universe_rows("AAA", d)
    assert u["date"].min() == idx[19]                       # 20 bars before the first dv20
    assert u["date"].iloc[4] == pd.Timestamp("2024-03-19")
    assert not ((u["date"] > "2024-02-20") & (u["date"] < "2024-03-19")).any()   # 20 new bars after the gap


def test_entry_point():
    day, t = L.entry_point(pd.Timestamp("2024-03-05 07:10:05"), CAL)
    assert day == pd.Timestamp("2024-03-05") and t == pd.Timestamp("2024-03-05 09:30")
    day, t = L.entry_point(pd.Timestamp("2024-03-05 11:00:30"), CAL)
    assert t == pd.Timestamp("2024-03-05 11:02")
    day, t = L.entry_point(pd.Timestamp("2024-03-08 16:05"), CAL)        # Friday evening -> Monday open
    assert day == pd.Timestamp("2024-03-11") and t == pd.Timestamp("2024-03-11 09:30")


def test_trade_and_net():
    b = pd.concat([minutes("2024-03-05", 10.0, -0.001), minutes("2024-03-06", 9.6)], ignore_index=True)
    spy = pd.concat([minutes("2024-03-05", 500.0), minutes("2024-03-06", 500.0)], ignore_index=True)
    tr = L.trade(b, spy, CAL, pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-05 11:02"),
                 headline_t=pd.Timestamp("2024-03-05 13:00"))
    assert tr["entry_t"] == pd.Timestamp("2024-03-05 11:02") and tr["exit_t"] == pd.Timestamp("2024-03-06 15:59")
    assert tr["r_spy"] == 0 and tr["r"] == pytest.approx(9.6 / b.loc[b["t"] == tr["entry_t"], "o"].iloc[0] - 1)
    assert tr["r_pre"] < 0
    n = L.net(-1, tr["r"], 0.0, 0.02, "penny", 1)
    assert n == pytest.approx(-tr["r"] - 0.02 - 0.20 / 365)


def test_sample_cells_caps_and_is_reproducible():
    ev = pd.DataFrame({"category": "offering", "group": ["penny"] * 500 + ["main"] * 10,
                       "accession": [f"a{i:04d}" for i in range(510)]})
    a, b = L.sample_cells(ev), L.sample_cells(ev.sample(frac=1, random_state=1))
    assert (a.groupby("group").size() == pd.Series({"main": 10, "penny": 400})).all()
    assert list(a["accession"]) == list(b["accession"])


def test_cell_decision():
    rng = np.random.default_rng(0)
    days = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 11)]
    m = pd.DataFrame({"entry_day": days, "net": 0.02 + rng.normal(0, 0.01, len(days)), "lag_min": 90.0})
    d = L.cell_decision(m)
    assert d["n"] == 120 and d["candidate"] and d["t"] > 3
    m["lag_min"] = 20.0
    assert not L.cell_decision(m)["candidate"]
    m["lag_min"] = np.inf                                     # no headline at all still counts as a long lag
    assert L.cell_decision(m)["median_lag_min"] == np.inf


def test_submissions_rows_reads_older_files():
    class R:
        def __init__(self, j):
            self._j = j

        def json(self):
            return self._j
    recent = {"accessionNumber": ["n1", "n0"], "form": ["8-K", "4"], "filingDate": ["2025-05-01", "2022-06-01"],
              "items": ["5.02,9.01", ""]}
    older = {"accessionNumber": ["o1"], "form": ["S-1"], "filingDate": ["2023-02-01"], "items": [""]}
    pages = {"CIK0000000042.json": {"filings": {"recent": recent, "files": [{"name": "old-1.json", "filingTo": "2023-03-01"}]}},
             "old-1.json": older}
    got = L.submissions_rows(42, get=lambda url: R(pages[url.rsplit("/", 1)[1]]))
    assert [(r["accession"], r["items"]) for r in got] == [("n1", ["5.02", "9.01"]), ("o1", [])]
