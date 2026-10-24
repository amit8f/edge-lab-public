"""L1 forward shadow books (FWD1) on synthetic data (no network)."""
import numpy as np
import pandas as pd

from tracks.news import l1_live as L
from tracks.news import l1_study as S

CAL = pd.DataFrame({"date": pd.bdate_range("2026-10-20", "2026-11-06"), "open": "09:30", "close": "16:00"})


def mbars(day, p0, step=0.0):
    t = pd.date_range(f"{day:%Y-%m-%d} 09:30", periods=390, freq="min")
    c = p0 + step * np.arange(390)
    return pd.DataFrame({"t": t, "o": c, "c": c})


def two_days(prev, day, nxt, p_prev, p_day, p_next):
    return pd.concat([mbars(prev, p_prev), mbars(day, p_day), mbars(nxt, p_next)], ignore_index=True)


def test_ready_days_wait_for_the_second_session_close():
    now = pd.Timestamp("2026-10-28 22:30", tz=L.ET)
    days = L.ready_days(CAL, now, L.FIRST_DAY - pd.Timedelta(days=1))
    assert days == [pd.Timestamp("2026-10-26")]          # Oct 27's filings exit Oct 29
    assert L.ready_days(CAL, now, pd.Timestamp("2026-10-26")) == []


def test_candidates_keep_the_two_cells_forms_with_a_ticker():
    idx = pd.DataFrame({"form": ["424B5", "NT 10-Q", "8-K", "S-1"], "company": ["A", "B", "C", "D"],
                        "cik": [1, 2, 3, 4], "filed": pd.Timestamp("2026-10-26"), "file": ["a", "b", "c", "d"]})
    tmap = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "cik": [1, 2, 3]})
    c = L.candidates(idx, tmap)
    assert list(zip(c["symbol"], c["category"])) == [("AAA", "offering"), ("BBB", "late_filing")]


def test_event_shorts_the_cell_with_a_control_and_r_open():
    prev, day, nxt = (pd.Timestamp(x) for x in ("2026-10-26", "2026-10-27", "2026-10-28"))
    uni = pd.DataFrame({"symbol": ["PEN", "CTL", "BIG"], "date": prev, "close": [3.0, 2.0, 50.0],
                        "dv20": [1e6, 1e6, 1e8], "group": ["penny", "penny", "main"]})
    bars = {"PEN": two_days(prev, day, nxt, 3.0, 3.3, 2.97), "CTL": two_days(prev, day, nxt, 2.0, 2.0, 2.0)}
    spy = two_days(prev, day, nxt, 500.0, 500.0, 500.0)
    row = {"accession": "0001-26-000001", "form": "424B5", "category": "offering", "symbol": "PEN", "cik": 7,
           "filed": prev}
    rec = L.event(row, "2026-10-27 10:00:00", CAL, uni, lambda s, w: bars[s], lambda d, w: spy, {"PEN": True})
    assert rec["status"] == "ok" and rec["entry_t"] == "2026-10-27 10:01:00" and rec["group"] == "penny"
    assert np.isclose(rec["r"], 2.97 / 3.3 - 1) and np.isclose(rec["r_open"], 3.3 / 3.0 - 1)
    assert np.isclose(rec["net"], S.net(-1, rec["r"], 0.0, 0.02, "penny", 1))
    assert rec["control_symbol"] == "CTL" and np.isclose(rec["control_net"], S.net(-1, 0.0, 0.0, 0.02, "penny", 1))
    big = dict(row, symbol="BIG")
    assert L.event(big, "2026-10-27 10:00:00", CAL, uni, None, None, {})["status"] == "other cell"
    assert L.event(row, None, CAL, uni, None, None, {})["status"] == "no header"
