import json

import pandas as pd

from core.ledger import Ledger
from tracks.main import sp1_live as S


def fake_index(published_through):
    def index(d):
        d = pd.Timestamp(d)
        if d > pd.Timestamp(published_through):
            return None
        rows = []
        if d == pd.Timestamp("2025-10-02"):
            rows = [("10-12B", "SpinCo A", 111, d), ("8-K", "Other", 222, d), ("10-12B", "SpinCo B", 333, d)]
        if d == pd.Timestamp("2025-10-06"):
            rows = [("10-12B/A", "SpinCo A", 111, d)]
        return pd.DataFrame(rows, columns=["form", "company", "cik", "filed"])
    return index


class Resp:
    def __init__(self, j):
        self._j = j

    def json(self):
        return self._j


def fake_get(url):
    if "CIK0000000111" in url:
        return Resp({"filings": {"recent": {"form": ["10-12B"], "filingDate": ["2025-10-02"]}}})
    if "CIK0000000333" in url:      # had a 10-12B half a year earlier: not a first filing
        return Resp({"filings": {"recent": {"form": ["10-12B", "10-12B"], "filingDate": ["2025-10-02", "2025-04-01"]}}})
    raise AssertionError(url)


def test_scan_and_candidates(tmp_path):
    now = pd.Timestamp("2025-10-08 03:00", tz="UTC")        # Oct 7 evening ET: yesterday is Oct 6
    rows, cand, ent = S.run(tmp_path, now, fake_index("2025-10-03"), fake_get)
    assert [r["cik"] for r in rows] == [111, 333]           # stopped at Oct 6, not yet published
    by = {c["date"]: json.loads((tmp_path / "sp1" / c["file"]).read_text()) for c in cand}
    assert by["111"]["candidate"] and not by["333"]["candidate"]
    rows, cand, _ = S.run(tmp_path, now, fake_index("2025-10-06"), fake_get)
    assert [r["form"] for r in rows] == ["10-12B", "10-12B", "10-12B/A"] and cand == []
    assert Ledger(tmp_path / "sp1").verify() == []


def test_entry_after_22nd_session(tmp_path, monkeypatch):
    cal = pd.DataFrame({"date": pd.bdate_range("2025-10-01", "2027-06-30")})
    dates = tmp_path / "dates.csv"
    dates.write_text("cik,symbol,regular_way_day,source\n111,SPNA,2026-01-05,release\n333,SPNB,,\n")
    monkeypatch.setattr(S, "DATES", dates)
    days = pd.DatetimeIndex(cal["date"])
    entry = days[days.searchsorted(pd.Timestamp("2026-01-05")) + 21]
    seen = []

    def opens(sym, d):
        seen.append((sym, d))
        return 12.5

    now = (entry - pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    _, _, ent = S.run(tmp_path, now, fake_index("2025-10-01"), fake_get, opens, cal)
    assert ent == [] and seen == []
    now = (entry + pd.Timedelta(hours=20)).tz_localize("America/New_York").tz_convert("UTC")
    _, _, ent = S.run(tmp_path, now, fake_index("2025-10-01"), fake_get, opens, cal)
    rec = json.loads((tmp_path / "sp1" / ent[0]["file"]).read_text())
    assert rec["entry_day"] == f"{entry:%Y-%m-%d}" and rec["entry_open_raw"] == 12.5
    assert rec["exit_day"] == f"{days[days.searchsorted(entry) + 250]:%Y-%m-%d}"
    _, _, ent = S.run(tmp_path, now, fake_index("2025-10-01"), fake_get, opens, cal)
    assert ent == []                                         # logged once
