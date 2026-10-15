"""SEC leads logger on fake EDGAR responses (no network)."""
import json

import pandas as pd

from core import leads_sec

HDR = ("&lt;SEC-HEADER&gt;\n&lt;ACCEPTANCE-DATETIME&gt;20261013163005\n&lt;TYPE&gt;8-K\n"
       "&lt;ITEMS&gt;5.02\n&lt;ITEMS&gt;9.01\n&lt;/SEC-HEADER&gt;")


class R:
    text = HDR


def fake_index(published):
    def index(d):
        if pd.Timestamp(d) not in published:
            return None
        return pd.DataFrame({"form": ["8-K", "4", "10-K"], "company": ["A", "B", "C"], "cik": [1, 2, 3],
                             "filed": [pd.Timestamp(d)] * 3,
                             "file": ["edgar/data/1/0000000001-26-000001.txt"] * 3})
    return index


def test_parse_header():
    assert leads_sec.parse_header(HDR) == ("2026-10-13 16:30:05", ["5.02", "9.01"])


def test_run_fills_days_skips_holiday_and_waits_for_unpublished(tmp_path):
    now = pd.Timestamp("2026-10-15T03:30:00Z")          # Oct 14 23:30 ET -> yesterday Oct 13
    idx = fake_index({pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-13")})   # Oct 12 holiday
    n = leads_sec.run(tmp_path, now, first_day="2026-10-09", index=idx, get=lambda u: R())
    assert n == 3
    rows = [json.loads(x) for x in (tmp_path / "leads/sec/2026-10-13.jsonl").read_text().splitlines()]
    assert [r["form"] for r in rows] == ["8-K", "4"]
    assert rows[0]["accepted_et"] == "2026-10-13 16:30:05" and rows[0]["items"] == ["5.02", "9.01"]
    assert (tmp_path / "leads/sec/2026-10-12.jsonl").read_text() == ""
    # next night: Oct 14 not published yet -> nothing written, cursor stays
    later = pd.Timestamp("2026-10-16T03:30:00Z")
    assert leads_sec.run(tmp_path, later, index=idx, get=lambda u: R()) == 0
