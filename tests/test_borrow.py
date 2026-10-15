"""IBKR borrow-fee file parsing and logging (no network)."""
import pandas as pd

from core import borrow

SAMPLE = """#BOF|2026.11.02|23:45:01
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|
AAPL|USD|APPLE INC|265598|US0378331005|3.5|0.25|>10000000|
XPEN|USD|XPEN INC|1|US1|-120.5|125.0|3500|
BRK B|USD|BERKSHIRE HATH-B|2|US2|3.6|0.25|NA|
#EOF|3
"""


def test_parse():
    when, d = borrow.parse(SAMPLE)
    assert when == "2026.11.02 23:45:01"
    r = d.set_index("symbol")
    assert r.loc["XPEN", "fee"] == 1.25 and r.loc["XPEN", "available"] == 3500
    assert r.loc["AAPL", "available"] == 10_000_000 and pd.isna(r.loc["BRK.B", "available"])


def test_run_logs_once_and_latest_reads_back(tmp_path):
    now = pd.Timestamp("2026-11-03 04:50", tz="UTC")
    assert len(borrow.run(tmp_path, now=now, get=lambda: SAMPLE)) == 3
    assert borrow.run(tmp_path, now=now, get=lambda: SAMPLE) is None
    d, day = borrow.latest(tmp_path)
    assert day == "2026-11-02" and d.set_index("symbol").loc["XPEN", "fee"] == 1.25
