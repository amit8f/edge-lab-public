"""IBKR borrow-fee file parsing and logging (no network)."""
import pandas as pd

from core import borrow

SAMPLE = """#BOF|2026.11.02|23:45:01
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|
049323AB4|USD|CB ATLAS FINL 06.625% 27|557991763|XXXXXXX3AB46|3.6300|0.2500|300000|BBG017QTG7P1|
AAPL|USD|APPLE INC|265598|US0378331005|3.5|0.25|>10000000|BBG1|
XPEN|USD|XPEN INC|1|US1|-120.5|125.0|3500|BBG2|
BRK B|USD|BERKSHIRE HATH-B|2|US2|3.6|0.25|NA|BBG3|
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
    assert len(borrow.run(tmp_path, now=now, get=lambda: SAMPLE)) == 4
    assert borrow.run(tmp_path, now=now, get=lambda: SAMPLE) is None
    d, day = borrow.latest(tmp_path)
    assert day == "2026-11-02" and d.set_index("symbol").loc["XPEN", "fee"] == 1.25


def test_fetch_falls_back_to_the_next_host(monkeypatch):
    seen = []

    class FakeFTP:
        def __init__(self, host, timeout):
            seen.append(host)
            if host == "down":
                raise TimeoutError("timed out")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, pw):
            pass

        def retrbinary(self, cmd, write):
            write(SAMPLE.encode("latin-1"))

    monkeypatch.setattr(borrow.ftplib, "FTP", FakeFTP)
    assert borrow.fetch(("down", "up")) == SAMPLE and seen == ["down", "up"]
