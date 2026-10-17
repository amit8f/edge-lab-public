import json

import pandas as pd

from core import registers as R

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>SEC Charges Trader</title><link>https://www.sec.gov/lr-1</link><pubDate>Fri, 16 Oct 2026 12:00:00 EDT</pubDate>
<description>&lt;p&gt;Traded ahead of the &lt;b&gt;merger&lt;/b&gt;&lt;/p&gt;</description></item></channel></rss>"""
FTC = """<table><thead><tr><th>Date</th></tr></thead><tbody>
<tr><td>10/15/2026</td><td><a href="#">Acquirer Inc.</a></td><td>Target &amp; Co</td></tr></tbody></table>"""


class Resp:
    def __init__(self, content):
        self.content = content


def test_parsers():
    it = R.parse_rss(RSS)
    assert it[0]["title"] == "SEC Charges Trader" and it[0]["summary"] == "Traded ahead of the  merger"
    assert R.parse_ftc(FTC) == [["10/15/2026", "Acquirer Inc.", "Target & Co"]]
    idx = pd.DataFrame({"form": ["144", "4", "144/A"], "company": ["A", "B", "C"], "cik": [1, 2, 3],
                        "filed": pd.to_datetime(["2026-10-16"] * 3), "file": ["x", "y", "z"]})
    assert [r["company"] for r in R.form144_rows(idx)] == ["A", "C"]
    assert R.form144_rows(None) is None


def test_run_logs_once_and_retries_late_index(tmp_path):
    def get(url):
        return Resp(RSS) if "rss" in url else Resp(FTC.encode())
    calls = []

    def index_fn(d):
        calls.append(d)
        if d == "2026-10-16":
            return None                       # not published yet
        return pd.DataFrame({"form": ["144"], "company": ["A"], "cik": [1], "filed": pd.to_datetime([d]), "file": ["x"]})
    R.run(tmp_path, pd.Timestamp("2026-10-16"), get=get, index_fn=index_fn)
    led = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    kinds = [(e["kind"], e["date"]) for e in led]
    assert ("form144", "2026-10-16") not in kinds and ("form144", "2026-10-15") in kinds
    assert ("sec_litigation", "2026-10-16") in kinds and ("ftc_early_termination", "2026-10-16") in kinds
    n = len(led)
    R.run(tmp_path, pd.Timestamp("2026-10-16"), get=get, index_fn=lambda d: index_fn("2026-10-15"))
    led2 = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(led2) == n + 1                 # only the late form 144 day is added


def test_last_business_day():
    assert R.last_business_day(pd.Timestamp("2026-10-17 01:00", tz="America/New_York")) == pd.Timestamp("2026-10-16")
    assert R.last_business_day(pd.Timestamp("2026-10-20 01:00", tz="America/New_York")) == pd.Timestamp("2026-10-19")
