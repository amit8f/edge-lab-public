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


HOUSE = b"""<?xml version="1.0" encoding="utf-8"?><FinancialDisclosure>
<Member><Prefix>Hon.</Prefix><Last>Doe</Last><First>Jane</First><Suffix /><FilingType>P</FilingType><StateDst>CA12</StateDst>
<Year>2026</Year><FilingDate>10/14/2026</FilingDate><DocID>20031234</DocID></Member>
<Member><Prefix /><Last>Roe</Last><First>Rick</First><Suffix /><FilingType>O</FilingType><StateDst>TX01</StateDst>
<Year>2026</Year><FilingDate>10/14/2026</FilingDate><DocID>10061234</DocID></Member>
<Member><Prefix /><Last>Old</Last><First>Ann</First><Suffix /><FilingType>P</FilingType><StateDst>NY03</StateDst>
<Year>2026</Year><FilingDate>08/01/2026</FilingDate><DocID>20030001</DocID></Member>
</FinancialDisclosure>"""


def test_house_ptr():
    rows = R.parse_house(HOUSE, pd.Timestamp("2026-10-16"))
    assert rows == [{"name": "Hon. Jane Doe", "state_district": "CA12", "filed": "2026-10-14", "doc_id": "20031234",
                     "pdf": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20031234.pdf"}]
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2026FD.xml", HOUSE)
    assert len(R.fetch_house(pd.Timestamp("2026-10-16"), get=lambda u: Resp(buf.getvalue()))) == 1
    assert R.fetch_house(pd.Timestamp("2026-10-16"), get=lambda u: Resp(b"not a zip")) is None
