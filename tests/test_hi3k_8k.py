import pandas as pd

from tracks.news import hi3k_8k as K

INDEX = """<div class="formGrouping"><div class="infoHead">Accepted</div>
<div class="info">2025-10-15 16:05:12</div></div>
<table class="tableFile" summary="Document Format Files">
<tr><th scope="col">Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td scope="row">1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/0001/x8k.htm">x8k.htm</a></td>
<td scope="row">8-K</td><td>10</td></tr>
<tr class="blueRow"><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/0001/ex991.htm">ex991.htm</a>
</td><td scope="row">EX-99.1</td><td>20</td></tr></table>"""

RELEASE = """<html><body><p>Exhibit 99.1</p><div><b>FOR IMMEDIATE RELEASE</b></div><p>October 15, 2025</p>
<p>Contact: ir@acme.com</p><p><b>ACME Corp Reports Record Third Quarter Revenue&nbsp;of $1.2 Billion</b></p>
<p>SPRINGFIELD, Oct. 15, 2025 -- ACME Corp today announced results for its third quarter ended September 30, 2025,
with revenue up 20% from a year earlier and earnings per share well above the company's prior outlook range.</p>
</body></html>"""


def test_parse_index_time_and_docs():
    t, docs = K.parse_index(INDEX)
    assert t == pd.Timestamp("2025-10-15 20:05:12", tz="UTC")     # EDT -> UTC
    assert docs[0] == ("8-K", "/Archives/edgar/data/1/0001/x8k.htm")
    assert K.first_ex99(docs) == "/Archives/edgar/data/1/0001/ex991.htm"
    assert K.first_ex99([("8-K", "a"), ("EX-10.1", "b")]) is None


def test_title_skips_boilerplate():
    assert K.title(RELEASE) == "ACME Corp Reports Record Third Quarter Revenue of $1.2 Billion"
    assert K.title("<p>Exhibit 99.1</p><p>Press Release</p><p>Short line</p>") is None


def test_ticker_by_cik_filters_exchange():
    j = {"fields": ["cik", "name", "ticker", "exchange"],
         "data": [[1, "A", "AAA", "Nasdaq"], [1, "A", "AAA-W", "Nasdaq"], [2, "B", "BBB", "OTC"],
                  [3, "C", "brk-b", "NYSE"]]}
    assert K.ticker_by_cik(j) == {1: "AAA", 3: "BRK.B"}


def test_overlap_share(tmp_path):
    import json
    (tmp_path / "2026-06-01.json").write_text(json.dumps({"day": "2026-06-01", "events": [
        {"symbol": "AAA", "status": "decided"}, {"symbol": "BBB", "status": "not in universe"}]}))
    tr = pd.DataFrame({"arm": ["X"] * 3, "part": ["overlap"] * 3, "role": ["picks"] * 3, "day": ["2026-06-01"] * 3,
                       "symbol": ["AAA", "BBB", "CCC"]})
    assert abs(K.overlap_share(tr, tmp_path)["X"] - 1 / 3) < 1e-12


def test_title_joins_broken_lines_and_skips_contacts():
    raw = ("<p>Investors: Jane Roe, VP, Investor Relations - 201-847-6927</p>"
           "<p>Bank7 Corp. (NASDAQ: BSVN) today announced results for the quarter ended in September</p>"
           "<p>Adagio Medical Announces Pricing of up to Approximately</p><p>$25 Million Private Placement</p>")
    assert K.title(raw) == "Adagio Medical Announces Pricing of up to Approximately $25 Million Private Placement"
