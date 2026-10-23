from tracks import hand_check

PAGE = """<div id="formName"><strong>Form SC 13D</strong> - General statement</div>
<div class="infoHead">Filing Date</div><div class="info">2023-03-28</div>
<div class="infoHead">Accepted</div>
<div class="info">2023-03-28 11:57:07</div>
<span class="companyName">THOR INDUSTRIES INC (Subject)</span>
<span class="companyName">Some Fund LP (Filed by)</span>"""


class Resp:
    status_code, text = 200, PAGE


def test_edgar_index_reads_acceptance_form_and_names(monkeypatch):
    seen = []
    monkeypatch.setattr(hand_check.requests, "get", lambda url, **k: seen.append(url) or Resp())
    monkeypatch.setattr(hand_check.time, "sleep", lambda s: None)
    r = hand_check.edgar_index("730263", "0001085146-23-001491", {})
    assert seen == ["https://www.sec.gov/Archives/edgar/data/730263/000108514623001491/0001085146-23-001491-index.htm"]
    assert r == {"accepted": "2023-03-28 11:57:07", "form": "Form SC 13D",
                 "names": ["THOR INDUSTRIES INC", "Some Fund LP"]}
