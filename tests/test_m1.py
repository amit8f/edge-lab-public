"""M1 pure helpers on synthetic data (no network)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tracks.main import m1_screen as m1


def test_clean_text_drops_number_cells_pages_and_xbrl_header():
    doc = ("<html><head><title>x</title></head><body><div style='display:none'><ix:header>hidden facts"
           "</ix:header></div><p>Our revenue grew &amp; margins held.</p><table><tr><td>Revenue</td>"
           "<td>$ 1,234</td><td>(56)</td></tr></table><p>12</p><p>Page iii</p><p>- 7 -</p></body></html>")
    t = m1.clean_text(doc)
    toks = m1.tokens(t)
    assert "hidden" not in toks and "facts" not in toks and "title" not in toks
    assert toks["revenue"] == 2 and toks["margins"] == 1 and "page" not in toks and "iii" not in toks
    assert "1,234" not in t


def test_item_1a_takes_body_not_table_of_contents():
    text = ("Item 1A. Risk Factors 12\nItem 1B. Unresolved 20\n...\nItem 1A. Risk Factors\nWe may lose "
            "customers and suppliers may fail.\nItem 1B. Unresolved Staff Comments\nNone.")
    assert "customers" in m1.item_1a(text) and "unresolved" not in m1.item_1a(text)
    assert m1.item_1a("no such section") == ""


def test_cosine_and_jaccard():
    a, b = m1.tokens("alpha beta beta gamma"), m1.tokens("alpha beta delta")
    assert np.isclose(m1.cosine(a, b), (1 + 2) / np.sqrt(6 * 3))
    assert np.isclose(m1.jaccard(a, b), 2 / 4)
    assert np.isnan(m1.cosine(a, m1.tokens("")))


def test_comparison_same_form_window_closest_to_365():
    f = pd.DataFrame({"form": ["10-Q", "10-K", "10-Q", "10-Q", "10-Q"],
                      "filed": pd.to_datetime(["2019-05-01", "2019-03-01", "2019-08-01", "2020-04-28",
                                               "2020-05-05"])})
    c = m1.comparison_index(f)
    assert c[3] == 0 and c[4] == 0              # 363 and 370 days back; 10-K never compared with a 10-Q
    assert c[0] is None and c[1] is None and c[2] is None


def test_scores_at_latest_filing_with_comparison_in_window():
    sim = pd.DataFrame({"cik": [1, 1, 2], "form": ["10-K", "10-Q", "10-Q"], "accn": ["a", "b", "c"],
                        "filed": pd.to_datetime(["2020-03-01", "2020-05-01", "2019-12-01"]),
                        "cos": [0.9, np.nan, 0.8]})
    s = m1.scores_at(sim, [pd.Timestamp("2020-05-29")])
    assert list(s["cik"]) == [1] and s["score"].iloc[0] == 0.9   # b has no comparison; c is >91 days old


def test_long_short_deciles_and_costs():
    t = pd.Timestamp("2020-01-31")
    df = pd.DataFrame({"date": t, "cik": range(20), "score": np.arange(20.0), "ret1": np.r_[np.zeros(18), 0.1, 0.1]})
    df.loc[0:1, "ret1"] = -0.05
    ex = m1.long_short(df, "ret1")
    assert np.isclose(ex["excess"].iloc[0], 0.1 - (-0.05) - (2 * m1.COST + m1.BORROW_MONTH))
    assert ex["n_long"].iloc[0] == 2 and ex["n_short"].iloc[0] == 2
    ctrl = m1.long_short(df, "ret1", rng_seed=m1.SEED)
    assert ctrl["n_long"].iloc[0] == 2


def test_cik_rows_pairs_and_skips_missing_doc():
    sub = {"filings": {"recent": {"form": ["10-K", "10-K/A", "10-K", "8-K", "10-K"],
                                  "filingDate": ["2016-03-01", "2016-04-01", "2017-03-02", "2017-03-03", "2018-03-01"],
                                  "accessionNumber": ["0-16-1", "0-16-2", "0-17-1", "0-17-2", "0-18-1"],
                                  "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm", "e.htm"]}, "files": []}}
    docs = {"a.htm": "<p>alpha beta gamma</p>", "c.htm": "<p>alpha beta delta</p>"}

    def get(url):
        if "submissions" in url:
            return SimpleNamespace(json=lambda: sub)
        d = docs.get(url.rsplit("/", 1)[1])
        return None if d is None else SimpleNamespace(content=d.encode())

    rows = pd.DataFrame(m1.cik_rows(7, get))
    assert list(rows["accn"]) == ["0-16-1", "0-17-1", "0-18-1"]
    assert np.isclose(rows["cos"].iloc[1], 2 / 3) and rows["prev_accn"].iloc[1] == "0-16-1"
    assert not rows["parsed"].iloc[2] and np.isnan(rows["cos"].iloc[2])
