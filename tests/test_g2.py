"""G2's stricter matcher on synthetic sentences (no network): exact names only, customer sentences only,
a reportable share, and links rebuilt from G1's stored sentences with G1's windows."""
import pandas as pd

from tracks.main import g2_links as G


NAMES = G.Names([(1, "Walmart Inc."), (2, "Rocky Brands, Inc."), (3, "Apple Inc."), (4, "AT&T Inc."), (9, "Heartland Express")])


def test_exact_names_only():
    assert NAMES.match("Walmart Inc.") == 1 and NAMES.match("Walmart") == 1
    assert NAMES.match("Rocky") is None                       # G1's prefix match is gone
    assert NAMES.match("AT&T") is None                        # one word under 5 letters


def test_sentence_rules():
    assert G.keep_sentence("One customer, Walmart Inc., accounted for 10.9% of our revenues.")
    assert not G.keep_sentence("Our Industrial segment represents 47% of sales to customers.")
    assert not G.keep_sentence("Sales to Walmart accounted for 8% of revenues.")
    assert not G.keep_sentence("Apple represented 30% of revenues.")         # no customer word


def test_links_rebuilt_from_g1_sentences_with_windows():
    s1 = "Sales to Apple Inc. and Walmart represented 22% and 11% of revenues, respectively."
    rows = pd.DataFrame([
        {"supplier_cik": 9, "filed": pd.Timestamp("2020-02-25"), "accn": "a1", "doc": "x.htm", "parsed": True,
         "customer_cik": 3, "phrase": "Apple Inc", "pct": 0.22, "sentence": s1},
        {"supplier_cik": 9, "filed": pd.Timestamp("2020-02-25"), "accn": "a1", "doc": "x.htm", "parsed": True,
         "customer_cik": 1, "phrase": "Walmart", "pct": 0.22, "sentence": s1},
        {"supplier_cik": 9, "filed": pd.Timestamp("2021-02-20"), "accn": "a2", "doc": "y.htm", "parsed": True,
         "customer_cik": 2, "phrase": "Rocky", "pct": 0.5, "sentence": "Customers in the Rocky Mountain area were 50% of sales."},
    ])
    lk = G.links(rows, NAMES)
    assert sorted(lk["customer_cik"]) == [1, 3] and (lk["pct"] == 0.22).all()
    assert (lk["until"] == pd.Timestamp("2021-02-20")).all()            # the next 10-K ends the window
