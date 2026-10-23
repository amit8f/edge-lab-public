"""G1 link extraction and signal construction, on synthetic data only (no network)."""
import numpy as np
import pandas as pd

from tracks.main import g1_links as g1

NAMES = g1.Names([(1, "WAL MART STORES INC"), (2, "AMAZON COM INC"), (3, "APPLE INC"),
                  (4, "APPLE HOSPITALITY REIT, INC."), (5, "ACME WIDGETS CORP"), (6, "GENERAL MILLS INC"),
                  (7, "GENERAL MOTORS CO")])


def test_key_and_match():
    assert g1.key("Wal-Mart Stores, Inc.") == g1.key("WAL MART STORES INC") == "walmartstores"
    assert g1.key("The Boeing Company") == "boeing"
    assert NAMES.match("Wal-Mart Stores") == 1
    assert NAMES.match("Amazon") == 2                     # prefix of exactly one company
    assert NAMES.match("Apple") == 3                      # exact beats the prefix shared with the REIT
    assert NAMES.match("General") is None                 # prefix of two companies
    assert NAMES.match("Sales") is None and NAMES.match("Acme") is None   # stop word; too short


def test_customer_sentences_and_extract():
    text = ("We sell widgets. Sales to Wal-Mart Stores, Inc. accounted for 23% of our net sales in 2019.\n"
            "Amazon represented approximately 11 percent of revenue. Apple is a great company.\n"
            "Our largest customer, Acme Widgets Corp, was 15% of revenues.")
    s = g1.customer_sentences(text)
    assert len(s) == 3                                  # "Apple is a great company." has no share
    rows = g1.extract(text, NAMES, self_cik=5)
    found = {c: p for c, _, p, _ in rows}
    assert found == {1: 0.23, 2: 0.11}                  # Acme is the filer itself
    rows2 = g1.extract("Customers Amazon and Wal-Mart Stores each accounted for 12% of net sales.", NAMES, 99)
    assert {c for c, *_ in rows2} == {1, 2}


def test_valid_links_window():
    rows = pd.DataFrame({"supplier_cik": [9, 9, 9], "filed": pd.to_datetime(["2017-03-01", "2018-02-20", "2020-03-01"]),
                         "accn": ["a1", "a2", "a3"], "doc": "d", "parsed": True,
                         "customer_cik": [1.0, np.nan, 2.0], "phrase": ["x", None, "y"], "pct": [0.2, np.nan, 0.1],
                         "sentence": ["s", None, "t"]})
    v = g1.valid_links(rows).set_index("accn")
    assert v.loc["a1", "until"] == pd.Timestamp("2018-02-20")              # the next 10-K came first
    assert v.loc["a3", "until"] == pd.Timestamp("2021-06-01")              # 15 months
    pairs = g1.customer_pairs(v.reset_index(), [pd.Timestamp("2018-01-31"), pd.Timestamp("2018-03-30")])
    assert list(pairs["date"]) == [pd.Timestamp("2018-01-31")]


def test_signal_uses_last_month_of_linked_customers():
    dates = list(pd.to_datetime(["2018-01-31", "2018-02-28", "2018-03-30"]))
    uni = pd.DataFrame({"date": np.repeat(dates, 3), "cik": [1, 2, 9] * 3,
                        "ret1": [0.10, -0.05, 0.0, 0.02, 0.04, 0.0, 0.0, 0.0, 0.0]})
    prev = g1.prev_returns(uni, dates)
    assert prev[(prev["date"] == dates[1]) & (prev["cik"] == 1)]["prev"].item() == 0.10
    pairs = pd.DataFrame({"date": [dates[1], dates[1]], "supplier_cik": [9, 9], "customer_cik": [1, 2]})
    pan = g1.linked_signal(uni[uni["date"] == dates[1]], prev, pairs, "customer_cik", "supplier_cik")
    assert pan.set_index("cik").loc[9, "score"] == 0.025 and pan.set_index("cik").loc[1, "score"] != pan.set_index("cik").loc[1, "score"]


def test_ls_skips_thin_dates():
    rng = np.random.default_rng(0)
    rows = []
    for t, n in ((pd.Timestamp("2018-01-31"), 60), (pd.Timestamp("2018-02-28"), 30)):
        s = rng.normal(size=n)
        rows.append(pd.DataFrame({"date": t, "cik": range(n), "score": s, "ret1": s * 0.01}))
    ls, skipped = g1.ls_quintiles(pd.concat(rows))
    assert skipped == 1 and len(ls) == 1 and ls["excess"].iloc[0] > 0


def test_insider_and_comention_links():
    owners = pd.DataFrame({"owner_cik": [7, 7, 7, 8], "owner_name": ["Doe Jane"] * 3 + ["Roe R"],
                           "issuer_cik": [1, 2, 3, 1], "filed": pd.to_datetime(["2018-01-10", "2017-06-01", "2015-01-01", "2018-01-02"])})
    il = g1.insider_links_at(owners, pd.Timestamp("2018-01-31"))
    assert list(zip(il["cik_a"], il["cik_b"])) == [(1, 2)]               # issuer 3 is older than 24 months
    news = pd.DataFrame({"created_at": ["2018-01-05T14:00:00Z"] * 3 + ["2018-01-06T14:00:00Z"],
                         "symbols": ["AAA,BBB", "BBB,AAA", "AAA,BBB,CCC", "AAA,BBB,CCC,DDD,EEE"]})
    c = g1.comention_counts(news)
    links = g1.comention_links_at(c, pd.Period("2018-01", "M"))
    assert list(zip(links["a"], links["b"], links["n"])) == [("AAA", "BBB", 3)]


def test_check_gate(tmp_path):
    assert g1.check_passed(tmp_path) is None
    pd.DataFrame({"ok": ["y"] * 41 + ["n"] * 8 + [""]}).to_csv(tmp_path / "G1_links_check.csv", index=False)
    assert g1.check_passed(tmp_path) is None
    pd.DataFrame({"ok": ["y"] * 41 + ["n"] * 9}).to_csv(tmp_path / "G1_links_check.csv", index=False)
    assert g1.check_passed(tmp_path) == 41


def test_item_101_contract_links():
    text = ("FORM 8-K. Item 1.01 Entry into a Material Definitive Agreement. On May 1, 2020, the Company entered "
            "into a Supply Agreement with General Motors Co. under which it will deliver widgets. "
            "Item 9.01 Financial Statements and Exhibits. Exhibit 10.1 Agreement with Apple Inc.")
    assert "General Motors" in g1.item_101(text) and "Apple" not in g1.item_101(text)
    assert [c for c, *_ in g1.contract_links(text, NAMES, self_cik=5)] == [7]
    assert g1.item_101("Item 2.02 Results of Operations.") == ""


def test_contract_hubs_and_pairs():
    filed = pd.Timestamp("2020-05-01")
    rows = pd.DataFrame({"filer_cik": list(range(100, 100 + g1.HUB_MAX + 1)) + [5],
                         "other_cik": [9.0] * (g1.HUB_MAX + 1) + [7.0], "filed": filed, "accn": "a", "doc": "d"})
    v = g1.contract_valid(rows)
    assert list(v["other_cik"]) == [7]                  # the bank named by 51 filers is a hub
    p = g1.contract_pairs(v, [pd.Timestamp("2020-06-30"), pd.Timestamp("2021-09-30")])
    assert set(zip(p["src"], p["dst"])) == {(5, 7), (7, 5)} and set(p["date"]) == {pd.Timestamp("2020-06-30")}


def test_company_names_creates_its_cache_folder(tmp_path):
    class R:
        content = b"ACME WIDGETS CORP                    10-K   5   2015-03-02  edgar/x.txt\n"
    d = g1.company_names(tmp_path / "m4" / "g1", [5], get=lambda url: R)
    assert d.to_dict("records") == [{"cik": 5, "name": "ACME WIDGETS CORP"}]
    assert (tmp_path / "m4" / "g1" / "names.parquet").exists()
