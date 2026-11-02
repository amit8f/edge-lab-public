"""W2's headline-naming rule and its filter on W1a's classifier (synthetic headlines, no network)."""
import pandas as pd

from tracks.news import w2_shorts as W


def test_named_by_ticker_or_short_name():
    assert W.named("Spruce Point publishes short report on XYZ", "XYZ", None)
    assert W.named("Citron short on $ABC", "ABC", None)
    assert W.named("Short report: (NASDAQ: QQ) shares fall", "QQ", None)
    assert W.named("Hindenburg Research says Super Micro Computer fraud report", "SMCI", "Super Micro Computer, Inc.")
    assert not W.named("Hindenburg short report on peers", "SMCI", "Super Micro Computer, Inc.")
    assert not W.named("Short seller targets XYZW", "XYZ", None)


def test_month_events_keeps_one_symbol_named_short_seller_articles(monkeypatch):
    news = pd.DataFrame({"id": [1, 2, 3], "created_at": ["2020-03-02T15:00:00Z"] * 3,
                         "headline": ["a", "b", "c"], "symbols": ["AAA", "AAA,BBB", "CCC"]})

    def fake(n):
        return pd.DataFrame({"id": [1, 2, 2, 3], "t": pd.Timestamp("2020-03-02 10:00"), "symbol": ["AAA", "AAA", "BBB", "CCC"],
                             "group": ["short_seller", "short_seller", "short_seller", "big"], "sign": -1,
                             "src_words": "x", "deal_words": "short", "n_symbols": [1, 2, 2, 1]})
    monkeypatch.setattr(W.W, "classified", fake)
    monkeypatch.setattr(W, "named", lambda h, s, n: True)
    assert W.month_events(news, {})["id"].tolist() == [1]


def test_fill_urls_matches_by_id_and_keeps_existing():
    chk = pd.DataFrame({"id": [10, 11], "t": ["2020-03-02 10:00:00", "2020-03-03 10:00:00"], "symbol": ["AAA", "BBB"],
                        "url": [None, "https://kept"]})
    calls = []

    def fetch(sym, a, b):
        calls.append(sym)
        return [{"id": 9, "url": "https://wrong"}, {"id": 10, "url": "https://right"}]
    out = W.fill_urls(chk, fetch)
    assert out["url"].tolist() == ["https://right", "https://kept"] and calls == ["AAA"]


def test_verdict_follows_the_registered_reading():
    ok = {"mean": 0.02, "t_hac": 3.4, "ci95": [0.004, 0.04]}
    good = {"alpha": 0.01, "t": 2.3}
    assert W.verdict(ok, good, 99) == "inconclusive"                      # under 100 events
    assert W.verdict(ok, good, 150) == "passed"
    assert W.verdict(dict(ok, t_hac=2.9), good, 150) == "failed"           # t below 3
    assert W.verdict(ok, {"alpha": 0.01, "t": 1.4}, 150) == "failed"       # alpha t below 2
    assert W.verdict(dict(ok, ci95=[-0.001, 0.04]), good, 150) == "failed"
    assert W.verdict(ok, None, 150) == "failed"


def test_measure_refuses_until_the_w2_hand_check_passes(tmp_path):
    chk = pd.DataFrame({"id": range(50), "ok": ["y"] * 39 + ["n"] * 11})
    chk.to_csv(tmp_path / "W2_check.csv", index=False)
    assert not W.W.check_passed(tmp_path, W.CHECK_FILE)[0]
    assert W.stage_measure("unused-cache", tmp_path, 1) is False
    chk["ok"] = ["y"] * 40 + ["n"] * 10
    chk.to_csv(tmp_path / "W2_check.csv", index=False)
    assert W.W.check_passed(tmp_path, W.CHECK_FILE)[0]


def test_diagnose_splits_gross_costs_and_the_move_before_entry():
    n = 40
    m = pd.DataFrame({"sign": -1, "r": [-0.02] * n, "r_spy": [0.0] * n, "net": [0.02 - 0.03] * n,
                      "ac": [100.0] * n, "entry_px": [95.0] * (n // 2) + [101.0] * (n // 2),
                      "t": pd.to_datetime(["2020-03-02 08:00"] * n)})
    d = W.diagnose(m)
    assert abs(d["gross_mean"] - 0.02) < 1e-9 and abs(d["costs_mean"] - 0.03) < 1e-9
    assert d["by_pre_entry_move"]["fell_over_3pct"]["n"] == 20 and d["by_pre_entry_move"]["rose"]["n"] == 20
    assert d["by_article_time"]["before_open"]["n"] == 40 and d["by_article_time"]["in_session"]["n"] == 0
