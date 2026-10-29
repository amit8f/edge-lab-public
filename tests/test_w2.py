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
