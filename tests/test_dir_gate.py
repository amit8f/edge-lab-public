import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tracks.news import dir_gate as G


def _vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(
        ["acme wins navy contract", "acme reports earnings", "acme names new ceo", "widget sales rise"])


def test_similarity_repeat_and_new():
    v = _vec()
    assert G.max_similarity(v, ["acme wins navy contract"], ["acme wins navy contract"]) > 0.99
    assert G.max_similarity(v, ["acme names new ceo"], ["widget sales rise"]) < 0.1
    assert G.max_similarity(v, ["acme names new ceo"], []) == 0.0


def test_gate_rule():
    p = pd.DataFrame({"sim": [0.1, 0.9, 0.1, 0.1], "f_earnings": [0, 0, 1, 0], "f_fda": [0, 0, 0, 1]})
    assert list(G.gated(p).index) == [0]


def test_control_seeded_and_too_few():
    e = pd.DataFrame({"net_long": np.arange(10, dtype=float)})
    a = G.control_net(e, 3, "2024-03-05")
    assert a == G.control_net(e, 3, pd.Timestamp("2024-03-05"))
    assert np.isnan(G.control_net(e, 11, "2024-03-05"))


def _longs(n_days=60, seed=1, gain=0.01):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2023-01-02", periods=n_days)):
        for j in range(12):
            novel = j % 2 == 0
            rows.append({"d": d, "symbol": f"S{j}", "sim": 0.2 if novel else 0.95, "f_earnings": 0, "f_fda": 0,
                         "net": rng.normal(0.0, 0.01) + (gain if novel else -gain)})
    return pd.DataFrame(rows)


def test_report_passes_planted_gain_and_too_few():
    lo = _longs()
    ev = lo.rename(columns={"net": "net_long"})[["d", "symbol", "net_long"]].assign(net_long=lambda x: x["net_long"] * 0 - 0.01)
    r = G.report(lo, ev)
    assert r["gated_trades"] == 360 and r["verdict"] == "too few"
    lo2 = _longs(n_days=100)
    ev2 = lo2.rename(columns={"net": "net_long"})[["d", "symbol", "net_long"]].assign(net_long=-0.01)
    r2 = G.report(lo2, ev2)
    assert r2["verdict"] == "passed" and r2["gain_over_all_longs"]["lo"] > 0
    assert len(r2["novelty_deciles_all_longs"]) == 10


def test_report_fails_without_gain():
    lo = _longs(n_days=100, gain=0.0)
    ev = lo.rename(columns={"net": "net_long"})[["d", "symbol", "net_long"]]
    assert G.report(lo, ev)["verdict"] == "failed"


def test_similarities_window():
    ses = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=30)})
    ses["prev_close_t"] = ses["date"] - pd.Timedelta(hours=8)        # 16:00 the day before
    longs = pd.DataFrame({"symbol": ["A", "A"], "d": [ses["date"][25], ses["date"][25]],
                          "t": [ses["date"][25] + pd.Timedelta(hours=9, minutes=31)] * 2})
    texts = pd.DataFrame({"symbol": ["A", "A", "A"],
                          "t": [ses["date"][2], ses["date"][20], ses["date"][25] + pd.Timedelta(hours=8)],
                          "text": ["acme names new ceo", "acme wins navy contract", "acme wins navy contract"]})
    s = G.similarities(longs, texts, _vec(), ses)
    assert (s > 0.99).all()
    texts.loc[1, "t"] = ses["date"][1]                                 # older than 20 sessions: not counted
    assert (G.similarities(longs, texts, _vec(), ses) < 0.99).all()
