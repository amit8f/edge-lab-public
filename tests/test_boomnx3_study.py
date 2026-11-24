import numpy as np
import pandas as pd

from tracks.news import boomnx3_study as BN


def test_tag_boom_ranks_within_the_night():
    d = pd.Timestamp("2024-02-01")
    cand = pd.DataFrame({"symbol": ["A", "B", "C"], "d": d - pd.Timedelta(days=1), "nxt": d,
                         "score": [0.9, 0.5, 0.7], "boom": [1, 0, 1]})
    tr = pd.DataFrame({"d": [d, d, d], "symbol": ["A", "B", "Z"], "side": 1, "net": [0.1, 0.0, -0.1]})
    out = BN.tag_boom(tr, cand, top=2)
    assert out["rank"].tolist()[:2] == [1.0, 3.0] and np.isnan(out["rank"].iloc[2])
    assert out["in_top"].tolist() == [True, False, False]


def test_matched_control_uses_unfired_same_bucket_events():
    d = pd.Timestamp("2024-03-01")
    ev = pd.DataFrame({"d": d, "symbol": list("ABCDEFGHIJKLMNO"), "ret": [0.05, 0.01, 0.03] + [0.0] * 12,
                       "price": [10.0] * 15, "days": 1, "vol20": [0.5, 0.51, 0.52] + list(np.linspace(0.01, 0.1, 12))})
    fired = pd.DataFrame({"d": [d], "symbol": ["A"], "side": [1], "net": [0.04]})
    m = BN.matched(ev, fired)
    assert len(m) == 1
    from tracks.news import nx1_study as N
    c = N.cost_frac([10.0], [False], [1])[0]
    assert np.isclose(m["ctrl"].iloc[0], np.mean([0.01, 0.03]) - c)


def test_top3_share():
    assert BN.top3_share([2, 1, 1, -1, 1]) == 1.0
