import numpy as np
import pandas as pd

from tracks.news import nxvote_study as V


def test_vote_needs_k_agreeing_and_a_majority():
    E = np.array([[1.2, 1.5, 0.3, -1.1],      # 2 up, 1 down
                  [1.2, -1.5, 1.0, -1.1],     # 2 up, 2 down: tie
                  [-2.0, -1.0, -1.4, 0.0],    # 3 down
                  [0.9, 0.9, 0.9, 0.9]])      # nobody fires
    side, conf = V.vote(E, 2)
    assert side.tolist() == [1, 0, -1, 0]
    assert np.isclose(conf[0], 1.35) and np.isclose(conf[2], 4.4 / 3) and conf[1] == 0
    side3, _ = V.vote(E, 3)
    assert side3.tolist() == [0, 0, -1, 0]


def test_alone():
    side, conf = V.alone([1.5, -0.5, -1.0])
    assert side.tolist() == [1, 0, -1] and conf.tolist() == [1.5, 0.0, 1.0]


def test_book_sizes_and_control():
    d = pd.Timestamp("2024-03-01")
    real = pd.DataFrame({"symbol": list("ABCDEF"), "d": d, "t": pd.date_range("2024-03-01 09:31", periods=6, freq="min"),
                         "ret": [0.02, -0.01, 0.0, 0.01, -0.02, 0.0], "price": 20.0, "days": 1})
    side = np.array([1, -1, 0, 0, 0, 0])
    conf = np.array([2.0, 1.0, 0, 0, 0, 0])
    days, tr = V.book(real, side, conf, 7)
    assert len(tr) == 2 and tr["w"].round(3).tolist() == [0.1, 0.05]
    assert days["n"].tolist() == [2] and np.isfinite(days["sized_pmc"]).all()
    assert np.isclose(days["sized_net"][0], (tr["w"] * tr["net"]).sum())
    days0, tr0 = V.book(real, np.zeros(6, int), np.zeros(6), 7)
    assert days0["sized_net"].tolist() == [0.0] and tr0.empty


def test_top3_share():
    assert V.top3_share([3, 1, 1, -1, 1]) == 1.0
    assert V.top3_share([-1, -1]) is None
