import numpy as np
import pandas as pd

from tracks.news import hi3_replay as R
from tracks.news import nx3_study as S


def test_news_start_needs_a_full_month_from_april():
    info = {"2026-03": (22, 22), "2026-04": (5, 22), "2026-05": (20, 21), "2026-06": (22, 22)}
    assert R.news_start(info) == "2026-05"
    assert R.news_start({"2026-04": (1, 22)}) is None


def test_price_at_uses_last_bar_starting_by_t():
    b = pd.DataFrame({"t": pd.to_datetime(["2026-06-01 09:30", "2026-06-01 09:31", "2026-06-01 09:32"]),
                      "o": [1, 2, 3], "c": [1.5, 2.5, 3.5], "v": [1, 1, 1]})
    assert R.price_at(b, pd.Timestamp("2026-06-01 09:31:30")) == 2.5
    assert np.isnan(R.price_at(b, pd.Timestamp("2026-06-01 09:29")))
    assert np.isnan(R.price_at(None, pd.Timestamp("2026-06-01 09:31")))


class Fake:
    cut = 1.0

    def __init__(self, e):
        self.e = np.asarray(e, float)

    def effect(self, X):
        return self.e[: len(X)]


def _dec(n):
    return [{"symbol": f"S{i}", "t": f"2026-06-01 09:{30 + i:02d}:00", "etb": True,
             "inputs": {k: 0.0 for k in S.INPUTS}} for i in range(n)]


def test_arm_book_hi_bar_and_slots():
    e = [1.2, 2.0, -1.6, -1.1, 0.5, 3.0, 1.7, 1.9]
    p, c = R.arm_book(_dec(8), Fake(e), 1, hi=False)
    assert [x["side"] for x in p] == [1, 1, -1, -1, 1] and len(c) == len(p)
    ph, _ = R.arm_book(_dec(8), Fake(e), 1, hi=True)
    assert [x["symbol"] for x in ph] == ["S1", "S2", "S5", "S6"]       # |e| >= 1.5, 3 longs max
    assert all(abs(x["e"]) >= 1.5 for x in ph)


def test_summarize_pick_minus_control():
    rows = []
    for d in range(10):
        day = f"2026-06-{d + 1:02d}"
        rows += [{"day": day, "role": "picks", "side": 1, "net": 0.01 + 0.001 * d},
                 {"day": day, "role": "control", "side": 1, "net": 0.0}]
    s = R.summarize(pd.DataFrame(rows), [f"d{i}" for i in range(12)])
    assert s["trades"] == 10 and s["days_with_picks"] == 10 and s["win"] == 1.0
    assert s["pick_minus_control"] > 0 and s["t"] > 1.5 and s["passes"]
    assert R.summarize(pd.DataFrame(rows).iloc[0:0], ["d"])["trades"] == 0
