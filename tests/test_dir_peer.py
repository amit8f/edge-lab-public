import numpy as np
import pandas as pd

from tracks.main import dir_peer as P


def test_moves_at_uses_only_visible_bars():
    day = pd.Timestamp("2024-03-05")
    b = pd.DataFrame({"t": [day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=9, minutes=14),
                            day + pd.Timedelta(hours=9, minutes=20), day + pd.Timedelta(hours=9, minutes=35),
                            day + pd.Timedelta(hours=9, minutes=36)],
                      "c": [101.0, 102.0, 999.0, 105.0, 999.0]})
    pm, m35 = P.moves_at(b, day + pd.Timedelta(hours=9, minutes=30), 100.0)
    assert np.isclose(pm, 0.02) and np.isclose(m35, 0.05)
    assert all(np.isnan(P.moves_at(None, day, 100.0)))


def test_peer_side_rule():
    # peers agree with an up stock leading its ETF -> follow (long); peers disagree -> fade (short)
    s = P.peer_side([0.05, 0.05, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01], [0.02, -0.02, 0.02, 0.0], [5, 5, 2, 5])
    assert list(s) == [1.0, -1.0, 0.0, 0.0]
    # stock lags its ETF on the way up, peers agree -> follow the negative residual (short)
    assert P.peer_side([0.01], [0.03], [0.02], [4])[0] == -1.0


def test_balanced_accuracy():
    side = np.array([1, 1, -1, -1, 1, 0])
    lab = np.array([1, 1, -1, 1, 0, 1])
    assert np.isclose(P.balanced_accuracy(side, lab), (2 / 3 + 1.0) / 2)


def test_random_sides_seeded_per_day():
    d = pd.DatetimeIndex(["2024-01-02"] * 5 + ["2024-01-03"] * 5)
    a = P.random_sides(d)
    assert (a == P.random_sides(d)).all() and set(a) <= {-1.0, 1.0}
    assert (a[:5] == P.random_sides(d[:5])).all()


def _df(n_days, edge, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2021-01-04", periods=n_days):
        for _ in range(20):
            lab = rng.choice([1, -1, 0])
            side = rng.choice([1, -1]) if lab == 0 else lab if rng.random() < 0.5 + edge else -lab
            ex = 100 * (1 + 0.08 * lab) if lab else 100 * (1 + rng.normal(0, 0.01))
            rows.append({"d": d, "open": 20.0, "side": side, "lab": lab, "gap": rng.choice([1.0, -1.0]),
                         "entry": 100.0, "exit": ex})
    return pd.DataFrame(rows)


def test_entry_result_pass_and_fail():
    df = _df(60, 0.4)
    r = P.entry_result(df, df["side"], df["gap"], df["entry"], df["exit"], df["lab"])
    assert r["verdict"] == "passed" and r["balanced_accuracy"] > 0.8
    df = _df(60, 0.0, seed=3)
    r = P.entry_result(df, df["side"], df["gap"], df["entry"], df["exit"], df["lab"])
    assert r["verdict"] == "failed"
    small = df[df["d"] < df["d"].min() + pd.Timedelta(days=10)]
    assert P.entry_result(small, small["side"], small["gap"], small["entry"], small["exit"], small["lab"])["verdict"] == "too few"
