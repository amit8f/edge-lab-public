"""PICK2: target, bars, the 2-a-day book, the kept rows and the choice (synthetic data, no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main import pick2_study as S2
from tracks.main import pick_study as S

D1, D2 = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-02")
WPT = {0: 1.5, 1: 2.0, 2: 2.0}


def rows(recs):
    out = []
    for d, hm, sym, p, net in recs:
        out.append({"symbol": sym, "d": d, "t": pd.Timestamp(f"{d.date()} {hm}"), "tier": 2, "bad": False, "p": p,
                    "net_1.5": -9.0, "net_2.0": net, "exit_1.5": "stop", "exit_2.0": "close"})
    return pd.DataFrame(out)


def test_target_is_net_over_swing_clipped_and_nan_where_undefined():
    y = S2.target([0.02, -0.5, 0.5, np.nan, 0.01], [0.01, 0.01, 0.01, 0.01, 0.0])
    assert y[0] == pytest.approx(2.0) and y[1] == -S2.CLIP and y[2] == S2.CLIP and np.isnan(y[3]) and np.isnan(y[4])


def test_row_net_takes_the_width_of_the_rows_tier():
    D = pd.DataFrame({"net_1.0": [9.0, 9.0], "net_1.5": [0.01, 0.02], "net_2.0": [0.03, 0.04]})
    assert list(S2.row_net(D, [0, 2], WPT)) == pytest.approx([0.01, 0.04])


def test_bars_are_top_shares_and_tops_sit_above_them():
    p = np.linspace(0, 1, 100001)
    b = S2.bars_of(p)
    assert b["0.001"] == pytest.approx(0.999) and b["0.0001"] == pytest.approx(0.9999)
    t = S2.tops_of(p, b)
    assert all(t[q] >= b[q] for q in b)


def test_book_buys_the_first_two_stocks_to_clear_one_row_a_stock():
    R = rows([(D1, "09:35", "A", 0.5, 0.01), (D1, "09:40", "A", 0.9, 0.05),     # A again: ignored
              (D1, "09:40", "B", 0.4, 0.02), (D1, "09:40", "C", 0.6, -0.03),    # same time: C before B
              (D1, "09:45", "D", 0.9, 0.08),                                    # third: not bought
              (D1, "09:30", "E", 0.1, 0.09),                                    # below the bar
              (D2, "10:00", "F", 0.5, np.nan), (D2, "10:05", "G", 0.5, 0.04)])  # no trade for F: G is bought
    tk = S2.book(R, "p", 0.3, WPT)
    assert list(tk["symbol"]) == ["A", "C", "G"]
    assert list(tk["net"]) == pytest.approx([0.01, -0.03, 0.04]) and set(tk["exit"]) == {"close"}


def test_kept_rows_give_the_same_book_as_all_rows():
    rng = np.random.default_rng(0)
    recs = [(d, f"{9 + (35 + 5 * k) // 60:02d}:{(35 + 5 * k) % 60:02d}", f"S{j}", float(rng.random()),
             float(rng.normal(0, 0.02))) for d in (D1, D2) for k in range(30) for j in range(40)]
    R = rows(recs)
    bars = {"p": {"0.1": 0.9, "0.01": 0.99}}
    K = pd.concat([S2.keep_first(g, bars) for _, g in R.groupby("d")])
    assert len(K) < len(R) / 10
    for thr in bars["p"].values():
        pd.testing.assert_frame_equal(S2.book(K, "p", thr, WPT), S2.book(R, "p", thr, WPT))


def test_choice_is_the_best_mean_net_with_enough_trades():
    table = {"XGBoost": {"0.001": {"trades": 300, "mean_net": 0.001}, "0.0001": {"trades": 40, "mean_net": 0.02}},
             "CatBoost": {"0.001": {"trades": 250, "mean_net": 0.003}}}
    assert S2.choose(table) == {"learner": "CatBoost", "q": "0.001", "trades": 250, "mean_net": 0.003}
    assert S2.choose({"XGBoost": {"0.001": {"trades": 99, "mean_net": 0.5}}}) is None


def test_fit_reads_2019_to_2023_only():
    with pytest.raises(ValueError):
        S.read_joined("nowhere", "2024-01", ["d"])


@pytest.mark.parametrize("name", S2.LEARNERS)
def test_learners_fit_a_number_and_reload(name, tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (2 * X[:, 0] + rng.normal(0, 0.5, 3000)).astype(np.float32)
    m = S2.make(name)
    m.fit(X, y)
    p = S2.predict(m, X)
    assert np.corrcoef(p, y)[0, 1] > 0.7
    path = S.save_model(name, m, str(tmp_path / "m"))
    assert np.allclose(S2.predict(S2.load_model(path), X), p, atol=1e-4)
