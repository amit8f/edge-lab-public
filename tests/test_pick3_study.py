"""PICK3: synthetic data only, no network."""
import numpy as np
import pandas as pd

from tracks.main import pick3_study as P3


def test_bars_wait_for_60_sessions_and_use_only_the_last_60():
    assert P3.bars_from([1.0] * 59) is None
    m = [100.0] * 10 + list(np.arange(60, dtype=float))
    n, b = P3.bars_from(m)
    assert abs(n - np.quantile(np.arange(60), 0.3)) < 1e-12 and abs(b - np.quantile(np.arange(60), 0.7)) < 1e-12


def test_picks_take_first_clear_per_stock_bad_rows_need_the_higher_bar_and_r_must_be_positive():
    t = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:00", "2024-01-02 10:05", "2024-01-02 10:05",
                        "2024-01-02 10:10"])
    g = pd.DataFrame({"symbol": ["A", "B", "A", "C", "D"], "t": t, "r": [0.5, 0.8, 0.9, 0.6, 0.7],
                      "bad": [False, True, False, True, False]})
    pk = P3.picks_of_day(g, 0.4, 0.7, n=5)
    assert list(pk["symbol"]) == ["B", "A", "D"]               # C is bad and under 0.7; B clears 0.7; A once
    assert list(P3.picks_of_day(g, 0.4, 0.7)["symbol"]) == ["B", "A"]
    assert P3.picks_of_day(g.assign(r=-1.0), -2.0, -2.0).empty


def test_check_rejects_too_few_trades_and_passes_a_clear_edge():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-02", periods=200)
    d = np.repeat(days, 1)
    tk = pd.DataFrame({"d": d, "net": 0.02 + rng.normal(0, 0.01, len(d)), "ctrl_net": rng.normal(0, 0.01, len(d)),
                       "bad": False, "exit": "close"})
    ok = P3.check(tk, days)
    assert ok["verdict"] == "not rejected" and ok["trades_per_day"] == 1.0
    few = P3.check(tk.head(100), days)
    assert few["verdict"] == "rejected" and not few["rules"]["trades_150"]


def test_a_synthetic_day_scores_with_the_committed_models_and_trades_with_controls():
    import json

    from tracks.main import pick_study as S
    from tracks.main import pick_test as T
    day = pd.Timestamp("2024-03-04")
    t = pd.date_range("2024-03-04 09:30", "2024-03-04 15:55", freq="5min")

    def b(seed):
        r = np.random.default_rng(seed)
        c = 50 * np.exp(np.cumsum(r.normal(0, 0.004, len(t))))
        o = np.r_[50.0, c[:-1]]
        return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) * 1.002, "l": np.minimum(o, c) * 0.998, "c": c,
                             "v": 40000.0})
    syms = [f"S{i}" for i in range(5)]
    bars = {s: b(i) for i, s in enumerate(syms + ["SPY"])}
    uni = {s: ("main", 50.0, 5e7) for s in syms}
    D = T.day_inputs(day, uni, bars, {s: [0.01] * 20 for s in syms})
    sess = np.array(pd.bdate_range("2024-01-01", "2024-03-05").values.astype("datetime64[ns]").astype(np.int64))
    spy = pd.DataFrame({"date": pd.bdate_range("2023-12-01", "2024-03-05"), "ao": 400.0, "ac": 400.0, "c": 400.0})
    model = P3.Model(json.loads(P3.FIT.read_text()))
    g = model.score(S.join_rows(D, pd.DataFrame(columns=S.NEWS_TONE), sess, S.spy_parts(spy)))
    assert np.isfinite(g["r"]).all() and len(g) == len(D)
    lo = float(g["r"].min()) - 1
    pk = P3.day_trades(g, bars, model, (lo, lo), day)
    assert len(pk) <= 2 and np.isfinite(pk["net"]).all() and pk["ctrl_net"].notna().all()
