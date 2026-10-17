"""S1 VWAP snap-back: indicators, signals, exits and the random twin on synthetic bars (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main import s1_vwap as s


def frame(closes, vols=None, opens=None, sessions=None):
    n = len(closes)
    c = np.asarray(closes, float)
    o = np.asarray(opens if opens is not None else c, float)
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"o": o, "h": np.maximum(o, c), "l": np.minimum(o, c), "c": c,
                         "v": np.asarray(vols if vols is not None else [100.0] * n, float),
                         "session": sessions if sessions is not None else ["s1"] * n}, index=idx)


def test_vwap_bands_match_hand_computation():
    d = s.indicators(frame([10, 11, 12], vols=[1, 2, 1]))
    tp = np.array([10, 11, 12.0])
    w = np.array([1, 2, 1.0])
    vw = (tp * w).sum() / w.sum()
    assert d["vwap"].iloc[-1] == pytest.approx(vw)
    assert d["sd"].iloc[-1] == pytest.approx(np.sqrt((w * (tp - vw) ** 2).sum() / w.sum()))
    assert d["n"].tolist() == [1, 2, 3] and d["last"].tolist() == [False, False, True]


def test_long_signal_after_a_flush_on_volume_and_exit_at_vwap():
    rng = np.random.default_rng(0)
    c = list(100 + rng.normal(0, 0.05, 80))
    c += [99.0, 98.0, 97.0, 96.0, 95.0, 94.0]          # flush lower, StochRSI pinned low
    c += [94.3]                                       # green bounce bar on volume
    c += [95, 97, 99.5, 101, 101, 101, 101, 101, 101]
    n = len(c)
    o = c.copy()
    o[86] = 94.0
    v = [100.0] * n
    v[86] = 400.0
    d = s.indicators(frame(c, v, o))
    d.loc[d.index[86], "k"] = 10.0                     # force K rising from low (the sign rule is what we test)
    d.loc[d.index[85], "k"] = 5.0
    sides = s.signal_sides(d)
    assert sides[86] == 1 and (sides[:86] >= 0).all()
    tr = s.simulate(d, sides, 0.002)
    t = next(x for x in tr if x["side"] == 1)
    j = 87 + next(i for i, x in enumerate(d["c"].to_numpy()[87:]) if x >= d["vwap"].to_numpy()[87 + i])
    assert t["bars"] == j - 86 and t["gross"] == pytest.approx(d["c"].iloc[j] / d["o"].iloc[87] - 1)
    assert t["net"] == pytest.approx(t["gross"] - 0.002)


def test_time_exit_session_end_and_one_position_at_a_time():
    c = np.full(40, 10.0)
    vw = np.full(40, 20.0)                            # never reached
    last = np.zeros(40, bool)
    last[30] = True
    j, g = s.trade_from(c, c, vw, last, 5, 1)
    assert j == 6 + s.MAX_BARS - 1 and g == 0.0
    j, _ = s.trade_from(c, c, vw, last, 25, 1)
    assert j == 30                                    # session end comes first
    d = frame(list(c), sessions=["a"] * 31 + ["b"] * 9)
    d = s.indicators(d)
    sides = np.zeros(40, int)
    sides[[21, 22, 23]] = 1
    tr = s.simulate(d, sides, 0.0)                   # flat prices: each trade exits on its entry bar (at VWAP)
    assert [x["t"] for x in tr] == [d.index[22], d.index[24]]   # the signal at 22 comes while 21's trade is open


def test_random_twin_same_session_and_shorts_blocked_when_not_shortable():
    c = list(np.linspace(10, 11, 60))
    d = s.indicators(frame(c, sessions=["a"] * 30 + ["b"] * 30))
    sides = np.zeros(60, int)
    sides[[40]] = 1
    sides[[50]] = -1
    tr = s.simulate(d, sides, 0.001, rng=np.random.default_rng(1), shortable=False)
    assert [x["side"] for x in tr] == [1] and "ctrl_net" in tr[0] and tr[0]["session"] == "b"


def test_primary_needs_200_trades_and_uses_day_differences():
    assert s.primary([{"session": "2024-01-02", "net": 0.01, "gross": 0.01, "ctrl_net": 0.0}], "crypto")["verdict"] == "no conclusion"
    days = pd.bdate_range("2023-01-02", periods=300)
    tr = [{"session": f"{d:%Y-%m-%d}", "side": 1, "bars": 3, "net": 0.002 + 0.0001 * (i % 3), "gross": 0.003,
           "ctrl_net": -0.001} for i, d in enumerate(days)]
    r = s.primary(tr, "crypto")
    assert r["trades"] == 300 and r["mean"] == pytest.approx(np.mean([x["net"] + 0.001 for x in tr]))
    assert r["verdict"] == "PASS"


def test_crypto_bars_binance_microseconds_and_period_cut(monkeypatch):
    def month(sym, ym):
        t0 = pd.Timestamp(f"{ym}-01", tz="UTC")
        idx = pd.date_range(t0, periods=3, freq="5min")
        return pd.DataFrame({"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": [1.0, 0.0, 2.0]}, index=idx)
    monkeypatch.setattr(s, "binance_month", month)
    d, src = s.crypto_bars("BTCUSDT")
    assert src == "binance" and (d["v"] > 0).all() and d.index.max() < pd.Timestamp("2025-10-01", tz="UTC")
    assert d["session"].iloc[0] == "2021-01-01"


def test_stock_bars_keep_regular_session_only(monkeypatch):
    day = pd.Timestamp("2024-01-03")
    sessions = pd.DataFrame({"open": [pd.Timestamp("2024-01-03 14:30", tz="UTC")],
                             "close": [pd.Timestamp("2024-01-03 21:00", tz="UTC")]}, index=pd.DatetimeIndex([day]))
    ts = pd.date_range("2024-01-03 14:00", "2024-01-03 21:30", freq="5min", tz="UTC")
    rows = [{"t": t.isoformat(), "o": 1, "h": 1, "l": 1, "c": 1, "v": 0 if i == 10 else 5} for i, t in enumerate(ts)]
    monkeypatch.setattr(s, "_pages", lambda url, params: rows)
    d = s.stock_bars("SPY", day, day, sessions)
    assert d.index.min() == pd.Timestamp("2024-01-03 14:30", tz="UTC")
    assert d.index.max() == pd.Timestamp("2024-01-03 20:55", tz="UTC") and len(d) == 77   # 78 bars, one zero-volume
    assert set(d["session"]) == {"2024-01-03"}


def test_stock_universe_requests_base_symbols(monkeypatch):
    import core.bars
    import core.env
    a = pd.DataFrame({"symbol": ["AAA", "BBB_DELISTED"], "base_symbol": ["AAA", "BBB"]})
    seen = []

    def fake_bars(syms, start, end, adjustment, headers, **kw):
        seen.extend(syms)
        return {}

    monkeypatch.setattr(s, "_assets", lambda: a)
    monkeypatch.setattr(core.env, "alpaca_headers", lambda: {})
    monkeypatch.setattr(core.bars, "fetch_daily_bars", fake_bars)
    monkeypatch.setattr(core.bars, "to_frames", lambda raw: {})
    s.stock_universe(log=lambda m: None)
    assert seen == ["AAA", "BBB"]
