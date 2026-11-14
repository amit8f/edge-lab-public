"""Crypto studies: archive parsing, the universe rule, the trend book and the day-trade simulation (synthetic)."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from tracks.crypto import cdata as C
from tracks.crypto import day_study as D
from tracks.crypto import trend_live as L
from tracks.crypto import trend_study as T


def zipped(text, name="x.csv"):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, text)
    return b.getvalue()


def test_klines_parse_with_or_without_header_and_in_microseconds():
    row = "1704067200000,10,12,9,11,5,1704068099999,55,7,2,22,0\n"
    a = C.parse_klines(zipped(row))
    b = C.parse_klines(zipped("open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbqv,ignore\n" + row))
    c = C.parse_klines(zipped(row.replace("1704067200000", "1704067200000000")))
    for d in (a, b, c):
        assert d["t"].iat[0] == pd.Timestamp("2024-01-01") and d["c"].iat[0] == 11 and d["tbqv"].iat[0] == 22


def test_funding_parse_takes_time_and_rate():
    f = C.parse_funding(zipped("calc_time,funding_interval_hours,last_funding_rate\n1704096000003,8,0.0001\n"))
    assert f["t"].iat[0] == pd.Timestamp("2024-01-01 08:00") and f["rate"].iat[0] == 0.0001


def test_universe_is_ranked_on_the_month_before_the_quarter_and_skips_stable_and_dated_pairs():
    assert C.rank_month("2024Q1") == "2023-12" and C.rank_month("2020Q2") == "2020-03"
    v = {"BTCUSDT": 9.0, "USDCUSDT": 99.0, "ETHUSDT_240329": 50.0, "ETHUSD": 70.0, "BTCDOMUSDT": 80.0, "LUNAUSDT": 5.0,
         "XRPUSDT": 0.0}
    assert C.top_symbols(v, 2) == ["BTCUSDT", "LUNAUSDT"]


def test_trend_signal_uses_only_the_close_28_days_back():
    c = np.r_[np.full(28, 10.0), 11.0, 9.0]
    s = T.signal(c)
    assert s[:28].sum() == 0 and s[28] == 1 and s[29] == 0


def test_trend_book_pays_costs_and_holds_cash_when_out():
    ret = np.array([[0.10, 0.0], [0.10, 0.0], [-0.5, 0.0]])
    sig = np.array([[1, 0], [1, 0], [0, 0]])
    eq = T.book(ret, sig, 0.01)
    assert eq[0] == pytest.approx(0.5 + 0.5 * 0.995 * 1.1)
    assert eq[2] == pytest.approx(0.5 + 0.5 * 0.995 * 1.21 * 0.995)          # sold before the fall
    assert T.hold(ret)[-1] == pytest.approx((1.21 * 0.5 + 1) / 2)


def test_random_twin_keeps_switches_and_days_held():
    rng = np.random.default_rng(0)
    s = (rng.random(400) > 0.6).astype(int)
    t = T.twin(s, rng)
    assert t.sum() == s.sum() and len(T.runs(t)[1]) == len(T.runs(s)[1]) and t[0] == s[0] and not (t == s).all()


def bars(n=3000, seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[100.0, c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.001, n)))
    v = rng.uniform(1, 5, n)
    return pd.DataFrame({"t": pd.date_range("2021-01-01", periods=n, freq="15min"), "o": o, "h": h, "l": lo, "c": c,
                         "v": v, "qv": v * c, "n": rng.integers(10, 50, n).astype(float), "tbqv": v * c * 0.5})


@pytest.mark.parametrize("side", ["L", "S"])
def test_vector_simulation_matches_the_slow_routine(side):
    df = bars()
    o, h, lo, c = (df[x].to_numpy() for x in ("o", "h", "l", "c"))
    sw = D.swing_of(h, lo, c)
    g, kind, xi = D.simulate(o, h, lo, c, sw, side)
    idx = np.flatnonzero(np.isfinite(g))
    assert len(idx) > 2000 and set(kind[idx]) == {"stop", "tp", "time"}
    for i in idx[::37]:
        rg, rk, rx = D.trade_ref(o, h, lo, c, i, sw[i], side)
        assert g[i] == pytest.approx(rg, abs=1e-12) and kind[i] == rk and xi[i] == rx


def test_a_bar_touching_stop_and_target_counts_as_the_stop_and_entry_is_the_next_open():
    n = 40
    o = np.full(n, 100.0); h = np.full(n, 100.0); lo = np.full(n, 100.0); c = np.full(n, 100.0)
    o[11] = 101.0                                     # the entry bar's open
    h[11], lo[11] = 110.0, 90.0
    sw = np.full(n, 0.01)
    g, kind, xi = D.simulate(o, h, lo, c, sw, "L")
    assert kind[10] == "stop" and g[10] == pytest.approx(-0.015) and xi[10] == 11
    g, kind, _ = D.simulate(o, h, lo, c, sw, "S")
    assert kind[10] == "stop" and g[10] == pytest.approx(-0.015)
    assert np.isnan(D.simulate(o, h, lo, c, sw, "L")[0][n - D.H:]).all()     # no room for 16 bars


def test_funding_is_charged_to_longs_only_for_payments_after_entry():
    df = bars(n=1500, seed=3, vol=0.0001)             # quiet: every trade exits on time
    fund = pd.DataFrame({"t": [df["t"].iat[1000]], "rate": [0.001]})
    R = D.coin_rows(df, fund, {"2021Q1"})
    r = R.set_index("t")
    held = r.loc[df["t"].iat[995]]                    # enters bar 996, holds through bar 1000
    late = r.loc[df["t"].iat[999]]                    # enters at bar 1000, the payment's own bar: not charged
    assert held["exit_L"] == "time" and held["bars_L"] == D.H
    assert held["net_L"] == pytest.approx(held["gross_L"] - D.COST - 0.001)
    assert held["net_S"] == pytest.approx(held["gross_S"] - D.COST + 0.001)
    assert late["net_L"] == pytest.approx(late["gross_L"] - D.COST)


def test_features_use_only_past_bars():
    df = bars(n=1200, seed=5)
    fund = pd.DataFrame({"t": df["t"].iloc[::32].to_numpy(), "rate": 0.0001})
    a = D.features(df, fund)
    df2 = df.copy()
    df2.loc[900:, ["o", "h", "l", "c", "qv", "tbqv", "n"]] *= 3.0
    b = D.features(df2, fund)
    cols = [c for c in a.columns if c != "t"]
    pd.testing.assert_frame_equal(a.loc[:899, cols], b.loc[:899, cols])


def test_rows_keep_only_universe_quarters_and_missing_bars_are_not_decisions():
    df = bars(n=1500, seed=7).drop(index=[1200]).reset_index(drop=True)
    none = pd.DataFrame({"t": pd.to_datetime([]), "rate": []})
    assert len(D.coin_rows(df, none, {"2022Q1"})) == 0
    R = D.coin_rows(df, none, {"2021Q1"})
    assert len(R) > 500 and bars(n=1500, seed=7)["t"].iat[1200] not in set(R["t"])


def test_book_holds_one_position_a_coin_and_three_in_all():
    t0 = pd.Timestamp("2024-01-01")
    c = pd.DataFrame({"t": [t0] * 4 + [t0 + D.BAR, t0 + 2 * D.BAR], "symbol": ["A", "B", "C", "D", "A", "D"],
                      "side": "L", "pred": [4, 3, 2, 1, 9, 9], "bars": [1, 16, 16, 16, 16, 16]})
    # A, B, C fill the book; D is refused; A's one-bar trade has ended by the next entry, so A is taken again;
    # then the book is full for D
    assert D.take(c) == [0, 1, 2, 4]


def test_candidates_take_the_better_side_and_long_only_drops_shorts():
    R = pd.DataFrame({"t": pd.Timestamp("2024-01-01"), "symbol": ["A", "B"], "net_L": [0.01, 0.02],
                      "net_S": [-0.01, 0.03], "gross_L": 0.0, "gross_S": 0.0, "exit_L": "time", "exit_S": "tp",
                      "bars_L": 16, "bars_S": 3})
    pred = {"L": np.array([0.5, 0.1]), "S": np.array([0.2, 0.9])}
    c = D.candidates(R, pred, 0.4)
    assert list(c["symbol"]) == ["B", "A"] and list(c["side"]) == ["S", "L"] and list(c["net"]) == [0.03, 0.01]
    assert list(c["bars"]) == [3, 16]
    assert list(D.candidates(R, pred, 0.4, long_only=True)["symbol"]) == ["A"]


def test_verdict_needs_every_rule():
    s = {"mean_net": 0.001, "trades": 150, "book_daily": {"t_nw": 3.2},
         "minus_control_daily": {"mean": 0.001, "t_nw": 2.5}, "alpha_on_btc": {"alpha": 0.001, "t": 2.1}}
    assert D.verdict(s)[1] == "PASS"
    s["book_daily"]["t_nw"] = 2.9
    assert D.verdict(s)[1] == "FAIL"


# ---------- the trend rule's forward book ----------

def _closes(n, seed=3):
    rng = np.random.default_rng(seed)
    days = [str(d.date()) for d in pd.date_range("2026-01-01", periods=n)]
    return {c: dict(zip(days, 100 * np.cumprod(1 + rng.normal(0, 0.04, n)), strict=True)) for c in L.COINS}, days


def test_live_trend_book_repeats_the_history_book_day_by_day():
    closes, days = _closes(200)
    P = pd.DataFrame(closes)
    sig = np.column_stack([T.signal(P[c].to_numpy()) for c in L.COINS])
    ret = P.pct_change().to_numpy()
    start = 40
    eq = T.book(ret[start + 1:], sig[start:-1], L.COST)
    book, _ = L.trade(L.new_book(), L.signal_at(closes, days[start]))
    assert L.signal_at(closes, days[start]) == dict(zip(L.COINS, sig[start].astype(int), strict=True))
    got = []
    for i in range(start + 1, len(days)):
        book = L.grow(book, {c: closes[c][days[i]] / closes[c][days[i - 1]] - 1 for c in L.COINS})
        got.append(L.equity(book) / L.START_USD)
        book, _ = L.trade(book, L.signal_at(closes, days[i]))
    assert np.allclose(got, eq)
    assert np.isclose(sum(book["twin"].values()) / L.START_USD, T.hold(ret[start + 1:])[-1])


def test_live_trend_job_starts_on_the_newest_day_and_catches_up(tmp_path):
    import json
    closes, days = _closes(60)
    cut = lambda n: {c: {d: v[d] for d in days[:n]} for c, v in closes.items()}
    L.run(tmp_path, fetch=lambda: cut(50))
    L.run(tmp_path, fetch=lambda: cut(50))                # nothing new: no second file, no error
    s = json.loads((tmp_path / "crytrend" / "summary.json").read_text())
    assert s["days"] == 1 and s["first_day"] == days[49] and s["rule_usd"] == 1000 and s["twin_usd"] == 1000
    L.run(tmp_path, fetch=lambda: cut(60))                # ten days later: every missed day is logged in order
    s = json.loads((tmp_path / "crytrend" / "summary.json").read_text())
    assert s["days"] == 11 and s["last_day"] == days[59] and len(s["curve"]) == 11 and "sharpe" in s["rule"]
    assert L.Ledger(tmp_path / "crytrend" / "ledger").verify() == []
    with pytest.raises(RuntimeError):                     # a hole in the candles stops the job
        L.run(tmp_path / "x", fetch=lambda: {c: {d: v[d] for d in days[40:50]} for c, v in closes.items()})
