"""NX3 on synthetic data (no network): decision times, the 1-minute entry, events, the zero effect without
news and the gate."""
import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import w1_data as W

DAYS = pd.bdate_range("2017-01-02", "2019-12-31")
CAL = pd.DataFrame({"date": DAYS, "open": "09:30", "close": "16:00"})
SES = X.sessions(CAL)


def head(symbol, t, i=0, **kw):
    r = {"id": i, "symbol": symbol, "t": pd.Timestamp(t), "hkey": f"h{i}", "head_len": 30, "lm_neg": 0, "lm_pos": 1,
         "fb_pos": 0.8, "fb_neg": 0.1, "fb_neu": 0.1, **{f"f_{k}": 0 for k in N.FLAGS}}
    return r | kw


def test_inputs_are_the_registered_37():
    assert len(X.INPUTS) == 37 and len(set(X.INPUTS)) == 37 and set(X.INTRADAY) <= set(X.BASE_INPUTS)


def test_decision_time_and_news_window():
    rows = pd.DataFrame([head("AAA", "2017-03-01 18:00", 1), head("AAA", "2017-03-02 08:00", 2, lm_neg=2),
                         head("BBB", "2017-03-02 10:15:20", 3), head("BBB", "2017-03-02 11:00", 4),
                         head("CCC", "2017-03-02 15:59", 5), head("DDD", "2017-03-01 15:00", 6)])
    e = X.session_events(rows, SES).set_index("symbol")
    assert e.loc["AAA", "d"] == pd.Timestamp("2017-03-02") and e.loc["AAA", "t"] == pd.Timestamp("2017-03-02 09:30")
    assert e.loc["AAA", "n_articles"] == 2 and e.loc["AAA", "lm_neg"] == 2 and e.loc["AAA", "time_bucket"] == 0
    assert e.loc["BBB", "t"] == pd.Timestamp("2017-03-02 10:15:20") and e.loc["BBB", "n_articles"] == 1
    assert e.loc["BBB", "time_bucket"] == 1 and "CCC" not in e.index         # later than 15:58
    assert e.loc["DDD", "d"] == pd.Timestamp("2017-03-01")


def bars(start="2017-03-02 09:30", n=390, p0=10.0, step=0.01):
    t = pd.date_range(start, periods=n, freq="min")
    c = p0 + step * np.arange(1, n + 1)
    return pd.DataFrame({"t": t, "o": c - step, "c": c, "v": 1000.0})


def test_trade_enters_one_minute_later_and_reads_the_reaction():
    b = bars()
    o = pd.Timestamp("2017-03-02 09:30")
    t = pd.Timestamp("2017-03-02 10:15:20")
    tr = X.trade(b, t, o, prev_ac=9.5, exit_ac=11.0, dv20=1e6 * 10, prev_c=10.0)
    assert tr["entry_t"] == pd.Timestamp("2017-03-02 10:17")            # first bar starting >= t + 1 minute
    px_t = float(b.loc[b["t"] == pd.Timestamp("2017-03-02 10:14"), "c"].iloc[0])   # last bar ended by t
    assert np.isclose(tr["r_open"], px_t / 9.5 - 1) and np.isclose(tr["r_react"], tr["entry"] / px_t - 1)
    assert np.isclose(tr["rv30"], 29 * 1000 / (1e6 * 30 / 390))          # bars 9:46-10:14 and np.isclose(tr["mins_open"], 45 + 20 / 60)
    pre = X.trade(b, o, o, 9.5, 11.0, 1e7, 10.0)
    assert pre["entry_t"] == pd.Timestamp("2017-03-02 09:31") and np.isnan(pre["rv30"])
    assert np.isclose(pre["r_open"], b["o"].iloc[0] / 9.5 - 1)
    late = b[b["t"] < pd.Timestamp("2017-03-02 10:16")]
    assert X.trade(pd.concat([late, b[b["t"] > pd.Timestamp("2017-03-02 10:40")]]), t, o, 9.5, 11, 1e7, 10) is None


def _rows(sym, price=20.0):
    rng = np.random.default_rng(len(sym) + ord(sym[0]))
    c = price * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS))))
    f = pd.DataFrame({"o": c * 0.995, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 5e5}, index=DAYS)
    u = W.daily_rows(sym, f, f.copy())
    u["cik"] = float(ord(sym[0]))
    return u


def _spy():
    c = 200 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, len(DAYS))))
    return pd.DataFrame({"date": DAYS, "ao": c * 0.999, "ac": c, "c": c})


def test_batch_events_inputs_from_the_previous_close_and_placebo_at_the_same_time(monkeypatch):
    monkeypatch.setattr(X, "FIRST_DAY", DAYS[0])
    rows = pd.concat([_rows(s) for s in ("AAA", "BBB", "CCC", "DDD")], ignore_index=True)
    d = DAYS[300]
    t = d + pd.Timedelta(hours=11)
    sev = pd.DataFrame([{"symbol": "AAA", "d": d, "t": t, **{k: 1.0 for k in N.NEWS_INPUTS}}])
    nd = pd.DataFrame({"symbol": ["AAA", "BBB"], "date": [d, d - pd.Timedelta(days=1)]})
    ev = X.batch_events(rows, sev, nd, _spy(), CAL, SES, 0)
    real, plac = ev[ev["kind"] == "real"].iloc[0], ev[ev["kind"] == "placebo"]
    prev = DAYS[299]
    a = rows[rows["symbol"] == "AAA"].set_index("date")
    assert real["prev"] == prev and np.isclose(real["r1"], a.loc[prev, "ac"] / a.loc[DAYS[298], "ac"] - 1)
    assert real["prev_ac"] == a.loc[prev, "ac"] and real["exit_ac"] == a.loc[DAYS[301], "ac"]
    assert len(plac) == 1 and plac["symbol"].iloc[0] in ("CCC", "DDD") and plac["t"].iloc[0] == t
    assert (plac[N.NEWS_INPUTS] == 0).all().all()


def events(news_effect, seed=0, per_day=24):
    rng = np.random.default_rng(seed)
    n = len(DAYS) * per_day
    kind = np.tile(np.r_[np.repeat("real", per_day // 2), np.repeat("placebo", per_day // 2)], len(DAYS))
    ev = pd.DataFrame({k: rng.normal(size=n) for k in X.INPUTS})
    real = kind == "real"
    nv = N.placebo_news()
    for k in N.NEWS_INPUTS:
        ev.loc[~real, k] = nv[k]
    ev.loc[real, "n_articles"] = rng.integers(1, 4, real.sum())
    ev["vol20"] = rng.uniform(0.005, 0.12, n)
    z = news_effect * np.where(real, np.tanh(ev["lm_net"]) * (1 + (ev["r_react"] > 0)), 0) + rng.normal(size=n)
    ev["mret"] = z * ev["vol20"] * X.HOLD_SCALE
    ev["ret"] = ev["mret"]
    ev["kind"] = kind
    ev["d"] = np.repeat(DAYS, per_day)
    ev["t"] = ev["d"] + pd.to_timedelta(rng.integers(570, 950, n), unit="min")
    ev["price"], ev["days"] = 20.0, 2
    return ev


def test_no_news_effect_is_zero():
    ev = events(1.0)
    m = X.Model(ev[ev["d"] < "2018-06-01"])
    pl = ev[ev["kind"] == "placebo"].head(500)
    assert np.all(m.effect(pl[X.INPUTS].to_numpy(float)) == 0)
    x = ev[X.INPUTS].head(3).to_numpy(float)
    keep = [X.INPUTS.index(k) for k in X.BASE_INPUTS + ["novelty"]]
    assert np.array_equal(X.no_news(x)[:, keep], x[:, keep])


def test_picks_go_in_time_order():
    day = pd.DataFrame({"t": pd.to_datetime(["2019-01-02 10:00", "2019-01-02 09:31", "2019-01-02 11:00",
                                             "2019-01-02 12:00", "2019-01-02 13:00", "2019-01-02 14:00"]),
                        "e": [5.0, 1.0, 2.0, 3.0, 4.0, -1.0], "fired": [True] * 5 + [True]})
    p, c = X.picks(day, np.random.default_rng(0))
    assert p.loc[p["side"] == 1, "e"].tolist() == [1.0, 5.0, 2.0] and p.loc[p["side"] == -1, "e"].tolist() == [-1.0]
    assert len(c) == 4 and c["side"].tolist() == [1, 1, 1, -1]


def test_gate_passes_a_news_effect_and_fails_without_one():
    _, _, gate = X.walk_forward(events(1.5, seed=1), CAL, years=[2019])
    assert gate == {"pick_minus_control": True, "pick_minus_shuffled": True}
    _, pooled, gate = X.walk_forward(events(0.0, seed=2), CAL, years=[2019])
    assert not any(gate.values())


def test_extremes_lists_the_walk_forward_picks_by_size():
    ev = events(1.5, seed=1)
    for k in ("symbol", "entry_t", "entry", "next", "exit_ac", "prev_c", "spy_ret"):
        ev[k] = 0.0
    ev["symbol"] = "S" + pd.Series(range(len(ev))).astype(str)
    top, pk = X.extremes(ev, CAL, years=[2019], n=5)
    assert list(top.columns) == X.EXTREME_COLS and len(top) == 5
    assert top["net"].abs().is_monotonic_decreasing and top["net"].abs().min() >= pk["net"].abs().drop(top.index).max()
    _, pk2, _, _ = X.evaluate(ev[ev["d"] >= "2019-01-01"], X.Model(ev[ev["d"] <= CAL["date"][
        CAL["date"].searchsorted(pd.Timestamp("2019-01-01")) - N.PURGE - 1]]), X.SEED + 2019)
    assert len(pk) == len(pk2)


def test_on_day_basis_restates_closes_in_the_day_raw_prices():
    d0, d1, d2 = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")
    # a 1:10 reverse split on d1: raw jumps 10x, all-adjusted is smooth
    dc = {("ABC", d0): (1.0, 10.0), ("ABC", d1): (10.5, 10.5), ("ABC", d2): (11.0, 11.0)}
    prev, exit_ = X.on_day_basis(dc, "ABC", d0, d1, d2)
    assert np.isclose(prev, 10.0) and np.isclose(exit_, 11.0)
    # a later liquidation shrinks all-adjusted prices; the ratio to d's raw close cancels it
    dc = {("ELM", d0): (15.0, 1.5), ("ELM", d1): (14.8, 1.48), ("ELM", d2): (15.1, 1.51)}
    prev, exit_ = X.on_day_basis(dc, "ELM", d0, d1, d2)
    assert np.isclose(prev, 15.0) and np.isclose(exit_, 15.1)
    assert X.on_day_basis(dc, "ELM", d0, d1, pd.Timestamp("2024-03-06")) is None


def test_month_trades_returns_use_one_price_basis(monkeypatch):
    cal = pd.DataFrame({"date": pd.bdate_range("2017-03-01", "2017-03-03"), "open": "09:30", "close": "16:00"})
    ses = X.sessions(cal)
    d, prev, nxt = pd.Timestamp("2017-03-02"), pd.Timestamp("2017-03-01"), pd.Timestamp("2017-03-03")
    dc = {("ABC", prev): (9.5, 0.95), ("ABC", d): (10.0, 1.0), ("ABC", nxt): (11.0, 1.1),
          ("SPY", prev): (100.0, 100.0), ("SPY", d): (100.0, 100.0), ("SPY", nxt): (101.0, 101.0)}
    monkeypatch.setattr(X, "daily_closes", lambda syms, a, b, h: dc)
    monkeypatch.setattr(X, "day_bars", lambda syms, day, o, c, h: {"ABC": bars(), "SPY": bars(p0=100.0, step=0.0)})
    ev = pd.DataFrame({"symbol": ["ABC"], "d": [d], "t": [pd.Timestamp("2017-03-02 10:15:20")], "kind": ["real"],
                       "prev_ac": [999.0], "exit_ac": [0.001], "prev_c": [9.5], "dv20": [1e7], "next": [nxt]})
    t = X.month_trades(ev, ses, {})
    assert len(t) == 1 and np.isclose(t["exit_ac"].iloc[0], 11.0) and np.isclose(t["prev_ac"].iloc[0], 9.5)
    assert np.isclose(t["ret"].iloc[0], 11.0 / t["entry"].iloc[0] - 1) and np.isclose(t["spy_ret"].iloc[0], 0.01)
