"""K2 AI stock trader: accounting, gaps, stops, margin calls, session bars, twins, on synthetic data."""
import json

import pandas as pd
import pytest

from core.ledger import Ledger
from tracks.main import ai_trader as k

T0 = int(pd.Timestamp("2026-10-16 14:51", tz="UTC").timestamp())   # 10:51 ET


def bar(t, o, h, lo, c):
    return [t, o, h, lo, c, 1000.0, t + 3600]


def data(prices, daily_up=True, t=T0):
    out = {}
    for s, p in prices.items():
        d = [[t - (61 - i) * 86400, p, p, p, p * ((0.97 + i * 0.001) if daily_up else (1.03 - i * 0.001)), 1.0]
             for i in range(60)]
        out[s] = {"1h": [bar(t - 7200 - (40 - i) * 3600, p, p, p, p) for i in range(40)], "1d": d, "last": p,
                  "last_t": pd.Timestamp(t, unit="s", tz="UTC").isoformat()}
    return out


PR = {s: 100.0 for s in k.SYMBOLS}


def test_open_close_costs_and_carry():
    b = k.new_book()
    k.open_pos(b, k.BOOKS["A"], "NVDA", 1, 100.0, T0)
    assert b["pos"]["notional"] == pytest.approx(125.0)
    assert b["cash"] == pytest.approx(100 - 0.125)
    year = 365 * 24 * 3600
    k.close_pos(b, 110.0, T0 + year, "decision")
    interest = 125 * k.MARGIN_RATE * 0.2                     # 8% on the borrowed fifth, one year
    assert b["cash"] == pytest.approx(100 - 0.125 + 12.5 - 1.25 * 110 * k.FEE - interest)


def test_short_pays_borrow_and_cash_buy_has_no_interest():
    b = k.new_book()
    k.open_pos(b, k.BOOKS["B"], "TSLA", -1, 100.0, T0)
    assert b["pos"]["carry"] == k.BORROW
    h = k.new_book()
    k.open_pos(h, k.BOOKS["HOLD"], "SPY", 1, 100.0, T0)
    assert h["pos"]["carry"] == 0 and h["pos"]["call"] is None
    assert k.equity(h, {"SPY": 100.0}) == pytest.approx(100 / (1 + k.FEE))


def test_gap_through_stop_fills_at_open():
    b = k.new_book()
    k.open_pos(b, k.BOOKS["B"], "AAPL", 1, 100.0, T0)
    f = k.settle(b, [bar(T0 + 600, 90.0, 91.0, 89.0, 90.5)])
    assert f[0]["why"] == "stop" and f[0]["exit"] == 90.0


def test_stop_inside_bar_fills_at_level_and_old_bar_ignored():
    b = k.new_book()
    k.open_pos(b, k.BOOKS["B"], "AAPL", -1, 100.0, T0)
    assert k.settle(b, [bar(T0 - 1800, 100, 200, 100, 100)]) == []     # started before the fill
    f = k.settle(b, [bar(T0 + 600, 101, 106, 100, 104)])
    assert f[0]["exit"] == pytest.approx(105.0)


def test_margin_call_keeps_the_rest():
    b = k.new_book()
    k.open_pos(b, k.BOOKS["A"], "AMD", 1, 100.0, T0)
    call = b["pos"]["call"]
    assert call == pytest.approx(100 * 0.2 / 0.75)
    f = k.settle(b, [bar(T0 + 600, 50.0, 50.0, 20.0, 21.0)])
    assert f[0]["why"] == "margin call" and f[0]["exit"] == pytest.approx(call)
    assert b["cash"] > 0                                    # the account keeps its remaining equity


def test_session_hours_aggregates_regular_session_only():
    day = pd.Timestamp("2026-10-15").date()
    o = pd.Timestamp("2026-10-15 09:30", tz=k.ET).tz_convert("UTC")
    c = pd.Timestamp("2026-10-15 16:00", tz=k.ET).tz_convert("UTC")
    rows = []
    for i in range(-2, 15):                                  # 08:30 .. 16:30 ET in 30-minute steps
        ts = o + pd.Timedelta(minutes=30 * i)
        rows.append({"t": ts.isoformat(), "o": 10 + i, "h": 11 + i, "l": 9 + i, "c": 10.5 + i, "v": 5})
    hours = k.session_hours(rows, {day: (o, c)})
    assert len(hours) == 7                                   # 6 full hours + 15:30-16:00
    assert hours[0][1] == 10 and hours[0][4] == 11.5 and hours[-1][6] == int(c.timestamp())


def test_parse():
    assert k.parse('{"action":"short","symbol":"tsla","reason":"x"}')["symbol"] == "TSLA"
    for bad in ("", "[1]", '{"action":"long","symbol":"GME"}', '{"action":"yolo"}'):
        assert k.parse(bad)["action"] == "hold"


def test_run_twins_hold_trend_and_ledger(tmp_path):
    answers = iter(['{"action":"long","symbol":"NVDA","reason":"a"}', "nonsense"])
    d = data(PR)
    rec = k.run(tmp_path, fetch=lambda now: d, ask=lambda s, t: next(answers))
    st = json.loads((tmp_path / "ai/books.json").read_text())["books"]
    assert st["A"]["pos"]["sym"] == "NVDA" and st["A-coin"]["pos"]["side"] == k.coin_side("A", T0)
    assert st["B"]["pos"] is None and rec["decisions"]["B"]["parsed"]["valid"] is False
    assert st["HOLD"]["pos"]["sym"] == "SPY" and st["TREND"]["pos"] is not None
    assert Ledger(tmp_path / "ai").verify() == []
    assert k.run(tmp_path, fetch=lambda now: d, ask=lambda s, t: "x") is None   # same hour: no-op

    t2 = T0 + 3600
    d2 = data(PR, daily_up=False, t=t2)
    rec2 = k.run(tmp_path, fetch=lambda now: d2, ask=lambda s, t: '{"action":"close"}')
    st = json.loads((tmp_path / "ai/books.json").read_text())["books"]
    assert st["A"]["pos"] is None and st["A-coin"]["pos"] is None and st["TREND"]["pos"] is None
    assert rec2["fills"]["A"][0]["why"] == "decision"
    assert Ledger(tmp_path / "ai").verify() == []


def test_closed_market_does_nothing(tmp_path):
    assert k.run(tmp_path, fetch=lambda now: None, ask=lambda s, t: "x") is None
    assert not (tmp_path / "ai").exists()
