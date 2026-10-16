"""K1 crypto paper bot: accounting, liquidation, stops, parsing and twins on synthetic candles."""
import json
from datetime import datetime, timezone

import pytest

from core.ledger import Ledger
from tracks.crypto import llm_bot as k

T0 = int(datetime(2026, 10, 16, 12, 7, tzinfo=timezone.utc).timestamp())


def flat(price, n=48, start=T0 - 48 * 3600):
    h0 = start - start % 3600
    return [[h0 + i * 3600, price, price, price, price, 1.0] for i in range(n)]


def data(prices, daily_up=True):
    out = {}
    for s, p in prices.items():
        d = [[T0 - (30 - i) * 86400, p, p, p, p * ((0.99 + i * 0.001) if daily_up else (1.02 - i * 0.002)), 1.0]
             for i in range(30)]
        out[s] = {"1h": flat(p), "4h": flat(p, 12), "1d": d, "last": p}
    return out


def test_open_and_close_charge_fees_and_pnl():
    b, cfg = k.new_book(), k.BOOKS["A"]
    k.open_pos(b, cfg, "BTC", 1, 100.0, T0)
    assert b["pos"]["notional"] == pytest.approx(125.0)          # 25% margin x 5
    assert b["cash"] == pytest.approx(100 - 125 * k.FEE)
    assert k.equity(b, {"BTC": 110.0}) == pytest.approx(100 - 125 * k.FEE + 12.5)
    k.close_pos(b, 110.0, T0 + 3600, "decision")
    assert b["cash"] == pytest.approx(100 - 125 * k.FEE + 12.5 - 1.25 * 110 * k.FEE)
    assert b["trips"][0]["why"] == "decision"


def test_liquidation_loses_the_margin_and_funding_is_charged():
    b, cfg = k.new_book(), k.BOOKS["A"]
    k.open_pos(b, cfg, "BTC", 1, 100.0, T0)
    liq = b["pos"]["liq"]
    assert liq == pytest.approx(100 * (1 - (0.2 - k.MAINT)))
    h1 = T0 - T0 % 3600 + 3600
    candles = [[h1, 100, 101, 99, 100, 1], [h1 + 3600, 100, 100, liq - 0.01, 90, 1]]
    f = k.settle(b, cfg, candles)
    assert [x["why"] for x in f] == ["liquidated"] and b["pos"] is None
    assert b["cash"] == pytest.approx(100 - 125 * k.FEE - 2 * 125 * k.FUND_PER_HOUR - 25)


def test_candle_before_the_open_is_not_used():
    b, cfg = k.new_book(), k.BOOKS["A"]
    k.open_pos(b, cfg, "BTC", 1, 100.0, T0)
    same_hour = [[T0 - T0 % 3600, 100, 100, 1.0, 100, 1]]          # started before the fill
    assert k.settle(b, cfg, same_hour) == [] and b["pos"] is not None


def test_short_stop_in_book_b():
    b, cfg = k.new_book(), k.BOOKS["B"]
    k.open_pos(b, cfg, "ETH", -1, 100.0, T0)
    assert b["pos"]["stop"] == pytest.approx(105.0)
    h1 = T0 - T0 % 3600 + 3600
    f = k.settle(b, cfg, [[h1, 100, 106, 99, 104, 1]])
    assert f[0]["why"] == "stop" and f[0]["exit"] == pytest.approx(105.0)
    assert b["cash"] == pytest.approx(100 - 50 * k.FEE - 50 * k.FUND_PER_HOUR - 2.5 - 0.5 * 105 * k.FEE)


def test_parse():
    assert k.parse('{"action":"long","symbol":"sol","reason":"up"}')["action"] == "long"
    assert k.parse('```json\n{"action": "close", "reason": "x"}\n```')["action"] == "close"
    for bad in ("", "buy BTC", '{"action":"long","symbol":"DOGE"}', '{"action":"moon"}'):
        p = k.parse(bad)
        assert p["action"] == "hold" and not p["valid"]


def test_switching_closes_first():
    b, cfg = k.new_book(), k.BOOKS["A"]
    prices = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}
    k.apply_action(b, cfg, "long", "BTC", prices, T0)
    f = k.apply_action(b, cfg, "short", "ETH", prices, T0 + 3600)
    assert [x["action"] for x in f] == ["close", "open"] and b["pos"]["sym"] == "ETH"
    assert k.apply_action(b, cfg, "short", "ETH", prices, T0 + 7200) == []


def test_step_twins_hold_trend_and_ledger(tmp_path):
    answers = iter(['{"action":"long","symbol":"BTC","reason":"a"}', "garbage"])
    now = datetime.fromtimestamp(T0, timezone.utc)
    d = data({"BTC": 100.0, "ETH": 50.0, "SOL": 10.0})
    rec = k.run(tmp_path, now=now, fetch=lambda: d, ask=lambda s, t: next(answers))
    st = json.loads((tmp_path / "crypto/books.json").read_text())["books"]
    assert st["A"]["pos"]["sym"] == "BTC" and st["A-coin"]["pos"]["sym"] == "BTC"
    assert st["A-coin"]["pos"]["side"] == k.coin_side("A", T0)
    assert st["B"]["pos"] is None and rec["decisions"]["B"]["parsed"]["valid"] is False
    assert st["HOLD"]["pos"]["sym"] == "BTC" and st["TREND"]["pos"] is not None
    assert len(rec["new_1h"]["BTC"]) == 48
    assert Ledger(tmp_path / "crypto").verify() == []
    assert k.run(tmp_path, now=now, fetch=lambda: d, ask=lambda s, t: "x") is None   # same hour: no-op

    # next hour: A closes, so its twin closes too; only new candles are logged
    d2 = data({"BTC": 101.0, "ETH": 50.0, "SOL": 10.0}, daily_up=False)
    d2["BTC"]["1h"] = d["BTC"]["1h"] + [[d["BTC"]["1h"][-1][0] + 3600, 100, 101, 100, 101, 1]]
    rec2 = k.run(tmp_path, now=datetime.fromtimestamp(T0 + 3600, timezone.utc), fetch=lambda: d2,
                 ask=lambda s, t: '{"action":"close"}')
    st = json.loads((tmp_path / "crypto/books.json").read_text())["books"]
    assert st["A"]["pos"] is None and st["A-coin"]["pos"] is None and st["TREND"]["pos"] is None
    assert len(rec2["new_1h"]["BTC"]) == 1
    assert Ledger(tmp_path / "crypto").verify() == []


def test_model_error_is_logged_as_hold(tmp_path):
    def boom(s, t):
        raise RuntimeError("quota")
    rec = k.run(tmp_path, now=datetime.fromtimestamp(T0, timezone.utc),
                fetch=lambda: data({"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}), ask=boom)
    assert rec["decisions"]["A"]["parsed"]["action"] == "hold" and "quota" in rec["decisions"]["A"]["error"]
