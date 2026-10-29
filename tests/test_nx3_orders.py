"""NX3 paper orders with a fake broker (no network): $500 entries in whole shares, sent once per pick; exits
on close for what was held before today's entries, once per session; fills logged after the close."""
import json

import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3_orders as O


class Day:
    def __init__(self, decisions, ctx=None):
        self.decisions, self.ctx = decisions, ctx or {}


class Broker:
    def __init__(self, pos=None):
        self.sent, self.pos = [], pos or {}

    def order(self, body):
        self.sent.append(body)
        return (403, {"message": "not shortable"}) if body["symbol"] == "BAD" else (200, {"status": "accepted"})

    def positions(self):
        return self.pos

    def by_client_id(self, cid):
        return {"status": "filled", "filled_avg_price": "10.01", "filled_qty": "49", "extra": 1}


D = pd.Timestamp("2026-10-30")


def dec(key, sym, side, px):
    return {"key": key, "symbol": sym, "status": "decided", "side": side, "entry_iex": px}


def test_entries_once_each_in_whole_shares():
    day = Day([dec("a", "AAA", 1, 10.2), dec("b", "BBB", -1, 33.0), {"key": "c", "status": "decided", "side": None},
               dec("d", "PRCY", 1, 900.0), dec("e", "BAD", -1, 5.0)])
    br = Broker()
    O.send_entries(day, D, br)
    O.send_entries(day, D, br)
    assert [(b["symbol"], b["side"], b["qty"]) for b in br.sent] == [("AAA", "buy", "49"), ("BBB", "sell", "15"), ("BAD", "sell", "100")]
    assert br.sent[0]["client_order_id"] == "nx3-20261030-AAA-in" and br.sent[0]["type"] == "market"
    assert day.ctx["sent"]["d"]["skip"] == "price above $500" and day.ctx["sent"]["e"]["http"] == 403


def test_exits_close_only_what_was_held_before_today():
    day = Day([dec("a", "AAA", 1, 10.0)])
    br = Broker({"AAA": {"qty": "80"}, "OLD": {"qty": "-12"}})
    O.send_entries(day, D, br)                 # today's 50 AAA stay open
    br.sent.clear()
    O.send_exits(day, D, br)
    O.send_exits(day, D, br)
    assert [(b["symbol"], b["side"], b["qty"], b["time_in_force"]) for b in br.sent] == [
        ("AAA", "sell", "30", "cls"), ("OLD", "buy", "12", "cls")]


def test_log_day_writes_fills_once(tmp_path):
    day = Day([dec("a", "AAA", 1, 10.0), dec("e", "BAD", -1, 5.0)])
    br = Broker()
    O.send_entries(day, D, br)
    O.send_exits(day, D, br)
    assert O.log_day(tmp_path, day, D, br) is not None and O.log_day(tmp_path, day, D, br) is None
    led = Ledger(tmp_path)
    rec = json.loads((tmp_path / led.entry("nx3_orders", "2026-10-30")["file"]).read_text())
    assert rec["orders"][0]["alpaca"] == {"status": "filled", "filled_qty": "49", "filled_avg_price": "10.01"} or \
        rec["orders"][0]["alpaca"]["filled_avg_price"] == "10.01"
    assert rec["orders"][1]["alpaca"] == {} and led.verify() == []


def test_no_secrets_no_broker(monkeypatch):
    monkeypatch.delenv("ALPACA_NX3_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_NX3_SECRET_KEY", raising=False)
    assert O.broker() is None


def test_guarded_swallows_errors():
    def boom():
        raise ValueError("x")
    assert O.guarded(boom) is None
