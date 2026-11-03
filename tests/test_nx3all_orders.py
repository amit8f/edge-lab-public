"""NX3-ALL paper orders with a fake broker (no network): confidence-sized whole-share entries, shorts only when
easy to borrow, skipped once the day passes the book, exits on close for what was held before today."""
import json

import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3all_orders as A

D = pd.Timestamp("2026-11-02")
CUT = 0.2


class Day:
    def __init__(self, decisions):
        self.decisions, self.ctx = decisions, {"cut": CUT}


class Broker:
    def __init__(self, pos=None):
        self.sent, self.pos = [], pos or {}

    def order(self, body):
        self.sent.append(body)
        return 200, {"status": "accepted"}

    def positions(self):
        return self.pos

    def by_client_id(self, cid):
        return {"status": "filled", "filled_avg_price": "10.01", "filled_qty": "5", "extra": 1}


def dec(key, sym, e, px, fired=True, etb=True):
    return {"key": key, "symbol": sym, "e": e, "fired": fired, "etb": etb, "entry_iex": px}


def test_sizes_follow_confidence_and_hard_to_borrow_shorts_are_skipped():
    day = Day([dec("a", "AAA", 2 * CUT, 50.0),             # confidence 2 -> 10% = $500
              dec("b", "BBB", -1.0 * CUT, 100.0),          # confidence 1 -> 5% = $250 short
              dec("c", "CCC", -1.5 * CUT, 10.0, etb=False),
              dec("d", "DDD", 0.5 * CUT, 10.0, fired=False),
              dec("e", "EEE", 2 * CUT, 6000.0)])           # under one share
    br = Broker()
    A.send_entries(day, D, br)
    A.send_entries(day, D, br)
    assert [(b["symbol"], b["side"], b["qty"]) for b in br.sent] == [("AAA", "buy", "10"), ("BBB", "sell", "2")]
    assert br.sent[0]["client_order_id"] == "nx3all-20261102-AAA-in"


def test_day_stops_at_the_book():
    day = Day([dec(str(i), f"S{i}", 3 * CUT, 10.0) for i in range(12)])      # 10% each: the 11th would pass 100%
    br = Broker()
    A.send_entries(day, D, br)
    assert len(br.sent) == 10 and day.ctx["all_sent"]["10"]["skip"].startswith("day's sizes")


def test_exits_close_only_what_was_held_before_today_and_fills_log_once(tmp_path):
    day = Day([dec("a", "AAA", 2 * CUT, 50.0)])
    br = Broker({"AAA": {"qty": "14"}, "OLD": {"qty": "-3"}})
    A.send_entries(day, D, br)                 # 10 AAA bought today
    br.sent.clear()
    A.send_exits(day, D, br)
    A.send_exits(day, D, br)
    assert [(b["symbol"], b["side"], b["qty"], b["time_in_force"]) for b in br.sent] == [
        ("AAA", "sell", "4", "day"), ("OLD", "buy", "3", "day")]
    assert A.log_day(tmp_path, day, D, br) is not None and A.log_day(tmp_path, day, D, br) is None
    rec = json.loads((tmp_path / Ledger(tmp_path).entry("nx3all_orders", "2026-11-02")["file"]).read_text())
    assert rec["account"] == "EL-NX3ALL" and len(rec["orders"]) == 3


def test_no_secrets_no_broker(monkeypatch):
    monkeypatch.delenv("ALPACA_NX3ALL_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_NX3ALL_SECRET_KEY", raising=False)
    assert A.broker() is None
