"""NX3 paper orders (research/prereg/NX3.md, clarification 2026-10-29; Amit chose "Start now"): the live book's
picks are sent to the Alpaca paper account EL-NX3 as they are decided. Fake money only. The forward test is
unchanged (the ledger book minus its control); the account shows real fills, costs and borrow.

Entry: a market day order at the decision (training enters at the first minute bar after t + 1 minute), $500
a trade (nx1_study.NOTIONAL) in whole shares at the IEX price the scorer read. Exit: a market-on-close order
on the next session (training exits at the next close), sent once each session for every position the
account held before that session's own entries. Nothing is sent without the ALPACA_NX3_* secrets, and only to
the paper API (core.alpaca.post).
"""
import json

import pandas as pd

from adaptive import accounts
from core import alpaca
from core.ledger import Ledger
from tracks.news import nx1_study as N

ACCOUNT = "EL-NX3"


class Broker:
    def __init__(self, h):
        self.h = h

    def positions(self):
        return {p["symbol"]: p for p in alpaca.get(f"{alpaca.TRADE}/positions", {}, headers=self.h)}

    def order(self, body):
        return alpaca.post(f"{alpaca.TRADE}/orders", body, headers=self.h)

    def by_client_id(self, cid):
        try:
            return alpaca.get(f"{alpaca.TRADE}/orders:by_client_order_id", {"client_order_id": cid},
                              headers=self.h, tries=3)
        except Exception as e:  # noqa: BLE001 - an unknown order is recorded, not fatal
            return {"error": str(e)[:200]}


def broker():
    h = accounts.headers(ACCOUNT)
    return Broker(h) if h else None


def entry_body(d, x):
    """The entry order for one pick (a decision with side and entry_iex), or None under one share."""
    qty = int(N.NOTIONAL // float(x["entry_iex"]))
    if qty < 1:
        return None
    return {"symbol": x["symbol"], "qty": str(qty), "side": "buy" if x["side"] > 0 else "sell", "type": "market",
            "time_in_force": "day", "client_order_id": f"nx3-{pd.Timestamp(d):%Y%m%d}-{x['symbol']}-in"}


def send_entries(day, d, br):
    """Sends every pick of the day not sent yet; each attempt is kept in day.ctx['sent'] (by event key)."""
    sent = day.ctx.setdefault("sent", {})
    for x in day.decisions:
        if not x.get("side") or x["key"] in sent:
            continue
        body = entry_body(d, x)
        if body is None:
            sent[x["key"]] = {"symbol": x["symbol"], "skip": "price above $500"}
            continue
        code, js = br.order(body)
        sent[x["key"]] = {"symbol": x["symbol"], "body": body, "http": code, "status": js.get("status"),
                          "message": None if code < 300 else js.get("message")}
        print(f"[nx3-orders] {body['side']} {body['qty']} {x['symbol']}: {code} {js.get('status') or js.get('message')}")


def send_exits(day, d, br):
    """Once per session: market-on-close orders closing what the account held before today's entries."""
    if "exits" in day.ctx:
        return
    today = {}
    for s in day.ctx.get("sent", {}).values():
        if s.get("http", 999) < 300:
            b = s["body"]
            today[b["symbol"]] = today.get(b["symbol"], 0) + int(b["qty"]) * (1 if b["side"] == "buy" else -1)
    out = []
    for sym, p in sorted(br.positions().items()):
        q = int(round(float(p["qty"]))) - today.get(sym, 0)
        if q == 0:
            continue
        body = {"symbol": sym, "qty": str(abs(q)), "side": "sell" if q > 0 else "buy", "type": "market",
                "time_in_force": "cls", "client_order_id": f"nx3-{pd.Timestamp(d):%Y%m%d}-{sym}-out"}
        code, js = br.order(body)
        out.append({"symbol": sym, "body": body, "http": code, "status": js.get("status"),
                    "message": None if code < 300 else js.get("message")})
        print(f"[nx3-orders] exit {body['side']} {body['qty']} {sym}: {code} {js.get('status') or js.get('message')}")
    day.ctx["exits"] = out


def log_day(root, day, d, br):
    """After the close: the day's orders with Alpaca's fills (ledger kind nx3_orders)."""
    led = Ledger(root)
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    if led.has("nx3_orders", day_s) or not ("sent" in day.ctx or "exits" in day.ctx):
        return None
    keep = ("status", "filled_qty", "filled_avg_price", "filled_at", "submitted_at", "error")
    rows = []
    for kind, recs in (("entry", list(day.ctx.get("sent", {}).values())), ("exit", day.ctx.get("exits", []))):
        for r in recs:
            o = br.by_client_id(r["body"]["client_order_id"]) if r.get("http", 999) < 300 else {}
            rows.append(r | {"kind": kind, "alpaca": {k: o.get(k) for k in keep if k in o}})
    return led.write_json("nx3_orders", day_s, f"nx3/orders/{day_s}.json",
                          json.loads(json.dumps({"day": day_s, "account": ACCOUNT, "orders": rows}, default=str)))


def guarded(f, *a):
    """Orders never stop the ledger book: an error is printed and the book carries on."""
    try:
        return f(*a)
    except Exception as ex:  # noqa: BLE001
        print(f"[nx3-orders] {f.__name__} failed: {type(ex).__name__}: {ex}")
        return None

