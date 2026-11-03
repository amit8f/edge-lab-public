"""NX3-ALL paper orders (research/prereg/NX3ALL.md, clarification 2026-11-02): the uncapped book's picks are sent to
the Alpaca paper account EL-NX3ALL as they are decided. Fake money only. The forward test is unchanged (the ledger
book minus its control, nx3all_live); the account shows real fills, costs and borrow.

Entry: a market day order at the decision, in whole shares at the IEX price the scorer read, sized by the rule at
decision time: 5% of the $5,000 book at confidence |e|/cut = 1 rising to 10% at 2 or more. The ledger book scales
the day's sizes down together after the close; an order can't wait for that, so orders are instead skipped once
the day's sent sizes would pass 100% of the book. Shorts only where the stock was easy to borrow. Exit: a
market order on the next session, sent 5 minutes before the close (market-on-close orders expired unfilled in Alpaca
paper on 2026-11-02), for every position held before that session's own entries.
Nothing is sent without the ALPACA_NX3ALL_* secrets, and only to the paper API (core.alpaca.post).
"""
import json

import pandas as pd

from adaptive import accounts
from core.ledger import Ledger
from tracks.news import nx3_orders as O
from tracks.news import nx3_study as S

ACCOUNT = "EL-NX3ALL"
BOOK = 5000.0


class Broker(O.Broker):
    pass


def broker():
    h = accounts.headers(ACCOUNT)
    return Broker(h) if h else None


def entry_body(d, x, cut):
    """The entry order for one fired event, or None (not a pick, hard-to-borrow short, or under one share)."""
    e = x.get("e")
    if e is None or not x.get("fired") or not x.get("entry_iex"):
        return None, 0.0
    side = 1 if e > 0 else -1
    if side < 0 and not x.get("etb"):
        return None, 0.0
    w = float(S.ALL_MIN + (S.ALL_MAX - S.ALL_MIN) * min(max(abs(e) / cut - 1.0, 0.0), 1.0))
    qty = int(BOOK * w // float(x["entry_iex"]))
    if qty < 1:
        return None, 0.0
    return {"symbol": x["symbol"], "qty": str(qty), "side": "buy" if side > 0 else "sell", "type": "market",
            "time_in_force": "day", "client_order_id": f"nx3all-{pd.Timestamp(d):%Y%m%d}-{x['symbol']}-in"}, w


def send_entries(day, d, br):
    """Sends every pick of the day not sent yet; each attempt is kept in day.ctx['all_sent'] (by event key)."""
    sent = day.ctx.setdefault("all_sent", {})
    cut = day.ctx["cut"]
    used = sum(s.get("w", 0.0) for s in sent.values() if s.get("http", 999) < 300)
    for x in day.decisions:
        if x["key"] in sent:
            continue
        body, w = entry_body(d, x, cut)
        if body is None:
            continue
        if used + w > 1.0:
            sent[x["key"]] = {"symbol": x["symbol"], "skip": "day's sizes would pass the book"}
            continue
        code, js = br.order(body)
        sent[x["key"]] = {"symbol": x["symbol"], "body": body, "w": w, "http": code, "status": js.get("status"),
                          "message": None if code < 300 else js.get("message")}
        if code < 300:
            used += w
        print(f"[nx3all-orders] {body['side']} {body['qty']} {x['symbol']}: {code} {js.get('status') or js.get('message')}")


def send_exits(day, d, br):
    """Once per session, sent 5 minutes before the close: market day orders closing what the account held before today's
    entries (market-on-close orders expired unfilled in Alpaca paper on 2026-11-02, so they are not used)."""
    if "all_exits" in day.ctx:
        return
    today = {}
    for s in day.ctx.get("all_sent", {}).values():
        if s.get("http", 999) < 300:
            b = s["body"]
            today[b["symbol"]] = today.get(b["symbol"], 0) + int(b["qty"]) * (1 if b["side"] == "buy" else -1)
    out = []
    for sym, p in sorted(br.positions().items()):
        q = int(round(float(p["qty"]))) - today.get(sym, 0)
        if q == 0:
            continue
        body = {"symbol": sym, "qty": str(abs(q)), "side": "sell" if q > 0 else "buy", "type": "market",
                "time_in_force": "day", "client_order_id": f"nx3all-{pd.Timestamp(d):%Y%m%d}-{sym}-out"}
        code, js = br.order(body)
        out.append({"symbol": sym, "body": body, "http": code, "status": js.get("status"),
                    "message": None if code < 300 else js.get("message")})
        print(f"[nx3all-orders] exit {body['side']} {body['qty']} {sym}: {code} {js.get('status') or js.get('message')}")
    day.ctx["all_exits"] = out


def log_day(root, day, d, br):
    """After the close: the day's orders with Alpaca's fills (ledger kind nx3all_orders)."""
    led = Ledger(root)
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    if led.has("nx3all_orders", day_s) or not ("all_sent" in day.ctx or "all_exits" in day.ctx):
        return None
    keep = ("status", "filled_qty", "filled_avg_price", "filled_at", "submitted_at", "error")
    rows = []
    for kind, recs in (("entry", list(day.ctx.get("all_sent", {}).values())), ("exit", day.ctx.get("all_exits", []))):
        for r in recs:
            o = br.by_client_id(r["body"]["client_order_id"]) if r.get("http", 999) < 300 else {}
            rows.append(r | {"kind": kind, "alpaca": {k: o.get(k) for k in keep if k in o}})
    return led.write_json("nx3all_orders", day_s, f"nx3all/orders/{day_s}.json",
                          json.loads(json.dumps({"day": day_s, "account": ACCOUNT, "orders": rows}, default=str)))
