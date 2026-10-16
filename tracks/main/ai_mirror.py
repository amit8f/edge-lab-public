"""Copies the $100 AI bot's positions (K2, books A and B) into a separate Alpaca PAPER account at 25x size,
so Amit can watch them in Alpaca's app (research/prereg/K2.md, amendment 2026-10-16).

Display only: the official K2 result stays the simulated $100 books. The two books' positions are netted per
symbol (one Alpaca account can't be long and short the same stock). Longs may be fractional; shorts are
rounded to whole shares because Alpaca can't short fractions. Orders go only to the paper API (core.alpaca.post
refuses anything else). Without the mirror keys this does nothing.
"""
import json
import os
import time

from core import alpaca

SCALE = 25            # $100 books -> a $2,500 paper account
BOOKS = ("A", "B")
MIN_USD = 1.0         # skip changes smaller than this


def mirror_headers():
    key, secret = os.environ.get("ALPACA_MIRROR_KEY_ID"), os.environ.get("ALPACA_MIRROR_SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret} if key and secret else None


def targets(state):
    """{symbol: signed share count} the account should hold."""
    out = {}
    for n in BOOKS:
        p = state["books"][n]["pos"]
        if p:
            out[p["sym"]] = out.get(p["sym"], 0.0) + p["side"] * p["qty"] * SCALE
    for s, q in out.items():
        out[s] = float(round(q)) if q < 0 else round(q, 4)
    return out


def plan_orders(target, current, prices):
    """Orders taking `current` to `target`, as (symbol, side, qty) in the order to send them. A position that
    changes direction is closed first, then reopened."""
    orders = []
    for s in sorted(set(target) | set(current)):
        t, c = target.get(s, 0.0), current.get(s, 0.0)
        legs = [(c, 0.0), (0.0, t)] if c * t < 0 else [(c, t)]
        for a, b in legs:
            d = b - a
            if abs(d) * prices.get(s, 0.0) < MIN_USD and b != 0.0:
                continue
            if abs(d) > 1e-9:
                orders.append((s, "buy" if d > 0 else "sell", round(abs(d), 4)))
    return orders


def _positions(h):
    rows = alpaca.get(f"{alpaca.TRADE}/positions", {}, headers=h)
    return {r["symbol"]: abs(float(r["qty"])) * (-1 if r.get("side") == "short" else 1) for r in rows}


def _wait(h, oid, seconds=30):
    for _ in range(seconds // 3):
        o = alpaca.get(f"{alpaca.TRADE}/orders/{oid}", {}, headers=h)
        if o.get("status") in ("filled", "canceled", "rejected", "expired"):
            return o.get("status")
        time.sleep(3)
    return "pending"


def run(state, prices, log_path, headers=None, post=alpaca.post, positions=_positions, wait=_wait):
    """Send the orders; append what happened to log_path. Never raises: the bot's own run must not fail."""
    h = headers or mirror_headers()
    if h is None:
        return None
    out = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "orders": []}
    try:
        tgt = targets(state)
        out["target"] = tgt
        for sym, side, qty in plan_orders(tgt, positions(h), prices):
            body = {"symbol": sym, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day"}
            code, js = post(f"{alpaca.TRADE}/orders", body, headers=h)
            row = {"symbol": sym, "side": side, "qty": qty, "status": code, "message": js.get("message")}
            if code < 300 and js.get("id"):
                row["fill"] = wait(h, js["id"])
            out["orders"].append(row)
    except Exception as e:  # logged, never fatal
        out["error"] = f"{type(e).__name__}: {e}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    return out
