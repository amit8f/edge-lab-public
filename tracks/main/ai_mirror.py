"""Copies each $100 AI bot's positions into its own Alpaca PAPER account at 25x size, so Amit can watch
and compare them in Alpaca's app: book A (K2, the TikTok setup) and book C (K4, whole market, up to five
stocks) each get a $2,500 paper account. C took over B's account when B was stopped (research/prereg/K2.md
amendments 2026-10-16, research/prereg/K4.md).

Display only: the official K2 result stays the simulated $100 books. Longs may be fractional; shorts are
rounded to whole shares because Alpaca can't short fractions. Orders go only to the paper API (core.alpaca.post
refuses anything else). Without the mirror keys this does nothing.
"""
import json
import os
import time

from core import alpaca

SCALE = 1             # $10,000 books -> $10,000 paper accounts (1x)
ACCOUNTS = {"A": ("ALPACA_MIRROR_KEY_ID", "ALPACA_MIRROR_SECRET_KEY"),
            "C": ("ALPACA_MIRROR_B_KEY_ID", "ALPACA_MIRROR_B_SECRET_KEY"),   # B's old account
            "B": ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")}   # bot B's account is the main key pair
MIN_USD = 1.0         # skip changes smaller than this


def mirror_headers(book):
    ki, si = ACCOUNTS[book]
    key, secret = os.environ.get(ki), os.environ.get(si)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret} if key and secret else None


def targets(state, book):
    """{symbol: signed share count} the book's account should hold."""
    p = state["books"][book]["pos"]
    if p and "sym" in p:          # K2 books hold one position
        p = {p["sym"]: p}
    out = {s: x["side"] * x["qty"] * SCALE for s, x in (p or {}).items()}
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


def run(state, prices, log_path, headers_for=mirror_headers, post=alpaca.post, positions=_positions, wait=_wait):
    """For each book with keys, send the orders to its account and append what happened to log_path.
    Never raises: the bot's own run must not fail."""
    results = {}
    for book in ACCOUNTS:
        if book not in state.get("books", {}):
            continue
        h = headers_for(book)
        if h is None:
            continue
        out = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "book": book, "orders": []}
        try:
            tgt = targets(state, book)
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
        results[book] = out
    return results or None
