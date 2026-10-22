"""Plan v3 nightly job for the ranking strategies (SYS1): M3 (residual momentum, tracks/main/m3_live.py),
M4 (the long-term composite) and M6 (analyst actions, the M8 weekly ranking). Each runs in its own Alpaca
paper account (adaptive/accounts.py).

Each account holds its strategy's fixed starting cash (SYS1 clarification 2), so the account shows that
strategy's own result; the learner's weighted book is ledger-only. On a rebalance night (a new M3_picks,
M4_picks or M8_picks entry in state/longterm/ledger.jsonl for tonight's session) a strategy targets its top TOP_K
picks, equal-sized from its account budget, and the job orders for the next open:
- names that left the list: cancel their open orders (the broker-held stop), then sell at the open;
- new names: a day limit buy at the close + 1%, with a broker-held stop 15% below the limit (OTO order).
The ledger-only random control takes the first TOP_K of the same entry's logged control list.

Dry run by default: orders are logged, not sent. Set ADAPTIVE_SEND=1 to send them.
Usage: python -m adaptive.nightly --state <dir>
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

from adaptive import accounts, orders, sizing
from core import alpaca
from core.ledger import Ledger

BOOK = {"main": 5000.0, "penny": 1000.0}
START_WEIGHTS = {"main": {"M3": 1 / 3, "M4": 1 / 3, "M6": 1 / 3}}
STRATEGIES = {"M3": {"account": "EL-M3", "kind": "M3_picks", "system": "main", "budget": 1250.0},
              "M4": {"account": "EL-M4", "kind": "M4_picks", "system": "main", "budget": 1250.0},
              "M6": {"account": "EL-M6", "kind": "M8_picks", "system": "main", "budget": 1250.0}}
TOP_K = 5
ENTRY_LIMIT = 0.01
STOP = sizing.MAIN.stop_cap


class Api:
    """Paper trading calls for one account."""

    def __init__(self, hdrs):
        self.h = hdrs

    def positions(self):
        return {p["symbol"]: p for p in alpaca.get(f"{alpaca.TRADE}/positions", {}, headers=self.h)}

    def open_orders(self):
        return alpaca.get(f"{alpaca.TRADE}/orders", {"status": "open", "nested": "true"}, headers=self.h)

    def cancel(self, order_id):
        url = f"{alpaca.TRADE}/orders/{order_id}"
        assert url.startswith("https://paper-api.alpaca.markets/")
        r = requests.delete(url, headers=self.h, timeout=60)
        return r.status_code

    def order(self, body):
        return alpaca.post(f"{alpaca.TRADE}/orders", body, headers=self.h)


def weights(root):
    p = root / "weights.json"
    return json.loads(p.read_text()) if p.exists() else START_WEIGHTS


def tonight_entry(longterm, kind, day):
    led = Ledger(longterm)
    e = led.entry(kind, day)
    return None if e is None else json.loads((longterm / e["file"]).read_text())


def plan(strategy, rec, held, day):
    """Exit and entry orders for one strategy's account. held: {symbol: qty} now in the account."""
    sysname = STRATEGIES[strategy]["system"]
    budget = STRATEGIES[strategy]["budget"]
    picks = sorted(rec["picks"], key=lambda p: p["rank"])[:TOP_K]
    want = {p["symbol"]: p for p in picks}
    out = []
    for sym, qty in sorted(held.items()):
        if sym not in want and qty > 0:
            out.append({"kind": "exit", "symbol": sym,
                        "body": {"symbol": sym, "qty": str(int(qty)), "side": "sell", "type": "market",
                                 "time_in_force": "opg", "client_order_id": f"el-{strategy}-{day}-{sym}-out"}})
    per_name = budget / max(len(picks), 1)
    for p in picks:
        if p["symbol"] in held:
            continue
        price = float(p["price"])
        limit = price * (1 + ENTRY_LIMIT)
        cap, _ = sizing.size(budget, "long", STOP, limit, float("inf"), 0, sizing.MAIN)
        shares = min(int(per_name // limit), cap)
        if shares < 1:
            out.append({"kind": "skip", "symbol": p["symbol"], "why": f"budget ${per_name:.0f} under one share"})
            continue
        out.append({"kind": "entry", "symbol": p["symbol"],
                    "body": orders.bracket(p["symbol"], "long", shares, limit, limit * (1 - STOP),
                                           client_id=f"el-{strategy}-{day}-{p['symbol']}-in")})
    return out


def control_targets(rec):
    return list(rec.get("control", []))[:TOP_K]


def run(state, now=None, send=None, api_for=None):
    send = (os.environ.get("ADAPTIVE_SEND") == "1") if send is None else send
    state = Path(state)
    root = state / "adaptive"
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(root)
    now = now if now is not None else alpaca.utcnow()
    from tracks.main.longterm.live import last_session
    t, _ = last_session(now)
    day = f"{t:%Y-%m-%d}"
    if led.has("adaptive_night", day):          # m4-daily can finish more than once a night
        print(f"{day} already handled")
        return None
    w_all = weights(root)
    report = {"day": day, "send": send, "strategies": {}}
    for s, cfg in STRATEGIES.items():
        rec = tonight_entry(state / "longterm", cfg["kind"], day)
        if rec is None:
            report["strategies"][s] = {"rebalance": False}
            continue
        hdrs = accounts.headers(cfg["account"])
        if hdrs is None and api_for is None:
            report["strategies"][s] = {"rebalance": True, "error": f"{cfg['account']} secrets not set"}
            continue
        api = api_for(s) if api_for else Api(hdrs)
        held = {k: float(v.get("qty", 0)) for k, v in api.positions().items()}
        orders_ = plan(s, rec, held, day)
        sent = []
        if send:
            exits = {o["symbol"] for o in orders_ if o["kind"] == "exit"}
            for o in api.open_orders():
                if o.get("symbol") in exits:
                    api.cancel(o["id"])
            for o in orders_:
                if "body" in o:
                    code, js = api.order(o["body"])
                    sent.append({"symbol": o["symbol"], "kind": o["kind"], "http": code,
                                 "status": js.get("status") if code < 300 else "rejected",
                                 "message": js.get("message")})
        report["strategies"][s] = {"rebalance": True, "weight": w_all[cfg["system"]].get(s, 0.0), "held": held,
                                   "orders": orders_, "sent": sent, "control": control_targets(rec)}
    if any(v.get("rebalance") for v in report["strategies"].values()) and not led.has("adaptive_night", day):
        led.write_json("adaptive_night", day, f"nights/{day}.json", report)
    print(json.dumps({k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items() if kk != "held"}
                      for k, v in report["strategies"].items()}, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    run(a.state)


if __name__ == "__main__":
    main()
