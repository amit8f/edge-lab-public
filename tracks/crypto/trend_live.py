"""CRY-TREND forward book (research/prereg/CRYPTO.md, Amendment 2): a $1,000 ledger book that holds BTC and ETH,
half each, only while the coin's daily close is above its close 28 days earlier, next to a buy-and-hold twin.
Ledger only: nothing is ever sent to an exchange.

Each run reads Kraken's public daily candles and logs every finished day not yet logged, in order, so a run
that GitHub skips is caught up by the next one.

    python -m tracks.crypto.trend_live --state state
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.ledger import Ledger
from tracks.crypto import llm_bot as K
from tracks.crypto import trend_study as T

COINS = {"BTC": "XBTUSD", "ETH": "ETHUSD"}
START_USD = 1000.0
COST = 0.005                  # round trip, half on each side
JUDGED = "2027-12-31"
MIN_ROUND_TRIPS = 8


# ---------- pure helpers (tested) ----------

def new_book():
    return {"cash": START_USD, "pos": dict.fromkeys(COINS, 0.0), "twin": dict.fromkeys(COINS, START_USD / len(COINS))}


def grow(book, ret):
    """The book after one day's returns ({coin: return})."""
    return {"cash": book["cash"], "pos": {c: v * (1 + ret[c]) for c, v in book["pos"].items()},
            "twin": {c: v * (1 + ret[c]) for c, v in book["twin"].items()}}


def trade(book, sig, cost=COST):
    """trend_study.book's rule at one close: sell whole every held coin whose signal is off, then enter every
    signalled coin not held with half the book's value (or the cash left, if less). -> (book, trades)."""
    cash, pos, trades = book["cash"], dict(book["pos"]), []
    for c in COINS:
        if pos[c] > 0 and not sig[c]:
            trades.append({"coin": c, "action": "sell", "usd": pos[c], "cost_usd": pos[c] * cost / 2})
            cash += pos[c] * (1 - cost / 2)
            pos[c] = 0.0
    total = cash + sum(pos.values())
    for c in COINS:
        if pos[c] == 0 and sig[c]:
            put = min(total / len(COINS), cash)
            cash -= put
            pos[c] = put * (1 - cost / 2)
            trades.append({"coin": c, "action": "buy", "usd": put, "cost_usd": put * cost / 2})
    return {"cash": cash, "pos": pos, "twin": book["twin"]}, trades


def equity(book):
    return book["cash"] + sum(book["pos"].values())


def new_days(closes, last):
    """Finished days still to log, oldest first. closes: {coin: {day: close}}. The first run logs only the
    newest finished day (the book's start)."""
    days = sorted(set.intersection(*(set(v) for v in closes.values())))
    return [d for d in days if d > last] if last else days[-1:]


def signal_at(closes, day):
    """{coin: 1 if the close of `day` is above the close 28 calendar days earlier}; None if a close is missing."""
    back = str(np.datetime64(day) - np.timedelta64(T.LOOKBACK, "D"))
    if any(back not in v for v in closes.values()):
        return None
    return {c: int(closes[c][day] > closes[c][back]) for c in COINS}


def summary(days):
    """The dashboard's view of the logged days (each a day file's dict, oldest first)."""
    rule = np.array([d["rule_equity"] for d in days]) / START_USD
    twin = np.array([d["twin_equity"] for d in days]) / START_USD
    out = {"first_day": days[0]["day"], "last_day": days[-1]["day"], "days": len(days),
           "rule_usd": days[-1]["rule_equity"], "twin_usd": days[-1]["twin_equity"],
           "held": [c for c, v in days[-1]["book"]["pos"].items() if v > 0],
           "round_trips": sum(1 for d in days for t in d["trades"] if t["action"] == "sell"),
           "judged": JUDGED, "min_round_trips": MIN_ROUND_TRIPS,
           "curve": [[d["day"], round(d["rule_equity"], 2), round(d["twin_equity"], 2)] for d in days]}
    if len(days) > 2:               # the start day is the base of the curve, as in the history check
        out["rule"], out["twin"] = T.stats(rule[1:]), T.stats(twin[1:])
    return out


# ---------- job ----------

def fetch():
    """{coin: {day: close}} from Kraken's finished daily candles (a candle is named by the UTC day it opens)."""
    return {c: {datetime.fromtimestamp(r[0], timezone.utc).strftime("%Y-%m-%d"): r[4] for r in K.candles(p, 1440)}
            for c, p in COINS.items()}


def run(state, fetch=fetch):
    root = Path(state) / "crytrend"
    led = Ledger(root / "ledger")
    logged = sorted(e["date"] for e in led.entries() if e["kind"] == "day")
    prev = json.loads((led.root / "days" / f"{logged[-1]}.json").read_text()) if logged else None
    closes = fetch()
    for day in new_days(closes, prev["day"] if prev else None):
        sig = signal_at(closes, day)
        if sig is None:
            raise RuntimeError(f"no close 28 days before {day}")
        if prev is None:
            book = new_book()
        else:
            yday = str(np.datetime64(day) - np.timedelta64(1, "D"))
            if yday != prev["day"]:
                raise RuntimeError(f"day {yday} is missing between {prev['day']} and {day}")
            book = grow(prev["book"], {c: closes[c][day] / prev["close"][c] - 1 for c in COINS})
        rec = {"day": day, "close": {c: closes[c][day] for c in COINS}, "signal": sig, "rule_equity": equity(book),
               "twin_equity": sum(book["twin"].values())}
        rec["book"], rec["trades"] = trade(book, sig)
        led.write_json("day", day, f"days/{day}.json", rec)
        print(f"[cry-trend] {day}: rule ${rec['rule_equity']:.2f} twin ${rec['twin_equity']:.2f} "
              f"signal {sig} trades {len(rec['trades'])}", flush=True)
        prev = rec
    probs = led.verify()
    if probs:
        raise RuntimeError(f"ledger problems: {probs[:3]}")
    days = [json.loads(f.read_text()) for f in sorted((led.root / "days").glob("*.json"))]
    if days:
        tmp = root / "summary.tmp"
        tmp.write_text(json.dumps(summary(days), indent=1))
        tmp.replace(root / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
