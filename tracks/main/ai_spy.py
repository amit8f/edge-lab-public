"""K5: the intraday momentum bot on SPY, $10,000 paper (research/prereg/K5.md). It replaces K2's AI book A.

No language model is involved. Each regular session: the SPY move from the open to 10:00 sets the side;
the bot goes long (up) or short (down) at the first 10-minute run from 15:30 and closes at the first run
from 15:57. Books: A (the rule), A-coin (same times, coin-flip side) and HOLD (SPY bought once and held).
Logs to the orphan branch `state-ai` under spy/, with the same hash-chained ledger as the other bots.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.ledger import Ledger
from tracks.main import ai_trader as k2
from tracks.main import ai_wide as w

START_USD = 10_000.0
SEED = 20261018
SYM = "SPY"
ET = k2.ET
FEE, BORROW = k2.FEE, k2.BORROW
ENTRY_AT, EXIT_AT = (15, 30), (15, 57)     # ET wall-clock marks: first run at or after each
OPEN_AT, MARK_AT = (9, 30), (10, 0)

NAMES = {"A": "AI bot A (K5: intraday momentum on SPY)", "A-coin": "Coin flip on A's times",
         "HOLD": "Just hold SPY"}


def new_book():
    return {"cash": START_USD, "pos": {}, "trips": [], "stopped": False}


def open_all(book, side, price, t):
    """Open one SPY position with all the equity (1x). Fees come off the cash at the open."""
    eq = w.equity(book, {SYM: price})
    notional = eq
    book["cash"] -= notional * FEE
    book["pos"][SYM] = {"side": side, "entry": price, "last": price, "qty": notional / price,
                        "notional": notional, "opened": t, "charged": t}
    return {"action": "open", "side": side, "price": price, "t": t}


def coin_side(day):
    return random.Random(f"{SEED}-{day}").choice([1, -1])


def mark_t(day, hh, mm):
    """Timestamp of a wall-clock ET mark on a session date (YYYY-MM-DD)."""
    return int(pd.Timestamp(f"{day} {hh:02d}:{mm:02d}", tz=ET).timestamp())


def signal(bars, day):
    """Return the open-to-10:00 return from SPY 1-minute bars [{t, o, c}] of one session, or None.

    The open is the first bar's open at 09:30 ET; the 10:00 mark is the first bar at or after 10:00 ET,
    at its open. Returns None when either is missing (too early in the session, or no bars)."""
    if not bars:
        return None
    ts = [pd.Timestamp(b["t"]).tz_convert(ET) for b in bars]
    start = pd.Timestamp(f"{day} 09:30", tz=ET)
    mark = pd.Timestamp(f"{day} 10:00", tz=ET)
    o = [b for b, t in zip(bars, ts) if t >= start]
    m = [b for b, t in zip(bars, ts) if t >= mark]
    if not o or not m:
        return None
    base, px = float(o[0]["o"]), float(m[0]["o"])
    return px / base - 1 if base > 0 else None


def step(state, day, now_et, r1, px, t):
    """Apply one run's decisions to the three books. Returns the fills and marks as a record."""
    rec = {"day": day, "r1": r1, "fills": [], "t": t}
    books = state["books"]
    if not state.get("started"):
        state["started"] = t
        rec["fills"].append({"book": "HOLD", **open_all(books["HOLD"], 1, px, t)})

    hm = (now_et.hour, now_et.minute)
    day_state = state.setdefault("days", {}).setdefault(day, {"entered": False, "exited": False})

    if (not day_state["entered"]) and hm >= ENTRY_AT and r1 not in (None, 0):
        side_a = 1 if r1 > 0 else -1
        side_c = coin_side(day)
        for name, side in (("A", side_a), ("A-coin", side_c)):
            b = books[name]
            if not b["stopped"] and SYM not in b["pos"]:
                rec["fills"].append({"book": name, **open_all(b, side, px, t)})
        day_state["entered"] = True

    day_open = int(pd.Timestamp(f"{day} 09:30", tz=ET).timestamp())
    for name in ("A", "A-coin"):
        p = books[name]["pos"].get(SYM)
        if p and (p["opened"] < day_open or (hm >= EXIT_AT and day_state["entered"])):
            # yesterday's position (a missed run) or today's session exit
            trip = w.close_pos(books[name], SYM, px, t, "session exit" if p["opened"] >= day_open else "stale")
            rec["fills"].append({"book": name, **trip})
    if day_state["entered"] and hm >= EXIT_AT:
        day_state["exited"] = True

    for b in books.values():
        if SYM in b["pos"]:
            b["pos"][SYM]["last"] = px
            w.charge(b, t)
    rec["equity"] = {n: round(w.equity(b, {SYM: px}), 6) for n, b in books.items()}
    return rec


def run(state_dir, now=None, fetch_bars=None, latest=None):
    """One 10-minute run. fetch_bars(day, end) -> SPY 1-minute SIP bars; latest() -> (price, time) IEX trade."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now_et = now.tz_convert(ET)
    day = f"{now_et:%Y-%m-%d}"
    open_t, close_t = pd.Timestamp(f"{day} 09:30", tz=ET), pd.Timestamp(f"{day} 16:00", tz=ET)
    if not (open_t <= now_et < close_t) or now_et.weekday() >= 5:
        print(f"{now_et:%H:%M} ET: market closed; nothing to do")
        return None
    root = Path(state_dir) / "spy"
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(root)
    key = f"{now_et:%Y-%m-%dT%H}:{now_et.minute // 10 * 10:02d}"
    if led.has("run", key):
        print(f"{key} already logged")
        return None
    bars = fetch_bars(day, now - pd.Timedelta(minutes=16))
    r1 = signal(bars, day)
    px, px_time = latest()
    t = int(px_time.timestamp())
    bpath = root / "books.json"
    if bpath.exists():
        state = json.loads(bpath.read_text())
    else:
        state = {"books": {n: new_book() for n in NAMES}, "started": None, "days": {}}
    rec = step(state, day, now_et, r1, px, t)
    rec["price_source"] = "iex"
    led.write_json("run", key, f"runs/{day}/{key.replace(':', '')}.json", rec)
    k2._atomic(bpath, json.dumps(state, indent=1, sort_keys=True))
    from tracks.main import ai_mirror
    ai_mirror.run(state, {SYM: px}, root / "mirror.jsonl")
    print(f"{key}: r1 {r1}, SPY {px}, equity {rec['equity']}")
    return rec


def live_bars(day, end):
    from core import alpaca
    j = alpaca.get("https://data.alpaca.markets/v2/stocks/bars",
                   {"symbols": SYM, "timeframe": "1Min", "start": f"{day}T13:30:00Z",
                    "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "feed": "sip", "adjustment": "raw",
                    "limit": 10000})
    return (j.get("bars") or {}).get(SYM, [])


def live_latest():
    px, t = w.iex_latest([SYM])[SYM]
    return px, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state, fetch_bars=live_bars, latest=live_latest)


if __name__ == "__main__":
    main()
