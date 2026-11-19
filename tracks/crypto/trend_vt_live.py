"""CRY-TREND-VT forward book (research/prereg/CRYTRENDVT.md): a $1,000 ledger book that holds BTC and ETH on
CRY-TREND's signal, each held coin sized (1/2) x min(1, 0.50 / its 20-day realized volatility), reset at entry,
at exit (sold whole) and at each Monday close. Next to CRY-TREND's own book. Ledger only: nothing is sent anywhere.

    python -m tracks.crypto.trend_vt_live --state state
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from core.ledger import Ledger
from tracks.crypto import trend_live as L
from tracks.crypto import trend_study as T

COINS = L.COINS
START_USD, COST, JUDGED, MIN_ROUND_TRIPS = L.START_USD, L.COST, L.JUDGED, L.MIN_ROUND_TRIPS


# ---------- pure helpers (tested) ----------

def vol_at(closes, day):
    """{coin: annualised st.dev. of its last 20 daily returns at `day`}; None if a close is missing."""
    d0 = np.datetime64(day)
    days = [str(d0 - np.timedelta64(i, "D")) for i in range(T.VT_WINDOW, -1, -1)]
    if any(d not in v for v in closes.values() for d in days):
        return None
    out = {}
    for c, v in closes.items():
        px = np.array([v[d] for d in days])
        out[c] = float(np.std(px[1:] / px[:-1] - 1, ddof=1) * np.sqrt(365))
    return out


def trade(book, sig, vol, monday, cost=COST):
    """trend_study.vt_book's rule at one close. -> (book, trades)."""
    cash, pos, trades = book["cash"], dict(book["pos"]), []
    total = cash + sum(pos.values())
    for c in COINS:
        w = float(T.vt_weight(np.array([vol[c]]), len(COINS))[0]) if sig[c] else 0.0
        if pos[c] > 0 and not sig[c]:
            delta = -pos[c]
        elif sig[c] and (pos[c] == 0 or monday):
            delta = min(w * total, cash + pos[c]) - pos[c]
        else:
            continue
        fee = abs(delta) * cost / 2
        cash -= delta + fee
        pos[c] += delta
        if delta:
            trades.append({"coin": c, "action": "buy" if delta > 0 else "sell", "usd": abs(delta), "cost_usd": fee})
    return {"cash": cash, "pos": pos, "twin": book["twin"]}, trades


def summary(days):
    out = L.summary(days)
    out["round_trips"] = sum(1 for d in days for t in d["trades"] if t["action"] == "sell" and d["signal"][t["coin"]] == 0)
    return out


# ---------- job ----------

def run(state, fetch=L.fetch):
    root = Path(state) / "crytrendvt"
    led = Ledger(root / "ledger")
    logged = sorted(e["date"] for e in led.entries() if e["kind"] == "day")
    prev = json.loads((led.root / "days" / f"{logged[-1]}.json").read_text()) if logged else None
    closes = fetch()
    for day in L.new_days(closes, prev["day"] if prev else None):
        sig, vol = L.signal_at(closes, day), vol_at(closes, day)
        if sig is None or vol is None:
            raise RuntimeError(f"not enough closes before {day}")
        if prev is None:
            book = L.new_book()
        else:
            yday = str(np.datetime64(day) - np.timedelta64(1, "D"))
            if yday != prev["day"]:
                raise RuntimeError(f"day {yday} is missing between {prev['day']} and {day}")
            book = L.grow(prev["book"], {c: closes[c][day] / prev["close"][c] - 1 for c in COINS})
        rec = {"day": day, "close": {c: closes[c][day] for c in COINS}, "signal": sig, "vol": vol,
               "rule_equity": L.equity(book), "twin_equity": sum(book["twin"].values())}
        rec["book"], rec["trades"] = trade(book, sig, vol, date.fromisoformat(day).weekday() == 0)
        led.write_json("day", day, f"days/{day}.json", rec)
        print(f"[cry-trend-vt] {day}: rule ${rec['rule_equity']:.2f} trades {len(rec['trades'])}", flush=True)
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
