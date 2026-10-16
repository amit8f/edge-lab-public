"""K1: an AI crypto trader with $100 on paper (research/prereg/K1.md).

Every hour: read Kraken's public candles for BTC, ETH and SOL, settle stops, liquidations and funding on
the candles closed since the last run, ask Gemini what to do for books A and B, mirror their trades in
coin-flip twins, run the HOLD and TREND baselines, and log everything in the hash-chained ledger.
Nothing is ever sent to an exchange.

    python -m tracks.crypto.llm_bot --state state
"""
import argparse
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from core.ledger import Ledger

PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
START_USD = 100.0
FLOOR_USD = 5.0              # a book below this stops trading (stop rule)
SEED = 20261016
MODEL = "gemini-2.5-flash"
KRAKEN = "https://api.kraken.com/0/public"

_risk = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/risk.yaml").read_text())["tracks"]["crypto"]
FEE = _risk["fee_per_side"]
FUND_PER_HOUR = _risk["funding_per_8h"] / 8
MAINT = _risk["maintenance_margin"]

# lev/margin: leverage and share of equity posted as margin. perp: funding and liquidation apply.
BOOKS = {
    "A": dict(kind="ai", lev=5, margin=0.25, stop=None, perp=True, daily=False),
    "B": dict(kind="ai", lev=2, margin=0.25, stop=0.05, perp=True, daily=True),
    "A-coin": dict(kind="coin", twin="A", lev=5, margin=0.25, stop=None, perp=True),
    "B-coin": dict(kind="coin", twin="B", lev=2, margin=0.25, stop=0.05, perp=True),
    "HOLD": dict(kind="hold", lev=1, margin=1.0, stop=None, perp=False),
    "TREND": dict(kind="trend", lev=1, margin=1.0, stop=None, perp=False),
}

PROMPT_A = """You trade a paper crypto account. Symbols: BTC, ETH, SOL (USD perpetuals).
Rules: one position at a time, 25% of equity as margin, 5x isolated leverage, 0.05% fee per side.
Below are the last closed candles (UTC; time,open,high,low,close,volume) and your account.
Read the last 24 hours and the open position. Reason about whether there is an edge. Decide one action:
long, short, close or hold. Reply with JSON only:
{"action": "long|short|close|hold", "symbol": "BTC|ETH|SOL", "reason": "one line"}"""

PROMPT_B = """You trade a paper crypto account. Symbols: BTC, ETH, SOL (USD perpetuals).
Rules: one position at a time, 25% of equity as margin, 2x isolated leverage, a 5% stop-loss on every
position, 0.05% fee per side and funding while a position is open.
Below are the last closed candles (UTC; time,open,high,low,close,volume) and your account.
Most hourly moves are noise, and every trade costs fees. Holding cash is a good answer: only open or
keep a position when the daily trend and the last day's price action both point the same way. Decide one
action: long, short, close or hold. Reply with JSON only:
{"action": "long|short|close|hold", "symbol": "BTC|ETH|SOL", "reason": "one line"}"""


# ---------- accounting ----------

def new_book():
    return {"cash": START_USD, "pos": None, "trips": [], "stopped": False}


def equity(book, prices):
    p = book["pos"]
    return book["cash"] + (p["side"] * p["qty"] * (prices[p["sym"]] - p["entry"]) if p else 0.0)


def liq_price(entry, side, lev):
    return entry * (1 - side * (1 / lev - MAINT))


def open_pos(book, cfg, sym, side, price, t):
    eq = book["cash"]
    margin = eq * cfg["margin"]
    notional = margin * cfg["lev"]
    if not cfg["perp"]:                      # spot: the fee comes out of what is bought
        notional = eq / (1 + FEE)
    book["cash"] -= notional * FEE
    book["pos"] = {"sym": sym, "side": side, "entry": price, "qty": notional / price, "notional": notional,
                   "margin": margin, "opened": t, "checked": t,
                   "stop": price * (1 - side * cfg["stop"]) if cfg["stop"] else None,
                   "liq": liq_price(price, side, cfg["lev"]) if cfg["perp"] else None}
    return {"action": "open", "sym": sym, "side": side, "price": price, "notional": round(notional, 6), "t": t}


def close_pos(book, price, t, why):
    p = book["pos"]
    if why == "liquidated":
        pnl, fee = -p["margin"], 0.0
    else:
        pnl, fee = p["side"] * p["qty"] * (price - p["entry"]), p["qty"] * price * FEE
    book["cash"] += pnl - fee
    trip = {"sym": p["sym"], "side": p["side"], "open_t": p["opened"], "close_t": t, "entry": p["entry"],
            "exit": price, "pnl": round(pnl - fee, 6), "why": why}
    book["trips"].append(trip)
    book["pos"] = None
    if book["cash"] < FLOOR_USD:
        book["stopped"] = True
    return dict(trip, action="close")


def settle(book, cfg, candles):
    """Funding, liquidation and stop on each closed 1h candle that started after the last check.
    candles: [[t, o, h, l, c, v], ...] for the position's symbol, t in epoch seconds."""
    p, out = book["pos"], []
    if not p:
        return out
    for t, _o, h, lo, _c, _v in candles:
        if t < p["checked"] or book["pos"] is None:
            continue
        if cfg["perp"]:
            book["cash"] -= p["notional"] * FUND_PER_HOUR
        worst = lo if p["side"] > 0 else h
        if p["liq"] is not None and (worst - p["liq"]) * p["side"] <= 0:
            out.append(close_pos(book, p["liq"], t + 3600, "liquidated"))
        elif p["stop"] is not None and (worst - p["stop"]) * p["side"] <= 0:
            out.append(close_pos(book, p["stop"], t + 3600, "stop"))
        else:
            p["checked"] = t + 3600
    return out


def settle_now(book, price, t):
    """The current price can also be past the stop or liquidation level (the unclosed hour)."""
    p = book["pos"]
    if not p:
        return []
    if p["liq"] is not None and (price - p["liq"]) * p["side"] <= 0:
        return [close_pos(book, p["liq"], t, "liquidated")]
    if p["stop"] is not None and (price - p["stop"]) * p["side"] <= 0:
        return [close_pos(book, p["stop"], t, "stop")]
    return []


def apply_action(book, cfg, action, sym, prices, t):
    """long/short/close/hold for one book; returns fills. Switching side or symbol closes first."""
    if book["stopped"] or action == "hold":
        return []
    fills, p = [], book["pos"]
    if action == "close":
        return [close_pos(book, prices[p["sym"]], t, "decision")] if p else []
    side = 1 if action == "long" else -1
    if p and p["sym"] == sym and p["side"] == side:
        return []
    if p:
        fills.append(close_pos(book, prices[p["sym"]], t, "decision"))
        if book["stopped"]:
            return fills
    fills.append(open_pos(book, cfg, sym, side, prices[sym], t))
    return fills


def coin_side(name, t):
    return random.Random(f"{SEED}-{name}-{t}").choice([1, -1])


# ---------- data ----------

def _get(path, params):
    r = requests.get(f"{KRAKEN}/{path}", params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"kraken {path}: {j['error']}")
    return j["result"]


def candles(pair, interval):
    """Closed candles only (Kraken's last row is the unfinished one)."""
    res = _get("OHLC", {"pair": pair, "interval": interval})
    rows = next(v for k, v in res.items() if k != "last")
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows[:-1]]


def last_price(pair):
    res = _get("Ticker", {"pair": pair})
    return float(next(iter(res.values()))["c"][0])


def fetch_all():
    return {s: {"1h": candles(p, 60), "4h": candles(p, 240), "1d": candles(p, 1440), "last": last_price(p)}
            for s, p in PAIRS.items()}


# ---------- model ----------

def fmt(rows):
    return "\n".join(f"{datetime.fromtimestamp(r[0], timezone.utc):%Y-%m-%d %H:%M},"
                     f"{r[1]:g},{r[2]:g},{r[3]:g},{r[4]:g},{r[5]:.2f}" for r in rows)


def context(book, data, daily, now_iso):
    parts = [f"Now: {now_iso} UTC"]
    for s, d in data.items():
        parts.append(f"## {s} last price {d['last']:g}\n1h (last 24):\n{fmt(d['1h'][-24:])}\n"
                     f"4h (last 12):\n{fmt(d['4h'][-12:])}")
        if daily:
            parts.append(f"{s} 1d (last 30):\n{fmt(d['1d'][-30:])}")
    prices = {s: d["last"] for s, d in data.items()}
    p = book["pos"]
    pos = "none" if not p else (f"{'long' if p['side'] > 0 else 'short'} {p['sym']} entry {p['entry']:g}, "
                                f"notional ${p['notional']:.2f}, unrealized "
                                f"${p['side'] * p['qty'] * (prices[p['sym']] - p['entry']):.2f}")
    parts.append(f"## Account\nequity ${equity(book, prices):.2f}, open position: {pos}")
    return "\n\n".join(parts)


def parse(text):
    """{"action", "symbol", "reason"} from a reply; anything unreadable is hold (pre-registered)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    act = str(d.get("action", "")).strip().lower()
    sym = str(d.get("symbol", "")).strip().upper()
    if act not in {"long", "short", "close", "hold"} or (act in {"long", "short"} and sym not in PAIRS):
        return {"action": "hold", "symbol": None, "reason": "unreadable answer", "valid": False}
    return {"action": act, "symbol": sym if sym in PAIRS else None,
            "reason": " ".join(str(d.get("reason", "")).split())[:300], "valid": True}


def ask_gemini(system, text):
    from tracks.news.llm import _post
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingBudget": 1024}}}
    j = _post(url, {"x-goog-api-key": key}, body, "gemini")
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


# ---------- one hourly step ----------

def decide(rec, name, cfg, b, data, prices, t, now_iso, ask):
    system = PROMPT_A if name == "A" else PROMPT_B
    text = context(b, data, cfg["daily"], now_iso)
    dec = {"prompt_sha256": hashlib.sha256((system + "\n\n" + text).encode()).hexdigest()}
    try:
        raw = ask(system, text)
        dec["raw"] = raw
    except Exception as e:  # quota, network: logged and treated as hold
        raw, dec["error"] = "", f"{type(e).__name__}: {str(e)[:200]}"
    dec["parsed"] = parse(raw)
    rec["decisions"][name] = dec
    f = apply_action(b, cfg, dec["parsed"]["action"], dec["parsed"]["symbol"], prices, t)
    rec["fills"][name] += f
    return f


def step(state, data, now, ask=ask_gemini):
    """Advance every book by one run. Mutates state; returns the record to log."""
    t = int(now.timestamp())
    prices = {s: d["last"] for s, d in data.items()}
    books = state["books"]
    rec = {"t": t, "prices": prices, "fills": {}, "decisions": {}, "new_1h": {}}
    for s, d in data.items():   # every closed hourly candle once, for the 180-day replay
        seen = state["seen_1h"].get(s, 0)
        rec["new_1h"][s] = [r for r in d["1h"] if r[0] > seen]
        if d["1h"]:
            state["seen_1h"][s] = max(seen, d["1h"][-1][0])

    ended = {}
    for name, cfg in BOOKS.items():
        b, f = books[name], []
        if b["pos"]:
            f += settle(b, cfg, data[b["pos"]["sym"]]["1h"])
        if b["pos"]:
            f += settle_now(b, prices[b["pos"]["sym"]], t)
        rec["fills"][name] = f
        ended[name] = any(x["action"] == "close" for x in f)

    now_iso = now.strftime("%Y-%m-%d %H:%M")
    for name in ("A", "B"):
        cfg, b, f = BOOKS[name], books[name], []
        if not b["stopped"]:
            f = decide(rec, name, cfg, b, data, prices, t, now_iso, ask)
        ended[name] = ended[name] or any(x["action"] == "close" for x in f)

        # the coin-flip twin follows the same timing and symbol with a random side
        tw, tcfg, tf = books[f"{name}-coin"], BOOKS[f"{name}-coin"], []
        opened = next((x for x in f if x["action"] == "open"), None)
        if tw["pos"] and (ended[name] or opened):
            tf.append(close_pos(tw, prices[tw["pos"]["sym"]], t, "twin"))
        if opened and not tw["stopped"]:
            tf.append(open_pos(tw, tcfg, opened["sym"], coin_side(name, t), prices[opened["sym"]], t))
        rec["fills"][f"{name}-coin"] += tf

    hb = books["HOLD"]
    if hb["pos"] is None and not hb["trips"]:
        rec["fills"]["HOLD"].append(open_pos(hb, BOOKS["HOLD"], "BTC", 1, prices["BTC"], t))

    tb, daily = books["TREND"], data["BTC"]["1d"]
    if len(daily) >= 20:
        above = daily[-1][4] > sum(r[4] for r in daily[-20:]) / 20
        if above and tb["pos"] is None:
            rec["fills"]["TREND"].append(open_pos(tb, BOOKS["TREND"], "BTC", 1, prices["BTC"], t))
        elif not above and tb["pos"] is not None:
            rec["fills"]["TREND"].append(close_pos(tb, prices["BTC"], t, "below 20-day average"))

    rec["equity"] = {n: round(equity(b, prices), 6) for n, b in books.items()}
    rec["positions"] = {n: b["pos"] for n, b in books.items()}
    return rec


def init_state(t):
    return {"started": t, "books": {n: new_book() for n in BOOKS}, "seen_1h": {}}


def summary(state, rec):
    started = datetime.fromtimestamp(state["started"], timezone.utc)
    lines = [f"# K1 crypto paper bot\n\nStarted {started:%Y-%m-%d %H:%M} UTC with $100 per book. "
             f"Last run {datetime.fromtimestamp(rec['t'], timezone.utc):%Y-%m-%d %H:%M} UTC. "
             "Paper only; verdict on 2027-04-14 (research/prereg/K1.md).\n",
             "| Book | Equity | Round trips | Liquidations | Open position |", "|---|---|---|---|---|"]
    for n, b in state["books"].items():
        p = b["pos"]
        pos = f"{'long' if p['side'] > 0 else 'short'} {p['sym']} @ {p['entry']:g}" if p else "cash"
        liq = sum(x["why"] == "liquidated" for x in b["trips"])
        lines.append(f"| {n} | ${rec['equity'][n]:.2f} | {len(b['trips'])} | {liq} | {pos} |")
    for n in ("A", "B"):
        d = rec["decisions"].get(n)
        if d:
            lines.append(f"\n{n} said: {d['parsed']['action']} {d['parsed']['symbol'] or ''}: {d['parsed']['reason']}")
    return "\n".join(lines) + "\n"


def _atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run(state_dir, now=None, fetch=fetch_all, ask=ask_gemini):
    now = now or datetime.now(timezone.utc)
    root = Path(state_dir) / "crypto"
    led = Ledger(root)
    hour = now.strftime("%Y-%m-%dT%H")
    if led.has("hour", hour):
        print(f"{hour} already logged")
        return None
    bpath = root / "books.json"
    state = json.loads(bpath.read_text()) if bpath.exists() else init_state(int(now.timestamp()))
    rec = step(state, fetch(), now, ask)
    led.write_json("hour", hour, f"hours/{hour[:10]}/{hour}.json", rec)
    _atomic(bpath, json.dumps(state, indent=1, sort_keys=True))
    _atomic(root / "README.md", summary(state, rec))
    print(summary(state, rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
