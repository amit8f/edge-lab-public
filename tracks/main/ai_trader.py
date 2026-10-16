"""K2: an AI stock trader with $100 on paper (research/prereg/K2.md).

Every hour of the regular session: read Alpaca SIP bars (raw, ending now - 16 minutes) for ten big
stocks, settle stops and margin calls on the bars since the last check, ask Gemini what books A and B
should do, mirror their trades in coin-flip twins, run the HOLD (SPY) and TREND baselines, and log
everything in the hash-chained ledger. No order is sent anywhere; fills are simulated.

    python -m tracks.main.ai_trader --state state
"""
import argparse
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from core.ledger import Ledger

SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
START_USD = 100.0
FLOOR_USD = 5.0
SEED = 20261016
MODEL = "gemini-2.5-flash"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ET = "America/New_York"

_risk = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/risk.yaml").read_text())["tracks"]["ai_trader"]
FEE = _risk["fee_per_side"]
BORROW = _risk["borrow_per_year"]
MARGIN_RATE = _risk["margin_rate_per_year"]
MAINT = {1: _risk["maint_long"], -1: _risk["maint_short"]}

# exposure: position notional as a multiple of the book's equity at the open
BOOKS = {
    "A": dict(kind="ai", exposure=1.25, stop=None, daily=False),
    "B": dict(kind="ai", exposure=0.5, stop=0.05, daily=True),
    "A-coin": dict(kind="coin", exposure=1.25, stop=None),
    "B-coin": dict(kind="coin", exposure=0.5, stop=0.05),
    "HOLD": dict(kind="hold", exposure=1.0, stop=None),
    "TREND": dict(kind="trend", exposure=1.0, stop=None),
}

SYMS_TXT = ", ".join(SYMBOLS)
PROMPT_A = f"""You trade a paper US stock account. Symbols: {SYMS_TXT}.
Rules: one position at a time, long or short, position size 1.25x account equity, no stop-loss, 0.1%
cost per side, shorts pay borrow and leverage pays margin interest.
Below are the last regular-session 1-hour bars (US/Eastern; time,open,high,low,close,volume) and your
account. Read the recent bars and the open position. Reason about whether there is an edge. Decide one
action: long, short, close or hold. Reply with JSON only:
{{"action": "long|short|close|hold", "symbol": "<one of the symbols>", "reason": "one line"}}"""

PROMPT_B = f"""You trade a paper US stock account. Symbols: {SYMS_TXT}.
Rules: one position at a time, long or short, position size 0.5x account equity, a 5% stop-loss on
every position, 0.1% cost per side, shorts pay borrow.
Below are the last regular-session 1-hour bars and 60 daily bars (US/Eastern;
time,open,high,low,close,volume) and your account.
Most hourly moves are noise, and every trade costs money. Holding cash is a good answer: only open or
keep a position when the daily trend and the recent hourly bars both point the same way. Decide one
action: long, short, close or hold. Reply with JSON only:
{{"action": "long|short|close|hold", "symbol": "<one of the symbols>", "reason": "one line"}}"""


# ---------- accounting ----------

def new_book():
    return {"cash": START_USD, "pos": None, "trips": [], "stopped": False}


def equity(book, prices):
    p = book["pos"]
    return book["cash"] + (p["side"] * p["qty"] * (prices[p["sym"]] - p["entry"]) if p else 0.0)


def call_price(entry, side, exposure):
    """Price at which the account's equity falls to the maintenance share of the position value."""
    m = MAINT[side]
    if side > 0:
        return entry * (1 - 1 / exposure) / (1 - m) if exposure > 1 else None
    return entry * (1 + 1 / exposure) / (1 + m)


def carry_rate(side, exposure):
    """Yearly cost as a share of notional: borrow for shorts, margin interest on the borrowed part."""
    return BORROW if side < 0 else MARGIN_RATE * max(0.0, 1 - 1 / exposure)


def charge_carry(book, t):
    p = book["pos"]
    if p and t > p["charged"]:
        book["cash"] -= p["notional"] * p["carry"] * (t - p["charged"]) / (3600 * 24 * 365)
        p["charged"] = t


def open_pos(book, cfg, sym, side, price, t):
    notional = book["cash"] * cfg["exposure"]
    if cfg["exposure"] == 1.0 and side > 0:     # cash purchase: the cost comes out of what is bought
        notional = book["cash"] / (1 + FEE)
    book["cash"] -= notional * FEE
    book["pos"] = {"sym": sym, "side": side, "entry": price, "qty": notional / price, "notional": notional,
                   "opened": t, "checked": t, "charged": t, "carry": carry_rate(side, cfg["exposure"]),
                   "stop": price * (1 - side * cfg["stop"]) if cfg["stop"] else None,
                   "call": call_price(price, side, cfg["exposure"])}
    return {"action": "open", "sym": sym, "side": side, "price": price, "notional": round(notional, 6), "t": t}


def close_pos(book, price, t, why):
    charge_carry(book, t)
    p = book["pos"]
    pnl, fee = p["side"] * p["qty"] * (price - p["entry"]), p["qty"] * price * FEE
    book["cash"] += pnl - fee
    trip = {"sym": p["sym"], "side": p["side"], "open_t": p["opened"], "close_t": t, "entry": p["entry"],
            "exit": price, "pnl": round(pnl - fee, 6), "why": why}
    book["trips"].append(trip)
    book["pos"] = None
    if book["cash"] < FLOOR_USD:
        book["stopped"] = True
    return dict(trip, action="close")


def _hit(level, side, o, worst):
    """Fill price if the bar reached the level (a bar opening past it fills at its open), else None."""
    if level is None:
        return None
    if (o - level) * side <= 0:
        return o
    if (worst - level) * side <= 0:
        return level
    return None


def settle(book, bars):
    """Stops and margin calls on each regular-session bar that started after the last check.
    bars: [[t_start, o, h, l, c, v, t_end], ...] for the position's symbol, epoch seconds."""
    p, out = book["pos"], []
    if not p:
        return out
    for t, o, h, lo, _c, _v, t_end in bars:
        if t < p["checked"]:
            continue
        worst = lo if p["side"] > 0 else h
        px = _hit(p["call"], p["side"], o, worst)
        if px is not None:
            out.append(close_pos(book, px, t_end, "margin call"))
            break
        px = _hit(p["stop"], p["side"], o, worst)
        if px is not None:
            out.append(close_pos(book, px, t_end, "stop"))
            break
        p["checked"] = t_end
    return out


def settle_now(book, price, t):
    """The current price can also be past the stop or margin-call level (the bar still forming)."""
    p = book["pos"]
    if not p:
        return []
    for level, why in ((p["call"], "margin call"), (p["stop"], "stop")):
        if level is not None and (price - level) * p["side"] <= 0:
            return [close_pos(book, price, t, why)]
    return []


def apply_action(book, cfg, action, sym, prices, t):
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

def _bars(symbols, timeframe, start, end):
    from core.alpaca import get
    out, token = {s: [] for s in symbols}, None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
                  "adjustment": "raw", "feed": "sip", "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        j = get(BARS_URL, params)
        for s, rows in (j.get("bars") or {}).items():
            out[s] += rows
        token = j.get("next_page_token")
        if not token:
            return out


def session_hours(rows, sessions):
    """30-minute bars -> regular-session hour bars (9:30-10:30, ..., 15:30-close).
    sessions: {date: (open_utc, close_utc)} as pandas Timestamps."""
    groups = {}
    for r in rows:
        if not r["v"]:
            continue
        ts = pd.Timestamp(r["t"])
        day = ts.tz_convert(ET).date()
        if day not in sessions:
            continue
        o, c = sessions[day]
        if not (o <= ts < c):
            continue
        k = int((ts - o).total_seconds() // 3600)
        groups.setdefault((day, k), []).append(r)
    out = []
    for (day, k), g in sorted(groups.items()):
        o, c = sessions[day]
        start = o + pd.Timedelta(hours=k)
        end = min(start + pd.Timedelta(hours=1), c)
        out.append([int(start.timestamp()), g[0]["o"], max(x["h"] for x in g), min(x["l"] for x in g),
                    g[-1]["c"], float(sum(x["v"] for x in g)), int(end.timestamp())])
    return out


def market_window(now):
    """(price time, sessions) when now - 16 min is inside a regular session, else (None, sessions)."""
    from core.alpaca import SIP_DELAY, trading_calendar
    cal = trading_calendar((now - pd.Timedelta(days=20)).date(), now.date())
    sessions = {d.date(): (pd.Timestamp(f"{d:%Y-%m-%d} {o}", tz=ET).tz_convert("UTC"),
                           pd.Timestamp(f"{d:%Y-%m-%d} {c}", tz=ET).tz_convert("UTC"))
                for d, o, c in zip(cal["date"], cal["open"], cal["close"])}
    px_time = now - SIP_DELAY
    o_c = sessions.get(px_time.tz_convert(ET).date())
    return (px_time if o_c and o_c[0] < px_time <= o_c[1] else None), sessions


def fetch_all(now):
    """{sym: {"1h": [...], "1d": [...], "last": price, "last_t": iso}} or None outside the session."""
    now = pd.Timestamp(now)
    px_time, sessions = market_window(now)
    if px_time is None:
        return None
    end = px_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    half = _bars(SYMBOLS, "30Min", (now - pd.Timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ"), end)
    daily = _bars(SYMBOLS, "1Day", (now - pd.Timedelta(days=100)).strftime("%Y-%m-%d"), end)
    mins = _bars(SYMBOLS, "1Min", (px_time - pd.Timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ"), end)
    today = px_time.tz_convert(ET).date()
    out = {}
    for s in SYMBOLS:
        m = [r for r in mins[s] if r["v"]]
        if not m:
            raise RuntimeError(f"{s}: no minute bars before {end}")
        hours = [h for h in session_hours(half[s], sessions) if h[6] <= int(px_time.timestamp())]
        days = [[int(pd.Timestamp(r["t"]).timestamp()), r["o"], r["h"], r["l"], r["c"], float(r["v"])]
                for r in daily[s] if r["v"] and pd.Timestamp(r["t"]).tz_convert(ET).date() < today]
        out[s] = {"1h": hours, "1d": days, "last": float(m[-1]["c"]), "last_t": m[-1]["t"]}
    return out


# ---------- model ----------

def fmt(rows, daily=False):
    f = "%Y-%m-%d" if daily else "%Y-%m-%d %H:%M"
    return "\n".join(f"{pd.Timestamp(r[0], unit='s', tz='UTC').tz_convert(ET):{f}},"
                     f"{r[1]:g},{r[2]:g},{r[3]:g},{r[4]:g},{r[5]:.0f}" for r in rows)


def context(book, data, daily, now_txt):
    parts = [f"Prices as of {now_txt} US/Eastern"]
    for s, d in data.items():
        parts.append(f"## {s} last price {d['last']:g}\n1h (last 7 sessions):\n{fmt(d['1h'][-49:])}")
        if daily:
            parts.append(f"{s} 1d (last 60):\n{fmt(d['1d'][-60:], True)}")
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
    if not isinstance(d, dict):
        d = {}
    act = str(d.get("action", "")).strip().lower()
    sym = str(d.get("symbol", "")).strip().upper()
    if act not in {"long", "short", "close", "hold"} or (act in {"long", "short"} and sym not in SYMBOLS):
        return {"action": "hold", "symbol": None, "reason": "unreadable answer", "valid": False}
    return {"action": act, "symbol": sym if sym in SYMBOLS else None,
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


# ---------- one run ----------

def decide(rec, name, cfg, b, data, prices, t, now_txt, ask):
    system = PROMPT_A if name == "A" else PROMPT_B
    text = context(b, data, cfg["daily"], now_txt)
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


def step(state, data, t, ask=ask_gemini):
    """Advance every book by one run at price time t (epoch seconds). Mutates state; returns the record."""
    prices = {s: d["last"] for s, d in data.items()}
    books = state["books"]
    rec = {"t": t, "prices": prices, "fills": {}, "decisions": {}, "new_1h": {}}
    for s, d in data.items():   # every session hour bar once, for the six-month replay
        seen = state["seen_1h"].get(s, 0)
        rec["new_1h"][s] = [r for r in d["1h"] if r[0] > seen]
        if d["1h"]:
            state["seen_1h"][s] = max(seen, d["1h"][-1][0])

    ended = {}
    for name in BOOKS:
        b, f = books[name], []
        if b["pos"]:
            f += settle(b, data[b["pos"]["sym"]]["1h"])
        if b["pos"]:
            f += settle_now(b, prices[b["pos"]["sym"]], t)
        if b["pos"]:
            charge_carry(b, t)
        rec["fills"][name] = f
        ended[name] = any(x["action"] == "close" for x in f)

    now_txt = f"{pd.Timestamp(t, unit='s', tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M}"
    for name in ("A", "B"):
        cfg, b, f = BOOKS[name], books[name], []
        if not b["stopped"]:
            f = decide(rec, name, cfg, b, data, prices, t, now_txt, ask)
        ended[name] = ended[name] or any(x["action"] == "close" for x in f)
        tw, tf = books[f"{name}-coin"], []
        opened = next((x for x in f if x["action"] == "open"), None)
        if tw["pos"] and (ended[name] or opened):
            tf.append(close_pos(tw, prices[tw["pos"]["sym"]], t, "twin"))
        if opened and not tw["stopped"]:
            tf.append(open_pos(tw, BOOKS[f"{name}-coin"], opened["sym"], coin_side(name, t),
                               prices[opened["sym"]], t))
        rec["fills"][f"{name}-coin"] += tf

    hb = books["HOLD"]
    if hb["pos"] is None and not hb["trips"]:
        rec["fills"]["HOLD"].append(open_pos(hb, BOOKS["HOLD"], "SPY", 1, prices["SPY"], t))

    tb, daily = books["TREND"], data["SPY"]["1d"]
    if len(daily) >= 20:
        above = daily[-1][4] > sum(r[4] for r in daily[-20:]) / 20
        if above and tb["pos"] is None:
            rec["fills"]["TREND"].append(open_pos(tb, BOOKS["TREND"], "SPY", 1, prices["SPY"], t))
        elif not above and tb["pos"] is not None:
            rec["fills"]["TREND"].append(close_pos(tb, prices["SPY"], t, "below 20-day average"))

    rec["equity"] = {n: round(equity(b, prices), 6) for n, b in books.items()}
    rec["positions"] = {n: b["pos"] for n, b in books.items()}
    return rec


def init_state(t):
    return {"started": t, "books": {n: new_book() for n in BOOKS}, "seen_1h": {}}


NAMES = {"A": "AI bot A (her setup: 1.25x, no stop)", "B": "AI bot B (careful: 0.5x, 5% stop, daily trend)",
         "A-coin": "Coin flip on A's timing", "B-coin": "Coin flip on B's timing", "HOLD": "Just hold SPY",
         "TREND": "SPY above its 20-day average"}


def summary(state, rec, fd=None):
    """The live page on the state-ai branch (GitHub renders it): books, then every AI decision with its reason."""
    def et(t):
        return f"{pd.Timestamp(t, unit='s', tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M}"
    lines = [f"# $100 AI stock bot (paper, K2)\n\nStarted {et(state['started'])} ET with $100 per book. "
             f"**Last update {et(rec['t'])} ET** (updates every trading hour). Paper money only; fills use real "
             "SIP prices with costs. The coin-flip twins trade at the same moments in a random direction: an AI book "
             "has to beat its twin to show skill. Verdict on 2027-04-16 (research/prereg/K2.md).\n",
             "| Book | Equity | Return | Closed trades | Won | Margin calls | Now holding |", "|---|---|---|---|---|---|---|"]
    for n, b in state["books"].items():
        p = b["pos"]
        pos = f"{'long' if p['side'] > 0 else 'short'} {p['sym']} from ${p['entry']:,.2f}" if p else "cash"
        calls = sum(x["why"] == "margin call" for x in b["trips"])
        wins = sum(x.get("pnl", 0) > 0 for x in b["trips"])
        eq = rec["equity"][n]
        lines.append(f"| {NAMES.get(n, n)} | ${eq:.2f} | {eq - 100:+.2f}% | {len(b['trips'])} | {wins} | {calls} | {pos} |")
    if fd and fd["log"]:
        lines += ["\n## Decisions and trades, newest first\n", "| Time (ET) | Bot | Decision | Trade | Why |",
                  "|---|---|---|---|---|"]
        for x in fd["log"]:
            fills = "; ".join(f"{'opened' if f['action'] == 'open' else 'closed'} "
                              f"{'long' if (f['side'] or 0) > 0 else 'short'} {f['sym']} @ ${f['price']:,.2f}"
                              + (f" ({f['why']})" if f.get("why") and f["why"] != "decision" else "")
                              for f in x["fills"]) or "none"
            act = x["action"] + (f" {x['symbol']}" if x["symbol"] and x["action"] != "hold" else "")
            why = (x["reason"] or "no answer, counted as hold").replace("|", "/").replace("\n", " ")
            lines.append(f"| {x['t']} | {x['book']} | {act} | {fills} | {why} |")
    return "\n".join(lines) + "\n"


def feed(state, led, keep=60):
    """Compact summary for the dashboard: each book's equity and open position, the equity path, and the
    AI books' latest decisions with their reasons and fills (newest first). Rebuilt from the ledger."""
    def et(t):
        return f"{pd.Timestamp(t, unit='s', tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M}"
    runs = [json.loads((led.root / e["file"]).read_text()) for e in led.entries() if e["kind"] == "run"]
    books = {}
    last = runs[-1] if runs else {"equity": {}}
    for n, b in state["books"].items():
        p = b["pos"]
        books[n] = {"equity": last["equity"].get(n), "trips": len(b["trips"]),
                    "wins": sum(x.get("pnl", 0) > 0 for x in b["trips"]),
                    "open": f"{'long' if p['side'] > 0 else 'short'} {p['sym']} @ {p['entry']:g}" if p else None}
    log = []
    for r in runs:
        for n in ("A", "B"):
            d = (r.get("decisions") or {}).get(n)
            if d:
                log.append({"t": et(r["t"]), "book": n, "action": d["parsed"]["action"],
                            "symbol": d["parsed"].get("symbol"), "reason": d["parsed"].get("reason"),
                            "fills": [{"action": f["action"], "side": f.get("side"), "sym": f.get("sym"),
                                       "price": f.get("price", f.get("exit")), "why": f.get("why")} for f in r["fills"].get(n, [])]})
    return {"started": et(state["started"]), "updated": et(last["t"]) if runs else None, "runs": len(runs),
            "books": books, "history": [{"t": et(r["t"]), "eq": r["equity"]} for r in runs],
            "log": log[::-1][:keep]}


def _atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run(state_dir, now=None, fetch=fetch_all, ask=ask_gemini):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    data = fetch(now)
    if data is None:
        print(f"{now}: market closed at now - 16 min; nothing to do")
        return None
    t = int(min(pd.Timestamp(d["last_t"]) for d in data.values()).timestamp())
    root = Path(state_dir) / "ai"
    led = Ledger(root)
    key = pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H")
    if led.has("run", key):
        print(f"{key} already logged")
        return None
    bpath = root / "books.json"
    state = json.loads(bpath.read_text()) if bpath.exists() else init_state(t)
    rec = step(state, data, t, ask)
    led.write_json("run", key, f"runs/{key[:10]}/{key}.json", rec)
    _atomic(bpath, json.dumps(state, indent=1, sort_keys=True))
    from tracks.main import ai_mirror
    m = ai_mirror.run(state, rec["prices"], root / "mirror.jsonl")
    if m is not None:
        print(f"Alpaca mirror: {len(m['orders'])} orders" + (f", error {m['error']}" if "error" in m else ""))
    fd = feed(state, led)
    _atomic(root / "feed.json", json.dumps(fd))
    _atomic(root / "README.md", summary(state, rec, fd))
    print(summary(state, rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
