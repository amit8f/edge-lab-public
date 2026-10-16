"""K4: the $100 AI stock bot that picks from the whole liquid market and holds up to five stocks
(research/prereg/K4.md).

Once a day: the 300 most traded $5+ US common stocks (median raw dollar volume over 20 sessions) and which
of them Alpaca can short. Every 10 minutes of the session: the 40 of those that moved most since the previous close
(plus anything held), shown to Gemini with their recent hourly closes; it answers with up to five
positions, long or short. Book C follows the answer, C-coin takes the same symbols at the same moments
with random sides, HOLD buys SPY once. Fills are simulated at the latest IEX trade (SIP 16 minutes back when IEX has nothing newer) with main-track costs and logged in
the hash-chained ledger; the Alpaca mirror only copies book C for display.

    python -m tracks.main.ai_wide --state state
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

from core.ledger import Ledger
from tracks.main import ai_trader as k2

N_UNIVERSE = 300
SHORTLIST = 40
MAX_POS = 5
SLOT = 1 / MAX_POS          # each new position: a fifth of equity, so gross exposure stays near 1x
MIN_PRICE = 5.0
DV_DAYS = 20
START_USD, FLOOR_USD = 100.0, 5.0
SEED = 20261019
FEE, BORROW = k2.FEE, k2.BORROW
ET = k2.ET

PROMPT = f"""You manage a $100 paper US stock account. Every 10 minutes of the trading day you see the {SHORTLIST} most traded US stocks
that moved most today (from the {N_UNIVERSE} most traded stocks over $5) plus anything you hold, with their
recent hourly closes, up to three news headlines from the last 24 hours, and your account.
Rules: hold up to {MAX_POS} positions, long or short. Each new position is sized at one fifth of equity; a
position you keep stays as it is. Only symbols marked shortable can be shorted. Costs: 0.1% per side, shorts
pay 3% a year borrow.
Name the full set of positions you want to hold now (an empty list means all cash). Anything
you hold and leave out is sold. Reply with JSON only:
{{"positions": [{{"symbol": "XYZ", "side": "long|short", "reason": "one line"}}], "summary": "one line"}}"""


# ---------- accounting ----------

def new_book():
    return {"cash": START_USD, "pos": {}, "trips": [], "stopped": False}


def equity(book, prices):
    return book["cash"] + sum(p["side"] * p["qty"] * (prices.get(s, p["last"]) - p["entry"])
                              for s, p in book["pos"].items())


def mark(book, prices):
    for s, p in book["pos"].items():
        if s in prices:
            p["last"] = prices[s]


def charge(book, t):
    for p in book["pos"].values():
        if p["side"] < 0 and t > p["charged"]:
            book["cash"] -= p["notional"] * BORROW * (t - p["charged"]) / (3600 * 24 * 365)
        p["charged"] = max(p["charged"], t)


def open_pos(book, sym, side, price, t, eq):
    notional = eq * SLOT
    book["cash"] -= notional * FEE
    book["pos"][sym] = {"side": side, "entry": price, "last": price, "qty": notional / price,
                        "notional": notional, "opened": t, "charged": t}
    return {"action": "open", "sym": sym, "side": side, "price": price, "notional": round(notional, 6), "t": t}


def close_pos(book, sym, price, t, why):
    p = book["pos"].pop(sym)
    pnl, fee = p["side"] * p["qty"] * (price - p["entry"]), p["qty"] * price * FEE
    book["cash"] += pnl - fee
    trip = {"sym": sym, "side": p["side"], "open_t": p["opened"], "close_t": t, "entry": p["entry"],
            "exit": price, "notional": p["notional"], "pnl": round(pnl - fee, 6), "why": why}
    book["trips"].append(trip)
    return dict(trip, action="close")


def rebalance(book, target, prices, t):
    """Close what isn't wanted (or wanted on the other side), then open the new names at a fifth of equity
    each. target: {symbol: side}. Positions kept on the same side are left alone."""
    if book["stopped"]:
        return []
    fills = []
    for s in sorted(book["pos"]):
        if target.get(s) != book["pos"][s]["side"]:
            fills.append(close_pos(book, s, prices[s], t, "decision"))
    if equity(book, prices) < FLOOR_USD:
        book["stopped"] = True
        return fills
    eq = equity(book, prices)
    for s in sorted(target):
        if s not in book["pos"] and len(book["pos"]) < MAX_POS:
            fills.append(open_pos(book, s, target[s], prices[s], t, eq))
    return fills


def coin_side(sym, t):
    return random.Random(f"{SEED}-{sym}-{t}").choice([1, -1])


# ---------- data ----------

def build_universe(now):
    """The day's universe from daily raw bars of the sessions before today, and the shortable set."""
    from core import alpaca
    from core.bars import fetch_daily_bars, to_frames
    from core.env import alpaca_headers
    a = alpaca.active_assets()
    a = a[a["excl_reason"].isna()]
    syms = sorted(a["symbol"])
    today = pd.Timestamp(now).tz_convert(ET).normalize().tz_localize(None)
    raw = to_frames(fetch_daily_bars(syms, f"{(today - pd.Timedelta(days=45)).date()}",
                                     f"{(today - pd.Timedelta(days=1)).date()}", adjustment="raw",
                                     headers=alpaca_headers(), batch_size=200))
    last = max(df.index[-1] for df in raw.values())
    rows = []
    for s, df in raw.items():
        d = df.tail(DV_DAYS)
        if d.index[-1] != last or d["c"].iloc[-1] < MIN_PRICE:
            continue
        c = df["c"]
        rows.append({"symbol": s, "dv": float((d["c"] * d["v"]).median()), "prev": float(c.iloc[-1]),
                     "c5": float(c.iloc[-6]) if len(c) > 5 else None, "c20": float(c.iloc[-21]) if len(c) > 20 else None})
    u = sorted(rows, key=lambda r: (-r["dv"], r["symbol"]))[:N_UNIVERSE]
    shortable = set(a.loc[a.get("shortable", False) & a.get("easy_to_borrow", False), "symbol"]) \
        if "shortable" in a and "easy_to_borrow" in a else set()
    return {"date": f"{today:%Y-%m-%d}", "as_of": f"{last:%Y-%m-%d}", "candidates": len(syms),
            "stocks": u, "shortable": sorted(shortable & {r["symbol"] for r in u})}


def _chunks(xs, n=100):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def fetch_hour(now, uni, held):
    """(price time, {sym: {"last", "last_t", "1h"}}) for the universe, holdings and SPY, or (None, None)
    when now - 16 minutes is outside the regular session."""
    now = pd.Timestamp(now)
    px_time, sessions = k2.market_window(now)
    if px_time is None:
        return None, None, sessions
    end = px_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    syms = sorted({r["symbol"] for r in uni["stocks"]} | set(held) | {"SPY"})
    mins = {}
    for c in _chunks(syms):
        mins.update(k2._bars(c, "1Min", (px_time - pd.Timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ"), end))
    out = {}
    for s in syms:
        m = [r for r in mins.get(s, []) if r["v"]]
        if m:
            out[s] = {"last": float(m[-1]["c"]), "last_t": m[-1]["t"], "src": "sip", "1h": []}
    o, c = sessions[px_time.tz_convert(ET).date()]
    latest = {}
    for ch in _chunks(syms):
        latest.update(iex_latest(ch))
    for s, (p, ts) in latest.items():
        if o <= ts <= min(c, now) and s in out and ts > pd.Timestamp(out[s]["last_t"]):
            out[s].update(last=p, last_t=ts.isoformat(), src="iex")
    return px_time, out, sessions


def iex_latest(syms):
    """{symbol: (price, time)} of the latest IEX trade: Alpaca's free real-time feed (one exchange, ~2-3% of
    volume). K4 uses it for current prices only (amendment 2026-10-16, Amit's OK); history stays SIP."""
    from core import alpaca
    j = alpaca.get("https://data.alpaca.markets/v2/stocks/trades/latest", {"symbols": ",".join(syms), "feed": "iex"})
    return {s: (float(x["p"]), pd.Timestamp(x["t"])) for s, x in (j.get("trades") or {}).items() if x.get("p")}


def add_hours(data, syms, now, px_time, sessions):
    end = px_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in _chunks(sorted(syms)):
        half = k2._bars(c, "30Min", (pd.Timestamp(now) - pd.Timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"), end)
        for s in c:
            if s in data:
                data[s]["1h"] = [h for h in k2.session_hours(half.get(s, []), sessions)
                                 if h[6] <= int(px_time.timestamp())]


def fetch_news(syms, px_time, per=3, pages=5):
    """{symbol: [headline, ...]} newest first, published in the 24 hours before the price time."""
    from core import alpaca
    out, token = {s: [] for s in syms}, None
    for _ in range(pages):
        p = {"symbols": ",".join(syms), "start": (px_time - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": px_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "sort": "desc", "limit": 50, "include_content": "false"}
        if token:
            p["page_token"] = token
        j = alpaca.get(alpaca.NEWS_URL, p)
        for n in j.get("news", []):
            for s in n.get("symbols", []):
                if s in out and len(out[s]) < per and n.get("headline"):
                    out[s].append(" ".join(n["headline"].split())[:200])
        token = j.get("next_page_token")
        if not token or all(len(v) >= per for v in out.values()):
            break
    return out


def shortlist(uni, data, held, n=SHORTLIST):
    """The n universe stocks with the largest absolute move from the previous close, plus holdings."""
    moves = []
    for r in uni["stocks"]:
        d = data.get(r["symbol"])
        if d:
            moves.append((-abs(d["last"] / r["prev"] - 1), r["symbol"]))
    top = [s for _, s in sorted(moves)[:n]]
    return top + sorted(s for s in held if s not in top)


# ---------- model ----------

def context(book, uni, data, syms, now_txt, news=None):
    info = {r["symbol"]: r for r in uni["stocks"]}
    short = set(uni["shortable"])
    prices = {s: d["last"] for s, d in data.items()}
    lines = [f"Prices as of {now_txt} US/Eastern. Columns: symbol, last price, % since previous close, "
             "% over 5 sessions, % over 20 sessions, shortable, last 7 hourly closes (oldest first)"]
    for s in syms:
        d, r = data[s], info.get(s, {})
        pct = lambda base: f"{(d['last'] / base - 1) * 100:+.1f}" if base else "n/a"
        closes = " ".join(f"{h[4]:g}" for h in d["1h"][-7:])
        lines.append(f"{s}, {d['last']:g}, {pct(r.get('prev'))}, {pct(r.get('c5'))}, {pct(r.get('c20'))}, "
                     f"{'yes' if s in short else 'no'}, {closes}")
        for h in (news or {}).get(s, []):
            lines.append(f"    news: {h}")
    pos = "; ".join(f"{'long' if p['side'] > 0 else 'short'} {s} from {p['entry']:g} "
                    f"(${p['side'] * p['qty'] * (prices.get(s, p['last']) - p['entry']):+.2f})"
                    for s, p in sorted(book["pos"].items())) or "none"
    lines.append(f"\n## Account\nequity ${equity(book, prices):.2f}, open positions: {pos}")
    return "\n".join(lines)


def parse(text, allowed, shortable):
    """{symbol: side} wanted, or None when the answer is unreadable (then nothing changes, pre-registered).
    Symbols outside the list, shorts of unshortable names and anything past five are dropped and noted."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict) or not isinstance(d.get("positions"), list):
        return None, [], "unreadable answer"
    want, dropped, reasons = {}, [], []
    for x in d["positions"]:
        if not isinstance(x, dict):
            continue
        s, side = str(x.get("symbol", "")).strip().upper(), str(x.get("side", "")).strip().lower()
        sd = {"long": 1, "short": -1}.get(side)
        if s not in allowed or sd is None or (sd < 0 and s not in shortable) or s in want or len(want) >= MAX_POS:
            dropped.append(f"{side} {s}")
            continue
        want[s] = sd
        reasons.append(f"{s}: {' '.join(str(x.get('reason', '')).split())[:200]}")
    return want, dropped, " ".join(str(d.get("summary", "")).split())[:300]


# ---------- one run ----------

def init_state(t):
    return {"started": t, "books": {"C": new_book(), "C-coin": new_book(), "HOLD": new_book()}}


def step(state, uni, data, syms, t, ask=k2.ask_gemini, news=None):
    prices = {s: d["last"] for s, d in data.items()}
    books = state["books"]
    for b in books.values():
        mark(b, prices)
        charge(b, t)
    rec = {"t": t, "universe_date": uni["date"], "shortlist": syms, "prices": {s: prices[s] for s in syms if s in prices},
           "fills": {n: [] for n in books}}
    now_txt = f"{pd.Timestamp(t, unit='s', tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M}"
    c = books["C"]
    if not c["stopped"]:
        text = context(c, uni, data, syms, now_txt, news)
        dec = {"prompt_sha256": hashlib.sha256((PROMPT + "\n\n" + text).encode()).hexdigest()}
        try:
            raw = ask(PROMPT, text)
            dec["raw"] = raw
        except Exception as e:  # quota, network: logged, nothing changes
            raw, dec["error"] = "", f"{type(e).__name__}: {str(e)[:200]}"
        want, dropped, summ = parse(raw, set(syms), set(uni["shortable"]))
        dec.update({"want": want, "dropped": dropped, "summary": summ})
        rec["decision"] = dec
        if want is not None:
            rec["fills"]["C"] = rebalance(c, want, prices, t)
            tw = books["C-coin"]
            opened = {f["sym"] for f in rec["fills"]["C"] if f["action"] == "open"}
            closed = {f["sym"] for f in rec["fills"]["C"] if f["action"] == "close"}
            tw_target = {s: p["side"] for s, p in tw["pos"].items() if s not in closed and s not in opened}
            tw_target.update({s: coin_side(s, t) for s in opened})
            rec["fills"]["C-coin"] = rebalance(tw, tw_target, prices, t)
    h = books["HOLD"]
    if not h["pos"] and not h["trips"] and "SPY" in prices:
        rec["fills"]["HOLD"].append(open_pos(h, "SPY", 1, prices["SPY"], t, START_USD / SLOT / (1 + FEE)))
    rec["equity"] = {n: round(equity(b, prices), 6) for n, b in books.items()}
    rec["positions"] = {n: b["pos"] for n, b in books.items()}
    return rec


NAMES = {"C": "AI bot C (whole market, up to 5 stocks)", "C-coin": "Coin flip on C's picks", "HOLD": "Just hold SPY"}


def summary(state, rec, runs):
    def et(t):
        return f"{pd.Timestamp(t, unit='s', tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M}"
    lines = [f"# $100 AI bot C: whole market, up to 5 stocks (paper, K4)\n\nStarted {et(state['started'])} ET "
             f"with $100 per book. **Last update {et(rec['t'])} ET.** Paper money only; fills use real SIP prices "
             "with costs. The coin-flip book buys or shorts the same stocks at the same moments in a random "
             "direction: C has to beat it to show skill (research/prereg/K4.md).\n",
             "| Book | Equity | Return | Closed trades | Won | Holding now |", "|---|---|---|---|---|---|"]
    for n, b in state["books"].items():
        pos = ", ".join(f"{'long' if p['side'] > 0 else 'short'} {s}" for s, p in sorted(b["pos"].items())) or "cash"
        eq = rec["equity"][n]
        lines.append(f"| {NAMES[n]} | ${eq:.2f} | {eq - 100:+.2f}% | {len(b['trips'])} | "
                     f"{sum(x['pnl'] > 0 for x in b['trips'])} | {pos} |")
    lines += ["\n## C's decisions, newest first\n", "| Time (ET) | Wants | Trades | Why |", "|---|---|---|---|"]
    for r in runs[::-1][:60]:
        d = r.get("decision") or {}
        want = d.get("want")
        wants = ("no answer, nothing changed" if want is None else
                 ", ".join(f"{'long' if v > 0 else 'short'} {s}" for s, v in sorted(want.items())) or "all cash")
        fills = "; ".join(f"{'bought' if (f['action'] == 'open') == (f['side'] > 0) else 'sold'} {f['sym']} @ "
                          f"${f.get('price', f.get('exit')):,.2f}" for f in r["fills"].get("C", [])) or "none"
        why = (d.get("summary") or d.get("error", "") or "").replace("|", "/")
        lines.append(f"| {et(r['t'])} | {wants} | {fills} | {why} |")
    return "\n".join(lines) + "\n"


def run(state_dir, now=None, universe=build_universe, fetch=fetch_hour, hours=add_hours, ask=k2.ask_gemini,
        headlines=fetch_news):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    root = Path(state_dir) / "wide"
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(root)
    bpath = root / "books.json"
    state = json.loads(bpath.read_text()) if bpath.exists() else None
    held = sorted(state["books"]["C"]["pos"]) + sorted(state["books"]["C-coin"]["pos"]) if state else []
    day = f"{now.tz_convert(ET):%Y-%m-%d}"
    upath = root / "universe" / f"{day}.json"
    if fetch is fetch_hour and k2.market_window(now)[0] is None:   # don't build a universe on closed hours
        print(f"{now}: market closed at now - 16 min; nothing to do")
        return None
    if upath.exists():
        uni = json.loads(upath.read_text())
    else:
        uni = universe(now)
        led.write_json("universe", day, f"universe/{day}.json", uni)
    px_time, data, sessions = fetch(now, uni, held)
    if px_time is None:
        print(f"{now}: market closed at now - 16 min; nothing to do")
        return None
    key = f"{now:%Y-%m-%dT%H}:{now.minute // 10 * 10:02d}"   # one decision per 10-minute slot
    if led.has("run", key):
        print(f"{key} already logged")
        return None
    syms = [s for s in shortlist(uni, data, held) if s in data]
    hours(data, set(syms) | {"SPY"}, now, px_time, sessions)
    t = int(now.timestamp()) if any(d.get("src") == "iex" for d in data.values()) else int(px_time.timestamp())
    state = state or init_state(t)
    missing = [s for b in state["books"].values() for s in b["pos"] if s not in data]
    if missing:   # a held stock with no trade in the last hour keeps its last price
        print(f"no recent price for {missing}; marked at their last price")
        for s in missing:
            p = next(b["pos"][s] for b in state["books"].values() if s in b["pos"])
            data[s] = {"last": p["last"], "last_t": None, "1h": []}
    try:
        news = headlines(syms, pd.Timestamp(t, unit="s", tz="UTC"))
    except Exception as e:  # news is optional input; logged, the run goes on
        print(f"news unavailable: {type(e).__name__}")
        news = {}
    rec = step(state, uni, data, syms, t, ask, news)
    rec["price_source"] = {s: data[s].get("src", "sip") for s in syms if s in data}
    rec["headline_counts"] = {s: len(v) for s, v in news.items() if v}
    led.write_json("run", key, f"runs/{key[:10]}/{key.replace(':', '')}.json", rec)
    k2._atomic(bpath, json.dumps(state, indent=1, sort_keys=True))
    from tracks.main import ai_mirror
    m = ai_mirror.run(state, rec["prices"] | {s: d["last"] for s, d in data.items()}, root / "mirror.jsonl")
    for b, o in (m or {}).items():
        print(f"Alpaca mirror {b}: {len(o['orders'])} orders" + (f", error {o['error']}" if "error" in o else ""))
    runs = [json.loads((root / e["file"]).read_text()) for e in led.entries() if e["kind"] == "run"]
    k2._atomic(root / "README.md", summary(state, rec, runs))
    print(summary(state, rec, runs[-1:]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
