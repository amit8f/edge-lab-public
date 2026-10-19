"""K6: the combined design (research/prereg/K6.md): tape, news attention and a co-mention peer check, top 100,
10 positions, $10,000 paper. Runs beside K5 (ai_spy) so the two can be compared. No language model.
Every 10 minutes, for the top 100 stocks at $5 or more:
  ret60 = price now / price 60 minutes ago (1-minute SIP bars, IEX real-time for now); rv = last-30-minute volume /
  this session's volume pace x 30; news2 = articles naming the stock in the last 2 hours; peers = stocks named in the
  same articles in the last 24 hours, and peer_ret = their mean ret60.
  Enter when rv >= 1.5, |ret60| >= 1%, news2 >= 1, and peer_ret has the same sign as ret60. Longs and shorts: the 5
  best each by |ret60| x rv, at most 2 new entries a run. Hold while rv >= 1.0 and the sign holds; exit after 60 min
  or at 15:57. Books: B (the rule), B-coin (the same counts per run, random stocks and sides), state under combo/.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import alpaca
from core.ledger import Ledger
from tracks.main import ai_trader as k2
from tracks.main import ai_wide as w

START_USD = 10_000.0
STOP_FRAC = 0.5
N_POS, PER_SIDE, NEW_PER_RUN = 10, 5, 2
MIN_RV, HOLD_RV, MIN_RET = 1.5, 1.0, 0.01
MAX_HOLD_S = 3600
SEED = 20261018
ET = k2.ET
FEE = k2.FEE
NAMES = {"B": "AI bot B (K6: combined tape, news and peers)", "B-coin": "Coin twin of B (same counts, random stocks)"}


def new_book():
    return {"cash": START_USD, "pos": {}, "trips": [], "stopped": False}


def open_at(book, sym, side, price, t, notional):
    book["cash"] -= notional * FEE
    book["pos"][sym] = {"side": side, "entry": price, "last": price, "qty": notional / price,
                        "notional": notional, "opened": t, "charged": t}
    return {"action": "open", "sym": sym, "side": side, "price": price, "t": t}


def close_at(book, sym, price, t, why):
    p = book["pos"].pop(sym)
    pnl, fee = p["side"] * p["qty"] * (price - p["entry"]), p["qty"] * price * FEE
    book["cash"] += pnl - fee
    trip = {"sym": sym, "side": p["side"], "open_t": p["opened"], "close_t": t, "entry": p["entry"],
            "exit": price, "notional": p["notional"], "pnl": round(pnl - fee, 6), "why": why}
    book["trips"].append(trip)
    return {"action": "close", **trip}


def features(minute_bars, now_price, t):
    """ret60 and rv from one stock's 1-minute bars [{t, c, v}] (epoch seconds), or None."""
    if not minute_bars or now_price is None:
        return None
    before = [b for b in minute_bars if b["t"] <= t - 3600]
    if not before or before[-1]["c"] <= 0:
        return None
    session_v = sum(b["v"] for b in minute_bars)
    minutes = max(1, (max(b["t"] for b in minute_bars) - min(b["t"] for b in minute_bars)) // 60 + 1)
    pace = session_v / minutes * 30
    if pace <= 0:
        return None
    last30 = sum(b["v"] for b in minute_bars if b["t"] > t - 1800)
    return {"ret60": now_price / before[-1]["c"] - 1, "rv": last30 / pace}


def peer_and_news(articles, t, universe):
    """news2 {sym: count in last 2 hours}, peers {sym: set of symbols named with it in the last 24 hours}."""
    news2 = {s: 0 for s in universe}
    peers = {s: set() for s in universe}
    for a in articles:
        syms = [s for s in a["symbols"] if s in peers]
        if not syms:
            continue
        for s in syms:
            if a["t"] >= t - 7200:
                news2[s] += 1
            peers[s].update(x for x in syms if x != s)
    return news2, peers


def eligible(f, news2, peer_ret, shortable):
    """(side, score) for the combined entry test, else None."""
    if f is None or f["rv"] < MIN_RV or abs(f["ret60"]) < MIN_RET or news2 < 1 or peer_ret is None:
        return None
    side = 1 if f["ret60"] > 0 else -1
    if peer_ret * side <= 0:
        return None
    if side < 0 and not shortable:
        return None
    return side, abs(f["ret60"] * f["rv"])


def peer_ret_of(sym, peers, feats):
    """Average 60-minute return of the peers named with sym that have features today (None if none)."""
    names = [p for p in peers.get(sym, ()) if feats.get(p)]
    return round(sum(feats[p]["ret60"] for p in names) / len(names), 5) if names else None


def funnel(feats, news2, peers, universe):
    """How many stocks pass each entry test this run (logged only; the rule is unchanged)."""
    out = {"stocks": len(universe), "features": 0, "volume": 0, "move": 0, "news": 0, "peers": 0, "all": 0}
    for sym in universe:
        f = feats.get(sym)
        if f is None:
            continue
        out["features"] += 1
        v, m, n = f["rv"] >= MIN_RV, abs(f["ret60"]) >= MIN_RET, news2.get(sym, 0) >= 1
        names = [p for p in peers.get(sym, ()) if feats.get(p)]
        pr = sum(feats[p]["ret60"] for p in names) / len(names) if names else None
        agree = pr is not None and pr * f["ret60"] > 0
        out["volume"] += v
        out["move"] += m
        out["news"] += n
        out["peers"] += agree
        out["all"] += v and m and n and agree
    return out


def step(state, day, feats, news2, peers, shortable, universe, prices, t, now_et, rng):
    books = state["books"]
    b, c = books["B"], books["B-coin"]
    rec = {"t": t, "fills": []}
    day_open = int(pd.Timestamp(f"{day} 09:30", tz=ET).timestamp())
    last_exit = (now_et.hour, now_et.minute) >= (15, 57)
    for sym in sorted(b["pos"]):
        p = b["pos"][sym]
        f = feats.get(sym)
        keep = (not last_exit and p["opened"] >= day_open and not b["stopped"] and f is not None
                and f["rv"] >= HOLD_RV and (f["ret60"] > 0) == (p["side"] > 0) and t - p["opened"] < MAX_HOLD_S)
        if not keep and sym in prices:
            rec["fills"].append({"book": "B", **close_at(b, sym, prices[sym], t, "exit")})
    if not b["stopped"] and w.equity(b, prices) < STOP_FRAC * START_USD:
        b["stopped"] = True
        for sym in sorted(b["pos"]):
            rec["fills"].append({"book": "B", **close_at(b, sym, prices[sym], t, "stop")})
    if not b["stopped"] and not last_exit:
        longs = sum(1 for p in b["pos"].values() if p["side"] > 0)
        shorts = len(b["pos"]) - longs
        cands = []
        for sym in universe:
            if sym in b["pos"] or sym not in prices:
                continue
            peer_ret = None
            names = [p for p in peers.get(sym, ()) if feats.get(p)]
            if names:
                peer_ret = sum(feats[p]["ret60"] for p in names) / len(names)
            q = eligible(feats.get(sym), news2.get(sym, 0), peer_ret, sym in shortable)
            if q:
                cands.append((q[1], sym, q[0]))
        cands.sort(key=lambda x: (-x[0], x[1]))
        opened = 0
        for _, sym, side in cands:
            if opened >= NEW_PER_RUN or len(b["pos"]) >= N_POS:
                break
            if (side > 0 and longs >= PER_SIDE) or (side < 0 and shorts >= PER_SIDE):
                continue
            rec["fills"].append({"book": "B", **open_at(b, sym, side, prices[sym], t, w.equity(b, prices) / N_POS)})
            longs += side > 0
            shorts += side < 0
            opened += 1
    # the coin twin: same long and short counts as B, random universe stocks and sides, same exits
    for sym in sorted(c["pos"]):
        if last_exit or c["pos"][sym]["opened"] < day_open or c["stopped"]:
            rec["fills"].append({"book": "B-coin", **close_at(c, sym, prices[sym], t, "exit")})
    want_long = sum(1 for p in b["pos"].values() if p["side"] > 0)
    want_short = len(b["pos"]) - want_long
    have_long = sum(1 for p in c["pos"].values() if p["side"] > 0)
    have_short = len(c["pos"]) - have_long
    pool = [s for s in universe if s in prices and s not in c["pos"]]
    rng.shuffle(pool)
    for side, need in ((1, want_long - have_long), (-1, want_short - have_short)):
        for _ in range(0 if last_exit else max(0, need)):
            if not pool:
                break
            sym = pool.pop()
            rec["fills"].append({"book": "B-coin", **open_at(c, sym, side, prices[sym], t, w.equity(c, prices) / N_POS)})
    for bk in (b, c):
        for sym, p in bk["pos"].items():
            if sym in prices:
                p["last"] = prices[sym]
            w.charge(bk, t)
    rec["equity"] = {n: round(w.equity(bk, prices), 6) for n, bk in books.items()}
    return rec


def articles(t, syms):
    """Alpaca articles from the last 24 hours that name at least one of syms: [{t, symbols}]."""
    out, token = [], None
    start = pd.Timestamp(t, unit="s", tz="UTC") - pd.Timedelta(hours=24)
    for _ in range(5):
        p = {"symbols": ",".join(syms), "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "sort": "desc", "limit": 50, "include_content": "false"}
        if token:
            p["page_token"] = token
        j = alpaca.get(alpaca.NEWS_URL, p)
        for n in j.get("news", []):
            out.append({"t": int(pd.Timestamp(n["created_at"]).timestamp()), "symbols": n.get("symbols", [])})
        token = j.get("next_page_token")
        if not token:
            break
    return out


def run(state_dir, now=None, universe_fn=w.build_universe, minute_fn=None, latest_fn=None, news_fn=articles):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    px_time, sessions = k2.market_window(now)
    if px_time is None:
        print(f"{now}: market closed at now - 16 min; nothing to do")
        return None
    now_et = now.tz_convert(ET)
    day = f"{now_et:%Y-%m-%d}"
    root = Path(state_dir) / "combo"
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(root)
    key = f"{now_et:%Y-%m-%dT%H}:{now_et.minute // 10 * 10:02d}"
    if led.has("run", key):
        print(f"{key} already logged")
        return None
    upath = root / "universe" / f"{day}.json"
    if upath.exists():
        uni = json.loads(upath.read_text())
    else:
        uni = universe_fn(now)
        led.write_json("universe", day, f"universe/{day}.json", uni)
    syms = [r["symbol"] for r in uni["stocks"]]
    t = int(px_time.timestamp())
    mins = minute_fn(day, syms + ["SPY"], px_time)
    latest = latest_fn(syms + ["SPY"])
    prices, feats = {}, {}
    for s in syms + ["SPY"]:
        mb = mins.get(s, [])
        px = latest[s][0] if s in latest else (mb[-1]["c"] if mb else None)
        if px is None:
            continue
        prices[s] = px
        f = features(mb, px, t)
        if f:
            feats[s] = f
    arts = news_fn(t, syms)
    news2, peers = peer_and_news(arts, t, syms)
    bpath = root / "books.json"
    state = json.loads(bpath.read_text()) if bpath.exists() else {"books": {n: new_book() for n in NAMES}, "days": {}}
    rng = random.Random(f"{SEED}-{key}")
    rec = step(state, day, feats, news2, peers, set(uni.get("shortable", [])), syms, prices, t, now_et, rng)
    rec["funnel"] = funnel(feats, news2, peers, syms)
    rec["features"] = {s: {"ret60": round(f["ret60"], 5), "rv": round(f["rv"], 3), "news": news2.get(s, 0),
                           "peers": peer_ret_of(s, peers, feats)} for s, f in sorted(feats.items())}
    led.write_json("run", key, f"runs/{day}/{key.replace(':', '')}.json", rec)
    k2._atomic(bpath, json.dumps(state, indent=1, sort_keys=True))
    from tracks.main import ai_mirror
    ai_mirror.run(state, prices, root / "mirror.jsonl")
    print(f"{key}: equity {rec['equity']}, passing each test {rec['funnel']}")
    return rec


def live_minutes(day, syms, px_time):
    open_t = pd.Timestamp(f"{day} 09:30", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    end = px_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for c in w._chunks(sorted(syms)):
        for s, rows in k2._bars(c, "1Min", open_t, end).items():
            out[s] = [{"t": int(pd.Timestamp(r["t"]).timestamp()), "c": float(r["c"]), "v": float(r["v"])}
                      for r in rows if r["v"]]
    return out


def live_latest(syms):
    out = {}
    for c in w._chunks(sorted(syms)):
        out.update(w.iex_latest(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state, minute_fn=live_minutes, latest_fn=live_latest)


if __name__ == "__main__":
    main()
