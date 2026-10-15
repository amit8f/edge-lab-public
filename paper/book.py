"""Paper book: turns each track's nightly signals into simulated trades under config/risk.yaml.

Rules (docs/PLAN_V2.md, "Risk rules"):
- A signal formed after the close of day t enters at the open of the next trading day.
- Size: notional = account equity x risk_per_trade / stop distance, where the stop distance is
  atr_mult x ATR(20) / close, capped at the track's long or short stop cap; notional is also capped at
  account equity / max_positions.
- The stop sits one stop distance away from the entry fill. Stops are checked on the close and filled
  at the next open (gaps go through intraday stops anyway).
- Time exit at the close of the hold_days-th trading day, counting the entry day as day 1.
- A signal may carry a `limit` price: the entry is then a day limit order (see limit_fill), filled or
  dropped on the entry day, and pays no entry cost when it fills (the X1 comparison book).
- No take-profit. Costs: half the round trip on entry, half on exit. Shorts pay borrow every held day and
  need the stock to be shortable and easy to borrow on the signal date.
- New entries stop when a track's P&L falls pause_drawdown x its sleeve (start x the track's share in
  account.sleeves, or start / 3 without it) below its peak, or when the account falls pause_drawdown
  below its peak.
- Account-wide caps across all tracks (PROTOCOL rule 15's one book), when set in account: total
  long + short notional <= max_gross x equity; notional in one symbol <= max_name x equity; the sum of
  notional x stop distance (money lost if every stop is hit) <= max_open_risk x equity.

Fills use raw prices (adjustment=raw), as a broker would. When a held stock splits, the nightly job calls
apply_split so entry, stop and last prices stay on the new share basis. Cash dividends are not modelled
(small over 1-20 day holds; noted in every report).

State is a plain dict (JSON-serialisable) so the nightly job can store it in the hash-chained ledger.
"""
import copy
from pathlib import Path

import yaml

TRACKS = ("main", "penny", "news")
CONFIG = Path(__file__).resolve().parents[1] / "config" / "risk.yaml"


def load_config(path=CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def new_state(cfg):
    start = float(cfg["account"]["paper_start_usd"])
    return {
        "cash_pnl": 0.0,                      # realised P&L, all tracks
        "peak_equity": start,
        "track_pnl": {t: 0.0 for t in TRACKS},          # realised + open P&L per track
        "track_peak": {t: 0.0 for t in TRACKS},
        "open": [],                           # filled positions
        "pending_entries": [],                # accepted signals waiting for the next open
        "closed": [],                         # finished trades
        "rejected": [],                       # signals refused, with the reason
        "last_day": None,
    }


def _bar(bars, symbol, day):
    return bars.get((symbol, day))


def _signed_return(pos, price):
    r = price / pos["entry_price"] - 1.0
    return r if pos["side"] == "long" else -r


def _open_pnl(pos):
    return pos["notional"] * _signed_return(pos, pos["last_price"]) - pos["costs"]


def equity(state, cfg):
    start = float(cfg["account"]["paper_start_usd"])
    return start + state["cash_pnl"] + sum(_open_pnl(p) for p in state["open"])


def _track_pnl_now(state, track):
    realised = sum(t["pnl"] for t in state["closed"] if t["track"] == track)
    return realised + sum(_open_pnl(p) for p in state["open"] if p["track"] == track)


def _close(state, pos, day, price, reason, half_cost):
    gross = pos["notional"] * _signed_return(pos, price)
    costs = pos["costs"] + pos["notional"] * half_cost
    trade = dict(pos, exit_date=day, exit_price=price, exit_reason=reason, gross=gross, costs=costs,
                 pnl=gross - costs)
    trade.pop("last_price", None)
    trade.pop("pending_exit", None)
    state["closed"].append(trade)
    state["cash_pnl"] += gross - costs


def limit_fill(side, limit, b):
    """Fill price of a day limit order placed before the open, from the day's raw bar: the open when it
    is already through the limit, the limit when the day's range trades strictly through it, else None.
    Buys (long entries) fill at or below the limit, sells (short entries) at or above it."""
    if side == "long":
        return b["o"] if b["o"] <= limit else (limit if b.get("l", b["o"]) < limit else None)
    return b["o"] if b["o"] >= limit else (limit if b.get("h", b["o"]) > limit else None)


def process_day(state, cfg, day, bars, signals, calendar):
    """Advance the book through trading day `day` (after its close).

    bars: {(symbol, day): {"o": open, "c": close}} with raw prices, covering `day` for every open
    or pending symbol that traded. signals: list of dicts formed after the close of `day` with keys
    track, idea, symbol, side ("long"/"short"), hold_days, atr20, close, and optionally industry,
    shortable, easy_to_borrow. calendar: sorted list of trading days (same type as `day`).
    Returns a new state; the input is not modified.
    """
    s = copy.deepcopy(state)
    tc = cfg["tracks"]
    idx = calendar.index(day)

    # 1. entries and stop exits scheduled for this open
    still_pending = []
    for e in s["pending_entries"]:
        if e["entry_date"] != day:
            still_pending.append(e)
            continue
        b = _bar(bars, e["symbol"], day)
        if b is None:
            s["rejected"].append(dict(e, reason="no bar on entry day"))
            continue
        half = tc[e["track"]]["round_trip_cost"] / 2
        px = b["o"]
        if e.get("limit") is not None:
            px = limit_fill(e["side"], e["limit"], b)
            if px is None:
                s["rejected"].append(dict(e, reason="limit not filled"))
                continue
            half = 0.0                        # a filled limit order crosses no spread on the way in
        d = e["stop_dist"]
        stop = px * (1 - d) if e["side"] == "long" else px * (1 + d)
        s["open"].append(dict(e, entry_price=px, last_price=px, stop_price=stop,
                              costs=e["notional"] * half, days_held=0, pending_exit=None))
    s["pending_entries"] = still_pending

    keep = []
    for p in s["open"]:
        if p.get("pending_exit") and p["pending_exit"]["date"] == day:
            b = _bar(bars, p["symbol"], day)
            if b is not None:
                _close(s, p, day, b["o"], p["pending_exit"]["reason"], tc[p["track"]]["round_trip_cost"] / 2)
                continue
        keep.append(p)
    s["open"] = keep

    # 2. mark to the close, borrow, stops and time exits
    keep = []
    next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
    for p in s["open"]:
        b = _bar(bars, p["symbol"], day)
        if b is None:
            p["stale_days"] = p.get("stale_days", 0) + 1
            keep.append(p)
            continue
        p["stale_days"] = 0
        p["last_price"] = b["c"]
        p["days_held"] += 1
        if p["side"] == "short":
            p["costs"] += p["notional"] * tc[p["track"]]["borrow_per_year"] / 252
        half = tc[p["track"]]["round_trip_cost"] / 2
        if p["days_held"] >= p["hold_days"]:
            _close(s, p, day, b["c"], "time", half)
            continue
        hit = (b["c"] <= p["stop_price"]) if p["side"] == "long" else (b["c"] >= p["stop_price"])
        if hit and not p.get("pending_exit") and next_day is not None:
            p["pending_exit"] = {"date": next_day, "reason": "stop"}
        keep.append(p)
    s["open"] = keep

    # 3. drawdown brakes
    eq = equity(s, cfg)
    s["peak_equity"] = max(s["peak_equity"], eq)
    account_paused = eq < s["peak_equity"] * (1 - cfg["account"]["pause_drawdown"])
    start = float(cfg["account"]["paper_start_usd"])
    shares = cfg["account"].get("sleeves") or {t: 1 / len(TRACKS) for t in TRACKS}
    paused = {}
    for t in TRACKS:
        pnl = _track_pnl_now(s, t)
        s["track_pnl"][t] = pnl
        s["track_peak"][t] = max(s["track_peak"][t], pnl)
        paused[t] = account_paused or (s["track_peak"][t] - pnl) > tc[t]["pause_drawdown"] * start * shares[t]
    acct = cfg["account"]
    book = s["open"] + s["pending_entries"]

    # 4. new signals -> pending entries for the next open
    for sig in signals:
        t = sig["track"]
        rc = tc[t]
        reason = None
        live = [x for x in s["open"] + s["pending_entries"] if x["track"] == t]
        if next_day is None:
            reason = "no next trading day in calendar"
        elif paused[t]:
            reason = "track or account paused (drawdown)"
        elif sig["side"] == "short" and not (sig.get("shortable") and sig.get("easy_to_borrow")):
            reason = "not shortable / not easy to borrow"
        elif any(x["symbol"] == sig["symbol"] for x in live):
            reason = "already held or pending in this track"
        elif len(live) >= rc["max_positions"]:
            reason = "max positions"
        elif sig.get("industry") and sum(x.get("industry") == sig["industry"] for x in live) >= rc["max_per_industry"]:
            reason = "max per industry"
        if reason:
            s["rejected"].append(dict(sig, signal_date=day, reason=reason))
            continue
        cap = rc["long_stop_cap"] if sig["side"] == "long" else rc["short_stop_cap"]
        dist = min(rc["atr_mult"] * sig["atr20"] / sig["close"], cap)
        notional = min(eq * rc["risk_per_trade"] / dist, eq / rc["max_positions"])
        if "max_gross" in acct and sum(x["notional"] for x in book) + notional > acct["max_gross"] * eq:
            reason = "account gross exposure cap"
        elif "max_name" in acct and sum(x["notional"] for x in book if x["symbol"] == sig["symbol"]) + notional \
                > acct["max_name"] * eq:
            reason = "account per-symbol cap"
        elif "max_open_risk" in acct and sum(x["notional"] * x["stop_dist"] for x in book) + notional * dist \
                > acct["max_open_risk"] * eq:
            reason = "account open-risk cap"
        if reason:
            s["rejected"].append(dict(sig, signal_date=day, reason=reason))
            continue
        entry = dict(sig, signal_date=day, entry_date=next_day, notional=notional, stop_dist=dist)
        s["pending_entries"].append(entry)
        book.append(entry)
    s["last_day"] = day
    return s


def apply_split(state, symbol, ratio):
    """Rescale prices of open positions in `symbol` after a split of `ratio` new shares per old share
    (2.0 for a 2-for-1, 0.1 for a 1-for-10 reverse split). Notional is unchanged. Returns a new state."""
    s = copy.deepcopy(state)
    for p in s["open"]:
        if p["symbol"] == symbol:
            for k in ("entry_price", "stop_price", "last_price"):
                p[k] = p[k] / ratio
    return s


def summary(state, cfg):
    """Per-track counts and P&L for the weekly report."""
    out = {"equity": round(equity(state, cfg), 2), "peak_equity": round(state["peak_equity"], 2), "tracks": {}}
    for t in TRACKS:
        closed = [x for x in state["closed"] if x["track"] == t]
        wins = sum(x["pnl"] > 0 for x in closed)
        out["tracks"][t] = {
            "open": sum(p["track"] == t for p in state["open"]),
            "closed": len(closed),
            "win_rate": round(wins / len(closed), 3) if closed else None,
            "pnl": round(state["track_pnl"][t], 2),
        }
    return out
