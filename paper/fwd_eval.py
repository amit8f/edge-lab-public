"""Scorer for the forward near-miss books (research/prereg/FWD1.md rule 20, FWD2.md) and COMBO1: reads their
append-only logs, prices every pick and control from SIP all-adjusted daily bars, and returns each book's
daily pick-minus-control series, its closed cohorts and the pass-rule numbers, plus COMBO1's daily mix.
Read-only: it never writes to a ledger. The dashboard calls it; nothing trades on it.

Trade path (each registration's entry and exit): entry at the open of the first session after the signal
date, the entry session counts as session 1, exit at the close of session `hold` (A1F: or the first close
35%+ above entry). Daily P&L is sign x (close - previous close) / entry, so a trade's days sum to its return;
the round-trip cost is charged on the entry day and A1F's borrow (20% a year) on each day held.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main.longterm.stats import nw_tstat

BOOKS = {"R1_picks": "R1", "I1F_picks": "I1F", "A1F_picks": "A1F"}
L1_CELLS = {("offering", "penny"): "L1-offering-penny", ("late_filing", "main"): "L1-late-main"}
COMBO = ["R1", "L1-offering-penny", "L1-late-main", "I1F", "A1F"]   # FWD2's members; NX2 joins once it logs
LAGS = {"R1": 3, "I1F": 3, "A1F": 3, "L1-offering-penny": 3, "L1-late-main": 3, "COMBO1": 5}
VOL_DAYS, VOL_TARGET = 60, 0.01


def _jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def records(state):
    """[(book, record)] from the longterm ledger (R1, I1F, A1F) and the L1 forward ledger, in log order."""
    out = []
    lt = Path(state) / "longterm"
    for e in _jsonl(lt / "ledger.jsonl"):
        if e.get("kind") in BOOKS:
            out.append((BOOKS[e["kind"]], json.loads((lt / e["file"]).read_text())))
    l1 = Path(state) / "l1fwd"
    for e in _jsonl(l1 / "ledger.jsonl"):
        if e.get("kind") == "L1_fwd":
            out.append(("L1", json.loads((l1 / e["file"]).read_text())))
    return out


def symbols(recs):
    s = set()
    for book, r in recs:
        if book == "L1":
            continue
        s |= {p["symbol"] for p in r.get("picks", [])} | set(r.get("control", []))
    return s


def trade_days(bars, cal, as_of, hold, sign, cost, stop=None, borrow=0.0):
    """{session: P&L as a fraction of entry} for one trade, and whether it has closed. Empty when the entry
    session hasn't traded yet or the stock has no bar on it."""
    after = cal[cal > pd.Timestamp(as_of)]
    if bars is None or not len(after) or after[0] not in bars.index:
        return {}, False
    sess = after[:hold]
    entry = float(bars.at[sess[0], "o"])
    prev, out = entry, {}
    for i, d in enumerate(sess):
        if d not in bars.index:                    # no bar (halt, delisting): the price stays where it was
            out[d] = -borrow / 252
            continue
        c = float(bars.at[d, "c"])
        out[d] = sign * (c - prev) / entry - borrow / 252 - (cost if i == 0 else 0.0)
        prev = c
        if stop is not None and c >= entry * (1 + stop):
            return out, True
    return out, len(sess) == hold                  # cal holds finished sessions only; a delisted stock stays at its last close


def bar_book(book, recs, bars, cal):
    """Daily and per-cohort numbers for one bar-priced book (R1, I1F, A1F)."""
    daily, cohorts = {}, []
    for r in recs:
        short = r.get("side") == "short"
        sign = -1 if short else 1
        hold = int(r.get("hold_sessions") or 5)
        cost = float(r.get("cost_round_trip") or 0)
        stop = r.get("stop") if short else None
        borrow = float(r.get("borrow_per_year") or 0)
        legs = {"pick": [p["symbol"] for p in r.get("picks", [])], "control": list(r.get("control", []))}
        paths = {k: [trade_days(bars.get(s), cal, r["as_of"], hold, sign, cost, stop, borrow) for s in v]
                 for k, v in legs.items()}
        for d in sorted({d for k in paths for p, _ in paths[k] for d in p}):
            m = {k: [p[d] for p, _ in paths[k] if d in p] for k in paths}
            if m["pick"] and m["control"]:
                daily.setdefault(d, []).append(np.mean(m["pick"]) - np.mean(m["control"]))
        tot = {k: [sum(p.values()) for p, c in paths[k] if p and c] for k in paths}
        done = all(c for k in paths for p, c in paths[k] if p) and tot["pick"] and tot["control"]
        cohorts.append({"as_of": r["as_of"], "picks": len(legs["pick"]), "closed": bool(done),
                        "pick_net": float(np.mean(tot["pick"])) if done else None,
                        "control_net": float(np.mean(tot["control"])) if done else None})
    return {d: float(np.mean(v)) for d, v in daily.items()}, cohorts


def l1_books(recs):
    """The two L1 cells: each traded event's net minus its control's net, booked on its exit day."""
    out = {v: ({}, []) for v in L1_CELLS.values()}
    for r in recs:
        for e in r.get("events", []):
            name = L1_CELLS.get((e.get("category"), e.get("group")))
            if not name or e.get("status") != "ok" or e.get("net") is None or e.get("control_net") is None:
                continue
            day = pd.Timestamp(str(e["exit_t"])[:10])
            daily, cohorts = out[name]
            daily.setdefault(day, []).append(e["net"] - e["control_net"])
            cohorts.append({"as_of": e["entry_day"], "picks": 1, "closed": True, "pick_net": e["net"],
                            "control_net": e["control_net"]})
    return {k: ({d: float(np.mean(v)) for d, v in daily.items()}, cohorts) for k, (daily, cohorts) in out.items()}


def status(name, daily, cohorts):
    """Rule-20 numbers so far: closed cohorts, trading days, mean and NW t of cohort pick-minus-control,
    the picks' own mean, and (A1F, FWD2 amendment 1) the mean without the best cohort."""
    done = [c for c in cohorts if c["closed"]]
    diff = np.array([c["pick_net"] - c["control_net"] for c in done])
    own = [c["pick_net"] for c in done]
    m, _, t = nw_tstat(diff, LAGS[name]) if len(diff) > 1 else (float(diff.mean()) if len(diff) else None, None, None)
    out = {"book": name, "cohorts": len(cohorts), "closed_cohorts": len(done),
           "closed_trades": int(sum(c["picks"] for c in done)), "days": len(daily),
           "mean_pick_minus_control": m, "t_nw": t, "own_mean": float(np.mean(own)) if own else None,
           "cum": float(sum(daily.values())) if daily else 0.0}
    if name == "A1F" and len(diff) > 1:
        out["mean_without_best"] = float(np.delete(diff, diff.argmax()).mean())
    return out


def combo(series):
    """COMBO1's daily return: the mean over member books open that day of each book's daily return, scaled
    to 1% daily volatility by its trailing 60-day realised volatility (unscaled before 60 days)."""
    days = sorted({d for s in series.values() for d in s})
    out = {}
    for d in days:
        xs = []
        for name, s in series.items():
            if d not in s:
                continue
            past = [s[k] for k in sorted(s) if k < d][-VOL_DAYS:]
            sd = np.std(past, ddof=1) if len(past) >= VOL_DAYS else 0
            xs.append(s[d] * (VOL_TARGET / sd if sd > 0 else 1.0))
        if xs:
            out[d] = float(np.mean(xs))
    return out


def evaluate(state, bars, cal):
    """Everything above for all books. bars: {symbol: DataFrame(o, c) indexed by session}; cal: sessions."""
    recs = records(state)
    cal = pd.DatetimeIndex(sorted(cal))
    series, books = {}, []
    for name in ("R1", "I1F", "A1F"):
        daily, cohorts = bar_book(name, [r for b, r in recs if b == name], bars, cal)
        series[name] = daily
        books.append(status(name, daily, cohorts))
    for name, (daily, cohorts) in l1_books([r for b, r in recs if b == "L1"]).items():
        series[name] = daily
        books.append(status(name, daily, cohorts))
    c = combo({k: v for k, v in series.items() if k in COMBO and v})
    members_open = sum(1 for k in COMBO if series.get(k))
    m, _, t = nw_tstat(list(c.values()), LAGS["COMBO1"]) if len(c) > 1 else (None, None, None)
    books.append({"book": "COMBO1", "days": len(c), "members_logged": members_open,
                  "closed_trades": int(sum(b["closed_trades"] for b in books if b["book"] in COMBO)),
                  "mean_daily": m, "t_nw": t, "cum": float(sum(c.values())) if c else 0.0})
    return books
