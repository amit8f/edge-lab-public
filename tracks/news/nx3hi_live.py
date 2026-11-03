"""NX3-HI shadow book (research/prereg/NX3HI.md), from 2026-11-04. Ledger only: no orders.

After the close, from NX3's own decided events (nx3_live.decide logs every one with its e, fired flag and
easy-to-borrow flag): NX3's 3 long + 3 short slots in time order, but an event only takes a slot if its
confidence |e| / cut is at least HI (1.5). A short only where the stock was easy to borrow at decision time.
Control: as many of the day's decided events at random (default_rng(SEED + yyyymmdd + 200)) with the picks'
sides. $500 per trade like NX3.

Kinds nx3hi_day and nx3hi_settle (NX3's settlement code).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

FIRST_DAY = "2026-11-04"
HI = 1.5
SEED_ADD = 200
SLIM = ["symbol", "t", "side", "e", "prev_c", "dv20"]


def book(decided, cut, seed):
    """The day's NX3-HI picks and the paired control from its decided events."""
    dec = sorted(decided, key=lambda x: (pd.Timestamp(x["t"]), x["symbol"]))
    picks, taken = [], []
    for x in dec:
        e = x.get("e")
        if e is None or not x.get("fired") or abs(e) < HI * cut:
            continue
        side = 1 if e > 0 else -1
        if taken.count(side) >= S.PICKS or (side < 0 and not x.get("etb")):
            continue
        taken.append(side)
        picks.append(dict(x, side=side))
    ctrl = []
    if picks and len(dec) >= len(picks):
        k = np.random.default_rng(seed).choice(len(dec), size=len(picks), replace=False)
        ctrl = [dict(dec[i], side=p["side"]) for i, p in zip(k, picks)]
    return picks, ctrl


def close_day(root, decisions, d, seed, cut):
    """After the close: the day's NX3-HI book (kind nx3hi_day). None before FIRST_DAY or if already logged."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nx3hi_day", day_s):
        return None
    dec = [x for x in decisions if x["status"] == "decided"]
    picks, ctrl = book(dec, cut, seed + SEED_ADD)
    rec = {"day": day_s, "fire_cut": cut, "hi": HI, "decided": len(dec), "fired": sum(bool(x.get("fired")) for x in dec),
           "picks": [{k: p.get(k) for k in SLIM} for p in picks], "control": [{k: c.get(k) for k in SLIM} for c in ctrl]}
    return led.write_json("nx3hi_day", day_s, f"nx3hi/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))


def settle(root, api, cal, today):
    """Settles every nx3hi_day whose exit session has closed (kind nx3hi_settle)."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3hi_day"]:
        if led.has("nx3hi_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, role=role) for role in ("picks", "control") for p in rec[role]]
        res = {"day": e["date"], "trades": [], "pick_minus_control": None}
        if rows:
            ev = pd.DataFrame(rows).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                p, c = t[t["role"] == "picks"], t[t["role"] == "control"]
                if len(p) and len(c):
                    res["pick_minus_control"] = float(p["net"].mean() - c["net"].mean())
                res["book_pnl"] = float((p["net"] * N.NOTIONAL).sum())
                cols = ["symbol", "role", "side", "t", "entry", "exit_ac", "ret", "net"]
                res["trades"] = t[[x for x in cols if x in t]].to_dict("records")
        out.append(led.write_json("nx3hi_settle", e["date"], f"nx3hi/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
