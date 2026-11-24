"""NXM-VOTE shadow book (research/prereg/NXMVOTE.md), from FIRST_DAY. Ledger only: no orders.

After the close, from NX3's decided events and the inputs its scorer logged at decision time: each voter's
(LGBM, MLP, ENS, RFR; the version 1 models nxm_live loads) effect over its own fire cut; an event is traded only
when all K voters fire the same way (k chosen on 2019-2023). Uncapped and sized like the -ALL books by the voters'
mean confidence, a short only where easy to borrow; paired same-day control. Kinds nxvote_day and nxvote_settle.
Forward test after 100 trades: sized book minus control, mean > 0 with NW t >= 2, own net > 0.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nxm_live as M
from tracks.news import nxvote_study as V

FIRST_DAY = "2026-11-25"
K = 4                       # research/results/NXMVOTE/NXMVOTE_result.json
BOOK = 5000.0
SEED_ADD = 300


class Vote:
    """A model-like wrapper: effect = side x mean confidence of the agreeing voters (0 without a vote), cut 1."""
    cut = 1.0

    def __init__(self, models, k=K):
        self.voters, self.k = [models[a] for a in V.VOTERS], k

    def effect(self, X):
        E = np.column_stack([m.effect(X) / m.cut for m in self.voters])
        side, conf = V.vote(E, self.k)
        return side * conf


def close_day(root, decisions, d, seed, models=None):
    """After the close: the day's NXM-VOTE book (kind nxvote_day). None before FIRST_DAY or if already logged."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nxvote_day", day_s):
        return None
    models = models or M.load()
    dec = [x for x in decisions if x["status"] == "decided"]
    p, c = M.book_all(dec, Vote(models), seed + SEED_ADD)
    keep = M.SLIM + ["i", "w", "conf"]
    sh = M.shas()
    rec = {"day": day_s, "k": K, "voters": list(V.VOTERS), "models_sha256": {a: sh.get(a) for a in V.VOTERS},
           "decided": len(dec), "picks": [{k: x.get(k) for k in keep} for x in p],
           "control": [{k: x.get(k) for k in keep} for x in c]}
    return led.write_json("nxvote_day", day_s, f"nxvote/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))


def settle(root, api, cal, today):
    """Settles every nxvote_day whose exit session has closed (kind nxvote_settle)."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nxvote_day"]:
        if led.has("nxvote_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, role=role) for role in ("picks", "control") for p in rec[role]]
        res = {"day": e["date"], "trades": [], "sized_pick_minus_control": None}
        if rows:
            ev = pd.DataFrame(rows).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                p = t[t["role"] == "picks"].set_index("i")
                c = t[t["role"] == "control"].set_index("i")
                both = p.index.intersection(c.index)
                if len(both):
                    res["sized_pick_minus_control"] = float((p.loc[both, "w"] * (p.loc[both, "net"] - c.loc[both, "net"])).sum())
                res["book_pnl"] = float((p["w"] * BOOK * p["net"]).sum())
                cols = ["symbol", "role", "i", "w", "side", "t", "entry", "exit_ac", "ret", "net"]
                res["trades"] = t[[x for x in cols if x in t]].to_dict("records")
        out.append(led.write_json("nxvote_settle", e["date"], f"nxvote/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
