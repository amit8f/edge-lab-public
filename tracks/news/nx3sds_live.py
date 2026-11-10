"""NX3-ALL-SDS shadow book (research/prereg/NX3SDS.md), from 2026-11-10. Ledger only: no orders.

The same picks, sizes and control as NX3-ALL (its logged nx3all_day record), with one change: a short is bought
back at 15:55 on the day it was opened, at the close of the 1-minute bar that ends at 15:55 (raw SIP, the
basis of NX3's entry price). Longs still exit at the next session's close (NX3's own settlement). The control
trades follow the same rule by side.

Kind nx3sds_settle, written once the next session has closed (the longs' exit).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

FIRST_DAY = "2026-11-10"
BOOK = 5000.0
SHORT_EXIT = pd.Timedelta(minutes=5)       # buy back 5 minutes before the close (15:55 on a normal day)


def exit_price(bars, close_t):
    """Close of the last 1-minute bar that ends by close_t - SHORT_EXIT (the bar starting 15:54), or NaN."""
    if bars is None or bars.empty:
        return np.nan
    b = bars[bars["t"] + pd.Timedelta(minutes=1) <= close_t - SHORT_EXIT]
    return float(b["c"].iloc[-1]) if len(b) else np.nan


def same_day_shorts(t, prices):
    """t from NX3's settlement: shorts take their same-day exit (ret, days 1); a short without a price is dropped."""
    t = t.copy()
    sh = (t["side"] < 0).to_numpy()
    px = t["symbol"].map(prices).astype(float).to_numpy()
    t.loc[sh, "exit_ac"] = px[sh]
    t.loc[sh, "ret"] = px[sh] / t.loc[sh, "entry"].to_numpy(float) - 1
    t.loc[sh, "days"] = 1
    t["exit"] = np.where(sh, "same_day", "next_close")
    return t[np.isfinite(t["ret"].to_numpy(float))]


def settle(root, api, cal, today):
    """Settles every nx3all_day from FIRST_DAY whose next session has closed (kind nx3sds_settle)."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    ses = S.sessions(cal).set_index("date")
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3all_day"]:
        if e["date"] < FIRST_DAY or led.has("nx3sds_settle", e["date"]):
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
                shorts = sorted(set(t.loc[t["side"] < 0, "symbol"]))
                close_t = ses.loc[d, "close_t"]
                bars = api.day_bars(shorts, d, ses.loc[d, "open_t"], close_t) if shorts else {}
                t = same_day_shorts(t, {s: exit_price(bars.get(s), close_t) for s in shorts})
                t["net"] = N.net(t)
                p = t[t["role"] == "picks"].set_index("i")
                c = t[t["role"] == "control"].set_index("i")
                both = p.index.intersection(c.index)
                if len(both):
                    res["sized_pick_minus_control"] = float((p.loc[both, "w"] * (p.loc[both, "net"] - c.loc[both, "net"])).sum())
                res["book_pnl"] = float((p["w"] * BOOK * p["net"]).sum())
                cols = ["symbol", "role", "i", "w", "side", "t", "entry", "exit_ac", "exit", "ret", "net"]
                res["trades"] = t[[x for x in cols if x in t]].to_dict("records")
        out.append(led.write_json("nx3sds_settle", e["date"], f"nx3sds/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
