"""COST1 forward shadow copy (research/prereg/COST1.md, Amendment 1). Ledger only: no orders.

For each NX3-ALL-HI day from FIRST_DAY, once it is settled: the quoted half-spread at each pick's entry minute and
one minute before the exit session's close (read as cost1_study.quote_at), every trade's net with those measured
spreads in place of the flat one, and COST1's copy that skips trades whose entry half-spread is above C. Kind
cost1_settle, one file per NX3-ALL-HI day.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import cost1_study as Q
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

FIRST_DAY = "2026-11-25"
C = 0.005                       # chosen on 2019-2023 (research/results/MODELS5/COST1_result.json)
BOOK = 5000.0


def naive_et(t):
    t = pd.Timestamp(t)
    return t.tz_convert(S.ET).tz_localize(None) if t.tzinfo else t


def day_result(tr, hs_in, hs_out):
    """tr: settled picks (symbol, i, w, side, ret, price, days). Per-trade flat and measured nets, the skip flag,
    and the day's sized book P&L with and without the skip."""
    flat = N.cost_frac(tr["price"], tr["side"] < 0, tr["days"])
    net_m = Q.measured_net(tr["side"] * tr["ret"], flat, tr["price"], hs_in, hs_out)
    keep = Q.keep_mask(hs_in, C)
    w = tr["w"].to_numpy(float)
    trades = [{"symbol": s, "i": int(i), "w": float(wi), "side": int(sd), "hs_in": _f(a), "hs_out": _f(b),
               "net_flat": float(r * sd - fc), "net_m": float(m), "kept": bool(k)}
              for s, i, wi, sd, a, b, r, fc, m, k in zip(tr["symbol"], tr["i"], w, tr["side"], hs_in, hs_out,
                                                         tr["ret"], flat, net_m, keep)]
    return {"trades": trades, "kept": int(keep.sum()), "skipped": int((~keep).sum()),
            "book_pnl_flat": float(BOOK * (w * (tr["side"] * tr["ret"] - flat)).sum()),
            "book_pnl_measured": float(BOOK * (w * net_m).sum()),
            "book_pnl_cost1": float(BOOK * (w * np.where(keep, net_m, 0.0)).sum()),
            "gain": float(-(w * np.where(keep, 0.0, net_m)).sum())}


def _f(x):
    return float(x) if np.isfinite(x) else None


def settle(root, api, cal, today, quotes=None):
    """api.settle_trades(ev, cal) and api.h as nx3_live's Api; quotes(symbols, t) -> {symbol: half-spread}."""
    quotes = quotes or (lambda syms, t: Q.quote_at(syms, t, api.h))
    led = Ledger(root)
    ses = S.sessions(cal).set_index("date")
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3allhi_settle" and x["date"] >= FIRST_DAY]:
        if led.has("cost1_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        nx = caldates[k + 1]
        day = json.loads((Path(root) / f"nx3allhi/days/{e['date']}.json").read_text())
        res = {"day": e["date"], "c": C, "trades": []}
        if day.get("picks"):
            ev = pd.DataFrame(day["picks"]).assign(role="picks", d=d, next=nx)
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            t = t[np.isfinite(t["ret"])].reset_index(drop=True) if len(t) else t
            if len(t):
                t["entry_t"] = [naive_et(x) for x in t["entry_t"]]
                hs_in = {}
                for et, g in t.groupby("entry_t"):
                    for s, v in quotes(sorted(set(g["symbol"])), et).items():
                        hs_in[(s, et)] = v
                hs_out = quotes(sorted(set(t["symbol"])), ses.loc[nx, "close_t"] - pd.Timedelta(minutes=1))
                a = np.array([hs_in.get((s, x), np.nan) for s, x in zip(t["symbol"], t["entry_t"])], float)
                b = np.array([hs_out.get(s, np.nan) for s in t["symbol"]], float)
                res |= day_result(t, a, b)
        out.append(led.write_json("cost1_settle", e["date"], f"cost1/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
