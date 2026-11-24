"""SURV1-LONG forward ledger (research/prereg/SURV1LONG.md). Ledger only: no orders.

For each NX3-ALL-HI day from FIRST_DAY, once five closes after its entry day exist: the longs follow SURV1's frozen
hazard model (models/surv1/hazard.txt, p* 0.3) close by close up to close 5, the shorts keep NX3-ALL-HI's next-close
result. Kind surv1long_settle, one file per NX3-ALL-HI day.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import models5_study as M
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

FIRST_DAY = "2026-11-25"
MODEL_DIR = Path("models/surv1")
BOOK = 5000.0
LOOKBACK = pd.Timedelta(days=90)


class Hazard:
    """The exported LightGBM booster behind the sklearn-style predict_proba that models5_study.surv_probs uses."""

    def __init__(self, d=MODEL_DIR):
        import lightgbm as lgb
        self.b = lgb.Booster(model_file=str(d / "hazard.txt"))
        self.meta = json.loads((d / "meta.json").read_text())

    def predict_proba(self, X):
        p = self.b.predict(X[self.meta["inputs"]])
        return np.c_[1 - p, p]


def long_exits(longs, paths, spaths, model, pstar):
    """Per long: (close index 0..4 it exits at, its signed path value there). longs needs conf, side, price, atr, d."""
    probs = M.surv_probs(model, longs, paths, spaths)
    out = []
    for i in range(len(longs)):
        k = M.surv_exit(paths[i], probs[i], pstar)
        out.append((k, float(paths[i, k])))
    return out


def settle(root, daily, cal, today, model=None):
    """daily(symbols, first, last) -> DataFrame (symbol, date, o, h, l, c, v, ao, ah, al, ac, av), all-adjusted and raw
    SIP bars. Settles each nx3allhi day whose fifth close after entry has passed."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3allhi_settle" and x["date"] >= FIRST_DAY]:
        if led.has("surv1long_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + M.SURV_MAX >= len(caldates) or caldates[k + M.SURV_MAX] >= today:
            continue
        ds = caldates[k + 1:k + 1 + M.SURV_MAX]
        rec = json.loads((Path(root) / e["file"]).read_text())
        day = json.loads((Path(root) / f"nx3allhi/days/{e['date']}.json").read_text())
        conf = {p["i"]: p.get("conf") for p in day.get("picks", [])}
        price = {p["i"]: p.get("prev_c") for p in day.get("picks", [])}
        tr = [t for t in rec.get("trades", []) if t.get("role") == "picks" and t.get("net") is not None]
        res = {"day": e["date"], "closes": [f"{x:%Y-%m-%d}" for x in ds], "trades": []}
        longs = [t for t in tr if t["side"] > 0]
        model = model or Hazard()
        if longs:
            syms = sorted({t["symbol"] for t in longs} | {"SPY"})
            bars = daily(syms, d - LOOKBACK, ds[-1])
            ac = bars.pivot_table(index="date", columns="symbol", values="ac")
            spy = ac["SPY"] if "SPY" in ac else None
            rows, paths, spaths = [], [], []
            for t in longs:
                s = t["symbol"]
                if s not in ac or spy is None or not ds.isin(ac.index).all() or d not in ac.index:
                    continue
                a = ac.loc[ds, s].to_numpy(float)
                if not np.isfinite(a).all():
                    continue
                from core import chart_features as C
                g = C.chart_features(bars[bars["symbol"] == s])
                g = g[g["date"] < d]
                rows.append({"symbol": s, "i": t["i"], "w": t["w"], "side": 1, "d": d, "conf": conf.get(t["i"]),
                             "price": price.get(t["i"]), "atr": float(g["atr14"].iloc[-1]) if len(g) else np.nan,
                             "net_plain": t["net"], "ret": t["ret"]})
                paths.append(M.surv_path(t["ret"], a))
                spaths.append(spy.loc[ds].to_numpy(float) / spy.loc[d] - 1)
            if rows:
                lp = pd.DataFrame(rows)
                ex = long_exits(lp, np.vstack(paths), np.vstack(spaths), model, model.meta["pstar"])
                for r, (kx, g) in zip(rows, ex):
                    days = (ds[kx] - d).days
                    net = g - N.cost_frac([r["price"]], [False], [days])[0]
                    res["trades"].append({"symbol": r["symbol"], "i": r["i"], "w": r["w"], "side": 1,
                                          "closes_held": kx + 1, "net": float(net), "net_plain": float(r["net_plain"])})
        done = {(t["symbol"], t["i"]) for t in res["trades"]}
        for t in tr:
            if t["side"] < 0 or (t["symbol"], t["i"]) not in done:
                res["trades"].append({"symbol": t["symbol"], "i": t["i"], "w": t["w"], "side": t["side"],
                                      "closes_held": 1, "net": t["net"], "net_plain": t["net"],
                                      "missing_path": t["side"] > 0})
        res["book_pnl"] = float(sum(BOOK * t["w"] * t["net"] for t in res["trades"]))
        res["plain_pnl"] = float(sum(BOOK * t["w"] * t["net_plain"] for t in res["trades"]))
        res["long_gain"] = float(sum(t["w"] * (t["net"] - t["net_plain"]) for t in res["trades"] if t["side"] > 0))
        out.append(led.write_json("surv1long_settle", e["date"], f"surv1long/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
