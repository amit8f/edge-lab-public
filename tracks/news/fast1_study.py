"""FAST1 (research/prereg/FAST1.md): NX3-ALL-HI events exited within 60 minutes.

Exit chosen on 2019-2023 (mean net per trade, at least 300 trades) among 30 / 60-minute time exits and (2%, 1%) /
(3%, 1.5%) brackets; that one exit is then tested once on 2024-01 to 2025-09: mean net > 0, day-clustered t >= 2,
300+ trades, beats the 95th percentile of 200 random-control draws (same days, counts and sides). Writes
<out>/FAST1_result.json (summaries only).

    python -m tracks.news.fast1_study --cache cache/nx1 --out research/results/FAST1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import alpaca_headers
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import s0_study as S

CHOOSE = (2019, 2020, 2021, 2022, 2023)
TEST = (2024, 2025)
HI = 1.5
WINDOW = 60
EXITS = ("t30", "t60", "b200", "b300")
BRACKET = {"b200": (0.02, 0.01), "b300": (0.03, 0.015)}
WAIT_INTRADAY_LONG = 15
MIN_TRADES, T_MIN, DRAWS = 300, 2.0, 200


# ---------- pure helpers (tested) ----------

def exits(bars, entry_t, side):
    """bars: one session's raw 1-minute bars (t, o, h, l, c). Entry at the open of the first bar at or after entry_t
    (within 5 minutes). Returns entry price and the signed gross return of every exit in EXITS, or None.
    Brackets: first touch within WINDOW minutes; the target fills at its price, the stop at its price or the
    first bar's open beyond it; a bar touching both is the stop; neither -> out at the WINDOW time exit."""
    if bars is None or bars.empty:
        return None
    b = bars[bars["t"] >= entry_t]
    if b.empty or b["t"].iloc[0] > entry_t + pd.Timedelta(minutes=5):
        return None
    start, entry = b["t"].iloc[0], float(b["o"].iloc[0])
    if not entry > 0:
        return None
    out = {"price": entry}
    for h in (30, 60):
        w = b[b["t"] < start + pd.Timedelta(minutes=h)]
        out[f"t{h}"] = side * (float(w["c"].iloc[-1]) / entry - 1)
    w = b[b["t"] < start + pd.Timedelta(minutes=WINDOW)]
    for k, (tp, sl) in BRACKET.items():
        res = out["t60"]
        for i, r in enumerate(w.itertuples(index=False)):
            o = side * (r.o / entry - 1) if i else 0.0
            worst = side * ((r.l if side > 0 else r.h) / entry - 1)
            best = side * ((r.h if side > 0 else r.l) / entry - 1)
            if worst <= -sl:
                res = min(-sl, o)
                break
            if best >= tp:
                res = tp
                break
        out[k] = res
    return out


def cluster_t(net, day):
    """Mean net per trade and its day-clustered t (sum of day residuals)."""
    x, d = np.asarray(net, float), np.asarray(day)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = x.mean()
    g = pd.Series(x - m).groupby(d).sum().to_numpy()
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(m), float(m / se) if se > 0 else float("nan")


def choose(t):
    """t: choice-year event rows with net_<exit> columns -> best exit by mean net (>= MIN_TRADES trades)."""
    best, rep = None, {}
    for k in EXITS:
        x = t[f"net_{k}"].dropna()
        rep[k] = {"n": len(x), "mean_net": float(x.mean()) if len(x) else float("nan")}
        if len(x) >= MIN_TRADES and (best is None or rep[k]["mean_net"] > rep[best]["mean_net"]):
            best = k
    return best, rep


# ---------- job ----------

def entry_time(r, o_t):
    pre = r.entry_t <= o_t + pd.Timedelta(minutes=2)
    wait = 0 if (pre or r.side < 0) else WAIT_INTRADAY_LONG
    return r.entry_t + pd.Timedelta(minutes=wait), ("preopen" if pre else "intraday")


def rows_for(fr, bars, o_t, d, grp, draw=-1):
    out = []
    for r in fr.itertuples(index=False):
        et, when = entry_time(r, o_t)
        x = exits(bars.get(r.symbol), et, int(r.side))
        if x is None:
            continue
        cost = N.cost_frac([x["price"]], [r.side < 0], [1])[0]
        row = {"d": d, "grp": grp, "draw": draw, "when": when, "side": int(r.side), "price": x["price"],
               "next_net": float(r.side * r.ret - N.cost_frac([r.price], [r.side < 0], [r.days])[0])
               if hasattr(r, "ret") else float("nan")}
        row.update({f"net_{k}": x[k] - cost for k in EXITS})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/FAST1")
    a = ap.parse_args()
    d3 = X.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    cal = X.calendar(a.cache)
    ses = X.sessions(cal).set_index("date")
    caldates = pd.DatetimeIndex(cal["date"])
    headers = alpaca_headers()
    rows = []
    for yr in CHOOSE + TEST:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr, te = ev[ev["d"] <= cut_d], ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = X.Model(tr)
        real = te[te["kind"] == "real"]
        e = model.effect(real[X.INPUTS].to_numpy(float))
        real = real.assign(e=e, fired=model.fires(e))
        rng = np.random.default_rng(X.SEED + yr)
        for d, day in real.groupby("d", sort=True):
            p, c = X.picks(day, rng, cap=None, hi=HI, cut=model.cut)
            if p.empty or d not in ses.index:
                continue
            o_t = ses.loc[d, "open_t"]
            draws = []
            if yr in TEST:     # 200 control draws from the day's real events, same count and sides
                draws = [day.iloc[rng.choice(len(day), size=len(p), replace=False)].assign(side=p["side"].to_numpy())
                         for _ in range(DRAWS)]
            syms = set(p["symbol"]) | set(c["symbol"]) | {s for x in draws for s in x["symbol"]}
            bars = S.day_bars_hl(sorted(syms), d, o_t, ses.loc[d, "close_t"], headers)
            rows += [dict(r, year=yr) for r in rows_for(p, bars, o_t, d, "event")]
            rows += [dict(r, year=yr) for r in rows_for(c, bars, o_t, d, "control")]
            for i, x in enumerate(draws):
                rows += [dict(r, year=yr) for r in rows_for(x, bars, o_t, d, "draw", i)]
        print(f"[fast1] {yr}: {sum(1 for r in rows if r['year'] == yr and r['grp'] == 'event')} events", flush=True)
    t = pd.DataFrame(rows)
    ch = t[(t["grp"] == "event") & t["year"].isin(CHOOSE)]
    best, choice = choose(ch)
    rep = {"choice_years": list(CHOOSE), "choice": choice, "chosen": best,
           "choice_control": {k: float(t[(t["grp"] == "control") & t["year"].isin(CHOOSE)][f"net_{k}"].mean())
                              for k in EXITS}}
    if best:
        col = f"net_{best}"
        te = t[(t["grp"] == "event") & t["year"].isin(TEST)]
        m, tt = cluster_t(te[col], te["d"])
        dr = t[(t["grp"] == "draw") & t["year"].isin(TEST)].groupby("draw")[col].mean()
        p95 = float(dr.quantile(0.95)) if len(dr) else float("nan")
        rep["test"] = {
            "n": len(te), "days": int(te["d"].nunique()), "mean_net": m, "t_day": tt,
            "control_draws_p95": p95, "control_draws_mean": float(dr.mean()) if len(dr) else float("nan"),
            "right_way": float((te[col] > -N.cost_frac(te["price"], te["side"] < 0, np.ones(len(te)))).mean()),
            "pass": bool(m > 0 and tt >= T_MIN and len(te) >= MIN_TRADES and m > p95),
            "beside": {
                "next_close_net": float(te["next_net"].mean()),
                "per_day": float(te.groupby("d").size().mean()),
                "longs": float(te[te["side"] > 0][col].mean()), "shorts": float(te[te["side"] < 0][col].mean()),
                "main": float(te[te["price"] >= 5][col].mean()), "penny": float(te[te["price"] < 5][col].mean()),
                "n_main": int((te["price"] >= 5).sum()), "n_penny": int((te["price"] < 5).sum()),
                "by_year": {str(y): float(g[col].mean()) for y, g in te.groupby("year")},
                "preopen": float(te[te["when"] == "preopen"][col].mean()),
                "intraday": float(te[te["when"] == "intraday"][col].mean())}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "FAST1_result.tmp"
    tmp.write_text(json.dumps(rep, indent=1, default=float))
    tmp.replace(out / "FAST1_result.json")
    print(json.dumps(rep, default=float))


if __name__ == "__main__":
    main()
