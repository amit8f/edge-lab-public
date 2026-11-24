"""NXM-VOTE (research/prereg/NXMVOTE.md): trade a news event only when at least k of LGBM, MLP, ENS and RFR fire the
same way. Uncapped, sized as NX3-ALL by the voters' mean confidence, exit at the next close.

    python -m tracks.news.nxvote_study --cache cache/nx1 --state cache/nxvote --out research/results/NXMVOTE

Each test year's voter confidences are saved under --state, so a rerun resumes. Summary numbers only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X3
from tracks.news import nxm_study as M
from tracks.news import nxr_study as R

VOTERS = ("LGBM", "MLP", "ENS", "RFR")
K_GRID = (2, 3, 4)
CHOOSE = (2019, 2020, 2021, 2022, 2023)
TEST = (2024, 2025)
KEEP = ["symbol", "d", "t", "ret", "price", "days"]


# ---------- pure helpers (tested) ----------

def vote(E, k):
    """E: events x voters, each voter's effect over its own fire cut. Side +1/-1 when at least k voters fire that
    way and more fire it than the other, else 0; confidence = mean |E| of the voters on the chosen side."""
    E = np.asarray(E, float)
    up, dn = E >= 1, E <= -1
    nu, nd = up.sum(1), dn.sum(1)
    side = np.where((nu >= k) & (nu > nd), 1, np.where((nd >= k) & (nd > nu), -1, 0))
    with np.errstate(invalid="ignore"):
        cu = np.where(up, E, 0).sum(1) / np.maximum(nu, 1)
        cd = np.where(dn, -E, 0).sum(1) / np.maximum(nd, 1)
    return side, np.where(side > 0, cu, np.where(side < 0, cd, 0.0))


def alone(e):
    """One voter by itself: fired when |e| >= 1."""
    e = np.asarray(e, float)
    side = np.where(e >= 1, 1, np.where(e <= -1, -1, 0))
    return side, np.abs(e) * (side != 0)


def book(real, side, conf, seed):
    """Per day (every day with a real event): sized net of the book (0 with no trade), sized pick minus same-day
    random control (NaN with no trade), trade count; and the trades."""
    day_all = real.assign(e=side * conf, fired=side != 0)
    rng = np.random.default_rng(seed)
    rows, trades = [], []
    for d, day in day_all.groupby("d", sort=True):
        p, c = X3.picks(day, rng, cap=None)
        if p.empty or c.empty:
            rows.append({"d": d, "sized_net": 0.0, "sized_pmc": np.nan, "n": 0})
            continue
        w = X3.conf_sizes(np.abs(p["e"].to_numpy(float)))
        pn, cn = N.net(p).to_numpy(float), N.net(c).to_numpy(float)
        rows.append({"d": d, "sized_net": float(w @ pn), "sized_pmc": float(w @ (pn - cn)), "n": len(p)})
        trades.append(p[["d", "symbol", "t", "side"]].assign(net=pn, w=w))
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["d", "symbol", "t", "side", "net", "w"])
    return pd.DataFrame(rows), tr


def top3_share(net):
    net = np.asarray(net, float)
    tot = net.sum()
    return float(np.sort(net)[::-1][:3].sum() / tot) if tot > 0 else None


# ---------- walk-forward ----------

def year_effects(ev, cal, yr):
    """The test year's real events with each voter's (and XGB's) effect over its fire cut, fit as NXM does."""
    caldates = pd.DatetimeIndex(cal["date"])
    start = pd.Timestamp(f"{yr}-01-01")
    cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
    tr = ev[ev["d"] <= cut]
    te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
    if tr.empty or te.empty:
        return None
    models = M.fit_arms(tr)
    models["RFR"] = X3.Model(tr, R.learner("RFR"))
    real = te[(te["kind"] == "real") & np.isfinite(te["ret"])]
    X = real[X3.INPUTS].to_numpy(float)
    out = real[KEEP].copy()
    for a in ("XGB", *VOTERS):
        out[f"e_{a}"] = (models[a].effect(X) / models[a].cut).astype("float32")
    return out.reset_index(drop=True)


def measured(trades, cache):
    """COST1's quoted spreads for vote trades that are NX3-ALL-HI trades with a quote (reported only)."""
    from tracks.news import cost1_study as Q
    files = sorted(Q.cdir(cache).glob("q_*.parquet")) if cache else []
    if not files or trades.empty:
        return None
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    x = trades.merge(q, on=["symbol", "d", "t"], how="inner")
    x = x[np.isfinite(x["hs_in"]) & np.isfinite(x["hs_out"])]
    if x.empty:
        return None
    flat = N.cost_frac(x["price"], x["side"] < 0, x["days"])
    m = Q.measured_net(x["side"] * x["ret"], flat, x["price"], x["hs_in"], x["hs_out"])
    k = Q.keep_mask(x["hs_in"], 0.005)
    return {"matched": int(len(x)), "share_of_trades": float(len(x) / len(trades)),
            "net_flat": float(x["net"].mean()), "net_measured": float(m.mean()),
            "net_measured_cost1_kept": float(m[k].mean()) if k.any() else None}


def summary(days, trades):
    return {"trades": int(len(trades)), "net_per_trade": float(trades["net"].mean()) if len(trades) else None,
            "median_net": float(trades["net"].median()) if len(trades) else None,
            "trades_per_day": float(days["n"].mean()), "sized_daily_net": X3.tstat(days["sized_net"].to_numpy(float)),
            "top3_share": top3_share(trades["net"])}


def report(eff, cache=None):
    eff = eff.assign(year=eff["d"].dt.year)
    E = lambda x: x[[f"e_{a}" for a in VOTERS]].to_numpy(float)
    ch, te = eff[eff["year"].isin(CHOOSE)].reset_index(drop=True), eff[eff["year"].isin(TEST)].reset_index(drop=True)
    choose = {}
    for k in K_GRID:
        d, _ = book(ch, *vote(E(ch), k), X3.SEED + k)
        choose[str(k)] = float(d["sized_net"].mean())
    k = int(max(K_GRID, key=lambda j: choose[str(j)]))
    vd, vt = book(te, *vote(E(te), k), X3.SEED + 100 + k)
    pmc = X3.tstat(vd["sized_pmc"].to_numpy(float))
    rep = {"rule": "NXM-VOTE passes if the sized daily return minus a same-day random control has mean > 0 with NW t >= 2 "
                   "and its own sized daily net > 0 (2024-01..2025-09)",
           "voters": list(VOTERS), "choose_2019_2023_sized_daily_net": choose, "k": k,
           "test": {"days": int(len(vd)), "pick_minus_control": pmc, **summary(vd, vt),
                    "per_side_net": {int(s): float(g["net"].mean()) for s, g in vt.groupby("side")}},
           "reported": {}}
    for a in ("XGB", *VOTERS):
        ad, at = book(te, *alone(te[f"e_{a}"]), X3.SEED + 200)
        rep["reported"][a] = summary(ad, at)
        both = vd.set_index("d")["sized_net"] - ad.set_index("d")["sized_net"]
        rep["reported"][a]["vote_minus_this"] = X3.tstat(both.to_numpy(float))
        overlap = vt.merge(at[["d", "symbol", "side"]].assign(x=1), on=["d", "symbol", "side"], how="left")
        rep["reported"][a]["share_of_vote_trades_also_here"] = float(overlap["x"].notna().mean()) if len(vt) else None
    vt2 = vt.merge(te[KEEP], on=["symbol", "d", "t"], how="left")
    rep["reported"]["cost1_measured"] = measured(vt2, cache)
    rep["verdict"] = ("passed" if pmc["mean"] > 0 and pmc["t_nw"] >= 2 and rep["test"]["sized_daily_net"]["mean"] > 0
                      else "failed")
    return rep


def run(ev, cal, state, out, cache=None, years=N.TEST_YEARS, deadline=float("inf")):
    state, out = Path(state), Path(out)
    state.mkdir(parents=True, exist_ok=True)
    for yr in years:
        f = state / f"y{yr}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return None
        t0 = time.monotonic()
        res = year_effects(ev, cal, yr)
        if res is not None:
            N.atomic_parquet(res, f)
        print(f"[nxvote] {yr} done in {time.monotonic() - t0:.0f}s", flush=True)
    eff = pd.concat([pd.read_parquet(state / f"y{yr}.parquet") for yr in years if (state / f"y{yr}.parquet").exists()],
                    ignore_index=True)
    rep = report(eff, cache)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NXMVOTE_result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--state", default="cache/nxvote")
    ap.add_argument("--out", default="research/results/NXMVOTE")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    d3 = X3.nx3dir(a.cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X3.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    if run(ev, X3.calendar(a.cache), a.state, a.out, cache=Path(a.cache), deadline=deadline) is None:
        print("[nxvote] time budget used; rerun to continue")


if __name__ == "__main__":
    main()
