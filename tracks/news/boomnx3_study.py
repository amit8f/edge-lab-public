"""BOOM1 x NX3 (research/prereg/BOOMNX3.md): NX3's long calls taken only on stocks in BOOM1's nightly top 20, plus a
descriptive check of NX3 against controls matched on side, price tier and volatility.

    python -m tracks.news.boomnx3_study --stage candidates|run --cache cache/nx1 --out research/results/BOOMNX3

candidates: BOOM1's walk-forward (as anom1_study), each night's top 100 by score -> cache nx3/boomnx3/candidates.parquet.
run: the intersection test, the reported lines and the matched-control check. Summaries only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main import boom1_study as B
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

POOL = 100
TOP = B.TOP
TEST = (2024, 2025)
EARLY = (2019, 2020, 2021, 2022, 2023)
HURDLE = 0.002
MIN_TRADES = 30
FIRED = Path("research/results/NX3/NX3_fired_trades.csv")


def bdir(cache):
    return X.nx3dir(cache) / "boomnx3"


# ---------- pure helpers (tested) ----------

def tag_boom(trades, cand, top=TOP):
    """trades (d, symbol, ...) with BOOM1's rank for session d (NaN when not in the night's pool) and the boom label."""
    c = cand.assign(rank=cand.groupby("nxt")["score"].rank(ascending=False, method="first"))
    c = c.drop(columns="d").rename(columns={"nxt": "d"})[["d", "symbol", "rank", "boom"]]
    out = trades.merge(c, on=["d", "symbol"], how="left")
    out["in_top"] = out["rank"] <= top
    return out


def daily_t(t, col="net"):
    return X.tstat(t.groupby("d")[col].mean().to_numpy(float))


def top3_share(net):
    net = np.asarray(net, float)
    return float(np.sort(net)[::-1][:3].sum() / net.sum()) if net.sum() > 0 else None


def tier(price):
    p = np.asarray(price, float)
    return np.where(p < 5, 0, np.where(p < 20, 1, 2))


def matched(ev, fired):
    """Per fired trade: its net minus the mean net, on the same side, of the same day's unfired real events in the same
    price tier and vol20 quintile (quintiles cut within the year). ev: real events (d, symbol, ret, price, days, vol20)."""
    ev = ev.assign(year=ev["d"].dt.year, tier=tier(ev["price"]))
    ev["vq"] = ev.groupby("year")["vol20"].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    f = fired.merge(ev[["d", "symbol", "tier", "vq"]], on=["d", "symbol"], how="inner")
    pool = ev.merge(fired[["d", "symbol"]].assign(_f=1), on=["d", "symbol"], how="left")
    pool = pool[pool["_f"].isna()]
    cost = {}
    for s in (1, -1):
        g = pool.assign(cn=s * pool["ret"] - N.cost_frac(pool["price"], np.full(len(pool), s < 0), pool["days"]))
        cost[s] = g.groupby(["d", "tier", "vq"])["cn"].mean()
    f["ctrl"] = [cost[s].get((d, t, q), np.nan) for s, d, t, q in zip(f["side"], f["d"], f["tier"], f["vq"])]
    return f.assign(diff=f["net"] - f["ctrl"]).dropna(subset=["ctrl"])


# ---------- stages ----------

def stage_candidates(cache):
    df = B.load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    out = []
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = df[df["d"] <= cut_d]
        te = df[(df["d"] >= start) & (df["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        if len(tr) > B.TRAIN_ROWS:
            tr = tr.iloc[np.sort(np.random.default_rng(B.SEED + yr).choice(len(tr), B.TRAIN_ROWS, replace=False))]
        m = B.classifier().fit(tr[B.INPUTS].to_numpy(np.float32), tr["boom"].to_numpy())
        te = te.assign(score=m.predict_proba(te[B.INPUTS].to_numpy(np.float32))[:, 1])
        for _, g in te.groupby("d", sort=True):
            if len(g) >= TOP * 5:
                out.append(g.nlargest(POOL, "score")[["symbol", "d", "nxt", "score", "boom"]])
        print(f"[boomnx3 candidates] {yr}: {sum(len(x) for x in out)} so far", flush=True)
    N.atomic_parquet(pd.concat(out, ignore_index=True), bdir(cache) / "candidates.parquet")


def block(t, base):
    """Summary of one set of trades beside the base set (same period)."""
    if t.empty:
        return {"trades": 0}
    return {"trades": int(len(t)), "days": int(t["d"].nunique()), "net_per_trade": float(t["net"].mean()),
            "median_net": float(t["net"].median()), "daily": daily_t(t),
            "minus_base_per_trade": float(t["net"].mean() - base["net"].mean()),
            "boomed_share": float(t["boom"].mean()) if t["boom"].notna().any() else None,
            "top3_share": top3_share(t["net"])}


def measured(t, ev, cache):
    from tracks.news import cost1_study as Q
    files = sorted(Q.cdir(cache).glob("q_*.parquet"))
    if not files or t.empty:
        return None
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates(["symbol", "d"])
    x = t.merge(ev[["d", "symbol", "ret", "price", "days"]], on=["d", "symbol"]).merge(q, on=["symbol", "d"])
    x = x[np.isfinite(x["hs_in"]) & np.isfinite(x["hs_out"])]
    if x.empty:
        return None
    flat = N.cost_frac(x["price"], x["side"] < 0, x["days"])
    m = Q.measured_net(x["side"] * x["ret"], flat, x["price"], x["hs_in"], x["hs_out"])
    return {"matched": int(len(x)), "net_flat": float(x["net"].mean()), "net_measured": float(m.mean()),
            "net_measured_cost1_kept": float(m[Q.keep_mask(x["hs_in"], 0.005)].mean()) if Q.keep_mask(x["hs_in"], 0.005).any() else None}


def stage_run(cache, out):
    cand = pd.read_parquet(bdir(cache) / "candidates.parquet")
    fired = pd.read_csv(FIRED, parse_dates=["d"])
    fired["side"] = fired["side"].map({"long": 1, "short": -1})
    t = tag_boom(fired, cand)
    ev = pd.concat([pd.read_parquet(f, columns=["symbol", "d", "kind", "ret", "price", "days", "vol20"])
                    for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[(ev["kind"] == "real") & np.isfinite(ev["ret"]) & (ev["d"] <= N.LAST_EXIT)].drop_duplicates(["symbol", "d"])
    rep = {"rule": "BOOM1xNX3 passes if NX3 longs on BOOM1's top 20 have net/trade > 0 with NW t >= 2 on daily means, "
                   "30+ trades, and beat all NX3 longs by >= +0.20 points/trade (2024-01..2025-09)",
           "candidate_nights": int(cand["nxt"].nunique())}
    for name, yrs in (("test_2024_2025", TEST), ("early_2019_2023", EARLY)):
        p = t[t["year"].isin(yrs)]
        longs = p[p["side"] == 1]
        inter = longs[longs["in_top"]]
        r = {"all_longs": block(longs, longs), "intersection": block(inter, longs),
             "top100_longs": block(longs[longs["rank"] <= POOL], longs),
             "hi_intersection": block(inter[inter["conf"] >= 1.5], longs[longs["conf"] >= 1.5]),
             "shorts_on_top20": block(p[(p["side"] == -1) & p["in_top"]], p[p["side"] == -1]),
             "intersection_measured_costs": measured(inter, ev, cache)}
        rep[name] = r
    x = rep["test_2024_2025"]["intersection"]
    rep["verdict"] = ("passed" if x.get("trades", 0) >= MIN_TRADES and x["daily"]["mean"] > 0 and x["daily"]["t_nw"] >= 2
                      and x["minus_base_per_trade"] >= HURDLE else "failed")
    mm = matched(ev, fired)
    mm["year"] = mm["d"].dt.year
    rep["matched_control"] = {
        "note": "descriptive, not counted: pick net minus same-day unfired news events, same side, price tier, vol20 quintile",
        "coverage": float(len(mm) / len(fired)),
        **{name: {"all": daily_t(mm[mm["year"].isin(yrs)], "diff"),
                  **{("long" if s > 0 else "short"): daily_t(mm[mm["year"].isin(yrs) & (mm["side"] == s)], "diff") for s in (1, -1)}}
           for name, yrs in (("test_2024_2025", TEST), ("early_2019_2023", EARLY))}}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "BOOMNX3_result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["candidates", "run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/BOOMNX3")
    a = ap.parse_args()
    stage_candidates(Path(a.cache)) if a.stage == "candidates" else stage_run(Path(a.cache), a.out)


if __name__ == "__main__":
    main()
