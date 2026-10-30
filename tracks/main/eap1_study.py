"""EAP1 (research/prereg/EAP1.md): the earnings announcement premium. Stage "events" fetches the earnings
8-Ks (item 2.02, acceptance time in New York time) of every company that was in EAP1's main universe, builds
each week's picks (expected to report next week, from the same week a year earlier) and controls, and writes
the 20-pick hand-check file (research/checks/EAP1.csv). Stage "measure" waits for that check (>= 19 right).

Shares W1's cache (`tracks/news/w1_data.py`) and PD1's 8-K code. Nothing in "events" reads a return.

Usage: python -m tracks.main.eap1_study --stage events --cache cache/w1 --out research/results/EAP1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import w1_data as D
from tracks.penny import pd1_study as P

SEED, CHECK_N, CHECK_PASS = 20261030, 20, 19
MIN_PRICE, MIN_DV20 = 5.0, 5e6
HOLD, HOLDS, COST, LAGS = 5, (5, 10), 0.002, 4
FIRST_T, HOLDOUT = pd.Timestamp("2017-01-06"), pd.Timestamp("2025-10-01")
TRIALS, MIN_PICKS = 83, 2000
CHECK = Path("research/checks/EAP1.csv")


def edir(cache):
    return D.w1dir(cache) / "eap1"


def main_rows(cache):
    """W1 rows with raw close >= $5 and dv20 >= $5M, one row per company and date (highest dollar volume)."""
    parts = [pd.read_parquet(p, columns=["symbol", "date", "c", "dv20", "cik"],
                             filters=[("uni", "==", True), ("c", ">=", MIN_PRICE), ("dv20", ">=", MIN_DV20)])
             for p in D.batch_files(cache)]
    u = pd.concat([p for p in parts if len(p)], ignore_index=True).dropna(subset=["cik"]).astype({"cik": "int64"})
    return u.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])


def stage_filings(cache, ciks):
    d = edir(cache) / "subs"
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), P.BATCH):
        f = d / f"e_{b // P.BATCH:04d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + P.BATCH] for r in P.earnings_rows(c)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=P.COLS).to_parquet(tmp)
        tmp.replace(f)
        print(f"[eap1 filings] batch {b // P.BATCH + 1}/{-(-len(ciks) // P.BATCH)}: {len(rows)} earnings 8-Ks")
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("e_*.parquet"))], ignore_index=True)


def week_ends(cal):
    """The last session of each week."""
    d = cal["date"]
    return pd.DatetimeIndex(d.groupby(d.dt.to_period("W-SUN")).max().sort_values())


def sessions_after(days, t, n):
    """Entry (first session after t) and the exit of each hold in `n` (the entry day counted as the first)."""
    i = days.searchsorted(t, side="right")
    return {h: (days[i + h - 1] if i + h - 1 < len(days) else pd.NaT) for h in n}, (days[i] if i < len(days) else pd.NaT)


def in_window(d0, t, lo, hi):
    """Companies with an event day 0 in (t - lo, t - hi] calendar days. d0: Series cik -> sorted day 0s."""
    a, b = np.datetime64(t - pd.Timedelta(days=lo)), np.datetime64(t - pd.Timedelta(days=hi))
    return {c for c, v in d0.items() if ((v > a) & (v <= b)).any()}


def build(ev, uni, cal):
    """(picks, controls) rows: t, cik, symbol, dv20, entry, exit5, exit10, and for picks the year-old event."""
    days = pd.DatetimeIndex(cal["date"])
    d0 = ev.groupby("cik")["day0"].apply(lambda s: np.sort(s.to_numpy()))
    by_day = {d: g for d, g in uni.groupby("date")}
    picks, ctrls = [], []
    for t in week_ends(cal):
        if t < FIRST_T:
            continue
        exits, entry = sessions_after(days, t, HOLDS)
        if pd.isna(exits[max(HOLDS)]) or exits[HOLD] >= HOLDOUT:
            continue
        u = by_day.get(t)
        if u is None:
            continue
        exp = in_window(d0, t, 364, 357)
        near = in_window(d0, t, 371, 350)
        p = u[u["cik"].isin(exp)]
        if p.empty:
            continue
        pool = u[~u["cik"].isin(near)].sort_values("cik")
        k = np.random.default_rng(SEED + t.toordinal()).permutation(len(pool))[:len(p)]
        base = {"t": t, "entry": entry, **{f"exit{h}": exits[h] for h in HOLDS}}
        lo, hi = t - pd.Timedelta(days=364), t - pd.Timedelta(days=357)
        old = ev[(ev["day0"] > lo) & (ev["day0"] <= hi)].drop_duplicates("cik").set_index("cik")
        for r in p.itertuples():
            o = old.loc[r.cik]
            picks.append(base | {"cik": r.cik, "symbol": r.symbol, "dv20": r.dv20, "old_accession": o["accession"],
                                 "old_accepted_et": o["accepted_et"], "old_items": o["items"], "old_day0": o["day0"]})
        for r in pool.iloc[np.sort(k)].itertuples():
            ctrls.append(base | {"cik": r.cik, "symbol": r.symbol, "dv20": r.dv20})
    return pd.DataFrame(picks), pd.DataFrame(ctrls)


def stage_events(cache, out):
    cal = D.calendar()
    uni = main_rows(cache)
    ciks = sorted(uni["cik"].unique().tolist())
    fil = stage_filings(cache, ciks)
    ev = P.first_per_company(fil.drop_duplicates("accession"))
    ev = ev.assign(day0=P.day0(ev["accepted_et"], cal)).dropna(subset=["day0"])
    picks, ctrls = build(ev, uni, cal)
    for name, df in (("picks", picks), ("controls", ctrls), ("events", ev)):
        tmp = edir(cache) / f"{name}.tmp"
        df.to_parquet(tmp)
        tmp.replace(edir(cache) / f"{name}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    if not CHECK.exists():
        k = np.sort(np.random.default_rng(SEED).choice(len(picks), size=min(CHECK_N, len(picks)), replace=False))
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        picks.iloc[k][["t", "cik", "symbol", "old_accession", "old_accepted_et", "old_items", "old_day0"]].assign(
            t=lambda x: x["t"].dt.strftime("%Y-%m-%d"),
            old_accepted_et=lambda x: x["old_accepted_et"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            old_day0=lambda x: x["old_day0"].dt.strftime("%Y-%m-%d"), ok="", note="").to_csv(CHECK, index=False)
    cov = {"main_companies": len(ciks), "earnings_8k": int(len(fil)), "events": int(len(ev)),
           "weeks": int(picks["t"].nunique()) if len(picks) else 0, "picks": int(len(picks)),
           "controls": int(len(ctrls)),
           "picks_per_year": {str(y): int(n) for y, n in picks["t"].dt.year.value_counts().sort_index().items()}
           if len(picks) else {}}
    (out / "EAP1_coverage.json").write_text(json.dumps(cov, indent=1))
    print("EAP1", json.dumps(cov))


# ---------- measure ----------

def returns(cache, rows):
    """r5, r10 per row (all-adjusted open of entry to the close of each exit, PD1's trade_ret)."""
    out = {h: np.full(len(rows), np.nan) for h in HOLDS}
    rows = rows.reset_index(drop=True)
    idx = {s: np.flatnonzero(rows["symbol"].to_numpy() == s) for s in rows["symbol"].unique()}
    for bars in P.per_batch(cache, set(rows["symbol"])):
        for s, g in bars.items():
            for i in idx.get(s, []):
                for h in HOLDS:
                    out[h][i] = P.trade_ret(g, rows.at[i, "entry"], rows.at[i, f"exit{h}"])
    return rows.assign(**{f"r{h}": out[h] for h in HOLDS})


def weekly(picks, ctrls, h=HOLD, mask=None):
    p = picks if mask is None else picks[mask]
    p = p.dropna(subset=[f"r{h}"]).assign(net=lambda x: x[f"r{h}"] - COST)
    c = ctrls.dropna(subset=[f"r{h}"]).assign(net=lambda x: x[f"r{h}"] - COST)
    w = p.groupby("t")["net"].agg(["mean", "size"]).join(c.groupby("t")["net"].mean().rename("ctrl"), how="inner")
    return w.assign(diff=w["mean"] - w["ctrl"])


def test(w, ff5, umd):
    m, se, t = nw_tstat(w["diff"], LAGS)
    lo, hi = block_bootstrap_ci(w["diff"], 4, 10_000, SEED)
    pm, _, pt = nw_tstat(w["mean"], LAGS)
    mon = w["diff"].groupby(w.index.to_period("M")).sum()
    a = stats.factor_alpha(mon.to_numpy(float), mon.index, ff5, umd, 1)
    return {"weeks": int(len(w)), "picks": int(w["size"].sum()), "mean_minus_control": m, "se_hac": se, "t_hac": t,
            "ci95": [lo, hi], "picks_mean_net": pm, "picks_t": pt, "control_mean_net": float(w["ctrl"].mean()),
            "ff5_mom_alpha_monthly": a}


def stage_measure(cache, out):
    n_ok = check_right()
    res = {"idea": "EAP1", "prereg": "research/prereg/EAP1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok is None:
        raise SystemExit("[eap1] fill in research/checks/EAP1.csv (ok = y/n) first")
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "EAP1_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    picks = returns(cache, pd.read_parquet(edir(cache) / "picks.parquet"))
    ctrls = returns(cache, pd.read_parquet(edir(cache) / "controls.parquet"))
    ev = pd.read_parquet(edir(cache) / "events.parquet")
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    w = weekly(picks, ctrls)
    prim = test(w, ff5, umd)
    a = prim["ff5_mom_alpha_monthly"]
    prim["passes_rule12"] = bool(prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0 and prim["picks_mean_net"] > 0
                                 and prim["picks"] >= MIN_PICKS)
    prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
    prim["deflated_sharpe"] = stats.deflated_sharpe(w["diff"], TRIALS)
    prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
    prim["near_miss_rule19"] = bool(not prim["passes"] and prim["t_hac"] >= 1.5 and prim["picks_mean_net"] > 0)
    p2, both = stats.idea_counts()
    # exploratory
    d0 = ev.groupby("cik")["day0"].apply(lambda s: np.sort(s.to_numpy()))
    did = np.array([c in d0.index and ((d0[c] >= np.datetime64(e)) & (d0[c] <= np.datetime64(x))).any()
                    for c, e, x in zip(picks["cik"], picks["entry"], picks[f"exit{HOLD}"])])
    terc = pd.qcut(picks["dv20"].rank(method="first"), 3, labels=["low", "mid", "high"])
    ex = {"hold_10": test(weekly(picks, ctrls, h=10), ff5, umd),
          "reported_in_week_share": float(did.mean()),
          "reported_in_week": test(weekly(picks, ctrls, mask=did), ff5, umd),
          "not_reported_in_week": test(weekly(picks, ctrls, mask=~did), ff5, umd),
          "dv20_tercile": {str(k): float(weekly(picks, ctrls, mask=(terc == k).to_numpy())["diff"].mean())
                           for k in ["low", "mid", "high"]},
          "by_year": {str(y): float(g["diff"].mean()) for y, g in w.groupby(w.index.year)},
          "days_around_day0": "not computed: the registration's wording does not fix a window, so it is left out"}
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"weeks": int(picks["t"].nunique()), "picks": int(len(picks)),
                       "picks_with_trade": int(picks[f"r{HOLD}"].notna().sum()),
                       "controls_with_trade": int(ctrls[f"r{HOLD}"].notna().sum())},
            "primary": prim, "exploratory": ex}
    (out / "EAP1_result.json").write_text(json.dumps(res, indent=1, default=str))
    w.to_csv(out / "EAP1_weekly.csv")
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def check_right():
    if not CHECK.exists():
        return None
    ok = pd.read_csv(CHECK)["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/EAP1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    (stage_events if a.stage == "events" else stage_measure)(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
