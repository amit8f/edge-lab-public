"""SP1 (research/prereg/SP1.md): spin-offs bought after the forced selling. Stage "events" finds first-time
Form 10-12B filers in W1's EDGAR table that start trading within a year, fixes each one's listing day, entry
day (the 22nd session) and same-day controls, and writes the 20-event hand-check file
(research/checks/SP1.csv). Stage "measure" waits for that check (>= 19 right).

Shares W1's cache (`tracks/news/w1_data.py`). Nothing in "events" reads a return.

Usage: python -m tracks.main.sp1_study --stage events --cache cache/w1 --out research/results/SP1
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

SEED, CHECK_N, CHECK_PASS = 20261031, 20, 19
FORMS = {"10-12B", "10-12B/A"}
MIN_PRICE, MIN_DV20, WAIT, N_CTRL = 5.0, 2e6, 21, 5
HOLD, HOLDS, COST, LAGS = 250, (125, 250), 0.002, 11
FIRST_F, FIRST_ENTRY, HOLDOUT = pd.Timestamp("2015-10-01"), pd.Timestamp("2016-01-01"), pd.Timestamp("2025-10-01")
TRIALS, MIN_EVENTS = 83, 60
CHECK = Path("research/checks/SP1.csv")
# Rule 23 fix (hand check round 1, 13/20: the first bar was a when-issued or record-date day, or late, or the
# cache's first day). Each candidate's first regular-way day is looked up by hand from the parent's or the
# company's completion release (LISTED lists the candidates; DATES holds cik, symbol, regular_way, source).
LISTED = Path("research/checks/SP1_listings.csv")
DATES = Path("research/checks/SP1_dates.csv")
COLS = ["symbol", "date", "c", "dv20", "cik"]


def sdir(cache):
    return D.w1dir(cache) / "sp1"


def first_10(fil):
    """cik -> first Form 10-12B filing date (from FIRST_F), and every 10-12B filer's filing dates."""
    f = fil[fil["form"].isin(FORMS)]
    first = f.groupby("cik")["filed"].min()
    return first[first >= FIRST_F], f.groupby("cik")["filed"].apply(lambda s: np.sort(s.to_numpy()))


def listings(rows, first, days, regular=None):
    """Per company: listing day L (first bar on or after f0, within 365 days, none in the 365 days before
    f0), entry (the WAIT+1-th session counting L), the symbol and price row on the session before entry.
    With `regular` (cik -> documented first regular-way day), L is that day and companies without one drop."""
    out = []
    for cik, g in rows.groupby("cik"):
        f0 = first.get(cik)
        if f0 is None:
            continue
        g = g.sort_values("date")
        if ((g["date"] < f0) & (g["date"] >= f0 - pd.Timedelta(days=365))).any():
            continue
        after = g[(g["date"] >= f0) & (g["date"] <= f0 + pd.Timedelta(days=365))]
        if after.empty:
            continue
        L = after["date"].iloc[0]
        L_bar = L
        if regular is not None:
            if cik not in regular.index or pd.isna(regular[cik]) or regular[cik] < days[0]:
                continue
            L = days[days.searchsorted(regular[cik])]
        i = days.searchsorted(L)
        if i + WAIT >= len(days):
            continue
        entry, prior = days[i + WAIT], days[i + WAIT - 1]
        pr = g[g["date"] == prior].sort_values("dv20", ascending=False)
        out.append({"cik": int(cik), "f0": f0, "listing_day": L, "first_bar": L_bar, "entry": entry, "prior_day": prior,
                    "symbol": pr["symbol"].iloc[0] if len(pr) else after["symbol"].iloc[0],
                    "prior_close": float(pr["c"].iloc[0]) if len(pr) else np.nan,
                    "prior_dv20": float(pr["dv20"].iloc[0]) if len(pr) else np.nan})
    return pd.DataFrame(out)


def exits(ev, days):
    pos = days.searchsorted(ev["entry"])
    return ev.assign(**{f"exit{h}": [days[p + h - 1] if p + h - 1 < len(days) else pd.NaT for p in pos] for h in HOLDS})


def controls(ev, uni, filers):
    """Per event, N_CTRL candidate lists: main-universe stocks on the prior day whose company filed no 10-12B
    in the 3 years before entry, in the order of default_rng(SEED + entry ordinal).permutation; the first
    3*N_CTRL are kept so that candidates without an entry bar can be skipped in measure."""
    by_day = {d: g for d, g in uni.groupby("date")}
    out = {}
    for i, r in ev.iterrows():
        pool = by_day.get(r["prior_day"])
        if pool is None:
            continue
        lo = np.datetime64(r["entry"] - pd.Timedelta(days=3 * 365))
        hi = np.datetime64(r["entry"])
        bad = {c for c in pool["cik"] if c in filers.index and ((filers[c] >= lo) & (filers[c] <= hi)).any()}
        pool = pool[~pool["cik"].isin(bad | {r["cik"]})].sort_values("cik")
        perm = np.random.default_rng(SEED + r["entry"].toordinal()).permutation(len(pool))
        out[i] = pool.iloc[perm[:3 * N_CTRL]]["symbol"].tolist()
    return out


def stage_events(cache, out):
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    fil = D.load_filings(cache, ["cik", "form", "filed"])
    first, filers = first_10(fil)
    need = sorted(int(c) for c in first.index)
    rows = pd.concat([pd.read_parquet(p, columns=COLS, filters=[("cik", "in", need)]) for p in D.batch_files(cache)],
                     ignore_index=True).dropna(subset=["cik"]).astype({"cik": "int64"})
    raw = listings(rows, first, days)
    out.mkdir(parents=True, exist_ok=True)
    LISTED.parent.mkdir(parents=True, exist_ok=True)
    raw[["cik", "symbol", "f0", "first_bar"]].assign(
        f0=raw["f0"].dt.strftime("%Y-%m-%d"), first_bar=raw["first_bar"].dt.strftime("%Y-%m-%d")).to_csv(LISTED, index=False)
    if not DATES.exists():
        print(f"[sp1] wrote {LISTED} ({len(raw)} companies); fill {DATES} with each first regular-way day, then rerun")
        return
    d = pd.read_csv(DATES, parse_dates=["regular_way"])
    regular = d.set_index("cik")["regular_way"]
    ev = listings(rows, first, days, regular)
    ev = exits(ev, days)
    ev["main"] = (ev["prior_close"] >= MIN_PRICE) & (ev["prior_dv20"] >= MIN_DV20)
    sig = ev[ev["main"] & (ev["entry"] >= FIRST_ENTRY) & ev[f"exit{HOLD}"].notna()
             & (ev[f"exit{HOLD}"] < HOLDOUT)].reset_index(drop=True)
    parts = [pd.read_parquet(p, columns=COLS, filters=[("uni", "==", True), ("c", ">=", MIN_PRICE),
                                                        ("dv20", ">=", MIN_DV20), ("date", "in", list(sig["prior_day"]))])
             for p in D.batch_files(cache)] if len(sig) else []
    uni = pd.concat([p for p in parts if len(p)], ignore_index=True).dropna(subset=["cik"]).astype({"cik": "int64"}) \
        if parts else pd.DataFrame(columns=COLS)
    uni = uni.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])
    cand = controls(sig, uni, filers)
    sig["control_candidates"] = [cand.get(i, []) for i in sig.index]
    sdir(cache).mkdir(parents=True, exist_ok=True)
    tmp = sdir(cache) / "events.tmp"
    sig.to_parquet(tmp)
    tmp.replace(sdir(cache) / "events.parquet")
    if not CHECK.exists() and len(sig):
        k = np.sort(np.random.default_rng(SEED).choice(len(sig), size=min(CHECK_N, len(sig)), replace=False))
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        sig.iloc[k][["cik", "symbol", "f0", "listing_day", "entry"]].assign(
            **{c: (lambda x, c=c: x[c].dt.strftime("%Y-%m-%d")) for c in ("f0", "listing_day", "entry")},
            ok="", note="").to_csv(CHECK, index=False)
    cov = {"first_10_12b_filers": int(len(first)), "with_new_listing": int(len(raw)),
           "with_documented_regular_way_day": int(len(ev)),
           "main_at_entry": int(ev["main"].sum()), "events_in_window": int(len(sig)),
           "per_year": {str(y): int(n) for y, n in sig["entry"].dt.year.value_counts().sort_index().items()}}
    (out / "SP1_coverage.json").write_text(json.dumps(cov, indent=1))
    print("SP1", json.dumps(cov))


# ---------- measure ----------

def check_right():
    if not CHECK.exists():
        return None
    ok = pd.read_csv(CHECK)["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def cohorts(ev, h=HOLD, mask=None):
    e = ev if mask is None else ev[mask]
    e = e.dropna(subset=[f"r{h}", f"c{h}"]).assign(net=lambda x: x[f"r{h}"] - COST, cnet=lambda x: x[f"c{h}"] - COST)
    c = e.groupby(e["entry"].dt.to_period("M")).agg(pick=("net", "mean"), ctrl=("cnet", "mean"), n=("net", "size"))
    return c.assign(diff=c["pick"] - c["ctrl"]).sort_index()


def test(c, ff5, umd, h=HOLD):
    m, se, t = nw_tstat(c["diff"], LAGS)
    lo, hi = block_bootstrap_ci(c["diff"], 12, 10_000, SEED)
    pm, _, pt = nw_tstat(c["pick"], LAGS)
    months = max(1, round(h / 21))
    f = ff5.join(umd, how="inner")
    fh = f.rolling(months).sum().shift(-(months - 1)).reindex(c.index)
    ok = fh.notna().all(axis=1).to_numpy()
    alpha = stats.ols_hac(c["diff"].to_numpy(float)[ok], fh.to_numpy()[ok], LAGS) if ok.sum() > 8 else None
    return {"events": int(c["n"].sum()), "months": int(len(c)), "mean_minus_control": m, "se_hac": se, "t_hac": t,
            "ci95": [lo, hi], "picks_mean_net": pm, "picks_t": pt, "control_mean_net": float(c["ctrl"].mean()),
            "ff5_mom_alpha": alpha}


def stage_measure(cache, out):
    n_ok = check_right()
    res = {"idea": "SP1", "prereg": "research/prereg/SP1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok is None:
        raise SystemExit("[sp1] fill in research/checks/SP1.csv (ok = y/n) first")
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "SP1_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    ev = pd.read_parquet(sdir(cache) / "events.parquet")
    ev["entry1"] = [days[days.searchsorted(L) + 1] for L in ev["listing_day"]]
    ev = ev.assign(**{f"exit1_{h}": [days[days.searchsorted(e) + h - 1] for e in ev["entry1"]] for h in (HOLD,)})
    need = set(ev["symbol"]) | {s for v in ev["control_candidates"] for s in v}
    rets = {}
    for bars in P.per_batch(cache, need):
        for i, r in ev.iterrows():
            for s in [r["symbol"]] + list(r["control_candidates"]):
                if s in bars:
                    for h in HOLDS:
                        rets[(i, s, h)] = P.trade_ret(bars[s], r["entry"], r[f"exit{h}"])
                    if s == r["symbol"]:
                        rets[(i, s, "L1")] = P.trade_ret(bars[s], r["entry1"], r[f"exit1_{HOLD}"])
    for h in HOLDS:
        ev[f"r{h}"] = [rets.get((i, s, h), np.nan) for i, s in zip(ev.index, ev["symbol"])]
        ctrl = []
        for i, r in ev.iterrows():
            got = [rets[(i, s, h)] for s in r["control_candidates"] if np.isfinite(rets.get((i, s, h), np.nan))][:N_CTRL]
            ctrl.append(float(np.mean(got)) if got else np.nan)
        ev[f"c{h}"] = ctrl
    ev["rL1"] = [rets.get((i, s, "L1"), np.nan) for i, s in zip(ev.index, ev["symbol"])]
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    c = cohorts(ev)
    prim = test(c, ff5, umd)
    a = prim["ff5_mom_alpha"] or {"alpha": float("nan"), "t": float("nan")}
    prim["passes_rule12"] = bool(prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0 and prim["picks_mean_net"] > 0
                                 and prim["events"] >= MIN_EVENTS)
    prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
    prim["deflated_sharpe"] = stats.deflated_sharpe(c["diff"], TRIALS)
    prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
    prim["near_miss_rule19"] = bool(not prim["passes"] and prim["t_hac"] >= 1.5 and prim["picks_mean_net"] > 0)
    p2, both = stats.idea_counts()
    small = (ev["prior_dv20"] < ev["prior_dv20"].median()).to_numpy()
    ev_l1 = ev.assign(**{f"r{HOLD}": ev["rL1"], "entry": ev["entry1"]})
    ex = {"hold_125": test(cohorts(ev, h=125), ff5, umd, h=125),
          "entry_L_plus_1": test(cohorts(ev_l1), ff5, umd),
          "below_median_dv20": float(cohorts(ev, mask=small)["diff"].mean()),
          "above_median_dv20": float(cohorts(ev, mask=~small)["diff"].mean()),
          "by_year": {str(y): float(g["diff"].mean()) for y, g in c.groupby(c.index.year)}}
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"events": int(len(ev)), "with_trade": int(ev[f"r{HOLD}"].notna().sum()),
                       "with_control": int(ev[f"c{HOLD}"].notna().sum())},
            "primary": prim, "exploratory": ex}
    (out / "SP1_result.json").write_text(json.dumps(res, indent=1, default=str))
    ev.drop(columns=["control_candidates"]).to_csv(out / "SP1_trades.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/SP1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    (stage_events if a.stage == "events" else stage_measure)(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
