"""PD1 (research/prereg/PD1.md): post-earnings drift in penny stocks. Stage "events" builds the earnings 8-K
events from SEC's submissions records (acceptance time and items) for every company that was in W1's penny
universe, and writes the 20-event hand-check file (research/checks/PD1.csv). Stage "measure" waits for that
check (>= 19 right): the reaction, the top-20% signal against the previous 365 days' events, the 60-session
long from day +2's open, the same-day random control and the monthly-cohort test.

Shares W1's cache (`tracks/news/w1_data.py`: daily bars, universe flag, CIKs). Nothing here reads a return.

Usage: python -m tracks.penny.pd1_study --stage events --cache cache/w1 --out research/results/PD1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, sec, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import l1_study as L1
from tracks.news import w1_data as D

SEED, CHECK_N, CHECK_PASS = 20261028, 20, 19
FIRST, LAST = pd.Timestamp("2015-10-01"), pd.Timestamp("2025-09-30")
DEDUP_DAYS, BATCH = 30, 500
TOP, BOTTOM, MIN_PRIOR, HOLD, HOLDS = 0.80, 0.20, 100, 60, (20, 40, 60)
COST, BORROW, LAGS, N_CAND, CTRL_DAYS = 0.02, 0.20, 3, 5, 15
SIGNAL_FIRST, HOLDOUT = pd.Timestamp("2017-01-01"), pd.Timestamp("2025-10-01")
TRIALS, MIN_EVENTS = 71, 300
CHECK = Path("research/checks/PD1.csv")
SUBS = "https://data.sec.gov/submissions/"
COLS = ["cik", "accession", "form", "filed", "accepted_et", "items"]
ET = "America/New_York"


def pd1dir(cache):
    return D.w1dir(cache) / "pd1"


def earnings_rows(cik, get=sec.get, first=FIRST, last=LAST):
    """Form 8-K filings listing item 2.02, filed first .. last, with their acceptance time in New York time
    (naive). The submissions record's acceptanceDateTime is UTC: the hand check found every one 4-5 hours
    ahead of the filing index page's Eastern time."""
    r = get(f"{SUBS}CIK{int(cik):010d}.json")
    if r is None:
        return []
    j = r.json()
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        if pd.Timestamp(f.get("filingTo", "1900-01-01")) >= first:
            rr = get(SUBS + f["name"])
            if rr is not None:
                blocks.append(rr.json())
    out = []
    for b in blocks:
        n = len(b.get("form", []))
        for acc, form, filed, items, at in zip(b.get("accessionNumber", []), b.get("form", []),
                                               b.get("filingDate", []), b.get("items", [""] * n),
                                               b.get("acceptanceDateTime", [""] * n)):
            its = [i.strip() for i in (items or "").split(",") if i.strip()]
            if form == "8-K" and "2.02" in its and first <= pd.Timestamp(filed) <= last and at:
                out.append({"cik": int(cik), "accession": acc, "form": form, "filed": filed,
                            "accepted_et": pd.Timestamp(at[:19], tz="UTC").tz_convert(ET).tz_localize(None), "items": ",".join(its)})
    return out


def day0(accepted, cal):
    """The first session whose close comes after acceptance (cal: date, open, close as 'HH:MM' ET)."""
    closes = pd.to_datetime(cal["date"].dt.strftime("%Y-%m-%d") + " " + cal["close"].astype(str))
    i = np.searchsorted(closes.to_numpy(), pd.DatetimeIndex(accepted).to_numpy(), side="right")
    d = cal["date"].to_numpy()
    return pd.DatetimeIndex([d[k] if k < len(d) else pd.NaT for k in i])


def first_per_company(ev, days=DEDUP_DAYS):
    """One event per company per `days` calendar days: the first, then the next one at least `days` later."""
    ev = ev.sort_values(["cik", "accepted_et", "accession"])
    keep, last = [], {}
    for c, t in zip(ev["cik"], ev["accepted_et"]):
        ok = c not in last or (t - last[c]) >= pd.Timedelta(days=days)
        keep.append(ok)
        if ok:
            last[c] = t
    return ev[keep]


def penny_rows(cache):
    """W1 universe rows with raw close under $5 (W1's universe is L1's: close >= $1, dv20 >= $500k)."""
    parts = [pd.read_parquet(p, columns=["symbol", "date", "c", "dv20", "cik"],
                             filters=[("uni", "==", True), ("c", "<", L1.PENNY_MAX)]) for p in D.batch_files(cache)]
    u = pd.concat([p for p in parts if len(p)], ignore_index=True).dropna(subset=["cik"])
    return u.astype({"cik": "int64"})


def stage_filings(cache, ciks):
    d = pd1dir(cache) / "subs_et"            # "subs" held UTC times as ET (hand check, 2026-10-30)
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), BATCH):
        f = d / f"e_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + BATCH] for r in earnings_rows(c)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=COLS).to_parquet(tmp)
        tmp.replace(f)
        print(f"[pd1 filings] batch {b // BATCH + 1}/{-(-len(ciks) // BATCH)}: {len(rows)} earnings 8-Ks")
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("e_*.parquet"))], ignore_index=True)


def events(filings, pen, cal):
    """Dedup per company, day 0, then the penny universe on the session before day 0 (one row per company
    per date, highest dollar volume)."""
    ev = first_per_company(filings.drop_duplicates("accession"))
    ev = ev.assign(day0=day0(ev["accepted_et"], cal)).dropna(subset=["day0"])
    days = cal["date"].reset_index(drop=True)
    pos = days.searchsorted(ev["day0"])
    ev["prior_day"] = [days[p - 1] if p > 0 else pd.NaT for p in pos]
    u = pen.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])
    u = u.rename(columns={"date": "prior_day", "c": "prior_close"})
    return ev.merge(u, on=["cik", "prior_day"], how="inner").sort_values(["day0", "cik"]).reset_index(drop=True)


def stage_events(cache, out):
    cal = D.calendar()
    pen = penny_rows(cache)
    ciks = sorted(pen["cik"].unique().tolist())
    fil = stage_filings(cache, ciks)
    ev = events(fil, pen, cal)
    tmp = pd1dir(cache) / "events.tmp"
    ev.to_parquet(tmp)
    tmp.replace(pd1dir(cache) / "events.parquet")
    out.mkdir(parents=True, exist_ok=True)
    sig = ev[ev["day0"] >= pd.Timestamp("2017-01-01")]
    chk_file = CHECK
    if not chk_file.exists():
        k = np.sort(np.random.default_rng(SEED).choice(len(sig), size=min(CHECK_N, len(sig)), replace=False))
        chk_file.parent.mkdir(parents=True, exist_ok=True)
        sig.iloc[k][["cik", "accession", "symbol", "accepted_et", "items", "day0"]].assign(
            accepted_et=lambda x: x["accepted_et"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            day0=lambda x: x["day0"].dt.strftime("%Y-%m-%d"), ok="", note="").to_csv(chk_file, index=False)
    cov = {"penny_companies": len(ciks), "earnings_8k": int(len(fil)), "events_in_universe": int(len(ev)),
           "events_from_2017": int(len(sig)),
           "per_year": {str(y): int(n) for y, n in ev["day0"].dt.year.value_counts().sort_index().items()},
           "after_close_share": float((ev["accepted_et"].dt.hour >= 16).mean()) if len(ev) else None}
    (out / "PD1_coverage.json").write_text(json.dumps(cov, indent=1))
    print("PD1", json.dumps(cov))


# ---------- measure ----------

def windows(ev, cal):
    """Session dates per event: prior_day (the close before day 0), day1, entry (day +2) and the exit of
    each hold (the close of the hold's last session, the entry day counted as the first)."""
    days = cal["date"].reset_index(drop=True)
    pos = days.searchsorted(ev["day0"])
    at = lambda k: [days[i] if i < len(days) else pd.NaT for i in k]
    out = ev.assign(day1=at(pos + 1), entry=at(pos + 2))
    for h in HOLDS:
        out[f"exit{h}"] = at(pos + 1 + h)
    return out


def trade_ret(g, entry, exit_day):
    """All-adjusted open of `entry` to the last close on or before `exit_day` in the same segment (a series
    that ends early exits at its last close). NaN without a bar on the entry day."""
    if entry not in g.index:
        return np.nan
    e = g.loc[entry]
    x = g[(g.index >= entry) & (g.index <= exit_day) & (g["seg"] == e["seg"])]
    return float(x["ac"].iloc[-1] / e["ao"] - 1)


def reaction(g, prior_day, day1):
    if prior_day not in g.index or day1 not in g.index or g.loc[prior_day, "seg"] != g.loc[day1, "seg"]:
        return np.nan
    return float(g.loc[day1, "ac"] / g.loc[prior_day, "ac"] - 1)


def signal(ev):
    """'top' / 'bottom' / None per event from the reactions of events with an earlier day 0 in the 365
    calendar days before (at least MIN_PRIOR of them)."""
    ev = ev.sort_values(["day0", "cik"]).reset_index(drop=True)
    d, x = ev["day0"].to_numpy(), ev["xr"].to_numpy(float)
    out = []
    for i in range(len(ev)):
        lo = np.searchsorted(d, d[i] - np.timedelta64(365, "D"), side="left")
        hi = np.searchsorted(d, d[i], side="left")
        prior = x[lo:hi][np.isfinite(x[lo:hi])]
        if len(prior) < MIN_PRIOR or not np.isfinite(x[i]):
            out.append(None)
        elif x[i] >= np.quantile(prior, TOP):
            out.append("top")
        elif x[i] <= np.quantile(prior, BOTTOM):
            out.append("bottom")
        else:
            out.append(None)
    return ev.assign(side=out)


def control_candidates(picks, pen, filings):
    """Per pick, up to N_CAND (symbol) candidates from the penny universe on its prior_day, with no item 2.02
    8-K whose day 0 is within CTRL_DAYS calendar days of the pick's, drawn in the order of one permutation
    per day 0 (default_rng(SEED + day-0 ordinal)); a day's k-th pick takes candidates k*N_CAND onward, so
    two picks on one day never share a control."""
    u = pen.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])
    by_day = {d: g for d, g in u.groupby("date")}
    f_day0 = filings.groupby("cik")["day0"].apply(lambda s: np.sort(s.dropna().to_numpy()))
    out = {}
    for d0, g in picks.groupby("day0", sort=True):
        prior = g["prior_day"].iloc[0]
        pool = by_day.get(prior)
        if pool is None:
            continue
        lo, hi = np.datetime64(d0 - pd.Timedelta(days=CTRL_DAYS)), np.datetime64(d0 + pd.Timedelta(days=CTRL_DAYS))
        ok = [not ((c in f_day0.index) and ((f_day0[c] >= lo) & (f_day0[c] <= hi)).any()) for c in pool["cik"]]
        pool = pool[ok]
        perm = pool.iloc[np.random.default_rng(SEED + d0.toordinal()).permutation(len(pool))]["symbol"].tolist()
        for k, i in enumerate(g.index):
            out[i] = perm[k * N_CAND:(k + 1) * N_CAND]
    return out


def per_batch(cache, need):
    """{symbol: rows indexed by date (ao, ac, seg)} for the symbols in `need`, one W1 batch file at a time,
    passed to the caller's generator so only one batch is in memory."""
    for p in D.batch_files(cache):
        r = pd.read_parquet(p, columns=["symbol", "date", "ao", "ac", "seg"], filters=[("symbol", "in", sorted(need))])
        if len(r):
            yield {s: g.set_index("date").sort_index() for s, g in r.groupby("symbol")}


def check_right():
    if not CHECK.exists():
        return None
    c = pd.read_csv(CHECK)
    ok = c["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def cohort_test(tr, ff5, umd, hold=HOLD, side="top"):
    """Monthly entry cohorts: mean pick net minus mean control net; NW t (3 lags), month-block bootstrap,
    picks' own mean net, FF5 + momentum alpha on the factors summed over the entry month and the next two."""
    t = tr[(tr["side"] == side) & tr[f"r{hold}"].notna() & tr[f"c{hold}"].notna()].copy()
    sign = 1 if side == "top" else -1
    fee = COST + (BORROW * hold / 252 if side == "bottom" else 0)
    t["net"] = sign * t[f"r{hold}"] - fee
    t["cnet"] = sign * t[f"c{hold}"] - fee
    t["m"] = t["entry"].dt.to_period("M")
    c = t.groupby("m").agg(pick=("net", "mean"), ctrl=("cnet", "mean"), n=("net", "size")).sort_index()
    c["diff"] = c["pick"] - c["ctrl"]
    m, se, tt = nw_tstat(c["diff"], LAGS)
    lo, hi = block_bootstrap_ci(c["diff"], 3, 10_000, SEED)
    pm, _, pt = nw_tstat(c["pick"], LAGS)
    f = ff5.join(umd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umd.columns[0]]]
    f3 = f.rolling(3).sum().shift(-2).reindex(c.index)
    alpha = stats.ols_hac(c["diff"].to_numpy(float), f3.to_numpy(), LAGS)
    alpha["factors"] = list(f.columns)
    res = {"events": int(len(t)), "months": int(len(c)), "mean_minus_control": m, "se_hac": se, "t_hac": tt,
           "ci95": [lo, hi], "picks_mean_net": pm, "picks_t": pt, "control_mean_net": float(c["ctrl"].mean()),
           "ff5_mom_alpha": alpha}
    return res, c


def stage_measure(cache, out):
    n_ok = check_right()
    if n_ok is None:
        raise SystemExit("[pd1] fill in research/checks/PD1.csv (ok = y/n) first")
    res = {"idea": "PD1", "prereg": "research/prereg/PD1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "PD1_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    cal = D.calendar()
    ev = windows(pd.read_parquet(pd1dir(cache) / "events.parquet"), cal)
    ev = ev[ev[f"exit{HOLD}"].notna() & (ev[f"exit{HOLD}"] < HOLDOUT)].reset_index(drop=True)
    h = alpaca_headers()
    iwm = to_frames(fetch_daily_bars(["IWM"], "2015-06-01", alpaca.sip_end("2025-10-15"), adjustment="all", headers=h))["IWM"]
    iwm_c = iwm["c"]
    ev["iwm"] = iwm_c.reindex(ev["day1"]).to_numpy() / iwm_c.reindex(ev["prior_day"]).to_numpy() - 1
    raw = np.full(len(ev), np.nan)
    sym_rows = {s: np.flatnonzero(ev["symbol"].to_numpy() == s) for s in ev["symbol"].unique()}
    for bars in per_batch(cache, set(ev["symbol"])):
        for s, g in bars.items():
            for i in sym_rows.get(s, []):
                raw[i] = reaction(g, ev.at[i, "prior_day"], ev.at[i, "day1"])
    ev["xr"] = raw - ev["iwm"].to_numpy()
    ev = signal(ev)
    picks = ev[ev["side"].notna() & (ev["day0"] >= SIGNAL_FIRST)].copy()
    fil = stage_filings(cache, [])  # the cached earnings 8-Ks
    fil = fil.assign(day0=day0(fil["accepted_et"], cal))
    cand = control_candidates(picks, penny_rows(cache), fil)
    need = set(picks["symbol"]) | {s for v in cand.values() for s in v}
    rets = {}
    for bars in per_batch(cache, need):
        for i, r in picks.iterrows():
            for s in [r["symbol"]] + cand.get(i, []):
                if s in bars:
                    for hh in HOLDS:
                        rets[(i, s, hh)] = trade_ret(bars[s], r["entry"], r[f"exit{hh}"])
    for hh in HOLDS:
        picks[f"r{hh}"] = [rets.get((i, s, hh), np.nan) for i, s in zip(picks.index, picks["symbol"])]
        ctrl, csym = [], []
        for i in picks.index:
            hit = next(((s, rets[(i, s, hh)]) for s in cand.get(i, []) if np.isfinite(rets.get((i, s, hh), np.nan))),
                       (None, np.nan))
            csym.append(hit[0])
            ctrl.append(hit[1])
        picks[f"c{hh}"] = ctrl
        if hh == HOLD:
            picks["control_symbol"] = csym
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    prim, coh = cohort_test(picks, ff5, umd)
    a = prim["ff5_mom_alpha"]
    prim["passes_rule12"] = bool(prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0 and prim["picks_mean_net"] > 0
                                 and prim["events"] >= MIN_EVENTS)
    prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
    prim["deflated_sharpe"] = stats.deflated_sharpe(coh["diff"], TRIALS)
    prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
    prim["near_miss_rule19"] = bool(not prim["passes"] and prim["t_hac"] >= 1.5 and prim["picks_mean_net"] > 0)
    p2, both = stats.idea_counts()
    top = picks[picks["side"] == "top"]
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"events": int(len(ev)), "with_reaction": int(ev["xr"].notna().sum()),
                       "signal_events_from_2017": int((ev["day0"] >= SIGNAL_FIRST).sum()),
                       "top": int(len(top)), "bottom": int((picks["side"] == "bottom").sum()),
                       "top_with_trade": int(top[f"r{HOLD}"].notna().sum()),
                       "top_with_control": int(top[f"c{HOLD}"].notna().sum())},
            "primary": prim}
    ex = {f"hold_{hh}": cohort_test(picks, ff5, umd, hold=hh)[0] for hh in HOLDS if hh != HOLD}
    ex["short_bottom_60"] = cohort_test(picks, ff5, umd, side="bottom")[0]
    t2 = top.assign(net=top[f"r{HOLD}"] - COST, cnet=top[f"c{HOLD}"] - COST).dropna(subset=["net", "cnet"])
    ex["by_year"] = {str(y): {"n": int(len(g)), "minus_control": float((g["net"] - g["cnet"]).mean())}
                     for y, g in t2.groupby(t2["entry"].dt.year)}
    band = np.where(t2["prior_close"] < 2, "1-2", "2-5")
    ex["by_price_band"] = {b: {"n": int(len(g)), "minus_control": float((g["net"] - g["cnet"]).mean())}
                           for b, g in t2.groupby(band)}
    res["exploratory"] = ex
    (out / "PD1_result.json").write_text(json.dumps(res, indent=1, default=str))
    picks[["cik", "symbol", "accession", "day0", "entry", "side", "xr", "control_symbol"] +
          [f"{k}{hh}" for hh in HOLDS for k in ("r", "c")]].to_csv(out / "PD1_trades.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/PD1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    (stage_events if a.stage == "events" else stage_measure)(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
