"""W1b (research/prereg/W1.md): unexplained heavy trading, main and penny. A stock-day with volume z >= 3,
a close near the high (buying) or the low (selling), a modest same-direction return, and no Benzinga
article or EDGAR filing from t-3 to t; long buying names and short selling names from the next open to
the close of the 10th trading day, after costs, against same-day random controls.

Stages (each resumable; run in order):
1. data: the shared W1 daily bars, filings and headline months (tracks/news/w1_data.py).
2. signals: signal days and their trades, batch by batch (no API calls).
3. report: the primary tests and controls per price group and direction; the SEC-case check if
   W1b_sec_cases.csv is filled; the footprint and FINRA tables if their stages have run.
4. proxy: writes W1b_proxy_sample.csv (100 random DEFM14A filings) for Claude to fill with the
   "Background of the Merger" dates.
5. footprints: SIP trades and quotes on trading days -10 .. -1 before sampled announcements and controls.
6. finra: FINRA OTC Transparency weekly rows per ATS and per non-ATS firm for the same events.

Usage: python -m tracks.main.w1b_volume --cache <dir> --out <dir> [--stage data|signals|report|proxy|footprints|finra]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.costs import QUOTES_URL
from core.env import alpaca_headers
from core.shortdata import UA
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import l1_study as L1
from tracks.news import w1_data as D

ET = D.ET
SEED, N_BOOT = 20261017, 10_000
FIRST, LAST = pd.Timestamp("2016-03-01"), pd.Timestamp("2025-09-15")
Z_MIN, Z_CTRL, LOOK, HOLD, QUIET = 3.0, 1.0, 60, 10, 3
CL_BUY, CL_SELL, RET_MAX = 0.7, 0.3, 0.10
MIN_EVENTS, LAGS = 200, 5
COST = {"main": 0.002, "penny": 0.02}
BORROW = L1.BORROW
FOOT_PER_CAT, FOOT_DV_MAX, BLOCK, FOOT_DAYS = 400, 5e7, 10_000, 10
SEC_CTRL_LO, SEC_CTRL_HI = 250, 500
EMPTY = pd.DataFrame({"symbol": pd.Series(dtype=str), "cik": pd.Series(dtype="int64"),
                      "date": pd.Series(dtype="datetime64[ns]"), "kind": pd.Series(dtype=str),
                      "group": pd.Series(dtype=str), "side": pd.Series(dtype="int64"),
                      "ctrl_side": pd.Series(dtype="int64"), "r": pd.Series(dtype=float),
                      "r_spy": pd.Series(dtype=float), "held": pd.Series(dtype="int64")})
TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
WHOLESALERS = r"(?i)citadel|virtu|jane street|g1 execution|two sigma|ubs securities|de minimis"


# ------------------------------------------------------------------ pure rules (tested offline)

def signal_frame(d):
    """Per-day volume z, close location, day return and signal side for one symbol's daily rows (sorted by
    date, one segment at a time: nothing is computed across a >15-day gap)."""
    out = []
    for _, g in d.sort_values(["symbol", "date"]).groupby(["symbol", "seg"], sort=False):
        g = g.copy()
        lv = np.log(g["v"].astype(float))
        mu = lv.shift(1).rolling(LOOK, min_periods=LOOK).mean()
        sd = lv.shift(1).rolling(LOOK, min_periods=LOOK).std()
        g["z"] = (lv - mu) / sd
        rng_ = (g["h"] - g["l"]).replace(0, np.nan)
        g["cl"] = (g["c"] - g["l"]) / rng_
        g["ret"] = g["ac"] / g["ac"].shift(1) - 1
        buy = (g["z"] >= Z_MIN) & (g["cl"] >= CL_BUY) & (g["ret"] >= 0) & (g["ret"] <= RET_MAX)
        sell = (g["z"] >= Z_MIN) & (g["cl"] <= CL_SELL) & (g["ret"] <= 0) & (g["ret"] >= -RET_MAX)
        g["side"] = np.where(buy, 1, np.where(sell, -1, 0))
        # trade: next open to the close of the HOLD-th trading day after the signal, inside the segment
        g["entry_date"], g["entry_px"] = g["date"].shift(-1), g["ao"].shift(-1)
        g["exit_date"], g["exit_px"] = g["date"].shift(-HOLD), g["ac"].shift(-HOLD)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else d.iloc[:0]


def spy_returns(spy, entry, exit_):
    """SPY all-adjusted open of each entry date to the close of its exit date (spy indexed by date with
    ao, ac); NaN where SPY has no bar."""
    return (spy["ac"].reindex(pd.DatetimeIndex(exit_)).to_numpy() / spy["ao"].reindex(pd.DatetimeIndex(entry)).to_numpy()) - 1


def net_return(side, r, r_spy, group, days_held):
    borrow = BORROW[group] * days_held / 365 if side < 0 else 0.0
    return side * (r - r_spy) - COST[group] - borrow


def sign_mid(trades, quotes):
    """+1 buy / -1 sell / 0 unsigned per trade by the prevailing NBBO midpoint (the quote at or before the
    trade, within 1 second); tick test (price vs the previous different price) only where no quote is near."""
    t = trades.sort_values("t").reset_index(drop=True)
    q = quotes[(quotes["bp"] > 0) & (quotes["ap"] > quotes["bp"])].sort_values("t")
    m = pd.merge_asof(t, q[["t", "bp", "ap"]], on="t", direction="backward", tolerance=pd.Timedelta(seconds=1))
    mid = (m["bp"] + m["ap"]) / 2
    s = np.sign(m["p"] - mid)
    tick = np.sign(m["p"].diff().replace(0, np.nan).ffill()).fillna(0)
    return pd.Series(np.where(mid.notna(), s.fillna(0), tick), index=t.index).astype(int), t


def subpenny_retail(p, x):
    """BJZZ identification: an off-exchange print (venue D) whose price has a sub-penny fraction other
    than 0 or 0.5 cents."""
    frac = np.round((np.asarray(p, float) * 100) % 1, 4)
    return (np.asarray(x) == "D") & ~np.isclose(frac, 0) & ~np.isclose(frac, 1) & ~np.isclose(frac, 0.5)


def footprint_day(trades, quotes):
    """One stock-day's footprint: off-exchange share, block share (10,000+ shares), midpoint buy-sell
    imbalance, retail share and retail imbalance (all by shares)."""
    if trades.empty:
        return None
    side, t = sign_mid(trades, quotes)
    vol = t["s"].sum()
    off = t["x"] == "D"
    ret = subpenny_retail(t["p"], t["x"])
    signed = (side * t["s"]).sum()
    signed_n = t["s"][side != 0].sum()
    r_signed = (side[ret] * t["s"][ret]).sum()
    r_n = t["s"][ret & (side != 0)].sum()
    return {"shares": float(vol), "off_share": float(t["s"][off].sum() / vol), "block_share": float(t["s"][t["s"] >= BLOCK].sum() / vol),
            "imbalance": float(signed / signed_n) if signed_n else float("nan"),
            "retail_share": float(t["s"][ret].sum() / vol), "retail_imbalance": float(r_signed / r_n) if r_n else float("nan"),
            "unsigned_share": float(t["s"][side == 0].sum() / vol)}


def recall_test(cases, sig_days, cal, rng):
    """Amendment 7: share of SEC cases with a signal day in the 10 trading days before the announcement,
    against (a) the same stocks' mean rate over every 10-day window ending 250-500 trading days earlier
    and (b) one random such window per case, used for the one-sided Fisher exact test."""
    from scipy.stats import fisher_exact
    days = pd.DatetimeIndex(sorted(cal))
    by = {s: set(g["date"]) for s, g in sig_days.groupby("symbol")}
    hit, ctrl_rate, ctrl_hit = [], [], []
    for c in cases.itertuples(index=False):
        i = days.searchsorted(pd.Timestamp(c.announce_date))
        if i - SEC_CTRL_HI - FOOT_DAYS < 0:
            continue
        s = by.get(c.symbol, set())
        win = lambda end: any(d in s for d in days[end - FOOT_DAYS:end])  # noqa: E731
        hit.append(win(i))
        ends = range(i - SEC_CTRL_HI, i - SEC_CTRL_LO + 1)
        ctrl_rate.append(np.mean([win(e) for e in ends]))
        ctrl_hit.append(win(int(rng.integers(i - SEC_CTRL_HI, i - SEC_CTRL_LO + 1))))
    n = len(hit)
    if not n:
        return {"cases": 0}
    a, b = sum(hit), sum(ctrl_hit)
    p = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")[1]
    recall, rate = a / n, float(np.mean(ctrl_rate))
    return {"cases": n, "recall": recall, "control_rate_all_windows": rate, "control_hits_random_window": b,
            "fisher_p": float(p), "validated": bool(recall >= 2 * rate and p < 0.05)}


def finra_shares(rows):
    """Per week: total OTC shares, the ATS share and the named wholesalers' share of a stock's OTC volume,
    from FINRA weekly firm-level rows (summaryTypeCode names ATS vs non-ATS OTC)."""
    if not rows:
        return pd.DataFrame(columns=["week", "otc_shares", "ats_share", "wholesaler_share"])
    f = pd.DataFrame(rows)
    f["week"] = pd.to_datetime(f["weekStartDate"])
    f["q"] = pd.to_numeric(f["totalWeeklyShareQuantity"], errors="coerce").fillna(0)
    f["ats"] = f["summaryTypeCode"].astype(str).str.contains("ATS")
    f["whole"] = ~f["ats"] & f.get("marketParticipantName", pd.Series("", index=f.index)).astype(str).str.contains(WHOLESALERS)
    f["q_ats"], f["q_whole"] = f["q"].where(f["ats"], 0), f["q"].where(f["whole"], 0)
    g = f.groupby("week")[["q", "q_ats", "q_whole"]].sum()
    return pd.DataFrame({"week": g.index, "otc_shares": g["q"].to_numpy(), "ats_share": (g["q_ats"] / g["q"]).to_numpy(),
                         "wholesaler_share": (g["q_whole"] / g["q"]).to_numpy()})


def stats_block(x, col="net"):
    ok = x.dropna(subset=[col])
    per_day = ok.groupby("date")[col].mean().sort_index()
    if len(per_day) < 3:
        return {"n": int(len(ok)), "dates": int(len(per_day))}
    mean, se, t = nw_tstat(per_day.to_numpy(), LAGS)
    ci = block_bootstrap_ci(per_day.to_numpy(), 5, N_BOOT, SEED) if len(per_day) > 5 else (float("nan"),) * 2
    return {"n": int(len(ok)), "dates": int(len(per_day)), "mean": mean, "se_hac": se, "t_hac": t, "ci95": list(ci),
            "hit_rate": float((ok[col] > 0).mean())}


# ------------------------------------------------------------------ stages

def spy_daily():
    fr = to_frames(fetch_daily_bars(["SPY"], D.BARS_START, alpaca.sip_end(D.BARS_END), adjustment="all",
                                    headers=alpaca_headers()))
    return fr["SPY"].rename(columns={"o": "ao", "c": "ac"})[["ao", "ac"]]


def stage_signals(cache):
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    spy = spy_daily()
    news_days = D.news_symbol_days(cache)
    fil = D.load_filings(cache, ["cik", "filed"]).rename(columns={"filed": "date"})[["cik", "date"]].drop_duplicates()
    sdir = D.w1dir(cache) / "w1b_signals"
    sdir.mkdir(parents=True, exist_ok=True)
    for p in D.batch_files(cache):
        f = sdir / p.name
        if f.exists():
            continue
        d = pd.read_parquet(p)
        fr = signal_frame(d) if len(d) else d
        fr = fr[(fr["date"] >= FIRST) & (fr["date"] <= LAST) & fr["uni"] & fr["cik"].notna()] if len(fr) else fr
        cand = fr[(fr["side"] != 0) | ((fr["z"] < Z_CTRL) & ((fr["cl"] >= CL_BUY) | (fr["cl"] <= CL_SELL)))] if len(fr) else fr
        if len(cand):
            cand = cand.assign(cik=cand["cik"].astype("int64"))
            nd = news_days[news_days["symbol"].isin(set(cand["symbol"]))]
            fd = fil[fil["cik"].isin(set(cand["cik"]))]
            sig = cand[cand["side"] != 0]
            q = D.quiet_mask(sig[["symbol", "cik", "date"]].reset_index(drop=True), nd, fd, QUIET, days)
            sig = sig[q]
            ctrl = cand[(cand["side"] == 0)].assign(ctrl_side=np.where(cand.loc[cand["side"] == 0, "cl"] >= CL_BUY, 1, -1))
            keep = ["symbol", "cik", "date", "c", "dv20", "z", "cl", "ret", "side", "entry_date", "entry_px",
                    "exit_date", "exit_px"]
            rows = pd.concat([sig[keep].assign(kind="signal", ctrl_side=0), ctrl[keep + ["ctrl_side"]].assign(kind="pool")],
                             ignore_index=True)
            rows = rows.dropna(subset=["entry_px", "exit_px"])
            rows["group"] = np.where(rows["c"] < L1.PENNY_MAX, "penny", "main")
            rows["r"] = rows["exit_px"] / rows["entry_px"] - 1
            rows["r_spy"] = spy_returns(spy, rows["entry_date"], rows["exit_date"])
            rows["held"] = (rows["exit_date"] - rows["entry_date"]).dt.days
        else:
            rows = EMPTY.copy()
        tmp = f.with_suffix(".tmp")
        rows.to_parquet(tmp)
        tmp.replace(f)
        print(f"[w1b signals] {p.name}: {int((rows.get('kind', pd.Series(dtype=str)) == 'signal').sum())} signals")


def load_signals(cache, kind, year=None, columns=None):
    f = [("kind", "==", kind)]
    if year is not None:
        f += [("date", ">=", pd.Timestamp(f"{year}-01-01")), ("date", "<", pd.Timestamp(f"{year + 1}-01-01"))]
    parts = [pd.read_parquet(p, columns=columns, filters=f) for p in sorted((D.w1dir(cache) / "w1b_signals").glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def controls(sig, pool):
    """One same-day random control per signal: same price group, z < 1, the same close-location side,
    a different stock; default_rng(SEED + date ordinal), signals taken in symbol order within a date."""
    out = []
    if pool.empty:
        return pd.DataFrame(columns=["date", "symbol", "group", "side", "net"])
    pools = {k: g.sort_values("symbol") for k, g in pool.groupby(["date", "group", "ctrl_side"])}
    for d, g in sig.sort_values(["date", "symbol"]).groupby("date"):
        rng = np.random.default_rng(SEED + pd.Timestamp(d).toordinal())
        for e in g.itertuples(index=False):
            p = pools.get((d, e.group, e.side))
            if p is None:
                continue
            p = p[p["symbol"] != e.symbol]
            if len(p):
                c = p.iloc[int(rng.integers(len(p)))]
                out.append({"date": d, "symbol": c["symbol"], "group": e.group, "side": e.side,
                            "net": net_return(e.side, c["r"], c["r_spy"], e.group, c["held"])})
    return pd.DataFrame(out)


def factor_alpha(per_day, cal):
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fac = ffd.join(umdd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]]
    days = pd.DatetimeIndex(cal["date"])
    wins = [days[days.searchsorted(pd.Timestamp(d)) + 1:days.searchsorted(pd.Timestamp(d)) + 1 + HOLD] for d in per_day.index]
    a = stats.ols_hac(per_day.to_numpy(), stats.window_factors(fac, wins), LAGS)
    a["factors"] = list(fac.columns) + [f"(summed over each trade's {HOLD} days)"]
    return a


def stage_report(cache, out):
    cal = D.calendar()
    sig = load_signals(cache, "signal")
    sig["net"] = [net_return(s, r, rs, g, h) for s, r, rs, g, h in zip(sig["side"], sig["r"], sig["r_spy"], sig["group"], sig["held"])]
    pcols = ["symbol", "date", "group", "ctrl_side", "r", "r_spy", "held"]
    ctrl = pd.concat([controls(sig[sig["date"].dt.year == y], load_signals(cache, "pool", y, pcols))
                      for y in sorted(sig["date"].dt.year.unique())], ignore_index=True)
    p2, both = stats.idea_counts()
    res = {"idea": "W1b", "prereg": "research/prereg/W1.md", "phase2_count": p2, "phase1_plus_phase2": both, "cells": {}}
    for (g, s), x in sig.groupby(["group", "side"]):
        k = f"{g}/{'buying_long' if s > 0 else 'selling_short'}"
        b = stats_block(x)
        if b.get("dates", 0) > 2:
            per_day = x.groupby("date")["net"].mean().sort_index()
            a = factor_alpha(per_day, cal)
            b.update(factor_alpha=a, deflated_sharpe=stats.deflated_sharpe(per_day.to_numpy(), both),
                     passes=bool(b["n"] >= MIN_EVENTS) and stats.screen_pass(b, a))
        b["control"] = stats_block(ctrl[(ctrl["group"] == g) & (ctrl["side"] == s)]) if len(ctrl) else None
        res["cells"][k] = b
    cases_f = out / "W1b_sec_cases.csv"
    if cases_f.exists():
        cases = pd.read_csv(cases_f, dtype=str).dropna(subset=["symbol", "announce_date"])
        res["sec_case_check"] = recall_test(cases, sig[["symbol", "date"]], cal["date"], np.random.default_rng(SEED))
    else:
        res["sec_case_check"] = "W1b_sec_cases.csv not filled yet: results are unvalidated"
    validated = isinstance(res["sec_case_check"], dict) and res["sec_case_check"].get("validated", False)
    res["label"] = "validated detector" if validated else "unvalidated detector (amendment 7)"
    fp = D.w1dir(cache) / "w1b_foot"
    if fp.exists():
        res["footprints"] = footprint_table(fp)
    res["passes_any"] = [k for k, v in res["cells"].items() if v.get("passes")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "W1b_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("label", "passes_any")}, indent=1))


def footprint_table(fp):
    """KS tests of each footprint measure, events vs controls, per category (descriptive)."""
    from scipy.stats import ks_2samp
    rows = [json.loads(p.read_text()) for p in fp.glob("*.json")]
    days = pd.DataFrame([dict(cat=r["category"], window=r.get("window", "pre"), **d) for r in rows for d in r.get("days", [])])
    if days.empty:
        return None
    out = {}
    ctrl = days[days["cat"] == "control"]
    for cat in sorted(set(days["cat"]) - {"control"}):
        x = days[days["cat"] == cat]
        out[cat] = {}
        for m in ("off_share", "block_share", "imbalance", "retail_share", "retail_imbalance"):
            a, b = x[m].dropna(), ctrl[m].dropna()
            if len(a) > 5 and len(b) > 5:
                ks = ks_2samp(a, b)
                out[cat][m] = {"event_median": float(a.median()), "control_median": float(b.median()),
                               "ks": float(ks.statistic), "p": float(ks.pvalue), "n_event_days": int(len(a))}
    out["note"] = ("Retail prints use the BJZZ sub-penny rule, signed by the quote midpoint. Barber et al. (2023) found "
                   "that rule identifies about 35% of retail trades; midpoint signing had about 5% sign errors.")
    return out


def announcements(filings):
    """(cik, date, category) for footprint sampling: 8-K item 1.01/2.01 agreements, SC 13D originals, and
    merger targets (DEFM14A, SC TO-T, SC 14D9) dated by the company's first 8-K item 1.01 in the 180 days
    before, else by the filing itself."""
    f = filings.copy()
    items = f["items"].map(lambda x: set(x) if x is not None else set())
    is8 = f["form"] == "8-K"
    a = f[is8 & items.map(lambda s: bool(s & {"1.01", "2.01"}))].assign(category="agreement_8k")
    b = f[f["form"].isin(["SC 13D", "SCHEDULE 13D"])].assign(category="activist_13d")
    tgt = f[f["form"].isin(["DEFM14A", "SC TO-T", "SC 14D9"])].sort_values("filed").drop_duplicates("cik")
    ann = []
    k8 = a.groupby("cik")["filed"].apply(lambda s: s.sort_values().to_numpy())
    for r in tgt.itertuples(index=False):
        prior = k8.get(r.cik, np.array([], dtype="datetime64[ns]"))
        prior = prior[(prior <= np.datetime64(r.filed)) & (prior >= np.datetime64(r.filed - pd.Timedelta(days=180)))]
        ann.append({"cik": r.cik, "filed": pd.Timestamp(prior[0]) if len(prior) else r.filed, "category": "merger_target"})
    c = pd.DataFrame(ann, columns=["cik", "filed", "category"])
    allx = pd.concat([a[["cik", "filed", "category"]], b[["cik", "filed", "category"]], c], ignore_index=True)
    return allx[(allx["filed"] >= FIRST) & (allx["filed"] <= LAST)].rename(columns={"filed": "date"})


def stage_proxy(cache, out):
    f = out / "W1b_proxy_sample.csv"
    if f.exists():
        return
    fil = D.load_filings(cache, ["cik", "accession", "form", "filed"])
    p = fil[(fil["form"] == "DEFM14A") & (fil["filed"] >= FIRST) & (fil["filed"] <= LAST)].sort_values("accession")
    rng = np.random.default_rng(SEED)
    s = p.iloc[np.sort(rng.choice(len(p), size=min(100, len(p)), replace=False))].copy()
    s["url"] = [f"https://www.sec.gov/Archives/edgar/data/{c}/{a.replace('-', '')}/" for c, a in zip(s["cik"], s["accession"])]
    out.mkdir(parents=True, exist_ok=True)
    s.assign(first_contact="", nda_date="", note="").to_csv(f, index=False)


def foot_sample(cache):
    """Up to 400 announcements per category plus 400 random control stock-days, among universe rows of
    the prior trading day with dv20 under $50M (clarification 14), default_rng(SEED)."""
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    fil = D.load_filings(cache, ["cik", "form", "filed", "items"])
    fil["items"] = fil["items"].map(lambda x: list(x) if x is not None else [])
    ann = announcements(fil)
    ann["prior_day"] = days[np.maximum(days.searchsorted(ann["date"]) - 1, 0)]
    parts = []
    for y in sorted(ann["prior_day"].dt.year.unique()):
        u = D.universe_year(cache, y).rename(columns={"date": "prior_day"}).dropna(subset=["cik"])
        u = u[u["dv20"] < FOOT_DV_MAX].astype({"cik": "int64"}).sort_values("dv20").drop_duplicates(["cik", "prior_day"], keep="last")
        parts.append(ann[ann["prior_day"].dt.year == y].merge(u, on=["cik", "prior_day"], how="inner"))
    ev = pd.concat(parts, ignore_index=True).sort_values(["category", "date", "symbol"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    out = []
    for cat in ("agreement_8k", "activist_13d", "merger_target"):
        g = ev[ev["category"] == cat]
        if len(g):
            out.append(g.iloc[np.sort(rng.choice(len(g), size=min(FOOT_PER_CAT, len(g)), replace=False))])
    pool_days = days[(days >= FIRST) & (days <= LAST)]
    ctrl, years = [], {}
    while len(ctrl) < FOOT_PER_CAT:
        d = pool_days[int(rng.integers(len(pool_days)))]
        if d.year not in years:
            years[d.year] = D.universe_year(cache, d.year)
        u = years[d.year]
        u = u[(u["date"] == d) & (u["dv20"] < FOOT_DV_MAX)].sort_values("symbol")
        if len(u):
            r = u.iloc[int(rng.integers(len(u)))]
            nxt = days[days.searchsorted(d) + 1]
            ctrl.append({"category": "control", "date": nxt, "symbol": r["symbol"], "cik": r["cik"], "prior_day": d})
    out.append(pd.DataFrame(ctrl))
    return pd.concat(out, ignore_index=True)


def fetch_ticks(url, key, symbol, start, end, headers, max_pages=40):
    rows, tok, pages = [], None, 0
    p = {"symbols": symbol, "start": pd.Timestamp(start, tz=ET).tz_convert("UTC").isoformat(),
         "end": pd.Timestamp(end, tz=ET).tz_convert("UTC").isoformat(), "feed": "sip", "limit": 10000}
    while True:
        if tok:
            p["page_token"] = tok
        j = alpaca.get(url, p, headers=headers)
        rows += (j.get(key) or {}).get(symbol, [])
        tok, pages = j.get("next_page_token"), pages + 1
        time.sleep(0.31)
        if not tok:
            return rows
        if pages >= max_pages:
            return None


def stage_footprints(cache, budget_min):
    deadline = time.monotonic() + budget_min * 60
    h = alpaca_headers()
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    fdir = D.w1dir(cache) / "w1b_foot"
    fdir.mkdir(parents=True, exist_ok=True)
    sp = D.w1dir(cache) / "w1b_foot_sample.parquet"
    if not sp.exists():
        foot_sample(cache).to_parquet(sp)
    smp = pd.read_parquet(sp)
    for i, e in enumerate(smp.to_dict("records")):
        f = fdir / f"{i:05d}.json"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            print("[footprints] time budget reached; rerun to continue")
            return False
        j = days.searchsorted(pd.Timestamp(e["date"]))
        rec = {"category": e["category"], "symbol": e["symbol"], "date": str(e["date"]), "days": []}
        for d in days[max(j - FOOT_DAYS, 0):j]:
            o, c = L1.session_bounds(cal, d)
            tr = fetch_ticks(TRADES_URL, "trades", e["symbol"], o, c, h)
            qu = fetch_ticks(QUOTES_URL, "quotes", e["symbol"], o, c, h) if tr else None
            if not tr or qu is None:
                rec["days"].append({"day": str(d), "status": "no trades" if not tr else "too dense"})
                continue
            t = pd.DataFrame(tr)[["t", "x", "p", "s"]]
            t["t"] = pd.to_datetime(t["t"], utc=True)
            q = pd.DataFrame(qu)
            q = q[["t", "bp", "ap"]] if len(q) else pd.DataFrame(columns=["t", "bp", "ap"])
            q["t"] = pd.to_datetime(q["t"], utc=True)
            fpd = footprint_day(t, q)
            rec["days"].append({"day": str(d), **(fpd or {})})
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, default=str))
        tmp.replace(f)
    return True


def finra_rows(symbol, start, end, post=requests.post, page=5000):
    body = {"limit": page, "compareFilters": [{"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol}],
            "dateRangeFilters": [{"fieldName": "weekStartDate", "startDate": f"{start:%Y-%m-%d}", "endDate": f"{end:%Y-%m-%d}"}]}
    for attempt in range(5):
        r = post(FINRA_URL, json=body, headers={**UA, "Accept": "application/json"}, timeout=120)
        if r.status_code == 200:
            return r.json() or []
        if r.status_code == 204:
            return []
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FINRA weekly {symbol}: HTTP {r.status_code}")


def stage_finra(cache, budget_min):
    deadline = time.monotonic() + budget_min * 60
    smp = pd.read_parquet(D.w1dir(cache) / "w1b_foot_sample.parquet")
    fdir = D.w1dir(cache) / "w1b_finra"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(smp.to_dict("records")):
        f = fdir / f"{i:05d}.json"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        d = pd.Timestamp(e["date"])
        if d < pd.Timestamp("2016-10-01"):      # firm-level weekly data starts in mid-2016
            continue
        rows = finra_rows(e["symbol"], d - pd.Timedelta(weeks=27), d)
        sh = finra_shares(rows)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"category": e["category"], "symbol": e["symbol"], "date": str(d),
                                   "weeks": sh.to_dict("records")}, default=str))
        tmp.replace(f)
        time.sleep(1)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="signals", choices=["data", "signals", "report", "proxy", "footprints", "finra"])
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + a.budget_min * 60
    if a.stage == "data":
        print("[data] done" if D.run(cache, deadline) else "[data] unfinished; rerun to continue")
    elif a.stage == "signals":
        stage_signals(cache)
    elif a.stage == "report":
        stage_report(cache, out)
    elif a.stage == "proxy":
        stage_proxy(cache, out)
    elif a.stage == "footprints":
        stage_footprints(cache, a.budget_min)
    elif a.stage == "finra":
        stage_finra(cache, a.budget_min)


if __name__ == "__main__":
    main()
