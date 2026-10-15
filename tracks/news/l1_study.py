"""L1 (research/prereg/L1.md): how long after EDGAR accepts a filing do the headline and the price move
come? Seven filing categories x two price groups, 2023-01-01 .. 2025-09-24.

Stages (each resumable from its cache; run in order with --stage all):
1. universe: Alpaca raw daily bars of every common stock named in SEC's insider data sets (DERA, which
   also gives the point-in-time ticker -> CIK map), reduced to (symbol, date, close, dv20) rows that pass
   the universe rule.
2. filings: each universe company's EDGAR submissions record (forms, filing dates, 8-K items).
3. events: the seven categories, the price group from the universe on the trading day before the filing
   date, and up to 400 random events per category and group.
4. measure: per event, the acceptance time from the filing's header, the first Benzinga headline naming
   the symbol, 1-minute SIP bars of the stock, SPY and a random control stock, and the returns.
5. report: the per-cell decision, lag percentiles, the control, the market-model CAR with the
   Kolari-Pynnonen correction (PROTOCOL rule 11) and 20 events per category for the hand check.

Usage: python -m tracks.news.l1_study --cache <dir> --out <dir> [--stage universe|filings|events|measure|report|all]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, leads_sec, sec, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

ET = "America/New_York"
FIRST, LAST = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-09-24")
BARS_START, BARS_END = "2022-10-01", "2025-09-30"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
SEED, PER_CELL, N_BOOT = 20261015, 400, 10_000
MIN_PRICE, MIN_DV20, PENNY_MAX, GAP_DAYS = 1.0, 5e5, 5.0, 15
LAG_MIN, T_MIN, N_MIN = 60.0, 3.0, 100
OFFER = {"S-1", "S-3", "424B1", "424B2", "424B3", "424B4", "424B5"}
CATEGORIES = [  # (name, sign); order fixes the sampling order
    ("offering", -1), ("late_filing", -1), ("delisting_notice", -1), ("auditor_change", -1),
    ("officer_departure", -1), ("activist_stake", +1), ("insider_buy", +1)]
SIGN = dict(CATEGORIES)
DEFAULT_COST = {"penny": 0.02, "main": 0.002}
BORROW = {"penny": 0.20, "main": 0.03}


# ------------------------------------------------------------------ pure rules (tested offline)

def category(form, items, accession, buys):
    """The L1 category of one filing, or None. items: list of 8-K item numbers; buys: set of accession
    numbers of original Form 4s with an officer/director open-market purchase (DERA)."""
    if form in OFFER:
        return "offering"
    if form in ("NT 10-K", "NT 10-Q"):
        return "late_filing"
    if form == "8-K":
        for it, cat in (("3.01", "delisting_notice"), ("4.01", "auditor_change"), ("5.02", "officer_departure")):
            if it in items:
                return cat
        return None
    if form in ("SC 13D", "SCHEDULE 13D"):
        return "activist_stake"
    if form == "4" and accession in buys:
        return "insider_buy"
    return None


def universe_rows(sym, d):
    """(symbol, date, close, dv20) rows of one symbol's raw daily bars where the stock passes the universe
    rule (dv20 = median raw dollar volume of the last 20 bars, never across a >15-day gap)."""
    seg = (d.index.to_series().diff().dt.days > GAP_DAYS).cumsum()
    dv = (d["c"] * d["v"]).groupby(seg.to_numpy()).transform(lambda x: x.rolling(20).median())
    ok = (d["c"] >= MIN_PRICE) & (dv >= MIN_DV20)
    return pd.DataFrame({"symbol": sym, "date": d.index[ok], "close": d["c"][ok].to_numpy(),
                         "dv20": dv[ok].to_numpy()})


def sample_cells(ev):
    """Up to PER_CELL events per (category, group), drawn with one default_rng(SEED) in CATEGORIES order,
    penny before main, each cell sorted by accession first."""
    rng = np.random.default_rng(SEED)
    out = []
    for cat, _ in CATEGORIES:
        for grp in ("penny", "main"):
            c = ev[(ev["category"] == cat) & (ev["group"] == grp)].sort_values("accession").reset_index(drop=True)
            if len(c):
                k = rng.choice(len(c), size=min(PER_CELL, len(c)), replace=False)
                out.append(c.iloc[np.sort(k)])
    return pd.concat(out, ignore_index=True) if out else ev.iloc[:0]


def session_bounds(cal, day):
    """(open, close) naive-ET timestamps of a trading day from the calendar (date, open, close)."""
    r = cal.loc[cal["date"] == day].iloc[0]
    return pd.Timestamp(f"{day:%Y-%m-%d} {r['open']}"), pd.Timestamp(f"{day:%Y-%m-%d} {r['close']}")


def entry_point(acc, cal):
    """(entry day, earliest bar time): the first regular-session minute opening at least 1 minute after
    acceptance; a filing accepted after the last minute of a session (or on a non-trading day) waits for
    the next open."""
    t = acc + pd.Timedelta(minutes=1)
    t = t.ceil("min")
    days = cal["date"]
    for day in days[days >= acc.normalize()].iloc[:6]:
        o, c = session_bounds(cal, day)
        if t <= c - pd.Timedelta(minutes=1):
            return day, max(t, o)
    return None, None


def first_at_or_after(bars, t, end):
    x = bars[(bars["t"] >= t) & (bars["t"] < end)]
    return x.iloc[0] if len(x) else None


def last_before(bars, end, start=None):
    x = bars[(bars["t"] + pd.Timedelta(minutes=1) <= end) & ((bars["t"] >= start) if start is not None else True)]
    return x.iloc[-1] if len(x) else None


def trade(bars, spy, cal, day, t0, headline_t=None):
    """Entry at the first bar at or after t0 on `day` (its open), exit at the last bar of the next trading
    day (its close); SPY over the same times. Returns dict(entry_t, exit_t, r, r_spy, r_pre, r_pre_spy) or
    None. r_pre runs from the entry to the last bar ending by the headline (None when the headline came
    first)."""
    o, c = session_bounds(cal, day)
    e = first_at_or_after(bars, t0, c)
    if e is None:
        return None
    i = cal.index[cal["date"] == day][0]
    nxt = cal["date"].iloc[cal.index.get_loc(i) + 1] if cal.index.get_loc(i) + 1 < len(cal) else None
    if nxt is None:
        return None
    o2, c2 = session_bounds(cal, nxt)
    x = last_before(bars, c2, o2)
    se, sx = first_at_or_after(spy, e["t"], c), last_before(spy, c2, o2)
    if x is None or se is None or sx is None:
        return None
    out = {"entry_t": e["t"], "exit_t": x["t"], "exit_day": nxt, "r": x["c"] / e["o"] - 1,
           "r_spy": sx["c"] / se["o"] - 1, "r_pre": None, "r_pre_spy": None}
    if headline_t is not None and headline_t > e["t"]:
        p, sp = last_before(bars, headline_t, e["t"]), last_before(spy, headline_t, e["t"])
        if p is not None and sp is not None:
            out["r_pre"], out["r_pre_spy"] = p["c"] / e["o"] - 1, sp["c"] / se["o"] - 1
    return out


def cost_of(group, close, dv20, c0=None):
    """Round-trip cost: C0's measured median (09:35 + 15:59 half-spreads) for the stock's bucket when C0's
    result exists, else the registered 0.2% / 2%."""
    if c0:
        b = "penny" if close < PENNY_MAX else "small" if dv20 < 1e7 else "mid" if dv20 < 5e7 else "large"
        v = (c0.get("buckets", {}).get(b) or {}).get("round_trip_0935_1559", {}).get("p50")
        if v is not None and math.isfinite(v):
            return float(v)
    return DEFAULT_COST[group]


def net(sign, r, r_spy, cost, group, days_held):
    borrow = BORROW[group] * days_held / 365 if sign < 0 else 0.0
    return sign * (r - r_spy) - cost - borrow


def cell_decision(m):
    """The registered decision for one cell's measured rows (lag_min with inf for no headline, net,
    entry_day)."""
    n = int(m["net"].notna().sum())
    lag = float(np.median(m["lag_min"])) if len(m) else float("nan")
    per_day = m.dropna(subset=["net"]).groupby("entry_day")["net"].mean().sort_index()
    mean, se, t = nw_tstat(per_day.to_numpy(), 2) if len(per_day) > 2 else (float("nan"),) * 3
    ci = block_bootstrap_ci(per_day.to_numpy(), 5, N_BOOT, SEED) if len(per_day) > 5 else (float("nan"),) * 2
    cand = bool(lag >= LAG_MIN and mean > 0 and t >= T_MIN and ci[0] > 0 and n >= N_MIN)
    return {"n": n, "median_lag_min": lag, "mean_net": mean, "t": t, "ci95": list(ci), "dates": int(len(per_day)),
            "candidate": cand}


# ------------------------------------------------------------------ data access

def minute_bars(symbol, start, end, headers):
    """1-minute SIP bars (adjustment=all) in [start, end] (naive ET) as (t naive ET, o, c)."""
    rows, tok = [], None
    s = pd.Timestamp(start, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = pd.Timestamp(end, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        p = {"symbols": symbol, "timeframe": "1Min", "start": s, "end": e, "feed": "sip", "adjustment": "all",
             "limit": 10000}
        if tok:
            p["page_token"] = tok
        j = alpaca.get(BARS_URL, p, headers=headers)
        rows += [(b["t"], b["o"], b["c"]) for b in (j.get("bars") or {}).get(symbol, [])]
        tok = j.get("next_page_token")
        time.sleep(0.3)
        if not tok:
            break
    d = pd.DataFrame(rows, columns=["t", "o", "c"])
    d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    return d.sort_values("t").reset_index(drop=True)


def first_headline(symbol, start, end, headers):
    """(created_at naive ET, id, headline) of the first Benzinga article naming `symbol` in [start, end]."""
    p = {"symbols": symbol, "start": pd.Timestamp(start, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end": pd.Timestamp(end, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "sort": "asc",
         "limit": 50, "include_content": "false"}
    j = alpaca.get(alpaca.NEWS_URL, p, headers=headers)
    time.sleep(0.3)
    for a in j.get("news") or []:
        if symbol in (a.get("symbols") or []):
            t = pd.Timestamp(a["created_at"]).tz_convert(ET).tz_localize(None)
            return t, a.get("id"), a.get("headline")
    return None, None, None


def submissions_rows(cik, get=sec.get):
    """(accession, form, filed, items) of one company's filings dated FIRST .. LAST, from the recent block
    and the older files of its submissions record."""
    base = "https://data.sec.gov/submissions/"
    r = get(f"{base}CIK{int(cik):010d}.json")
    if r is None:
        return []
    j = r.json()
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        if pd.Timestamp(f.get("filingTo", "1900-01-01")) >= FIRST:
            rr = get(base + f["name"])
            if rr is not None:
                blocks.append(rr.json())
    out = []
    for b in blocks:
        for acc, form, filed, items in zip(b.get("accessionNumber", []), b.get("form", []), b.get("filingDate", []),
                                           b.get("items", [""] * len(b.get("form", [])))):
            if FIRST <= pd.Timestamp(filed) <= LAST:
                out.append({"cik": int(cik), "accession": acc, "form": form, "filed": filed,
                            "items": [i.strip() for i in (items or "").split(",") if i.strip()]})
    return out


# ------------------------------------------------------------------ stages

def stage_universe(cache):
    h = alpaca_headers()
    tmap = sec_data.dera_ticker_map(2021, pd.Period("2025Q3"), cache / "dera")
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]) & set(tmap["symbol"]))
    (cache / "uni").mkdir(parents=True, exist_ok=True)
    for b in range(0, len(syms), 500):
        f = cache / "uni" / f"u_{b // 500:03d}.parquet"
        if f.exists():
            continue
        fr = to_frames(fetch_daily_bars(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END), adjustment="raw", headers=h))
        parts = [universe_rows(s, d) for s, d in fr.items()]
        u = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "date", "close", "dv20"])
        u["cik"] = sec_data.map_ciks(u[["symbol", "date"]], tmap) if len(u) else []
        tmp = f.with_suffix(".tmp")
        u.dropna(subset=["cik"]).astype({"cik": "int64"}).to_parquet(tmp)
        tmp.replace(f)
        print(f"[universe] batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(u)} rows")


def load_universe(cache):
    return pd.concat([pd.read_parquet(p) for p in sorted((cache / "uni").glob("u_*.parquet"))], ignore_index=True)


def stage_filings(cache):
    ciks = sorted(load_universe(cache)["cik"].unique())
    (cache / "subs").mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), 500):
        f = cache / "subs" / f"s_{b // 500:03d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + 500] for r in submissions_rows(c)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=["cik", "accession", "form", "filed", "items"]).to_parquet(tmp)
        tmp.replace(f)
        print(f"[filings] batch {b // 500 + 1}/{-(-len(ciks) // 500)}: {len(rows)} filings")


def stage_events(cache):
    subs = pd.concat([pd.read_parquet(p) for p in sorted((cache / "subs").glob("s_*.parquet"))], ignore_index=True)
    buys = set(sec_data.dera_insider_buys(2022, pd.Period("2025Q3"), cache / "dera")["accn"])
    subs["items"] = subs["items"].map(list)
    subs["category"] = [category(f, i, a, buys) for f, i, a in zip(subs["form"], subs["items"], subs["accession"])]
    ev = subs.dropna(subset=["category"]).drop_duplicates("accession").copy()
    ev["filed"] = pd.to_datetime(ev["filed"])
    cal = alpaca.trading_calendar(BARS_START, BARS_END)
    days = pd.DatetimeIndex(cal["date"])
    ev["prior_day"] = days[np.maximum(days.searchsorted(ev["filed"].values) - 1, 0)]
    u = load_universe(cache).sort_values("dv20", ascending=False).drop_duplicates(["cik", "date"])
    ev = ev.merge(u.rename(columns={"date": "prior_day"}), on=["cik", "prior_day"], how="inner")
    ev["group"] = np.where(ev["close"] < PENNY_MAX, "penny", "main")
    smp = sample_cells(ev)
    smp.to_parquet(cache / "events.parquet")
    counts = ev.groupby(["category", "group"]).size().rename("eligible").reset_index()
    counts.to_csv(cache / "event_counts.csv", index=False)
    print(counts.to_string(index=False))
    print(f"[events] sampled {len(smp)}")


def stage_measure(cache, budget_min=330):
    deadline = time.time() + budget_min * 60
    h = alpaca_headers()
    cal = alpaca.trading_calendar(BARS_START, "2025-10-15")
    cal["date"] = pd.to_datetime(cal["date"])
    ev = pd.read_parquet(cache / "events.parquet").sort_values(["filed", "accession"]).reset_index(drop=True)
    u = load_universe(cache)
    by_day = {d: g.sort_values("symbol") for d, g in u.groupby("date")}
    c0p = Path("research/results/C0/C0_costs.json")
    c0 = json.loads(c0p.read_text()) if c0p.exists() else None
    mdir = cache / "measure"
    mdir.mkdir(parents=True, exist_ok=True)
    spy_cache = {}
    done = {p.stem for p in mdir.glob("*.json")}
    rngs = {}
    for e in ev.to_dict("records"):
        key = e["accession"]
        if key in done:
            continue
        if time.time() > deadline:
            print("[measure] time budget reached; rerun to continue")
            return False
        _, acc, _ = leads_sec.header(e["cik"], key)
        rec = {"accession": key, "category": e["category"], "group": e["group"], "symbol": e["symbol"],
               "cik": int(e["cik"]), "filed": f"{e['filed']:%Y-%m-%d}", "accepted_et": acc, "close": e["close"],
               "dv20": e["dv20"]}
        if acc is None:
            rec["status"] = "no header"
        else:
            acc = pd.Timestamp(acc)
            day, t0 = entry_point(acc, cal)
            idx = cal.index[cal["date"] == day][0] if day is not None else None
            if day is None or idx + 4 >= len(cal):
                rec["status"] = "no entry day"
            else:
                end3 = session_bounds(cal, cal["date"].iloc[idx + 3])[1]
                ht, hid, hl = first_headline(e["symbol"], acc - pd.Timedelta(minutes=60), end3, h)
                rec.update(headline_t=None if ht is None else str(ht), headline_id=hid, headline=hl,
                           lag_min=None if ht is None else (ht - acc).total_seconds() / 60, entry_day=f"{day:%Y-%m-%d}")
                nxt_close = session_bounds(cal, cal["date"].iloc[idx + 1])[1]
                win = (session_bounds(cal, day)[0], nxt_close)
                if day not in spy_cache:
                    spy_cache[day] = minute_bars("SPY", *win, h)
                b = minute_bars(e["symbol"], *win, h)
                tr = trade(b, spy_cache[day], cal, day, t0, ht)
                cost = cost_of(e["group"], e["close"], e["dv20"], c0)
                if tr:
                    held = (tr["exit_day"] - day).days
                    rec.update(entry_t=str(tr["entry_t"]), exit_t=str(tr["exit_t"]), r=tr["r"], r_spy=tr["r_spy"],
                               r_pre=tr["r_pre"], r_pre_spy=tr["r_pre_spy"], cost=cost,
                               net=net(SIGN[e["category"]], tr["r"], tr["r_spy"], cost, e["group"], held))
                # control: same day's universe and group (prior trading day's row), its own rng per date
                pool = by_day.get(cal["date"].iloc[idx - 1], u.iloc[:0])
                pool = pool[((pool["close"] < PENNY_MAX) == (e["group"] == "penny")) & (pool["symbol"] != e["symbol"])]
                g = rngs.setdefault(day, np.random.default_rng(SEED + day.toordinal()))
                if len(pool):
                    cs = pool.iloc[int(g.integers(len(pool)))]
                    cb = minute_bars(cs["symbol"], *win, h)
                    ct = trade(cb, spy_cache[day], cal, day, t0)
                    rec["control_symbol"] = cs["symbol"]
                    if ct:
                        held = (ct["exit_day"] - day).days
                        ccost = cost_of(e["group"], cs["close"], cs["dv20"], c0)
                        rec["control_net"] = net(SIGN[e["category"]], ct["r"], ct["r_spy"], ccost, e["group"], held)
                rec["status"] = "ok" if tr else "no bars"
        tmp = mdir / f"{key}.tmp"
        tmp.write_text(json.dumps(rec, default=str))
        tmp.replace(mdir / f"{key}.json")
        done.add(key)
        if len(done) % 100 == 0:
            print(f"[measure] {len(done)}/{len(ev)} events")
    return True


def car_table(m, h):
    """Market-model CAR over [entry day, next day] with estimation days -250..-30, per cell, with the
    BMP t and its Kolari-Pynnonen correction (rule 11)."""
    ok = m[m["status"] == "ok"].copy()
    ok["entry_day"] = pd.to_datetime(ok["entry_day"])
    syms = sorted(set(ok["symbol"]) | {"SPY"})
    fr = to_frames(fetch_daily_bars(syms, "2021-12-01", alpaca.sip_end(BARS_END), adjustment="all", headers=h))
    spy = fr["SPY"]["c"].pct_change()
    out = {}
    for (cat, grp), g in ok.groupby(["category", "group"]):
        sc, res = [], []
        for e in g.itertuples(index=False):
            d = fr.get(e.symbol)
            if d is None:
                continue
            r = d["c"].pct_change().reindex(spy.index)
            if e.entry_day not in r.index:
                continue
            i = r.index.get_loc(e.entry_day)
            if i < 251:
                continue
            est, win = slice(i - 250, i - 29), slice(i, i + 2)
            ra, ma = r.to_numpy(), spy.to_numpy()
            if np.isfinite(ra[est]).sum() < 100:
                continue
            car, scar, rs = stats.market_model_car(ra, ma, est, win)
            sc.append(SIGN[cat] * scar)
            res.append(rs[-100:] if len(rs) >= 100 else None)
        res = [x for x in res if x is not None]
        if len(sc) > 2:
            out[f"{cat}/{grp}"] = stats.bmp_kp(sc, stats.mean_residual_corr(res) if len(res) > 1 else 0.0)
    return out


def stage_report(cache, out):
    rows = [json.loads(p.read_text()) for p in sorted((cache / "measure").glob("*.json"))]
    m = pd.DataFrame(rows)
    m["lag_min"] = pd.to_numeric(m.get("lag_min"), errors="coerce").fillna(np.inf)
    p2, both = stats.idea_counts()
    res = {"idea": "L1", "prereg": "research/prereg/L1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "cells": {}}
    for (cat, grp), g in m[m["status"].isin(["ok", "no bars"])].groupby(["category", "group"]):
        d = cell_decision(g)
        lag = g["lag_min"]
        d.update(lag_p25=float(lag.quantile(.25)), lag_p75=float(lag.quantile(.75)),
                 share_no_headline=float(np.isinf(lag).mean()),
                 mean_pre_move_signed=float((SIGN[cat] * (pd.to_numeric(g.get("r_pre"), errors="coerce")
                                                          - pd.to_numeric(g.get("r_pre_spy"), errors="coerce"))).mean()),
                 control_mean_net=float(pd.to_numeric(g.get("control_net"), errors="coerce").mean()))
        per_day = g.dropna(subset=["net"]).groupby("entry_day")["net"].mean()
        if len(per_day) > 2:
            d["deflated_sharpe"] = stats.deflated_sharpe(per_day.to_numpy(), both)
        res["cells"][f"{cat}/{grp}"] = d
    res["car_kp"] = car_table(m, alpaca_headers())
    res["status_counts"] = m["status"].value_counts().to_dict()
    res["candidates"] = [k for k, v in res["cells"].items() if v["candidate"]]
    out.mkdir(parents=True, exist_ok=True)
    chk = m[m["status"] == "ok"].groupby("category", group_keys=False).apply(
        lambda g: g.sample(n=min(20, len(g)), random_state=SEED))
    chk[["category", "group", "symbol", "cik", "accession", "accepted_et", "headline_t", "headline"]].assign(
        checked_by="", ok="", note="").to_csv(out / "L1_events_check.csv", index=False)
    (out / "L1_result.json").write_text(json.dumps(res, indent=1, default=str))
    m.to_csv(out / "L1_events.csv", index=False)
    print(json.dumps(res, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["universe", "filings", "events", "measure", "report", "all"])
    ap.add_argument("--budget-min", type=float, default=330)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("universe", "all"):
        stage_universe(cache)
    if st in ("filings", "all"):
        stage_filings(cache)
    if st in ("events", "all"):
        stage_events(cache)
    if st in ("measure", "all"):
        if not stage_measure(cache, a.budget_min) and st == "all":
            return
    if st in ("report", "all"):
        stage_report(cache, out)


if __name__ == "__main__":
    main()
