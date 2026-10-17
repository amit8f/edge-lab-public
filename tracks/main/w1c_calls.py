"""W1c (research/prereg/W1.md): unusual short-dated out-of-the-money call buying with no news predicts the
stock over the next 10 trading days. Alpaca option bars (OPRA), 2024-02-01 .. 2025-09-15, main track.

Stages (each resumable; run in order):
1. data: the shared W1 daily bars, filings and headline months (tracks/news/w1_data.py).
2. probe: checks that Alpaca lists expired contracts and has their daily bars; writes W1c_probe.json.
   If not, W1c is logged "no data" and the later stages refuse to run.
3. options: per underlying (main-track stocks), every call and put expiring 2024-02-01 .. 2025-11-15 and
   their daily bars, reduced to one row per underlying-day; 50 underlyings per batch file.
4. report: signals, trades, controls and the primary test, plus the pre-deal call-buying replication
   (clarification 23).

Usage: python -m tracks.main.w1c_calls --cache <dir> --out <dir> [--stage data|probe|options|report]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.env import alpaca_headers
from tracks.main import w1b_volume as B
from tracks.news import l1_study as L1
from tracks.news import w1_data as D

SEED = 20261017
FIRST, LAST = pd.Timestamp("2024-02-01"), pd.Timestamp("2025-09-15")
EXP_LAST = "2025-11-15"
DTE, OTM, RATIO, MIN_CALLS, CALL_PUT, MEAN_DAYS, CTRL_RATIO = 45, 1.05, 5.0, 1000, 3.0, 20, 1.5
HOLD, QUIET, COST, MIN_EVENTS, LAGS, BATCH = 10, 3, 0.002, 150, 5, 50
CONTRACTS_URL = f"{alpaca.TRADE}/options/contracts"
OPT_BARS_URL = "https://data.alpaca.markets/v1beta1/options/bars"
OPT_QUOTES_URL = "https://data.alpaca.markets/v1beta1/options/quotes"
PRE_DAYS = 5


# ------------------------------------------------------------------ pure rules (tested offline)

def day_agg(contracts, bars, prior_close):
    """One underlying's per-day option volumes: short-dated OTM calls (expiring within 45 calendar days
    after the day, strike >= 1.05 x the prior raw close), all calls and all puts in the same expiries.
    contracts: symbol, type, strike, expiration; bars: symbol, date, v; prior_close: Series by date."""
    b = bars.merge(contracts, on="symbol", how="inner")
    b["dte"] = (b["expiration"] - b["date"]).dt.days
    b = b[(b["dte"] > 0) & (b["dte"] <= DTE)]
    b["pc"] = prior_close.reindex(pd.DatetimeIndex(b["date"])).to_numpy()
    b = b.dropna(subset=["pc"])
    call, put = b["type"] == "call", b["type"] == "put"
    otm = call & (b["strike"] >= OTM * b["pc"])
    g = pd.DataFrame({"date": b["date"], "call_otm": b["v"].where(otm, 0), "calls": b["v"].where(call, 0),
                      "puts": b["v"].where(put, 0)}).groupby("date").sum()
    return g.reset_index()


def add_ratio(agg, cal):
    """Call-OTM volume over its own mean of the previous 20 trading days (days with no option volume
    count as 0 from the first day the underlying had any)."""
    days = pd.DatetimeIndex(cal)
    a = agg.set_index("date").reindex(days[(days >= agg["date"].min()) & (days <= agg["date"].max())], fill_value=0)
    a["mean20"] = a["call_otm"].shift(1).rolling(MEAN_DAYS, min_periods=MEAN_DAYS).mean()
    a["ratio"] = a["call_otm"] / a["mean20"].replace(0, np.nan)
    return a.rename_axis("date").reset_index()


def unusual(a):
    return (a["ratio"] >= RATIO) & (a["call_otm"] >= MIN_CALLS) & (a["call_otm"] >= CALL_PUT * a["puts"])


def one_per_window(sig, days, hold=HOLD):
    """One event per stock per 10 trading days: a signal is dropped if the same stock had a kept signal in
    the previous 10 trading days."""
    days = pd.DatetimeIndex(sorted(days))
    keep, last = [], {}
    for i, (s, d) in enumerate(zip(sig["symbol"], sig["date"])):
        k = days.searchsorted(d)
        if s in last and k - last[s] < hold:
            keep.append(False)
            continue
        last[s] = k
        keep.append(True)
    return sig[keep]


# ------------------------------------------------------------------ stages

def pre_deal_test(targets, unusual_days, days, rng, pre=PRE_DAYS):
    """Clarification 23 (the Augustin, Brenner & Subrahmanyam pattern): share of merger targets with an
    unusual call day (news filter not applied) in the `pre` trading days before the announcement, against
    one random earlier `pre`-day window of the same stock that ends at least 20 trading days before it and
    lies inside the option data; one-sided Fisher exact test."""
    from scipy.stats import fisher_exact
    by = {s: set(g["date"]) for s, g in unusual_days.groupby("symbol")}
    have = {s: (g["date"].min(), g["date"].max()) for s, g in unusual_days.groupby("symbol")}
    lo = days.searchsorted(FIRST)
    hit, ctrl = [], []
    for t in targets.itertuples(index=False):
        i = days.searchsorted(pd.Timestamp(t.date))
        if i - 20 - pre < lo:
            continue
        s = by.get(t.symbol, set())
        win = lambda end: any(d in s for d in days[end - pre:end])  # noqa: E731
        hit.append(win(i))
        ctrl.append(win(int(rng.integers(lo + pre, i - 20 + 1))))
    n = len(hit)
    if not n:
        return {"targets": 0}
    a, b = sum(hit), sum(ctrl)
    p = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")[1]
    return {"targets": n, "share_before": a / n, "share_control_window": b / n, "fisher_p": float(p),
            "reproduces": bool(a >= 2 * max(b, 1) and p < 0.05), "symbols_with_unusual_days": len(have)}


def contracts_for(underlying, headers, status):
    rows, tok = [], None
    p = {"underlying_symbols": underlying, "status": status, "expiration_date_gte": f"{FIRST:%Y-%m-%d}",
         "expiration_date_lte": EXP_LAST, "limit": 10000}
    while True:
        if tok:
            p["page_token"] = tok
        j = alpaca.get(CONTRACTS_URL, p, headers=headers)
        rows += j.get("option_contracts") or []
        tok = j.get("next_page_token")
        time.sleep(0.31)
        if not tok:
            return rows


def option_bars(symbols, headers):
    out = []
    for i in range(0, len(symbols), 100):
        p, tok = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": "1Day", "start": f"{FIRST:%Y-%m-%d}",
                  "end": f"{LAST + pd.Timedelta(days=1):%Y-%m-%d}", "limit": 10000}, None
        while True:
            if tok:
                p["page_token"] = tok
            j = alpaca.get(OPT_BARS_URL, p, headers=headers)
            for s, rows in (j.get("bars") or {}).items():
                out += [(s, r["t"], r["v"]) for r in rows]
            tok = j.get("next_page_token")
            time.sleep(0.31)
            if not tok:
                break
    b = pd.DataFrame(out, columns=["symbol", "date", "v"])
    b["date"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert(None).dt.normalize()
    return b


def contracts_frame(rows):
    c = pd.DataFrame(rows)
    if c.empty:
        return pd.DataFrame(columns=["symbol", "type", "strike", "expiration"])
    c = c.rename(columns={"strike_price": "strike", "expiration_date": "expiration"})
    c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
    c["expiration"] = pd.to_datetime(c["expiration"])
    return c[["symbol", "type", "strike", "expiration"]].drop_duplicates("symbol")


def stage_probe(out):
    h = alpaca_headers()
    rows = contracts_for("AAPL", h, "inactive")
    c = contracts_frame(rows)
    old = c[c["expiration"] < pd.Timestamp("2024-06-30")]
    res = {"inactive_contracts_listed": int(len(c)), "expired_before_2024_07": int(len(old))}
    if len(old):
        b = option_bars(old["symbol"].head(20).tolist(), h)
        res["bars_rows_for_20_expired"] = int(len(b))
    res["usable"] = bool(res.get("bars_rows_for_20_expired", 0) > 0)
    if len(old):                      # clarification 24: the IV smirk needs historical option quotes
        try:
            q = alpaca.get(OPT_QUOTES_URL, {"symbols": old["symbol"].iloc[0], "limit": 10,
                                            "start": f"{old['expiration'].iloc[0] - pd.Timedelta(days=10):%Y-%m-%d}"}, headers=h)
            res["historical_quotes_rows"] = int(sum(len(v) for v in (q.get("quotes") or {}).values()))
        except Exception as e:        # noqa: BLE001 (any refusal means no quotes history)
            res["historical_quotes_rows"], res["historical_quotes_error"] = 0, type(e).__name__
    out.mkdir(parents=True, exist_ok=True)
    (out / "W1c_probe.json").write_text(json.dumps(res, indent=1))
    print(res)
    return res["usable"]


def probe_ok(out):
    f = out / "W1c_probe.json"
    return f.exists() and json.loads(f.read_text()).get("usable", False)


def main_underlyings(cache):
    """Symbols with at least one main-track universe day (raw close >= $5) in the window."""
    f = [("date", ">=", FIRST), ("date", "<=", LAST), ("uni", "==", True), ("c", ">=", L1.PENNY_MAX)]
    return sorted(set(D.read_rows(cache, ["symbol"], f)["symbol"]))


def stage_options(cache, out, budget_min):
    if not probe_ok(out):
        print("[options] probe not passed (W1c_probe.json): W1c has no data and does not run")
        return False
    deadline = time.monotonic() + budget_min * 60
    h = alpaca_headers()
    cal = D.calendar()
    odir = D.w1dir(cache) / "w1c_opt"
    odir.mkdir(parents=True, exist_ok=True)
    syms = main_underlyings(cache)
    for b in range(0, len(syms), BATCH):
        f = odir / f"o_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            print("[options] time budget reached; rerun to continue")
            return False
        part = syms[b:b + BATCH]
        raw = D.read_rows(cache, ["symbol", "date", "c"], [("symbol", "in", part), ("date", ">=", FIRST - pd.Timedelta(days=60))])
        aggs = []
        for s in part:
            c = contracts_frame(contracts_for(s, h, "inactive") + contracts_for(s, h, "active"))
            if c.empty:
                continue
            bars = option_bars(c["symbol"].tolist(), h)
            if bars.empty:
                continue
            px = raw[raw["symbol"] == s].set_index("date")["c"].sort_index()
            prior = px.shift(1).reindex(pd.DatetimeIndex(cal["date"])).ffill(limit=5)   # last raw close before each day
            a = add_ratio(day_agg(c, bars, prior), cal["date"])
            aggs.append(a.assign(symbol=s))
        df = pd.concat(aggs, ignore_index=True) if aggs else pd.DataFrame(
            columns=["date", "call_otm", "calls", "puts", "mean20", "ratio", "symbol"])
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(f)
        print(f"[options] batch {b // BATCH + 1}/{-(-len(syms) // BATCH)}")
    return True


def stage_report(cache, out):
    if not probe_ok(out):
        print("[report] W1c has no data")
        return
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    a = pd.concat([pd.read_parquet(p) for p in sorted((D.w1dir(cache) / "w1c_opt").glob("o_*.parquet"))], ignore_index=True)
    a = a[(a["date"] >= FIRST) & (a["date"] <= LAST)]
    d = D.read_rows(cache, ["symbol", "date", "c", "ao", "ac", "uni", "seg", "cik"],
                    [("symbol", "in", sorted(a["symbol"].unique())), ("date", ">=", FIRST - pd.Timedelta(days=10))])
    d = d.sort_values(["symbol", "date"])
    g = d.groupby(["symbol", "seg"])
    d["prev_c"], d["prev_uni"] = g["c"].shift(1), g["uni"].shift(1)
    d["entry_date"], d["entry_px"] = g["date"].shift(-1), g["ao"].shift(-1)
    d["exit_date"], d["exit_px"] = g["date"].shift(-HOLD), g["ac"].shift(-HOLD)
    m = a.merge(d, on=["symbol", "date"], how="inner")
    m = m[(m["prev_uni"] == True) & (m["prev_c"] >= L1.PENNY_MAX)].dropna(subset=["entry_px", "exit_px", "cik"])  # noqa: E712
    spy = B.spy_daily()
    m["r"] = m["exit_px"] / m["entry_px"] - 1
    m["r_spy"] = B.spy_returns(spy, m["entry_date"], m["exit_date"])
    m["net"] = m["r"] - m["r_spy"] - COST
    sig = m[unusual(m)].copy()
    sig = sig.assign(cik=sig["cik"].astype("int64"))
    q = D.quiet_mask(sig[["symbol", "cik", "date"]].reset_index(drop=True),
                     D.news_symbol_days(cache), D.load_filings(cache, ["cik", "filed"]).rename(columns={"filed": "date"})[["cik", "date"]],
                     QUIET, days)
    sig = one_per_window(sig[q].sort_values(["date", "symbol"]), days)
    pool = m[m["ratio"] < CTRL_RATIO]
    ctrl = []
    for dt, gs in sig.groupby("date"):
        rng = np.random.default_rng(SEED + pd.Timestamp(dt).toordinal())
        p = pool[pool["date"] == dt].sort_values("symbol")
        for e in gs.itertuples(index=False):
            pp = p[p["symbol"] != e.symbol]
            if len(pp):
                ctrl.append(pp.iloc[int(rng.integers(len(pp)))][["date", "symbol", "net"]])
    ctrl = pd.DataFrame(ctrl)
    p2, both = stats.idea_counts()
    prim = B.stats_block(sig)
    if prim.get("dates", 0) > 2:
        per_day = sig.groupby("date")["net"].mean().sort_index()
        al = B.factor_alpha(per_day, cal)
        prim.update(factor_alpha=al, deflated_sharpe=stats.deflated_sharpe(per_day.to_numpy(), both),
                    passes=bool(prim["n"] >= MIN_EVENTS) and stats.screen_pass(prim, al))
    fil = D.load_filings(cache, ["cik", "form", "filed", "items"])
    fil["items"] = fil["items"].map(lambda x: list(x) if x is not None else [])
    ann = B.announcements(fil)
    ann = ann[ann["category"].isin(["agreement_8k", "activist_13d"])]

    def followed(rows):
        if rows.empty:
            return float("nan")
        j = rows[["cik", "date"]].astype({"cik": "int64"}).merge(ann[["cik", "date"]].rename(columns={"date": "a"}), on="cik")
        j = j[(j["a"] > j["date"]) & (j["a"] <= j["date"] + pd.Timedelta(days=30))]
        return float(j.drop_duplicates(["cik", "date"]).shape[0] / len(rows))

    ctrl_cik = ctrl.merge(m[["symbol", "date", "cik"]], on=["symbol", "date"], how="left") if len(ctrl) else ctrl
    res = {"idea": "W1c", "prereg": "research/prereg/W1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "underlying_days": int(len(m)), "unusual_days": int(unusual(m).sum()), "events": int(len(sig)),
           "primary": prim, "control": B.stats_block(ctrl) if len(ctrl) else None,
           "exploratory": {"share_followed_by_8k_or_13d_30d": followed(sig),
                           "control_share": followed(ctrl_cik.dropna(subset=["cik"])) if len(ctrl) else None}}
    tg = B.announcements(fil)
    tg = tg[(tg["category"] == "merger_target") & (tg["date"] >= FIRST) & (tg["date"] <= D.FILE_LAST)]
    tg = tg.merge(m[["cik", "symbol"]].drop_duplicates("cik").astype({"cik": "int64"}), on="cik")
    ud = m[unusual(m)][["symbol", "date"]]
    res["replication_augustin"] = pre_deal_test(tg, ud, days, np.random.default_rng(SEED))
    res["passes"] = bool(prim.get("passes", False))
    out.mkdir(parents=True, exist_ok=True)
    (out / "W1c_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("events", "passes")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="probe", choices=["data", "probe", "options", "report"])
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    if a.stage == "data":
        ok = D.run(cache, time.monotonic() + a.budget_min * 60)
        print("[data] done" if ok else "[data] unfinished; rerun to continue")
    elif a.stage == "probe":
        stage_probe(out)
    elif a.stage == "options":
        stage_options(cache, out, a.budget_min)
    elif a.stage == "report":
        stage_report(cache, out)


if __name__ == "__main__":
    main()
