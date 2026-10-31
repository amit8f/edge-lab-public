"""CH (research/prereg/CH.md): data for the chart-model arms, main and penny. Shared data only; the arms never
share a model, a cut or a scale.

Stage "candidates": from W1's daily cache (`tracks/news/w1_data.py`), each universe symbol-day 2016-01 ..
2025-09-23 that passes CH's download filter (volume >= 1.5x its 20-day average, |gap| >= 3%, a new 20-day
high or low, or a close within 2% of the 252-day high or low), with its track (main or penny), written per
month to <cache>/ch/cand/<YYYY-MM>.parquet. Gaps and 20/252-day levels use the all-adjusted series (raw
o/h/l scaled by ac/c), so a split day is not a 50% "gap"; price and dollar-volume floors use raw values.

Stage "context": SIC codes (EDGAR), SPY and SPDR sector ETF daily bars, VIX closes (FRED).

Stage "build": 2016's volume share curve, then month by month from 2017: Alpaca SIP 5-minute bars (feed=sip,
adjustment=raw), 04:00-20:00 ET, of each candidate symbol-day, its previous session, and SPY and the sector
ETFs; `tracks/main/ch_build.py` turns them into rows and image windows under <cache>/ch/rows and
<cache>/ch/img; the bars themselves are not kept. Resumes month by month.

Nothing here reads a return.

Usage: python -m tracks.main.ch_data --stage candidates|context|build --cache cache/w1 --budget-min 330
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.news import w1_data as D

ET = "America/New_York"
FIRST, LAST_ENTRY = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-09-23")
MAIN_PRICE, MAIN_DV, PENNY_LO, PENNY_DV = 5.0, 5e6, 1.0, 1e6
VOL_X, GAP, NEAR52 = 1.5, 0.03, 0.02
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
PER_REQ = 50
COLS = ["symbol", "date", "o", "h", "l", "c", "v", "ao", "ac", "dv20", "seg", "cik"]


def chdir(cache):
    return Path(cache) / "ch"


def _atomic(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


def flag_days(d):
    """One symbol's daily rows (sorted, one segment at a time) -> the same rows with prior-day levels, the
    download filter `cand` and the track. Everything a row uses is known by that day's close; the floors
    use the previous session's raw close and dv20."""
    d = d.sort_values("date").copy()
    f = d["ac"] / d["c"]
    ah, al, ao, av = d["h"] * f, d["l"] * f, d["o"] * f, d["v"] / f
    prev = lambda s: s.groupby(d["seg"]).shift(1)
    roll = lambda s, n, fn: s.groupby(d["seg"]).transform(lambda x: getattr(x.rolling(n, min_periods=n), fn)().shift(1))
    d["prev_c"], d["prev_dv20"], d["prev_ac"] = prev(d["c"]), prev(d["dv20"]), prev(d["ac"])
    d["vol20"] = roll(av, 20, "mean")
    d["hi20"], d["lo20"] = roll(ah, 20, "max"), roll(al, 20, "min")
    d["hi252"], d["lo252"] = roll(ah, 252, "max"), roll(al, 252, "min")
    d["gap"] = ao / d["prev_ac"] - 1
    d["cand"] = ((av >= VOL_X * d["vol20"]) | (d["gap"].abs() >= GAP) | (ah > d["hi20"]) | (al < d["lo20"])
                 | (d["ac"] >= (1 - NEAR52) * d["hi252"]) | (d["ac"] <= (1 + NEAR52) * d["lo252"]))
    d["track"] = np.where((d["prev_c"] >= MAIN_PRICE) & (d["prev_dv20"] >= MAIN_DV), "main",
                          np.where((d["prev_c"] >= PENNY_LO) & (d["prev_c"] < MAIN_PRICE) & (d["prev_dv20"] >= PENNY_DV),
                                   "penny", ""))
    return d


def stage_candidates(cache, out):
    keep = ["symbol", "date", "cik", "track", "prev_c", "prev_dv20", "prev_ac", "vol20", "hi20", "lo20", "hi252",
            "lo252", "gap", "c", "ac"]
    parts = []
    for p in D.batch_files(cache):
        rows = pd.read_parquet(p, columns=COLS)
        rows = rows[rows["date"] >= FIRST - pd.Timedelta(days=400)]
        fl = pd.concat([flag_days(g) for _, g in rows.groupby("symbol", sort=False)], ignore_index=True)
        fl = fl[fl["cand"] & (fl["track"] != "") & (fl["date"] >= FIRST) & (fl["date"] <= LAST_ENTRY)]
        parts.append(fl[keep])
        print(f"[ch cand] {p.name}: {len(fl)} candidate days")
    c = pd.concat(parts, ignore_index=True)
    c = c.dropna(subset=["cik"]).sort_values("prev_dv20", ascending=False).drop_duplicates(["date", "cik"])
    c = c.sort_values(["date", "symbol"]).reset_index(drop=True)
    for m, g in c.groupby(c["date"].dt.to_period("M")):
        _atomic(g.reset_index(drop=True), chdir(cache) / "cand" / f"{m}.parquet")
    per_day = c.groupby(["date", "track"]).size().unstack(fill_value=0)
    cov = {"candidate_days": int(len(c)), "by_track": {k: int(v) for k, v in c["track"].value_counts().items()},
           "per_day_median": {k: float(per_day[k].median()) for k in per_day},
           "per_day_max": {k: int(per_day[k].max()) for k in per_day},
           "per_year": {str(y): int(n) for y, n in c["date"].dt.year.value_counts().sort_index().items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "CH_candidates.json").write_text(json.dumps(cov, indent=1))
    print("CH", json.dumps(cov))


def day_bars(syms, day, headers):
    """5-minute SIP raw bars of `syms` on `day`, 04:00-20:00 ET, as rows (symbol, t in ET, o, h, l, c, v, vw)."""
    s = pd.Timestamp(f"{day:%Y-%m-%d} 04:00", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = pd.Timestamp(f"{day:%Y-%m-%d} 20:00", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, tok = [], None
    while True:
        p = {"symbols": ",".join(syms), "timeframe": "5Min", "start": s, "end": e, "feed": "sip",
             "adjustment": "raw", "limit": 10000}
        if tok:
            p["page_token"] = tok
        j = alpaca.get(BARS_URL, p, headers=headers)
        for sym, bars in (j.get("bars") or {}).items():
            rows += [(sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", np.nan)) for b in bars]
        tok = j.get("next_page_token")
        if not tok:
            break
    d = pd.DataFrame(rows, columns=["symbol", "t", "o", "h", "l", "c", "v", "vw"])
    if len(d):
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        d = d[d["v"] > 0]
    return d


def fetch_days(pairs, headers):
    """5-minute bars of (symbol, date) pairs, PER_REQ symbols a request."""
    parts = []
    for day, g in pairs.groupby("date"):
        syms = sorted(set(g["symbol"]))
        for k in range(0, len(syms), PER_REQ):
            b = day_bars(syms[k:k + PER_REQ], day, headers)
            if len(b):
                b["date"] = day
                parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "t", "o", "h", "l", "c", "v", "vw", "date"])


# ---------- market context ----------

ETFS = ["SPY", "XLK", "XLV", "XLF", "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLC", "XLRE"]
SECTORS = [  # SPDR sector of a SIC code (first match); anything else uses SPY
    ("XLE", [(1300, 1399), (2900, 2999)]),
    ("XLU", [(4900, 4999)]),
    ("XLRE", [(6500, 6553), (6798, 6798)]),
    ("XLF", [(6000, 6799)]),
    ("XLV", [(2830, 2836), (3841, 3851), (8000, 8099)]),
    ("XLK", [(3570, 3579), (3661, 3679), (3812, 3812), (7370, 7379)]),
    ("XLC", [(2710, 2741), (4800, 4899), (7810, 7819)]),
    ("XLP", [(2000, 2199), (2840, 2844), (5400, 5499)]),
    ("XLB", [(1000, 1299), (1400, 1499), (2600, 2699), (2800, 2829), (2850, 2899), (3300, 3399)]),
    ("XLI", [(1500, 1799), (3400, 3569), (3580, 3599), (3700, 3709), (3720, 3799), (4000, 4799), (8700, 8799)]),
    ("XLY", [(2300, 2399), (2500, 2599), (3710, 3719), (3900, 3999), (5000, 5399), (5500, 5999), (7000, 7099)]),
]
VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def sector_etf(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "SPY"
    for etf, ranges in SECTORS:
        if any(lo <= s <= hi for lo, hi in ranges):
            return etf
    return "SPY"


def stage_context(cache):
    """SIC codes of every candidate company (EDGAR), ETF daily bars (raw and all) and VIX closes (FRED)."""
    from core import industry
    from core.bars import fetch_daily_bars, to_frames
    import io
    import requests
    d = chdir(cache)
    ciks = set()
    for f in sorted((d / "cand").glob("*.parquet")):
        ciks |= set(pd.read_parquet(f, columns=["cik"])["cik"].astype(int))
    industry.sic_codes(ciks, d)
    h = alpaca_headers()
    for adj in ("raw", "all"):
        f = d / f"etf_{adj}.parquet"
        if f.exists():
            continue
        fr = to_frames(fetch_daily_bars(ETFS, "2015-06-01", alpaca.sip_end("2025-10-15"), adjustment=adj, headers=h))
        _atomic(pd.concat([x.assign(symbol=k) for k, x in fr.items()]).reset_index(names="date"), f)
    if not (d / "vix.csv").exists():
        r = requests.get(VIX_URL, timeout=60)
        r.raise_for_status()
        v = pd.read_csv(io.StringIO(r.text))
        v.columns = ["date", "vix"]
        v["vix"] = pd.to_numeric(v["vix"], errors="coerce")
        v.dropna().to_csv(d / "vix.csv", index=False)
    print(f"[ch context] {len(ciks)} companies, ETFs and VIX cached")


# ---------- build ----------

def load_daily(cache, syms, lo, hi):
    f = [("symbol", "in", sorted(syms)), ("date", ">=", lo), ("date", "<=", hi)]
    return D.read_rows(cache, ["symbol", "date", "o", "h", "l", "c", "v", "ac", "seg"], f)


def stage_curve(cache, deadline):
    """2016's regular-session volume share curve per track (inputs only; 2016 has no entries)."""
    from tracks.main import ch_build as B
    d = chdir(cache)
    if (d / "curve.json").exists():
        return True
    h = alpaca_headers()
    for f in sorted((d / "cand").glob("2016-*.parquet")):
        dest = d / "curve_parts" / f.name
        if dest.exists():
            continue
        if time.monotonic() > deadline:
            return False
        c = pd.read_parquet(f, columns=["symbol", "date", "track"])
        b = fetch_days(c[["symbol", "date"]], h)
        b = b[(b["t"] >= b["date"] + B.OPEN) & (b["t"] < b["date"] + pd.Timedelta(hours=16))]
        b = b.assign(slot=((b["t"] - b["date"] - B.OPEN) / B.SLOT).astype(int))
        _atomic(b[["symbol", "date", "slot", "v"]].merge(c, on=["symbol", "date"]), dest)
        print(f"[ch curve] {f.stem}: {len(b)} bars")
    parts = pd.concat([pd.read_parquet(f) for f in sorted((d / "curve_parts").glob("*.parquet"))])
    cur = {t: B.share_curve(g).tolist() for t, g in parts.groupby("track")}
    tmp = d / "curve.tmp"
    tmp.write_text(json.dumps(cur))
    tmp.replace(d / "curve.json")
    return True


def build_month(cand, bars, daily, ctxd, curve, cal):
    """Rows (first trigger, inputs, labels) and image windows of one month's candidates."""
    from tracks.main import ch_build as B
    closes = dict(zip(cal["date"], cal["close"]))
    days = pd.DatetimeIndex(cal["date"])
    bars = bars.sort_values("t")
    by_sd = {k: g for k, g in bars.groupby(["symbol", "date"])}
    dly = {k: g.sort_values("date").reset_index(drop=True) for k, g in daily.groupby("symbol")}
    etf_raw, etf_all, vix, sic = ctxd["etf_raw"], ctxd["etf_all"], ctxd["vix"], ctxd["sic"]
    rows, intr, dimg = [], [], []
    for r in cand.itertuples():
        day = r.date
        close_t = pd.Timedelta(hours=int(closes[day][:2]), minutes=int(closes[day][3:5]))
        b = B.session(by_sd.get((r.symbol, day), pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw"])), day, close_t)
        if b.empty:
            continue
        f0 = r.ac / r.c
        ctx = {"prev_c": r.prev_ac / f0, "prev_dv20": r.prev_dv20, "vol20": r.vol20 * f0, "hi20": r.hi20 / f0,
               "lo20": r.lo20 / f0, "hi252": r.hi252 / f0, "lo252": r.lo252 / f0}
        i, trig = B.first_trigger(b, ctx, curve[r.track], r.track, close_t - pd.Timedelta(minutes=30))
        if i is None:
            continue
        s_end = b["end"].iloc[i]
        after = b[b["t"] >= s_end + B.LAG]
        if after.empty:
            continue
        g = dly.get(r.symbol)
        if g is None:
            continue
        seg = g.loc[g["date"] == day, "seg"]
        g = g[g["seg"] == seg.iloc[0]] if len(seg) else g
        k = g["ac"] / g["c"] / f0                     # today's raw basis
        base = g.assign(o=g["o"] * k, h=g["h"] * k, l=g["l"] * k, c=g["c"] * k, v=g["v"] / k)
        prior = base[base["date"] < day].tail(300)
        later = base[base["date"] > day].head(B.HOLD - 1)
        if len(prior) < 30:
            continue
        cik = int(r.cik) if pd.notna(r.cik) else -1
        mkt = market(day, s_end, by_sd, etf_raw, etf_all, vix, sector_etf(sic.get(cik)), days)
        A = B.atr(prior["h"], prior["l"], prior["c"])
        feat = B.features(b, i, prior[["o", "h", "l", "c", "v"]], ctx, mkt, curve[r.track], trig)
        lab = B.labels(after[["o", "h", "l", "c"]], float(after["o"].iloc[0]), A, later[["date", "h", "l", "c"]],
                       r.track, day)
        rows.append({"symbol": r.symbol, "date": day, "cik": cik, "track": r.track, "trig": trig,
                     "s": s_end, "entry_t": after["t"].iloc[0], "entry": float(after["o"].iloc[0]), "atr": A,
                     **feat, **lab})
        pi = days.searchsorted(day) - 1
        yb = by_sd.get((r.symbol, days[pi]) if pi >= 0 else None)
        ky = float(k[g["date"] == days[pi]].iloc[0]) if pi >= 0 and (g["date"] == days[pi]).any() else np.nan
        y = B.session(yb, days[pi], pd.Timedelta(hours=16)) if yb is not None else b.iloc[:0]
        win = np.full((156, 5), np.nan, np.float32)
        yy = y[["o", "h", "l", "c"]].to_numpy(float) * ky
        tt = b.iloc[: i + 1][["o", "h", "l", "c", "v"]].to_numpy(float)
        if len(y):
            win[: min(78, len(y)), :4] = yy[:78]
            win[: min(78, len(y)), 4] = y["v"].to_numpy(float)[:78] / ky
        win[78: 78 + min(78, len(tt))] = tt[:78]
        dw = np.full((B.DAILY_N, 5), np.nan, np.float32)
        pv = prior[["o", "h", "l", "c", "v"]].to_numpy(float)[-B.DAILY_N:]
        dw[B.DAILY_N - len(pv):] = pv
        intr.append(win)
        dimg.append(dw)
    out = pd.DataFrame(rows)
    return out, (np.stack(intr) if intr else np.zeros((0, 156, 5), np.float32)), \
        (np.stack(dimg) if dimg else np.zeros((0, 60, 5), np.float32))


def market(day, s_end, by_sd, etf_raw, etf_all, vix, etf, days):
    """Market inputs at s: SPY and the sector ETF today (5-minute bars vs yesterday's raw close), SPY's 5-day
    return to yesterday (all-adjusted), whether SPY is above its session VWAP, VIX's last close and 5-day change,
    and the day of week."""
    from tracks.main import ch_build as B
    out = {"dow": float(day.dayofweek)}
    for sym, key in (("SPY", "spy_ret_today"), (etf, "sector_ret_today")):
        b = by_sd.get((sym, day))
        pr = etf_raw[(etf_raw["symbol"] == sym) & (etf_raw["date"] < day)]
        if b is None or pr.empty:
            out[key] = np.nan
            continue
        ss = B.session(b, day, pd.Timedelta(hours=16))
        ss = ss[ss["end"] <= s_end]
        out[key] = float(ss["c"].iloc[-1] / pr["c"].iloc[-1] - 1) if len(ss) else np.nan
        if sym == "SPY" and len(ss):
            vw = (ss["vw"].fillna(ss["c"]) * ss["v"]).sum() / max(ss["v"].sum(), 1)
            out["spy_above_vwap"] = float(ss["c"].iloc[-1] > vw)
    if "sector_ret_today" in out and np.isnan(out["sector_ret_today"]):
        out["sector_ret_today"] = out.get("spy_ret_today", np.nan)
    sa = etf_all[(etf_all["symbol"] == "SPY") & (etf_all["date"] < day)]["c"].to_numpy()
    out["spy_ret5"] = float(sa[-1] / sa[-6] - 1) if len(sa) >= 6 else np.nan
    vv = vix[vix["date"] < day]["vix"].to_numpy()
    out["vix"] = float(vv[-1]) if len(vv) else np.nan
    out["vix_chg5"] = float(vv[-1] - vv[-6]) if len(vv) >= 6 else np.nan
    return out


def stage_build(cache, deadline):
    d = chdir(cache)
    if not stage_curve(cache, deadline):
        print("[ch build] time budget used in the 2016 curve; rerun to continue")
        return False
    curve = {k: np.array(v) for k, v in json.loads((d / "curve.json").read_text()).items()}
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    ctxd = {"etf_raw": pd.read_parquet(d / "etf_raw.parquet"), "etf_all": pd.read_parquet(d / "etf_all.parquet"),
            "vix": pd.read_csv(d / "vix.csv", parse_dates=["date"]),
            "sic": {int(k): v for k, v in json.loads((d / "sic.json").read_text()).items()}}
    h = alpaca_headers()
    for f in sorted((d / "cand").glob("*.parquet")):
        if f.stem < "2017-01":
            continue
        dest = d / "rows" / f.name
        if dest.exists():
            continue
        if time.monotonic() > deadline:
            print("[ch build] time budget used; rerun to continue")
            return False
        cand = pd.read_parquet(f)
        pi = days.searchsorted(cand["date"]) - 1
        prev = pd.DataFrame({"symbol": cand["symbol"], "date": days[np.maximum(pi, 0)]})
        mdays = pd.Series(sorted(cand["date"].unique()))
        etfs = pd.DataFrame([(e, x) for x in mdays for e in ETFS], columns=["symbol", "date"])
        bars = fetch_days(pd.concat([cand[["symbol", "date"]], prev, etfs]).drop_duplicates(), h)
        lo, hi = cand["date"].min() - pd.Timedelta(days=450), cand["date"].max() + pd.Timedelta(days=15)
        daily = load_daily(cache, set(cand["symbol"]), lo, hi)
        rows, intr, dimg = build_month(cand, bars, daily, ctxd, curve, cal)
        np.savez_compressed(d / "img" / f"{f.stem}.tmp.npz", intraday=intr, daily=dimg) \
            if (d / "img").mkdir(parents=True, exist_ok=True) is None else None
        (d / "img" / f"{f.stem}.tmp.npz").replace(d / "img" / f"{f.stem}.npz")
        _atomic(rows, dest)
        print(f"[ch build] {f.stem}: {len(cand)} candidates, {len(rows)} triggered, {len(bars)} bars fetched")
    return True


REBASE_TOL = 0.01


def stage_rebase(cache, out):
    """Bug fix (2026-10-31): until then build_month compared today's raw bars with yesterday's RAW close, so a
    split day showed a fake gap (a 2:1 split read as -50%). Deletes the rows and images of every month holding a
    candidate whose price basis moved more than 1% overnight (splits, large distributions), so stage=build
    rebuilds just those months (this stage runs it next); ordinary dividends move the basis by less and their rows stay. Writes counts."""
    d = chdir(cache)
    rep = {"tol": REBASE_TOL, "months": [], "affected_rows": 0, "affected_gap_rows": 0}
    for f in sorted((d / "rows").glob("*.parquet")):
        c = pd.read_parquet(d / "cand" / f.name, columns=["symbol", "date", "prev_c", "prev_ac", "c", "ac"])
        moved = c[((c["prev_ac"] / c["prev_c"]) / (c["ac"] / c["c"]) - 1).abs() > REBASE_TOL]
        if moved.empty:
            continue
        rows = pd.read_parquet(f, columns=["symbol", "date", "trig"]).merge(moved[["symbol", "date"]])
        rep["months"].append(f.stem)
        rep["affected_rows"] += len(rows)
        rep["affected_gap_rows"] += int((rows["trig"] == "gap").sum())
        f.unlink()
        (d / "img" / f"{f.stem}.npz").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CH_rebase.json").write_text(json.dumps(rep, indent=1))
    print(f"[ch rebase] {len(rep['months'])} months to rebuild, {rep['affected_rows']} rows "
          f"({rep['affected_gap_rows']} fired the gap trigger)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["candidates", "context", "build", "rebase"], required=True)
    ap.add_argument("--cache", default="cache/w1")
    ap.add_argument("--out", default="research/results/CH")
    ap.add_argument("--budget-min", type=float, default=330)
    a = ap.parse_args()
    deadline = time.monotonic() + 60 * a.budget_min
    if a.stage == "candidates":
        stage_candidates(a.cache, Path(a.out))
    elif a.stage == "context":
        stage_context(a.cache)
    else:
        if a.stage == "rebase":                   # then rebuild the deleted months in the same run
            stage_rebase(a.cache, Path(a.out))
        done = stage_build(a.cache, deadline)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        n = len(list((chdir(a.cache) / "rows").glob("*.parquet")))
        (Path(a.out) / "CH_build_progress.json").write_text(json.dumps({"months_done": n, "finished": bool(done)}))


if __name__ == "__main__":
    main()
