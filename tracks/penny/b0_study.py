"""B0 burst base rates (research/prereg/B0.md and its clarifications): what was visible at t's close
before a +30%-in-10-trading-days jump, against matched non-jumpers, per price group (penny $1-5 and
main $5+). B0 trades nothing.

Stages, run in order; each resumes from its cache (files are written atomically) and the long ones
stop starting new work once --budget-min has passed:
  bars      per ~500-symbol batch: daily universe rows (liquidity, volatility, price setup, forward
            maximum, minimum and return, first-burst flags) -> <cache>/rows/
  match     per year: first burst days and their 5 matched controls, per variant -> <cache>/keys/
  sources   FINRA short interest and Reg SHO short volume, SEC fails-to-deliver, the EDGAR form index,
            the DERA ticker map and insider buys, companyfacts share counts, Benzinga news -> <cache>/src/
  features  per year: features for matched rows and, in the test years, the whole universe -> <cache>/feat/
  report    fit on the development period, score the test period once -> <out>/B0_result.json

Usage: python -m tracks.penny.b0_study --stage bars|match|sources|features|report --cache <dir> --out <dir>
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, sec, shortdata, stats

BARS_START, BARS_END, CAL_END = "2014-12-01", "2025-09-30", "2025-12-31"
DEV = (pd.Timestamp("2016-01-04"), pd.Timestamp("2022-12-30"))
TEST = (pd.Timestamp("2023-01-03"), pd.Timestamp("2025-09-15"))
YEARS = range(2016, 2026)
SEED, N_BOOT, C, N_CTRL = 20261015, 2000, 1.0, 5
GAP_DAYS, SPACING, MIN_PRICE, MIN_DV = 15, 20, 1.0, 5e5
VARIANTS = {"b30_10": (0.30, 10), "b20_10": (0.20, 10), "b50_10": (0.50, 10), "b30_5": (0.30, 5)}
PRIMARY = "b30_10"
SI_LAG, SI_MAX_AGE = 8, 45              # trading days after settlement; calendar days at t
FTD_LAG, FTD_MAX_AGE = 30, 75           # calendar days after the half-month's end; at t
SHARES_MAX_AGE, CIK_MAX_AGE = 400, 730
OFFER_FORMS = {"S-1", "S-3", "424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "424B8"}
INDEX_FORMS = OFFER_FORMS | {"8-K"}
FEATURES = ["dtc", "shvr5", "ftd", "lockup", "insider90", "n8k20", "offer90", "ret20", "ret60", "dist52", "news5"]
LOG1P = {"dtc", "n8k20", "news5"}
BINARY = {"lockup", "insider90", "offer90"}
FIRST_MONTH, LAST_MONTH = "2015-11", "2025-09"
FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"


def atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def calendar():
    return alpaca.trading_calendar(BARS_START, CAL_END)


# ---------------------------------------------------------------- bars

def _forward(values, h, how):
    """Max or min of values[i+1 .. i+h] for each i (NaNs skipped; NaN when the window is empty)."""
    s = pd.Series(values[::-1])
    r = getattr(s.rolling(h, min_periods=1), how)().to_numpy()[::-1]
    return np.append(r[1:], np.nan)


def symbol_rows(sym, raw, adj, cal):
    """Universe rows for one symbol. raw/adj: frames indexed by date with o, h, l, c, v (zero-volume bars
    dropped); cal: DatetimeIndex of trading days. A >GAP_DAYS calendar gap ends a series; nothing is
    computed across it."""
    if raw is None or adj is None:
        return None
    idx = raw.index.intersection(adj.index)
    if len(idx) < 61:
        return None
    raw, adj = raw.loc[idx], adj.loc[idx]
    seg = (idx.to_series().diff().dt.days > GAP_DAYS).cumsum().to_numpy()
    out = []
    for s in np.unique(seg):
        r, a = raw[seg == s], adj[seg == s]
        if len(r) < 61:
            continue
        d = pd.DataFrame(index=r.index)
        d["close"] = r["c"]
        d["dv20"] = (r["c"] * r["v"]).rolling(20).median()
        d["avgvol20"] = r["v"].rolling(20).mean()
        d["vol60"] = np.log(a["c"]).diff().rolling(60).std()
        d["ret20"] = a["c"] / a["c"].shift(20) - 1
        d["ret60"] = a["c"] / a["c"].shift(60) - 1
        d["dist52"] = a["c"] / a["c"].rolling(252, min_periods=60).max() - 1
        f = r["c"] / a["c"]
        d["split30"] = (np.log(f / f.shift(1)).abs() > np.log(1.2)).astype(float).rolling(30, min_periods=1).max() > 0
        days = cal[(cal >= r.index[0]) & (cal <= r.index[-1])]
        ac, ao = a["c"].reindex(days).to_numpy(), a["o"].reindex(days).to_numpy()
        pos = days.get_indexer(r.index)
        keep = pos >= 0
        d, pos = d[keep], pos[keep]
        n = len(days)
        for h in (5, 10):
            d[f"maxup{h}"] = _forward(ac, h, "max")[pos] / ac[pos] - 1
        d["mindn10"] = _forward(ac, 10, "min")[pos] / ac[pos] - 1
        ff = pd.Series(ac).ffill().to_numpy()
        entry = np.append(ao[1:], np.nan)[pos]
        d["ret10"] = ff[np.minimum(pos + 10, n - 1)] / entry - 1
        out.append(d)
    if not out:
        return None
    d = pd.concat(out)
    d = d[(d.index >= DEV[0]) & (d.index <= TEST[1]) & (d["close"] >= MIN_PRICE) & (d["dv20"] >= MIN_DV)
          & d["vol60"].notna()]
    if not len(d):
        return None
    d = d.rename_axis("date").reset_index()
    d.insert(1, "symbol", sym)
    d.insert(2, "group", np.where(d["close"] < 5.0, "penny", "main"))
    d["first_bar"] = idx[0]
    cpos = cal.get_indexer(d["date"])
    for v, (thr, h) in VARIANTS.items():
        hit = (d[f"maxup{h}"] >= thr).to_numpy()
        first, last = np.zeros(len(d), bool), -10 ** 9
        for i in np.flatnonzero(hit):
            if cpos[i] - last >= SPACING:
                first[i], last = True, cpos[i]
        d[f"first_{v}"] = first
    fl = d.select_dtypes("float64").columns
    return d.astype({c: "float32" for c in fl})


def stage_bars(cache, deadline):
    from core.bars import fetch_daily_bars, to_frames
    from core.env import alpaca_headers
    from tracks.main.longterm import bars_batch
    rd = cache / "rows"
    rd.mkdir(parents=True, exist_ok=True)
    sf = rd / "symbols.json"
    if not sf.exists():
        a = bars_batch.assets_all()
        sf.write_text(json.dumps(sorted(set(a.loc[a["excl_reason"].isna(), "base_symbol"]))))
    syms = json.loads(sf.read_text())
    cal = pd.DatetimeIndex(calendar()["date"])
    h = alpaca_headers()
    nb = -(-len(syms) // 500)
    for b in range(nb):
        f = rd / f"rows_{b:03d}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            print(f"[b0] bars incomplete: stopped before batch {b + 1}/{nb} (budget); run again")
            return
        batch = syms[b * 500:(b + 1) * 500]
        end = alpaca.sip_end(BARS_END)
        raw = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="raw", headers=h))
        adj = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="all", headers=h))
        parts = [symbol_rows(s, raw.get(s), adj.get(s), cal) for s in batch]
        parts = [p for p in parts if p is not None]
        atomic_parquet(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), f)
        print(f"[b0] bars batch {b + 1}/{nb}: {sum(len(p) for p in parts)} universe rows")
    (rd / "complete.json").write_text(json.dumps({"batches": nb, "symbols": len(syms)}))
    print("[b0] bars complete")


def year_rows(cache, y, columns=None):
    """All universe rows dated in year y, read batch by batch."""
    lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    parts = []
    for f in sorted((cache / "rows").glob("rows_*.parquet")):
        p = pd.read_parquet(f, columns=columns)
        if len(p):
            parts.append(p[(p["date"] >= lo) & (p["date"] <= hi)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------- match

def cells(d):
    """Dollar-volume quintile and 60-day volatility decile within each date and price group."""
    g = d.groupby(["date", "group"])
    dvq = np.ceil(g["dv20"].rank(pct=True, method="first") * 5).clip(1, 5)
    vold = np.ceil(g["vol60"].rank(pct=True, method="first") * 10).clip(1, 10)
    return d.assign(dvq=dvq.astype(int), vold=vold.astype(int))


def match_day(day, variant, rng):
    """(symbol, match_id, role, group) rows for one date's universe rows (with dvq, vold): each first
    burst and up to N_CTRL same-cell non-bursts, drawn without replacement within the day."""
    thr, h = VARIANTS[variant]
    up = day[f"maxup{h}"]
    pool = day[up.notna() & (up < thr)]
    by_cell = {k: sorted(g["symbol"]) for k, g in pool.groupby(["group", "dvq", "vold"])}
    used, out = set(), []
    for e in day[day[f"first_{variant}"]].sort_values(["group", "symbol"]).itertuples():
        cand = [s for s in by_cell.get((e.group, e.dvq, e.vold), []) if s not in used]
        k = min(N_CTRL, len(cand))
        pick = [cand[i] for i in np.sort(rng.choice(len(cand), size=k, replace=False))] if k else []
        used.update(pick)
        out.append((e.symbol, e.symbol, "burst", e.group))
        out += [(p, e.symbol, "control", e.group) for p in pick]
    return out


def match_year(d):
    d = cells(d)
    rows = []
    for date, day in d.groupby("date", sort=True):
        for v in VARIANTS:
            rng = np.random.default_rng(SEED + pd.Timestamp(date).toordinal())
            rows += [(date, *r, v) for r in match_day(day, v, rng)]
    return pd.DataFrame(rows, columns=["date", "symbol", "match_id", "role", "group", "variant"])


def need_complete(cache):
    if not (cache / "rows" / "complete.json").exists():
        raise SystemExit("[b0] the bars stage isn't complete; run it first")


def stage_match(cache):
    need_complete(cache)
    cols = ["date", "symbol", "group", "dv20", "vol60", "maxup5", "maxup10"] + [f"first_{v}" for v in VARIANTS]
    for y in YEARS:
        f = cache / "keys" / f"keys_{y}.parquet"
        if f.exists():
            continue
        k = match_year(year_rows(cache, y, cols))
        atomic_parquet(k, f)
        p = k[k["variant"] == PRIMARY]
        print(f"[b0] {y}: {int((p['role'] == 'burst').sum())} primary bursts, {int((p['role'] == 'control').sum())} controls")


# ---------------------------------------------------------------- sources

def parse_full_index(text):
    """EDGAR quarterly form.idx -> frame (form, cik, filed) for INDEX_FORMS."""
    rows = []
    for ln in text.splitlines():
        m = re.match(r"^(\S+(?: \S+)*?)\s{2,}.*?\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2}|\d{8})\s+\S+\s*$", ln)
        if m and m.group(1) in INDEX_FORMS:
            rows.append(m.groups())
    d = pd.DataFrame(rows, columns=["form", "cik", "filed"])
    d["cik"] = d["cik"].astype("int64")
    d["filed"] = pd.to_datetime(d["filed"].str.replace("-", ""), format="%Y%m%d")
    return d


def months():
    return pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")


def src_short_interest(src, deadline, fetch=shortdata.short_interest):
    for m in months():
        f = src / "si" / f"si_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        d = fetch(m.start_time, m.end_time.normalize())
        atomic_parquet(d, f)
        print(f"[b0] short interest {m}: {len(d)} rows, settlements {sorted(d['settlement'].dt.strftime('%m-%d').unique())}")
    return True


def src_ftd(src, deadline, get=shortdata.ftd_file):
    for m in months():
        for half in "ab":
            f = src / "ftd" / f"ftd_{m}{half}.parquet"
            if f.exists():
                continue
            if time.monotonic() > deadline:
                return False
            d, lm = get(m.start_time, half)
            meta = {"month": str(m), "half": half, "posted": d is not None, "last_modified": lm,
                    "n_dates": int(d["date"].nunique()) if d is not None else 0}
            mf = shortdata.ftd_mean_fails(d).reset_index() if d is not None else pd.DataFrame(columns=["symbol", "mean_fails"])
            (src / "ftd").mkdir(parents=True, exist_ok=True)
            (src / "ftd" / f"ftd_{m}{half}.json").write_text(json.dumps(meta))
            atomic_parquet(mf, f)
    return True


def src_regsho(src, deadline, cal, day_fn=shortdata.regsho_day):
    for m in months():
        f = src / "regsho" / f"regsho_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        parts, srcs = [], {}
        for d in cal[(cal >= m.start_time) & (cal <= m.end_time)]:
            x, how = day_fn(d)
            srcs[f"{d:%Y-%m-%d}"] = how
            if x is not None:
                parts.append(x.assign(date=d))
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "short", "total", "date"])
        (src / "regsho").mkdir(parents=True, exist_ok=True)
        (src / "regsho" / f"regsho_{m}.json").write_text(json.dumps(srcs))
        atomic_parquet(out.astype({"short": "float32", "total": "float32"}), f)
        print(f"[b0] reg sho {m}: {len(srcs)} days, {sum(v is None for v in srcs.values())} without a file")
    return True


def src_index(src, get=sec.get):
    for per in pd.period_range("2015Q1", "2025Q3", freq="Q"):
        f = src / "index" / f"form_{per}.parquet"
        if not f.exists():
            r = get(FULL_INDEX.format(y=per.year, q=per.quarter))
            atomic_parquet(parse_full_index(r.content.decode("latin-1")), f)


def src_sec(src):
    from tracks.main.longterm import fundamentals, sec_data
    end_q = pd.Period("2025Q3")
    tmap = sec_data.dera_ticker_map(2014, end_q, src / "dera")
    sec_data.dera_insider_buys(2015, end_q, src / "dera")
    sf = src / "shares.parquet"
    if not sf.exists():
        z = src / "companyfacts.zip"
        sec_data.download_companyfacts(z)
        facts = sec_data.facts_for(z, tmap["cik"].unique(), src / "facts")
        sh = facts[facts["concept"] == fundamentals.SHARES][["cik", "filed", "val"]].dropna()
        atomic_parquet(sh.sort_values(["cik", "filed"]).reset_index(drop=True), sf)
        z.unlink()                         # 1.3 GB; the cache only keeps the share counts
        for p in (src / "facts").glob("*.parquet"):
            p.unlink()


def stage_sources(cache, deadline):
    from tracks.main.longterm import news_history
    src = cache / "src"
    cal = pd.DatetimeIndex(calendar()["date"])
    src_index(src)
    src_sec(src)
    done = (src_ftd(src, deadline) and src_short_interest(src, deadline) and src_regsho(src, deadline, cal)
            and news_history.fetch_range("2015-12", LAST_MONTH, src / "news", deadline=deadline))
    if done:
        (src / "complete.json").write_text("{}")
        print("[b0] sources complete")
    else:
        print("[b0] sources incomplete (budget); run again")


# ---------------------------------------------------------------- features

def by_cik_count(cik, date, ev, days):
    """For each row: events (cik, filed) with filed in (date - days, date]; NaN where cik is NaN."""
    out = np.where(np.isnan(cik), np.nan, 0.0)
    evs = {c: np.sort(g.to_numpy()) for c, g in ev.groupby("cik")["filed"]}
    dates = date.to_numpy()
    hv = np.flatnonzero(~np.isnan(cik))
    for c, rows in pd.Series(hv).groupby(cik[hv]):
        arr = evs.get(int(c))
        if arr is None:
            continue
        rows = rows.to_numpy()
        t = dates[rows]
        out[rows] = np.searchsorted(arr, t, "right") - np.searchsorted(arr, t - np.timedelta64(days, "D"), "right")
    return out


def asof(left, right, on_left, on_right, by, tolerance_days, direction="backward"):
    """merge_asof that keeps left's row order."""
    l = left.reset_index(drop=True).assign(_row=lambda x: range(len(x))).sort_values(on_left)
    r = right.sort_values(on_right)
    m = pd.merge_asof(l, r, left_on=on_left, right_on=on_right, by=by, direction=direction,
                      tolerance=pd.Timedelta(days=tolerance_days))
    return m.sort_values("_row").reset_index(drop=True)


def feat_dtc(d, si, cal):
    if not len(si):
        return np.full(len(d), np.nan)
    pos = cal.searchsorted(si["settlement"]) + SI_LAG
    si = si[pos < len(cal)].assign(avail=cal[pos[pos < len(cal)]])
    m = asof(d[["date", "symbol"]], si[["symbol", "avail", "settlement", "short_qty"]], "date", "avail",
             "symbol", 366)
    q = np.where((m["date"] - m["settlement"]).dt.days <= SI_MAX_AGE, m["short_qty"], np.nan)
    return np.where(d["split30"].to_numpy(), np.nan, q / d["avgvol20"].to_numpy())


def feat_shvr(d, rs, cal):
    if not len(rs):
        return np.full(len(d), np.nan)
    rs = rs[rs["total"] > 0]
    p = rs.assign(r=rs["short"] / rs["total"]).pivot_table(index="date", columns="symbol", values="r")
    p = p.reindex(cal[(cal >= p.index.min()) & (cal <= p.index.max())])
    m5 = p.rolling(5, min_periods=3).mean().stack().rename("shvr5").reset_index()
    m5.columns = ["date", "symbol", "shvr5"]
    return d[["date", "symbol"]].merge(m5, on=["date", "symbol"], how="left")["shvr5"].to_numpy()


def ftd_files(src, cal):
    """Frame (key, period_end, avail, last_modified) for every posted half-month file."""
    rows = []
    for j in sorted((src / "ftd").glob("ftd_*.json")):
        m = json.loads(j.read_text())
        if not m["posted"]:
            continue
        pe = shortdata.ftd_period_end(m["month"], m["half"])
        i = cal.searchsorted(pe + pd.Timedelta(days=FTD_LAG))
        if i < len(cal):
            rows.append((j.stem, pe, cal[i], m["last_modified"]))
    return pd.DataFrame(rows, columns=["key", "period_end", "avail", "last_modified"]).sort_values("avail")


def feat_ftd(d, files, fails, shares):
    """Mean daily fails in the latest file known by t, in basis points of shares outstanding. fails:
    frame (key, symbol, mean_fails); a symbol missing from a posted file has zero fails."""
    if not len(files):
        return np.full(len(d), np.nan)
    m = asof(d[["date", "symbol"]], files[["avail", "key", "period_end"]], "date", "avail", None, 3650)
    m = m.merge(fails, on=["key", "symbol"], how="left")
    fresh = ((m["date"] - m["period_end"]).dt.days <= FTD_MAX_AGE).to_numpy()
    mf = np.where(fresh, m["mean_fails"].fillna(0.0).to_numpy(), np.nan)
    sh = asof(d[["date", "cik"]].dropna().astype({"cik": "int64"}), shares.rename(columns={"filed": "sfiled"}),
              "date", "sfiled", "cik", SHARES_MAX_AGE)
    shv = pd.Series(np.nan, index=range(len(d)))
    shv[np.flatnonzero(d["cik"].notna().to_numpy())] = sh["val"].to_numpy()
    shv = shv.to_numpy()
    return np.where(shv > 0, mf / shv * 1e4, np.nan)


def feat_lockup(d, ipo):
    """1 when a 424B4 was filed 150-200 days before t and the symbol's first bar came within 5 days
    before to 30 days after it (so it was the IPO); NaN when the CIK is unknown."""
    have = d["cik"].notna().to_numpy()
    out = np.where(have, 0.0, np.nan)
    if not len(ipo) or not have.any():
        return out
    x = d.loc[have, ["date", "cik", "first_bar"]].astype({"cik": "int64"}).assign(probe=lambda z: z["date"] - pd.Timedelta(days=150))
    m = asof(x, ipo[["cik", "filed"]], "probe", "filed", "cik", 50)
    hit = m["filed"].notna() & (m["first_bar"] >= m["filed"] - pd.Timedelta(days=5)) & \
        (m["first_bar"] <= m["filed"] + pd.Timedelta(days=30))
    out[np.flatnonzero(have)] = hit.astype(float).to_numpy()
    return out


def feat_insider(d, buys):
    have = d["cik"].notna().to_numpy()
    out = np.where(have, 0.0, np.nan)
    if not len(buys) or not have.any():
        return out
    x = d.loc[have, ["date", "cik"]].astype({"cik": "int64"})
    b = buys[["cik", "filed"]].drop_duplicates().rename(columns={"filed": "bfiled"})
    m = asof(x, b, "date", "bfiled", "cik", 89)
    out[np.flatnonzero(have)] = m["bfiled"].notna().astype(float).to_numpy()
    return out


def session_closes(caldf):
    """UTC close time of each trading day (early closes included)."""
    t = pd.to_datetime(caldf["date"].dt.strftime("%Y-%m-%d") + " " + caldf["close"])
    return pd.DatetimeIndex(t).tz_localize("America/New_York").tz_convert("UTC")


def feat_news(d, news, caldf):
    """Articles tagging the symbol created after the close 5 trading days before t, up to t's close."""
    closes = session_closes(caldf)
    cal = pd.DatetimeIndex(caldf["date"])
    n = news.dropna(subset=["symbols"])
    t = pd.to_datetime(n["created_at"], utc=True)
    sess = closes.searchsorted(t, side="left")
    n = n.assign(sess=sess)[sess < len(cal)]
    n = n.assign(symbol=n["symbols"].str.split(",")).explode("symbol")
    cnt = n.groupby(["sess", "symbol"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(range(cnt.index.min(), cnt.index.max() + 1), fill_value=0)
    roll = cnt.rolling(5, min_periods=1).sum().stack()
    roll = roll[roll > 0].rename("n").rename_axis(["sess", "symbol"]).reset_index()
    x = pd.DataFrame({"sess": cal.get_indexer(d["date"]), "symbol": d["symbol"].to_numpy()})
    return x.merge(roll, on=["sess", "symbol"], how="left")["n"].fillna(0.0).to_numpy()


def load_parts(folder, pattern, keys):
    fs = [folder / pattern.format(k) for k in keys]
    parts = [pd.read_parquet(f) for f in fs if f.exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stage_features(cache):
    from tracks.main.longterm import news_history, sec_data
    need_complete(cache)
    src = cache / "src"
    if not (src / "complete.json").exists():
        raise SystemExit("[b0] the sources stage isn't complete; run it first")
    caldf = calendar()
    cal = pd.DatetimeIndex(caldf["date"])
    tmap = sec_data.dera_ticker_map(2014, pd.Period("2025Q3"), src / "dera")
    buys = sec_data.dera_insider_buys(2015, pd.Period("2025Q3"), src / "dera")
    shares = pd.read_parquet(src / "shares.parquet")
    files = ftd_files(src, cal)
    for y in YEARS:
        out = cache / "feat" / f"feat_{y}.parquet"
        if out.exists():
            continue
        rows = year_rows(cache, y)
        keys = pd.read_parquet(cache / "keys" / f"keys_{y}.parquet")[["date", "symbol"]].drop_duplicates()
        want = rows.merge(keys, on=["date", "symbol"], how="left", indicator=True)
        d = want[(want["_merge"] == "both") | ((want["date"] >= TEST[0]) & (want["date"] <= TEST[1]))]
        d = d.drop(columns="_merge").reset_index(drop=True)
        ms = pd.period_range(f"{y - 1}-10", f"{y}-12", freq="M")
        ms = [m for m in ms if str(m) <= LAST_MONTH]
        d["cik"] = sec_data.map_ciks(d[["symbol", "date"]], tmap, CIK_MAX_AGE)
        si = load_parts(src / "si", "si_{}.parquet", ms)
        d["dtc"] = feat_dtc(d, si, cal)
        d["shvr5"] = feat_shvr(d, load_parts(src / "regsho", "regsho_{}.parquet", ms), cal)
        keep = files[(files["avail"] >= pd.Timestamp(f"{y - 1}-09-01")) & (files["avail"] <= pd.Timestamp(f"{y}-12-31"))]
        fails = pd.concat([pd.read_parquet(src / "ftd" / f"{k}.parquet").assign(key=k) for k in keep["key"]]
                          + [pd.DataFrame(columns=["symbol", "mean_fails", "key"])], ignore_index=True)
        d["ftd"] = feat_ftd(d, keep, fails, shares)
        idx = load_parts(src / "index", "form_{}.parquet", pd.period_range(f"{y - 1}Q1", f"{y}Q4", freq="Q"))
        cik = d["cik"].to_numpy(dtype=float)
        d["lockup"] = feat_lockup(d, idx[idx["form"] == "424B4"])
        d["insider90"] = feat_insider(d, buys)
        d["n8k20"] = by_cik_count(cik, d["date"], idx[idx["form"] == "8-K"], 20)
        d["offer90"] = (by_cik_count(cik, d["date"], idx[idx["form"].isin(OFFER_FORMS)], 90) > 0).astype(float)
        d.loc[np.isnan(cik), "offer90"] = np.nan
        news = news_history.load_range(f"{y - 1}-12", min(f"{y}-12", LAST_MONTH), src / "news", columns=["created_at", "symbols"])
        d["news5"] = feat_news(d, news, caldf)
        atomic_parquet(d, out)
        print(f"[b0] features {y}: {len(d)} rows")


# ---------------------------------------------------------------- report

def fit_prep(X):
    """Per feature: (log1p?, winsor low, high, mean, sd) from development rows."""
    prep = {}
    for c in X.columns:
        v = X[c].astype(float)
        if c in LOG1P:
            v = np.log1p(v.clip(lower=0))
        if c in BINARY:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = (float(v.quantile(0.01)), float(v.quantile(0.99))) if v.notna().any() else (0.0, 0.0)
        w = v.clip(lo, hi)
        sd = float(w.std()) if w.notna().sum() > 1 else 0.0
        prep[c] = (c in LOG1P, lo, hi, float(w.mean()) if w.notna().any() else 0.0, sd if sd > 0 else 1.0)
    return prep


def transform(X, prep):
    cols = []
    for c, (lg, lo, hi, mu, sd) in prep.items():
        v = X[c].astype(float)
        if lg:
            v = np.log1p(v.clip(lower=0))
        cols.append(((v.clip(lo, hi) - mu) / sd).fillna(0.0).to_numpy())
    return np.column_stack(cols)


def date_weights(dates, rng, n_boot):
    """Bootstrap weights per row: how often its date was drawn, dates resampled with replacement."""
    codes, uniq = pd.factorize(pd.Series(dates))
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]


def auc_ci(y, s, dates, n_boot=N_BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, s))
    boots = []
    for w in date_weights(dates, np.random.default_rng(seed), n_boot):
        k = w > 0
        if len(np.unique(y[k])) == 2:
            boots.append(roc_auc_score(y[k], s[k], sample_weight=w[k]))
    return auc, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def mean_ci(x, dates, n_boot=N_BOOT, seed=SEED):
    """Row mean with a date-bootstrap CI (each drawn date brings all its rows)."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, dates = x[ok], np.asarray(dates)[ok]
    g = pd.DataFrame({"d": dates, "x": x}).groupby("d")["x"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    S, N, n = g["sum"].to_numpy(), g["size"].to_numpy(), len(g)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        boots.append((w * S).sum() / (w * N).sum())
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "n": int(len(x))}


def top_decile(u, score):
    """Rows in the top 10% of scores within each date (at least the top row)."""
    r = pd.Series(score, index=u.index).groupby(u["date"]).rank(ascending=False, method="first")
    n = u.groupby("date")["date"].transform("size")
    return (r <= np.maximum(1, np.floor(n * 0.1))).to_numpy()


def outcomes(u, cost):
    return {"p_up30_10d": mean_ci((u["maxup10"] >= 0.30).astype(float), u["date"]),
            "p_down20_10d": mean_ci((u["mindn10"] <= -0.20).astype(float), u["date"]),
            "mean_ret10_net": mean_ci(u["ret10"] - cost, u["date"])}


def coverage(m):
    return {int(y): {c: round(float(g[c].notna().mean()), 3) for c in FEATURES}
            for y, g in m.groupby(m["date"].dt.year)}


def evaluate(m, u, group, variant, costs):
    """Fit on development matched rows, score the test matched rows (and, for the primary, the test
    universe)."""
    from sklearn.linear_model import LogisticRegression
    x = m[(m["group"] == group) & (m["variant"] == variant)]
    dev = x[(x["date"] >= DEV[0]) & (x["date"] <= DEV[1])]
    test = x[(x["date"] >= TEST[0]) & (x["date"] <= TEST[1])]
    prep = fit_prep(dev[FEATURES])
    yd, yt = (dev["role"] == "burst").to_numpy().astype(int), (test["role"] == "burst").to_numpy().astype(int)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(transform(dev[FEATURES], prep), yd)
    st = mdl.decision_function(transform(test[FEATURES], prep))
    auc, ci = auc_ci(yt, st, test["date"].to_numpy())
    burst_dev = dev[dev["role"] == "burst"]
    n_ctrl = dev[dev["role"] == "control"].groupby(["date", "match_id"]).size()
    r = {"dev_bursts": int(yd.sum()), "dev_controls": int((1 - yd).sum()), "test_bursts": int(yt.sum()),
         "test_controls": int((1 - yt).sum()), "dev_bursts_short_of_controls": int(len(burst_dev) - (n_ctrl >= N_CTRL).sum()),
         "test_auc": auc, "test_auc_ci95": ci,
         "coefficients": {c: round(float(b), 4) for c, b in zip(FEATURES, mdl.coef_[0])},
         "passes": bool(auc >= 0.60 and ci[0] >= 0.55)}
    if variant != PRIMARY:
        return r
    xt = transform(test[FEATURES], prep)
    r["univariate_test_auc"] = {c: round(auc_point(yt, xt[:, i]), 4)
                                if len(np.unique(xt[:, i])) > 1 else None for i, c in enumerate(FEATURES)}
    r["coverage_by_year"] = coverage(x)
    ug = u[u["group"] == group].reset_index(drop=True)
    su = mdl.decision_function(transform(ug[FEATURES], prep))
    top = top_decile(ug, su)
    r["universe_test_rows"] = int(len(ug))
    r["top_decile"] = outcomes(ug[top], costs[group])
    r["base_rate"] = outcomes(ug, costs[group])
    return r


def auc_point(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def stage_report(cache, out):
    from paper.book import load_config
    feats = pd.concat([pd.read_parquet(cache / "feat" / f"feat_{y}.parquet") for y in YEARS], ignore_index=True)
    keys = pd.concat([pd.read_parquet(cache / "keys" / f"keys_{y}.parquet") for y in YEARS], ignore_index=True)
    m = keys.merge(feats.drop(columns="group"), on=["date", "symbol"], how="inner")
    u = feats[(feats["date"] >= TEST[0]) & (feats["date"] <= TEST[1])]
    tc = load_config()["tracks"]
    costs = {"penny": tc["penny"]["round_trip_cost"], "main": tc["main"]["round_trip_cost"]}
    p2, both = stats.idea_counts()
    res = {"idea": "B0", "prereg": "research/prereg/B0.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "costs_round_trip": costs, "PRIMARY": {}, "exploratory_variants": {}}
    for g in ("penny", "main"):
        res["PRIMARY"][g] = evaluate(m, u, g, PRIMARY, costs)
        res["exploratory_variants"][g] = {v: evaluate(m, u, g, v, costs) for v in VARIANTS if v != PRIMARY}
    res["passes_any_group"] = any(res["PRIMARY"][g]["passes"] for g in ("penny", "main"))
    res["ftd_publication_check"] = ftd_lag_check(cache / "src", pd.DatetimeIndex(calendar()["date"]))
    out.mkdir(parents=True, exist_ok=True)
    pb = m[(m["variant"] == PRIMARY) & (m["role"] == "burst")]
    chk = pb.sample(n=min(20, len(pb)), random_state=SEED)[["date", "symbol", "group", "close", "maxup10", "split30"]]
    chk.assign(checked_by="", ok="", note="").to_csv(out / "B0_bursts_check.csv", index=False)
    (out / "B0_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def ftd_lag_check(src, cal):
    """Files whose Last-Modified header is later than the date B0 assumed them known (listed, not used
    to move the dates)."""
    f = ftd_files(src, cal)
    lm = pd.to_datetime(f["last_modified"], errors="coerce", utc=True).dt.tz_convert(None)
    late = f[lm.dt.normalize() > f["avail"]]
    return {"files": int(len(f)), "with_last_modified": int(lm.notna().sum()),
            "later_than_assumed": late.assign(avail=late["avail"].dt.strftime("%Y-%m-%d"))[["key", "avail", "last_modified"]]
            .to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bars", "match", "sources", "features", "report"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    cache, deadline = Path(a.cache), time.monotonic() + 60 * a.budget_min
    if a.stage == "bars":
        stage_bars(cache, deadline)
    elif a.stage == "match":
        stage_match(cache)
    elif a.stage == "sources":
        stage_sources(cache, deadline)
    elif a.stage == "features":
        stage_features(cache)
    else:
        stage_report(cache, Path(a.out))


if __name__ == "__main__":
    main()
