"""V1 (research/prereg/V1.md): rebuild published factors from our data and compare them with Kenneth
French's series; reconcile a sample of prices and SEC values with outside sources.

Bars are fetched in ~500-symbol batches and reduced at once to one row per symbol and month-end
(cached per batch, atomic writes, resumable under a time budget).

Usage: python -m validation.v1_known_answers --cache <dir> [--out <dir>] [--budget-min N]
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers, sec_headers
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import align, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals
from tracks.main.longterm.screen import month_ends
from validation import french

ROOT = Path(__file__).resolve().parents[1]
FIRST, LAST = pd.Timestamp("2016-01-29"), pd.Timestamp("2025-08-29")
LABEL_END = pd.Timestamp("2025-10-01")
BARS_START, BARS_END = "2014-06-01", "2025-09-30"
SEED = 20261015
MIN_MAX_DAYS = 10
PASS = {"mkt": 0.98, "smb": 0.85, "hml": 0.70, "umd": 0.90, "strev": 0.85}
MKT_MAD = 0.005
FS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
SEC_CONCEPTS = {"Assets": 0, "StockholdersEquity": 0, "NetIncomeLoss": 4}     # concept: qtrs in num.txt


# ---------- per-symbol month rows (pure) ----------

def month_rows(sym, d, dates, month_end):
    """One row per formation date t (a month-end trading day) on which `sym` traded in t's last week:
    raw price, month-t return, return over months t-11..t-1, largest daily return in month t, 20-day
    return to t, median raw dollar volume in month t, and the next month's return (to the last close
    of month t+1 in the same series; to the series' last close when it ends inside t+1, flagged).
    `month_end` maps each month period to the calendar's last trading day in it."""
    if len(d) < 2:
        return []
    seg = segment_ids(d.index)
    per = d.index.to_period("M")
    idx = pd.Series(np.arange(len(d)), index=d.index)
    last = idx.groupby(per).max()
    first = idx.groupby(per).min()
    c, rc, v = d["c"].to_numpy(), d["rc"].to_numpy(), d["v"].to_numpy()
    seg_end = pd.Series(np.arange(len(d))).groupby(seg).max().to_dict()
    out = []
    for t in dates:
        m = t.to_period("M")
        if m not in last.index:
            continue
        i = int(last[m])
        if (t - d.index[i]).days > 7:                       # stopped trading before the last week
            continue
        s = seg[i]
        same = lambda k: k in last.index and seg[int(last[k])] == s
        row = {"symbol": sym, "date": t, "price": float(rc[i]), "r_next": np.nan, "r_next_trunc": False,
               "r_cur": np.nan, "mom": np.nan, "maxret": np.nan, "r20": np.nan}
        e = seg_end[s]
        j = None
        if e > i and per[i + 1] == m + 1:
            j = int(last[m + 1]) if seg[int(last[m + 1])] == s else e
        if j is not None:
            row["r_next"] = c[j] / c[i] - 1
            end_next = month_end.get(m + 1)
            row["r_next_trunc"] = bool(end_next is not None and (end_next - d.index[j]).days > 7)
        if same(m - 1):
            row["r_cur"] = c[i] / c[int(last[m - 1])] - 1
        if same(m - 1) and same(m - 12) and int(last[m - 12]) >= np.searchsorted(seg, s):
            row["mom"] = c[int(last[m - 1])] / c[int(last[m - 12])] - 1
        a = max(int(first[m]), int(np.searchsorted(seg, s)) + 1)
        if i - a + 1 >= MIN_MAX_DAYS:
            row["maxret"] = float(np.max(c[a:i + 1] / c[a - 1:i] - 1))
        if i - 20 >= np.searchsorted(seg, s):
            row["r20"] = c[i] / c[i - 20] - 1
        mb = np.arange(int(first[m]), i + 1)
        row["dvm"] = float(np.median(rc[mb] * v[mb]))
        out.append(row)
    return out


# ---------- factor builds (pure) ----------

def _vw(g, col="r_next"):
    w = g["mcap"].to_numpy()
    return float((g[col].to_numpy() * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def sort_2x3(g, char, lo=0.3, hi=0.7):
    """Value-weighted next-month returns of the 6 portfolios from a size (NYSE median) x `char` (NYSE
    30th/70th percentiles) sort; keys like ('S', 'L'). None when NYSE breakpoints or a portfolio is
    missing."""
    g = g[g[char].notna()]
    ny = g[g["exchange"] == "NYSE"]
    if len(ny) < 20:
        return None
    size_bp = ny["mcap"].median()
    c_lo, c_hi = ny[char].quantile([lo, hi])
    size = np.where(g["mcap"] <= size_bp, "S", "B")
    bucket = np.where(g[char] <= c_lo, "L", np.where(g[char] > c_hi, "H", "M"))
    out = {}
    for sz in "SB":
        for bk in "LMH":
            p = g[(size == sz) & (bucket == bk)]
            if p.empty:
                return None
            out[(sz, bk)] = _vw(p)
    return out


def factors(uni):
    """Monthly factor returns keyed by the return month (formation month + 1). `uni` rows need date,
    exchange, mcap, bm, mom, r_cur, maxret, r_next, inactive."""
    rows = []
    for t, g in uni.groupby("date", sort=True):
        g = g[g["r_next"].notna() & (g["mcap"] > 0)]
        r = {"m": t.to_period("M") + 1, "n": len(g), "n_nyse": int((g["exchange"] == "NYSE").sum()),
             "mkt": _vw(g), "mkt_survivors": _vw(g[~g["inactive"]])}
        p = sort_2x3(g[g["bm"] > 0], "bm")
        if p:
            r["smb"] = np.mean([p["S", k] for k in "LMH"]) - np.mean([p["B", k] for k in "LMH"])
            r["hml"] = (p["S", "H"] + p["B", "H"]) / 2 - (p["S", "L"] + p["B", "L"]) / 2
        p = sort_2x3(g, "mom")
        if p:
            r["umd"] = (p["S", "H"] + p["B", "H"]) / 2 - (p["S", "L"] + p["B", "L"]) / 2
        p = sort_2x3(g, "r_cur")
        if p:
            r["strev"] = (p["S", "L"] + p["B", "L"]) / 2 - (p["S", "H"] + p["B", "H"]) / 2
        mx = g[g["maxret"].notna()]
        if len(mx) >= 100:
            dec = pd.qcut(mx["maxret"].rank(method="first"), 10, labels=False)
            r["max_hl_vw"] = _vw(mx[dec == 9]) - _vw(mx[dec == 0])
            r["max_hl_ew"] = float(mx.loc[dec == 9, "r_next"].mean() - mx.loc[dec == 0, "r_next"].mean())
        rows.append(r)
    cols = ["n", "n_nyse", "mkt", "mkt_survivors", "smb", "hml", "umd", "strev", "max_hl_vw", "max_hl_ew"]
    return pd.DataFrame(rows).set_index("m").reindex(columns=cols)


def compare(ours, theirs):
    """Our factors next to French's (decimals, keyed by month), with the pre-registered checks."""
    j = ours.join(theirs, how="inner")
    j["mkt_rf"] = j["mkt"] - j["rf"]
    pairs = {"mkt": ("mkt_rf", "mktrf"), "smb": ("smb", "smb_ff"), "hml": ("hml", "hml_ff"),
             "umd": ("umd", "mom_ff"), "strev": ("strev", "strev_ff")}
    res = {}
    for k, (a, b) in pairs.items():
        x = j[[a, b]].dropna()
        corr = float(x[a].corr(x[b])) if len(x) > 2 else np.nan
        res[k] = {"months": len(x), "corr": corr, "mean_ours": float(x[a].mean()), "mean_french": float(x[b].mean()),
                  "mean_abs_diff": float((x[a] - x[b]).abs().mean()), "pass": bool(corr >= PASS[k])}
    res["mkt"]["pass"] = bool(res["mkt"]["pass"] and res["mkt"]["mean_abs_diff"] <= MKT_MAD)
    mx = j["max_hl_vw"].dropna()
    res["max_effect"] = {"months": len(mx), "mean_vw": float(mx.mean()) if len(mx) else None,
                         "mean_ew": float(j["max_hl_ew"].mean()) if len(mx) else None,
                         "pass": bool(len(mx) and mx.mean() < 0)}
    s = (j["mkt_survivors"] - j["mkt"]).dropna()
    res["survivorship"] = {"mean_monthly_gap": float(s.mean()), "annualised": float(s.mean() * 12)}
    res["all_pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    return res, j


def french_monthly(cache):
    hashes, frames = {}, []
    for name, rename in (("ff3_monthly", {"smb": "smb_ff", "hml": "hml_ff"}), ("mom_monthly", None),
                         ("strev_monthly", None)):
        raw, h = french.fetch(name, cache)
        hashes[name] = h
        f = french.parse(raw, "M")
        if name == "mom_monthly":
            f = f.rename(columns={f.columns[0]: "mom_ff"})
        elif name == "strev_monthly":
            f = f.rename(columns={f.columns[0]: "strev_ff"})
        else:
            f = f.rename(columns=rename)
        frames.append(f)
    return pd.concat(frames, axis=1), hashes


# ---------- reconciliation with outside sources ----------

_FAILS = {}


def _safe_get(host, get, url, **kw):
    """get(url) or None on a network error; after 5 straight errors from one host, stop asking it."""
    if _FAILS.get(host, 0) >= 5:
        return None
    try:
        r = get(url, **kw)
    except requests.RequestException as e:
        _FAILS[host] = _FAILS.get(host, 0) + 1
        print(f"[reconcile] {host}: {type(e).__name__} ({_FAILS[host]} in a row)")
        return None
    _FAILS[host] = 0
    return r


def stooq_closes(sym, get=requests.get):
    r = _safe_get("stooq", get, f"https://stooq.com/q/d/l/?s={sym.lower().replace('.', '-')}.us&i=d", timeout=30,
            headers={"User-Agent": "edge-lab research"})
    if r is None or r.status_code != 200 or not r.text.startswith("Date"):
        return None
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
    return df.set_index("Date")["Close"] if len(df) else None


def nasdaq_closes(sym, get=requests.get):
    url = (f"https://api.nasdaq.com/api/quote/{sym.replace('.', '/')}/historical?assetclass=stocks"
           f"&fromdate=2015-01-01&todate=2025-09-30&limit=9999")
    r = _safe_get("nasdaq", get, url, timeout=30,
                  headers={"User-Agent": "Mozilla/5.0 edge-lab research", "Accept": "application/json"})
    if r is None or r.status_code != 200:
        return None
    rows = (((r.json() or {}).get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if not rows:
        return None
    s = pd.Series({pd.Timestamp(x["date"]): float(str(x["close"]).replace("$", "").replace(",", "")) for x in rows})
    return s.sort_index()


def reconcile_prices(uni, sources, n=50, max_draws=150, tol=0.01):
    """20-day returns at sampled (symbol, month-end) rows vs an outside daily source."""
    cand = uni[uni["r20"].notna()].reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(cand))[:max_draws]
    got, missing, cache = [], [], {}
    for k in order:
        if len(got) >= n:
            break
        row = cand.iloc[k]
        sym, t = row["symbol"], row["date"]
        if sym not in cache:
            cache[sym] = next(((nm, s) for nm, fn in sources if (s := fn(sym)) is not None), (None, None))
        name, closes = cache[sym]
        if closes is None or t not in closes.index:
            missing.append({"symbol": sym, "date": str(t.date()), "reason": "not at source"})
            continue
        i = closes.index.get_loc(t)
        if i < 20:
            missing.append({"symbol": sym, "date": str(t.date()), "reason": "short history at source"})
            continue
        theirs = closes.iloc[i] / closes.iloc[i - 20] - 1
        got.append({"symbol": sym, "date": str(t.date()), "source": name, "ours": float(row["r20"]),
                    "theirs": float(theirs), "diff": float(row["r20"] - theirs),
                    "ok": bool(abs(row["r20"] - theirs) <= tol)})
    ok = sum(g["ok"] for g in got)
    return {"compared": len(got), "within_1pt": int(ok), "pass": bool(len(got) >= n and ok >= n - 2),
            "rows": got, "unavailable": missing}


def quarters(first="2017Q1", last="2025Q2"):
    return [str(q) for q in pd.period_range(first, last, freq="Q")]


def sample_sec(facts, rng):
    """4 quarters, then 5 first-filed 10-K values of the screened concepts per quarter."""
    r = facts[facts["concept"].isin(list(SEC_CONCEPTS)) & (facts["form"] == "10-K")].copy()
    r = r.sort_values("filed").drop_duplicates(["cik", "concept", "end"], keep="first")
    r["q"] = r["filed"].dt.to_period("Q").astype(str)
    picks = []
    for q in rng.choice(quarters(), size=4, replace=False):
        g = r[r["q"] == q]
        if len(g):
            picks.append(g.iloc[rng.choice(len(g), size=min(5, len(g)), replace=False)])
    return pd.concat(picks, ignore_index=True) if picks else r.iloc[0:0]


def fs_values(q, adshs, cache, get=requests.get):
    """num.txt rows of one quarter's Financial Statement Data Set for the given accession numbers."""
    path = Path(cache) / "dera_fs" / f"{q.lower()}.zip"
    if not path.exists():
        r = get(FS_URL.format(q=q.lower()), headers=sec_headers(), timeout=600)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
    with zipfile.ZipFile(path) as z:
        num = pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, keep_default_na=False)
    num = num[num["adsh"].isin(set(adshs))]
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    return num


def reconcile_sec(sample, fetch_q, tol=0.005):
    rows = []
    for q, g in sample.groupby("q"):
        num = fetch_q(q, g["accn"].unique())
        for _, f in g.iterrows():
            m = num[(num["adsh"] == f["accn"]) & (num["tag"] == f["concept"])
                    & (num["ddate"] == f"{f['end']:%Y%m%d}") & (num["qtrs"] == str(SEC_CONCEPTS[f["concept"]]))
                    & (num["uom"] == "USD") & (num.get("coreg", "") == "")]
            if "segments" in num:
                m = m[m["segments"] == ""]
            theirs = float(m["value"].iloc[0]) if len(m) else np.nan
            ok = bool(len(m) and abs(f["val"] - theirs) <= tol * max(abs(theirs), 1.0))
            rows.append({"cik": int(f["cik"]), "concept": f["concept"], "end": str(f["end"].date()), "accn": f["accn"],
                         "quarter": q, "ours": float(f["val"]), "theirs": theirs, "ok": ok})
    ok = sum(r["ok"] for r in rows)
    return {"compared": len(rows), "equal": int(ok), "pass": bool(len(rows) >= 20 and ok >= 19), "rows": rows}


# ---------- pipeline ----------

def build(cache, deadline, fetch=None):
    """Month rows for the whole universe, from cached ~500-symbol batches; None if the time budget ran
    out before every batch was done (the finished batches stay cached)."""
    cache = Path(cache)
    fetch = fetch or bars_batch.fetch_three
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    cal = spy.index
    dates = month_ends(cal, FIRST, LAST)
    month_end = {pd.Timestamp(t).to_period("M"): pd.Timestamp(t) for t in month_ends(cal, FIRST, cal.max())}
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    d = cache / "bars"
    d.mkdir(parents=True, exist_ok=True)
    n_batches = -(-len(syms) // 500)
    for b in range(n_batches):
        f, fr_path = d / f"v1_{b:04d}.parquet", d / f"v1rtab_{b:04d}.parquet"
        if f.exists() and fr_path.exists():
            continue
        if time.monotonic() > deadline:
            print(f"[bars] budget spent at batch {b + 1}/{n_batches}; rerun to continue")
            return None
        batch = syms[b * 500:(b + 1) * 500]
        fr = fetch(batch, BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in batch:
            if all(s in fr[a] for a in ("all", "raw", "split")):
                al = align(fr["all"][s], fr["raw"][s], fr["split"][s])
                if len(al):
                    rows += month_rows(s, al, dates, month_end)
                    rts.append(split_points(s, al))
        rt = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
        for df, path in ((rt, fr_path), (pd.DataFrame(rows), f)):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
        print(f"[bars] batch {b + 1}/{n_batches}: {len(rows)} month rows")
    rows = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("v1_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("v1rtab_*.parquet"))], ignore_index=True)
    meta = common.sort_values("status").drop_duplicates("base_symbol")      # 'active' sorts before 'inactive'
    rows["exchange"] = rows["symbol"].map(meta.set_index("base_symbol")["exchange"])
    rows["inactive"] = rows["symbol"].map(meta.set_index("base_symbol")["status"]).ne("active")
    rows["cik"] = sec_data.map_ciks(rows[["symbol", "date"]], tmap)
    funnel = {"symbols": len(syms), "month_rows": int(len(rows)), "with_cik": int(rows["cik"].notna().sum())}
    rows = rows.dropna(subset=["cik"]).astype({"cik": "int64"})
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, rows["cik"].unique(), cache / "facts")
    uni = attach_fundamentals(rows, facts, rtab)
    uni = uni[uni["mcap"].notna() & (uni["mcap"] > 0)]
    uni = uni.sort_values("dvm", ascending=False).drop_duplicates(["date", "cik"]).sort_values(["date", "symbol"])
    funnel.update({"with_mcap_one_per_company": int(len(uni)), "per_month_median": float(uni.groupby("date").size().median()),
                   "r_next_truncated": int(uni["r_next_trunc"].sum()), "r_next_missing": int(uni["r_next"].isna().sum())})
    return uni.reset_index(drop=True), facts, funnel


def run(cache, out, budget_min=290, sources=None, fs_fetch=None, fetch=None):
    out = Path(out)
    got = build(cache, time.monotonic() + budget_min * 60, fetch=fetch)
    if got is None:
        print("PANEL INCOMPLETE: rerun this workflow to continue (cached batches are kept)")
        return None
    uni, facts, funnel = got
    print("FUNNEL", json.dumps(funnel))
    ff, hashes = french_monthly(cache)
    ours = factors(uni)
    res, joined = compare(ours, ff)
    sources = sources if sources is not None else [("stooq", stooq_closes), ("nasdaq", nasdaq_closes)]
    res["prices"] = reconcile_prices(uni, sources)
    sample = sample_sec(facts, np.random.default_rng(SEED))
    res["sec_values"] = reconcile_sec(sample, fs_fetch or (lambda q, a: fs_values(q, a, cache)))
    res["all_pass"] = bool(res["all_pass"] and res["prices"]["pass"] and res["sec_values"]["pass"])
    res = {"study": "V1", "prereg": "research/prereg/V1.md", "french_sha256": hashes, "funnel": funnel, **res}
    out.mkdir(parents=True, exist_ok=True)
    joined.rename_axis("month").to_csv(out / "V1_monthly_factors.csv")
    (out / "V1_results.json").write_text(json.dumps(res, indent=1, default=str))
    for k in ("mkt", "smb", "hml", "umd", "strev", "max_effect", "survivorship"):
        print(k, json.dumps(res[k]))
    print("PRICES", res["prices"]["compared"], res["prices"]["within_1pt"], "SEC", res["sec_values"]["compared"],
          res["sec_values"]["equal"], "ALL PASS", res["all_pass"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "V1"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
