"""VETO1 (research/prereg/VETO1.md): do NX3-ALL-HI's buys lose when the company filed a late-filing notice
(NT 10-K or NT 10-Q, SEC Form 12b-25) in the 60 days before? History 2019-01..2025-09 from EDGAR's quarterly form
indexes and SEC's ticker file; trades and costs as LONG (tracks/news/long_study.py).

    python -m tracks.news.veto1_study --stage index --cache cache/nx1      (EDGAR quarterly indexes -> NT filings)
    python -m tracks.news.veto1_study --stage run --cache cache/nx1 --out research/results/VETO1
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core import sec
from tracks.main.longterm.stats import nw_tstat
from tracks.news import hi3k_8k as K
from tracks.news import long_study as LS
from tracks.news import nx1_study as N

NT_FORMS = ("NT 10-K", "NT 10-Q", "NT 10-K/A", "NT 10-Q/A")
WINDOW_DAYS = 60
FIRST_Q, LAST_Q = (2018, 4), (2025, 3)
MIN_VETOED = 30


def quarters(first=FIRST_Q, last=LAST_Q):
    y, q = first
    while (y, q) <= last:
        yield y, q
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)


def nt_rows(text):
    """Late-filing notices from one EDGAR form.idx (form, cik, filed)."""
    rows = []
    for ln in text.splitlines():
        if not ln.startswith("NT 10"):
            continue
        m = re.match(r"^(\S+(?: \S+)*?)\s{2,}(.*?)\s{2,}(\d+)\s+(\d{4}-?\d{2}-?\d{2})\s+(\S+)\s*$", ln)
        if m and m.group(1) in NT_FORMS:
            rows.append((m.group(1), int(m.group(3)), pd.Timestamp(m.group(4))))
    return pd.DataFrame(rows, columns=["form", "cik", "filed"])


def vetoed(trades, nt, window=WINDOW_DAYS):
    """True for a trade (symbol, d) whose company filed a notice in the `window` calendar days before d (filing
    dates up to d - 1, so a notice filed the same evening never counts)."""
    if nt.empty:
        return np.zeros(len(trades), bool)
    by = {s: np.sort(g["filed"].to_numpy("datetime64[ns]")) for s, g in nt.groupby("symbol")}
    out = np.zeros(len(trades), bool)
    for i, (s, d) in enumerate(zip(trades["symbol"], pd.to_datetime(trades["d"]).to_numpy("datetime64[ns]"), strict=True)):
        f = by.get(s)
        if f is None:
            continue
        lo, hi = d - np.timedelta64(window, "D"), d - np.timedelta64(1, "D")
        j = np.searchsorted(f, lo)
        out[i] = j < len(f) and f[j] <= hi
    return out


def stage_index(cache):
    out = Path(cache) / "veto1"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for y, q in quarters():
        f = out / f"nt_{y}Q{q}.parquet"
        if not f.exists():
            r = sec.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx")
            N.atomic_parquet(nt_rows(r.content.decode("latin-1")) if r is not None else
                             pd.DataFrame(columns=["form", "cik", "filed"]), f)
            print(f"[veto1] {y}Q{q}: {len(pd.read_parquet(f))} notices", flush=True)
        parts.append(pd.read_parquet(f))
    nt = pd.concat(parts, ignore_index=True)
    tick = K.ticker_by_cik(sec.get(K.TICKERS).json())
    nt["symbol"] = nt["cik"].map(tick)
    nt = nt.dropna(subset=["symbol"])
    N.atomic_parquet(nt, out / "nt.parquet")
    print(f"[veto1] {len(nt)} notices mapped to a listed ticker", flush=True)


def stage_run(cache, out):
    nt = pd.read_parquet(Path(cache) / "veto1" / "nt.parquet")
    p = LS.load(cache)
    p = p[p["exit_5"].notna() & (p["exit_5"] <= N.LAST_EXIT)].reset_index(drop=True)
    res = {"notices": len(nt), "window_days": WINDOW_DAYS}
    for name, lo in (("uncapped_longs", p[p["side"] > 0]), ("nx3_hi_buys", LS.cap_longs(p))):
        lo = lo.reset_index(drop=True)
        v = vetoed(lo, nt)
        net = lo["base"].to_numpy(float)
        a, b = net[v & np.isfinite(net)], net[~v & np.isfinite(net)]
        cell = {"buys": len(lo), "vetoed": int(v.sum()), "vetoed_net": float(a.mean()) if len(a) else None,
                "kept_net": float(b.mean()) if len(b) else None}
        if len(a) >= 2 and len(b) >= 2:
            w = stats.ttest_ind(a, b, equal_var=False)
            cell |= {"gap": float(a.mean() - b.mean()), "gap_t": float(w.statistic)}
            kd = pd.Series(np.where(v, np.nan, net), index=pd.DatetimeIndex(lo["d"])).dropna().groupby(level=0).mean()
            ad = pd.Series(net, index=pd.DatetimeIndex(lo["d"])).dropna().groupby(level=0).mean()
            diff = (kd.reindex(ad.index).fillna(0.0) - ad).to_numpy()
            cell["book_gain_daily"] = dict(zip(("mean", "se", "t"), nw_tstat(diff, 5))) if len(diff) > 2 else None
        res[name] = cell
    u = res["uncapped_longs"]
    res["passes"] = bool(u["vetoed"] >= MIN_VETOED and u.get("gap") is not None and u["gap"] < 0
                         and u["gap_t"] <= -1.5 and u["vetoed_net"] < 0)
    if u["vetoed"] < MIN_VETOED:
        res["note"] = f"under {MIN_VETOED} vetoed buys: no verdict"
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "VETO1.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["index", "run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/VETO1")
    a = ap.parse_args()
    stage_index(Path(a.cache)) if a.stage == "index" else stage_run(Path(a.cache), a.out)


if __name__ == "__main__":
    main()
