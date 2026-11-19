"""SECTOR1 stage 2 (research/prereg/SECTOR1.md, Amendment 1): what drives the moves, by sector, 2019-2023.

Descriptive only. Marks each stage-1 stock-day with an earnings window (8-K item 2.02), Benzinga news and its keyword
types, and relates sector returns to 10-year yield and WTI oil changes (FRED). Writes <out>/SECTOR1_stage2.json.

    python -m tracks.main.sector2_study --w1 cache/w1 --out research/results/SECTOR1 --cache cache/sector1
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import industry
from tracks.main import sector1_study as S1
from tracks.news import w1_data as W

TYPES = {
    "earnings": r"earnings|\beps\b|revenue|results|quarter",
    "guidance": r"guidance|outlook|forecast|raises|lowers|\bcuts\b",
    "analyst": r"upgrade|downgrade|price target|initiates|reiterates",
    "deal": r"acquire|acquisition|merger|to buy|buyout|takeover|\bdeal\b",
    "fda": r"\bfda\b|approval|\btrial\b|\bphase\b",
    "offering": r"offering|priced|dilution|private placement|warrants",
    "contract": r"contract|award|partnership|agreement",
}
GROUPS = ["earn", "news", "none"] + [f"t_{k}" for k in TYPES]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


# ---------- pure helpers (tested) ----------

def to_session(ts_et, cal):
    """ET timestamps -> the trading day whose 16:00 close is the first at or after them (NaT past the calendar)."""
    ts = pd.DatetimeIndex(ts_et)
    day = ts.normalize() + pd.to_timedelta((ts.hour >= 16).astype(int), unit="D")
    i = np.searchsorted(cal.values, day.values)
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = i < len(cal)
    out[ok] = cal.values[i[ok]]
    return out


def earn_days(filings, cal):
    """8-K filings with item 2.02 -> (cik, date) for the first trading day on or after filing and the next one."""
    f = filings[(filings["form"].str.startswith("8-K")) & filings["items"].fillna("").str.contains(r"\b2\.02\b")]
    i = np.searchsorted(cal.values, f["filed"].values)
    rows = []
    for c, k in zip(f["cik"].to_numpy(), i, strict=True):
        for j in (k, k + 1):
            if j < len(cal):
                rows.append((int(c), cal[j]))
    return pd.DataFrame(rows, columns=["cik", "date"]).drop_duplicates()


def news_flags(news, cal):
    """Headlines (created_at, headline, symbols) -> (symbol, date, n, t_<type>...) per trading session."""
    n = news[news["symbols"].fillna("") != ""].copy()
    n["date"] = to_session(W.et_time(n["created_at"]), cal)
    n = n.dropna(subset=["date"])
    h = n["headline"].fillna("").str.lower()
    for k, pat in TYPES.items():
        n[f"t_{k}"] = h.str.contains(pat, regex=True)
    n["symbol"] = n["symbols"].str.split(",")
    n = n.explode("symbol")
    agg = {"headline": "count", **{f"t_{k}": "max" for k in TYPES}}
    out = n.groupby(["symbol", "date"]).agg(agg).rename(columns={"headline": "n"}).reset_index()
    return out


def group_table(d):
    """Per sector and driver group: the stage-2 measures. d has ret, sector and the bool group columns."""
    rows = []
    for sec, x in d.groupby("sector"):
        big = x["ret"].abs() >= S1.BIG[1]
        up, dn = big & (x["ret"] > 0), big & (x["ret"] < 0)
        for g in GROUPS:
            m = x[g].to_numpy(bool)
            y = x[m]
            b = big[m]
            rows.append({"sector": industry.NAMES[int(sec)], "group": g, "stock_days": int(m.sum()),
                         "share_of_days": float(m.mean()),
                         "mean_abs_ret": float(y["ret"].abs().mean()) if len(y) else None,
                         "mean_ret": float(y["ret"].mean()) if len(y) else None,
                         "share_ge_8pct": float(b.mean()) if len(y) else None,
                         "up_share_of_8pct": float((y["ret"][b] > 0).mean()) if b.any() else None,
                         "share_of_all_8pct_up": float((m & up).sum() / up.sum()) if up.any() else None,
                         "share_of_all_8pct_down": float((m & dn).sum() / dn.sum()) if dn.any() else None})
    return rows


def macro_table(sec_ret, macro):
    """sec_ret: date x sector equal-weight returns; macro: date -> dgs10_chg, oil_ret."""
    j = sec_ret.join(macro, how="inner")
    rows = []
    for sec in sec_ret.columns:
        r = {"sector": industry.NAMES[int(sec)]}
        for k in ("dgs10_chg", "oil_ret"):
            x = j[[sec, k]].dropna()
            if len(x) < 60:
                continue
            v = x[k].var()
            r[f"corr_{k}"] = float(x[sec].corr(x[k]))
            r[f"beta_{k}"] = float(x[sec].cov(x[k]) / v) if v > 0 else None
            hi, lo = x[k].quantile(0.9), x[k].quantile(0.1)
            r[f"mean_ret_top10_{k}"] = float(x.loc[x[k] >= hi, sec].mean())
            r[f"mean_ret_bottom10_{k}"] = float(x.loc[x[k] <= lo, sec].mean())
        rows.append(r)
    return rows


def parse_fred(text, name):
    d = pd.read_csv(io.StringIO(text))
    d.columns = ["date", name]
    d["date"] = pd.to_datetime(d["date"])
    d[name] = pd.to_numeric(d[name], errors="coerce")
    return d.set_index("date")[name]


# ---------- job ----------

def macro(cache):
    f = Path(cache) / "fred.parquet"
    if not f.exists():
        s = {k: parse_fred(requests.get(FRED.format(k), timeout=60).text, k) for k in ("DGS10", "DCOILWTICO")}
        pd.DataFrame(s).to_parquet(f)
    m = pd.read_parquet(f)
    m = m[m.index <= S1.LAST]
    out = pd.DataFrame({"dgs10_chg": m["DGS10"].dropna().diff(), "oil_ret": m["DCOILWTICO"].dropna().pct_change()})
    return out.replace([np.inf, -np.inf], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    ciks = set()
    for p in W.batch_files(a.w1):
        ciks |= {int(c) for c in pd.read_parquet(p, columns=["cik"])["cik"].dropna().unique()}
    sectors = industry.industries(ciks, a.cache).to_dict()
    filings = W.load_filings(a.w1, ["cik", "form", "filed", "items"])
    out, frames, secret = {"years": {}}, [], []
    for y in S1.YEARS:
        d = S1.year_frame(a.w1, y, sectors)
        cal = pd.DatetimeIndex(sorted(d["date"].unique()))
        e = earn_days(filings[(filings["filed"] >= cal[0] - pd.Timedelta(days=7)) & (filings["filed"] <= cal[-1])], cal)
        e["earn"] = True
        d["cik_i"] = d["cik"].astype("Int64")
        d = d.merge(e.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
        news = pd.concat([W.news_month(a.w1, p) for p in pd.period_range(f"{y - 1}-12", f"{y}-12", freq="M")],
                         ignore_index=True)
        nf = news_flags(news, cal)
        d = d.merge(nf, on=["symbol", "date"], how="left")
        d["earn"] = d["earn"].eq(True)
        d["news"] = d["n"].fillna(0) > 0
        for k in TYPES:
            d[f"t_{k}"] = d[f"t_{k}"].eq(True)
        d["none"] = ~d["earn"] & ~d["news"]
        out["years"][str(y)] = group_table(d)
        frames.append(d[["sector", "ret"] + GROUPS])
        secret.append(d.groupby(["date", "sector"])["ret"].mean().unstack())
        print(f"[sector2] {y}: {len(d)} stock-days, earnings {d['earn'].mean():.3f}, news {d['news'].mean():.3f}",
              flush=True)
    out["pooled"] = group_table(pd.concat(frames, ignore_index=True))
    out["macro"] = macro_table(pd.concat(secret).sort_index(), macro(a.cache))
    out["note"] = ("2019-2023, PICK main universe, FF12 from current SIC codes; news window 16:00 ET previous session "
                   "to 16:00 ET; earnings = 8-K item 2.02 on the day or the day before; descriptive only")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage2.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(Path(a.out) / "SECTOR1_stage2.json")


if __name__ == "__main__":
    main()
