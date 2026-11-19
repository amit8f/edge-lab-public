"""SECTOR1 stage 1 (research/prereg/SECTOR1.md): how stocks move, by Fama-French 12 sector, 2019-2023.

Descriptive only. Reads W1's cached daily rows year by year (nothing after 2023-12-31), keeps PICK main's universe
(previous raw close >= $5 and previous 20-day median dollar volume >= $5M) and writes one table per sector for each
year and pooled to <out>/SECTOR1_stage1.json.

    python -m tracks.main.sector1_study --w1 cache/w1 --out research/results/SECTOR1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry
from tracks.news import w1_data as W

YEARS = range(2019, 2024)
MIN_PX, MIN_DV = 5.0, 5e6
BIG = (0.05, 0.08)
COST = 0.002                      # main-track round trip (config/risk.yaml)
MIN_STOCKS, MIN_DAYS = 30, 60
LAST = pd.Timestamp("2023-12-31")


# ---------- pure helpers (tested) ----------

def prepare(d):
    """Daily rows (symbol, date, o, h, l, c, v, ao, ac, dv20, seg, cik) -> universe stock-days with ret (adjusted
    close over the previous adjusted close), gap (adjusted open over it), intra (adjusted close over adjusted
    open) and rng ((high - low) / close). A stock-day is in when the previous day's raw close and dv20 pass; the
    previous day must be in the same series."""
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby(["symbol", "seg"], sort=False)
    d["pc"], d["pdv"], d["pac"] = g["c"].shift(), g["dv20"].shift(), g["ac"].shift()
    d = d[(d["pc"] >= MIN_PX) & (d["pdv"] >= MIN_DV) & (d["pac"] > 0) & (d["ao"] > 0)].copy()
    d["ret"], d["gap"], d["intra"] = d["ac"] / d["pac"] - 1, d["ao"] / d["pac"] - 1, d["ac"] / d["ao"] - 1
    d["rng"] = (d["h"] - d["l"]) / d["c"]
    return d


def comove(d):
    """Adds mkt (equal-weight mean return of the day's universe) and sec (the stock's sector mean that day, the
    stock itself left out; NaN when it is alone)."""
    d = d.copy()
    d["mkt"] = d.groupby("date")["ret"].transform("mean")
    g = d.groupby(["date", "sector"])["ret"]
    n, s = g.transform("count"), g.transform("sum")
    d["sec"] = ((s - d["ret"]) / (n - 1)).where(n > 1)
    return d


def r2(d, col):
    """Median over stocks of the squared correlation of ret with `col` (stocks with at least MIN_DAYS days)."""
    out = []
    for _, x in d.dropna(subset=[col]).groupby("symbol"):
        if len(x) >= MIN_DAYS and x[col].std() > 0 and x["ret"].std() > 0:
            out.append(float(np.corrcoef(x["ret"], x[col])[0, 1] ** 2))
    return float(np.median(out)) if out else None


def sector_table(d):
    """One dict per sector for a comove() frame."""
    rows = []
    for sec, x in d.groupby("sector"):
        a = x["ret"].abs()
        big = x[a >= BIG[1]]
        quiet = big[(big["mkt"].abs() <= 0.01) & (big["sec"].abs() <= 0.02)]
        ga, ia = x["gap"].abs().mean(), x["intra"].abs().mean()
        rows.append({
            "sector": industry.NAMES[int(sec)], "stocks": int(x["symbol"].nunique()), "stock_days": len(x),
            "thin": bool(x["symbol"].nunique() < MIN_STOCKS),
            "median_price": float(x["pc"].median()), "median_dv20": float(x["pdv"].median()),
            "mean_abs_ret": float(a.mean()), "median_abs_ret": float(a.median()), "p95_abs_ret": float(a.quantile(0.95)),
            **{f"share_abs_ge_{int(t * 100)}pct": float((a >= t).mean()) for t in BIG},
            "up_share_of_big": float((big["ret"] > 0).mean()) if len(big) else None,
            "mean_abs_gap": float(ga), "mean_abs_intra": float(ia), "overnight_share": float(ga / (ga + ia)),
            "r2_market": r2(x, "mkt"), "r2_sector": r2(x, "sec"),
            "big_days": len(big), "stock_specific_big_share": float(len(quiet) / len(big)) if len(big) else None,
            "median_range": float(x["rng"].median()), "cost_over_median_abs_intra": float(COST / x["intra"].abs().median()),
        })
    return sorted(rows, key=lambda r: r["sector"])


# ---------- job ----------

def year_frame(w1, year, sectors):
    f = [("date", ">=", pd.Timestamp(f"{year - 1}-12-01")), ("date", "<=", min(pd.Timestamp(f"{year}-12-31"), LAST))]
    d = W.read_rows(w1, ["symbol", "date", "o", "h", "l", "c", "ao", "ac", "dv20", "seg", "cik"], f)
    d = d.drop_duplicates(["symbol", "date"])
    d = prepare(d)
    d = d[d["date"].dt.year == year]
    d["sector"] = [sectors.get(int(c), 12) if pd.notna(c) else 12 for c in d["cik"]]
    return comove(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    a = ap.parse_args()
    ciks = set()
    for p in W.batch_files(a.w1):
        ciks |= {int(c) for c in pd.read_parquet(p, columns=["cik"])["cik"].dropna().unique()}
    sectors = industry.industries(ciks, a.cache).to_dict()
    print(f"[sector1] {len(ciks)} CIKs, {len(sectors)} with an industry", flush=True)
    out, frames = {"years": {}}, []
    for y in YEARS:
        d = year_frame(a.w1, y, sectors)
        out["years"][str(y)] = sector_table(d)
        frames.append(d[["symbol", "date", "sector", "ret", "gap", "intra", "rng", "pc", "pdv", "mkt", "sec"]])
        print(f"[sector1] {y}: {len(d)} stock-days", flush=True)
    out["pooled"] = sector_table(pd.concat(frames, ignore_index=True))
    out["note"] = ("2019-2023, PICK main universe, FF12 from current SIC codes; pooled R-squared is the median over "
                   "stocks across all five years")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage1.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(Path(a.out) / "SECTOR1_stage1.json")


if __name__ == "__main__":
    main()
