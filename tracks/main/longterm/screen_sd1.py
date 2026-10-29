"""SD1 (research/prereg/SD1.md): does accounting alpha survive year-old data on our free SEC data?

RL1's month-end panel (M4's price features and filters, M3's dates, next-month return) and SEC companyfacts.
Two composites of M4's four accounting families (quality, value, net issuance, asset growth), each family a
within-date percentile rank:
- Fresh: the latest 10-K (and share count) filed by t, as M4.
- Stale: the latest filed by t minus 12 months, with today's price (value ratios use t's market cap).
Same stocks for both (M4's filters at t on the fresh market cap). Monthly long-short quintile, 0.2% round trip
per side; FF5+momentum and CAPM alphas, Newey-West 3 lags. Runs through study.yml on the shared cache.

Usage: python -m tracks.main.longterm.screen_sd1 --cache <dir> [--out research/results/SD1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main.longterm import sec_data
from tracks.main.longterm import screen_rl1 as R1
from tracks.main.longterm.score import attach_fundamentals, eligible

COST, SEED, LAGS, Q = 0.002, 20261024, 3, 5
LAG = pd.Timedelta(days=365)
FAMS = ["f_quality", "f_value", "f_issuance", "f_assetgrowth"]
MIN_FAMS = 3


# ---------- pure helpers (tested) ----------

def stale_ratios(feat, facts, rtab, fresh):
    """The accounting ratios from filings available by t - 12 months, priced at t: gpa, roa and asset growth
    as filed then, issuance from the share counts then, E/P and B/M with the year-old earnings and equity
    over t's market cap. `fresh` is attach_fundamentals on the same rows (for t's market cap)."""
    old = attach_fundamentals(feat.assign(date=feat["date"] - LAG), facts, rtab)
    out = pd.DataFrame(index=feat.index)
    for k in ("gpa", "roa", "assetgrowth", "issuance"):
        out[k] = old[k].to_numpy()
    scale = old["mcap"].to_numpy(float) / fresh["mcap"].to_numpy(float)
    out["ep"], out["bm"] = old["ep"].to_numpy() * scale, old["bm"].to_numpy() * scale
    return out


def composite4(df):
    """Rows with >= 3 of the 4 families; the mean of the available family ranks within each date."""
    g = df.groupby("date")
    r = lambda c, hi=True: g[c].rank(pct=True) if hi else (-df[c]).groupby(df["date"]).rank(pct=True)
    out = df.assign(f_quality=pd.concat([r("gpa"), r("roa")], axis=1).mean(axis=1),
                    f_value=pd.concat([r("ep"), r("bm")], axis=1).mean(axis=1),
                    f_issuance=r("issuance", False), f_assetgrowth=r("assetgrowth", False))
    out["n_fams"] = out[FAMS].notna().sum(axis=1)
    out = out[out["n_fams"] >= MIN_FAMS].copy()
    out["score"] = out[FAMS].mean(axis=1)
    return out


def long_short(df, col, seed=None):
    """Per date: equal-weight top minus bottom quintile of `col` (or two random quintiles with `seed`),
    minus 0.2% round trip on each leg."""
    rows = []
    for t, g in df[df["ret1"].notna() & df[col].notna()].groupby("date", sort=True):
        k = len(g) // Q
        if k < 1:
            continue
        if seed is None:
            o = g.sort_values(col, kind="stable")
            lo, hi = o.head(k), o.tail(k)
        else:
            idx = np.random.default_rng(seed + t.toordinal()).permutation(len(g))
            lo, hi = g.iloc[idx[:k]], g.iloc[idx[k:2 * k]]
        rows.append({"date": t, "ls": hi["ret1"].mean() - lo["ret1"].mean() - 2 * COST, "n_leg": k})
    return pd.DataFrame(rows, columns=["date", "ls", "n_leg"])


def capm_alpha(ls, ff5):
    f = ff5.reindex(pd.PeriodIndex(stats.month_after(ls["date"]), freq="M"))
    return stats.ols_hac(ls["ls"].to_numpy(float), f[["Mkt-RF"]].to_numpy(), LAGS)


def verdict(fresh_ls, stale_ls, ff5, umd):
    res = {}
    for name, ls in (("fresh", fresh_ls), ("stale", stale_ls)):
        r = R1.summary(ls["ls"])
        r["ff5_mom_alpha"] = stats.factor_alpha(ls["ls"], stats.month_after(ls["date"]), ff5, umd, LAGS)
        r["capm_alpha"] = capm_alpha(ls, ff5)
        res[name] = r
    fa, sa = res["fresh"]["ff5_mom_alpha"], res["stale"]["ff5_mom_alpha"]
    if not (fa["alpha"] > 0 and fa["t"] >= 2.0):
        res["verdict"] = "nothing to replicate on our data (fresh alpha t < 2.0): fail"
    else:
        res["verdict"] = "replication holds" if sa["alpha"] >= 0.5 * fa["alpha"] else "fails: stale keeps under half"
    res["stale_share_of_fresh_alpha"] = float(sa["alpha"] / fa["alpha"]) if fa["alpha"] else None
    return res


# ---------- data ----------

def run(cache, out, budget_min=290):
    cache, out = Path(cache), Path(out)
    deadline = time.monotonic() + budget_min * 60
    p = R1.prepare(cache, deadline)            # builds (or resumes) RL1's bars and the companyfacts cache
    if p is None:
        return None
    _, ends1, funnel = p
    feat = pd.concat([pd.read_parquet(f) for f in sorted((cache / "bars").glob("rl1_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(f) for f in sorted((cache / "bars").glob("rl1rtab_*.parquet"))], ignore_index=True)
    tmap = sec_data.dera_ticker_map(2013, pd.Period(R1.LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= R1.LAST_SIGNAL]
    feat["cik"] = sec_data.map_ciks(feat[["symbol", "date"]], tmap)
    feat = feat.dropna(subset=["cik"]).astype({"cik": "int64"}).reset_index(drop=True)
    facts = sec_data.facts_for(cache / "companyfacts.zip", feat["cik"].unique(), cache / "facts")
    fresh = attach_fundamentals(feat, facts, rtab)
    st = stale_ratios(feat, facts, rtab, fresh)
    fresh = fresh[fresh["date"] >= pd.Timestamp("2017-01-01")]
    el = eligible(fresh)
    key = ["date", "cik", "symbol"]
    base = el[key + ["ret1", "price", "mcap", "dv20"]]
    acc = ["gpa", "roa", "ep", "bm", "issuance", "assetgrowth"]
    f_rows = el[key + acc]
    s_rows = feat.loc[:, key].join(st)[lambda x: x["date"] >= pd.Timestamp("2017-01-01")]
    s_rows = base[key].merge(s_rows, on=key, how="left")
    cf = composite4(base.merge(f_rows, on=key))
    cs = composite4(base.merge(s_rows, on=key))
    fl, sl = long_short(cf, "score"), long_short(cs, "score")
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    res = {"idea": "SD1", "prereg": "research/prereg/SD1.md", "ff5_sha256": sha5, "umd_sha256": shau,
           "funnel": funnel | {"fresh_rows": int(len(cf)), "stale_rows": int(len(cs)),
                               "months": int(fl["date"].nunique()), "median_per_date_fresh": float(cf.groupby("date").size().median()),
                               "median_per_date_stale": float(cs.groupby("date").size().median())}}
    res |= verdict(fl, sl, ff5, umd)
    res["control_random"] = R1.summary(long_short(cf, "score", seed=SEED)["ls"])
    both = fl.merge(sl, on="date", suffixes=("_fresh", "_stale"))
    res["per_year"] = {str(y): {"fresh": float(g["ls_fresh"].mean()), "stale": float(g["ls_stale"].mean())}
                       for y, g in both.groupby(both["date"].dt.year)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "SD1_result.json").write_text(json.dumps(res, indent=1, default=str))
    both.to_csv(out / "SD1_monthly.csv", index=False)
    print(json.dumps({k: res[k] for k in ("verdict", "stale_share_of_fresh_alpha")} |
                     {k: {"mean": res[k]["mean"], "t": res[k]["t_hac"], "alpha_t": res[k]["ff5_mom_alpha"]["t"]}
                      for k in ("fresh", "stale")}, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/SD1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
