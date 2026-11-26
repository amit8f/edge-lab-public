"""SECTOR1 stage 3 (research/prereg/SECTOR1.md, Amendment 2): how much of each sector's daily return variance is the
market, the big stocks, the sector, style factors, and each kind of company event; and how much of tomorrow a chart
predicts. 2019-2023 only. Descriptive. Writes <out>/SECTOR1_stage3.json.

    python -m tracks.main.sector3_study --w1 cache/w1 --out research/results/SECTOR1 --cache cache/sector1
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry
from tracks.main import sector1_study as S1
from tracks.main import sector2_study as S2
from tracks.news import w1_data as W
from validation import french

BLOCKS = {"market": ["mkt"], "big_stocks": ["big"], "sector": ["sec"], "style": ["smb", "hml", "rmw", "cma", "mom"]}
MIN_DAYS = 250
N_BIG = 10
HEAVY = 2.0
NEWS_ORDER = ["deal", "offering", "fda", "earnings", "analyst", "contract"]     # earnings type includes guidance
EVENTS = (["earn"] + [f"news_{k}" for k in NEWS_ORDER] + ["news_other", "other_8k", "offering_filing", "insider",
          "other_filing", "heavy_volume", "quiet"])
CHART = ["r1", "r5", "r20", "r60", "hi20", "lo20", "vol20", "rng1", "vr1", "mkt1", "sec1"]
LGB = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "random_state": 7, "verbose": -1, "n_jobs": 4}


# ---------- pure helpers (tested) ----------

def big_stock_return(d, n=N_BIG):
    """Per stock-day: equal-weight return of the day's n stocks with the highest previous-day dv20 (pdv), the stock
    itself left out (n-1 others when it is one of them)."""
    rk = d.groupby("date")["pdv"].rank(ascending=False, method="first")
    top = rk <= n
    s = d["ret"].where(top, 0.0).groupby(d["date"]).transform("sum")
    c = top.groupby(d["date"]).transform("sum")
    own = top.to_numpy()
    num = s - np.where(own, d["ret"], 0.0)
    den = c - own.astype(int)
    return (num / den).where(den > 0)


def r2_of(y, X):
    if X.shape[1] == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    tss = ((y - y.mean()) ** 2).sum()
    return float(1 - (e ** 2).sum() / tss) if tss > 0 else 0.0


def shapley(y, cols_by_block):
    """Shapley split of the full-model R-squared over the blocks: {block: share of variance}, plus the full R2."""
    names = list(cols_by_block)
    k = len(names)
    r2 = {}
    for m in range(k + 1):
        for s in itertools.combinations(range(k), m):
            X = np.column_stack([cols_by_block[names[i]] for i in s]) if s else np.empty((len(y), 0))
            r2[s] = r2_of(y, X)
    w = {m: math.factorial(m) * math.factorial(k - m - 1) / math.factorial(k) for m in range(k)}
    out = {}
    for i, nm in enumerate(names):
        v = 0.0
        for s, val in r2.items():
            if i in s:
                continue
            v += w[len(s)] * (r2[tuple(sorted(s + (i,)))] - val)
        out[nm] = v
    out["full"] = r2[tuple(range(k))]
    return out


def residuals(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def event_group(d):
    """The first matching group per stock-day (EVENTS order). d has bool columns earn, t_<type>, news, k8_other,
    f_offer, f_insider, f_other and float vr (volume over its 20-day median)."""
    conds = [d["earn"]]
    for k in NEWS_ORDER:
        c = d[f"t_{k}"] | (d["t_guidance"] if k == "earnings" else False)
        conds.append(d["news"] & c)
    conds += [d["news"], d["k8_other"], d["f_offer"], d["f_insider"], d["f_other"], d["vr"] >= HEAVY]
    g = np.select([np.asarray(c, bool) for c in conds], EVENTS[:-1], default="quiet")
    return pd.Series(g, index=d.index)


def filing_flags(filings, cal):
    """(cik, date) -> k8_other, f_offer, f_insider, f_other for the filing day's first session and the next one."""
    f = filings.dropna(subset=["cik", "filed"]).copy()
    items = f["items"].map(lambda v: ",".join(v) if isinstance(v, (list, np.ndarray)) else (v or ""))
    form = f["form"].fillna("")
    f["k8_other"] = form.str.startswith("8-K") & ~items.str.contains(r"\b2\.02\b")
    f["f_offer"] = form.str.match(r"^(S-1|S-3|424B)")
    f["f_insider"] = form.isin(["4", "4/A"])
    f["f_other"] = ~form.str.startswith("8-K") & ~f["f_offer"] & ~f["f_insider"]
    i = np.searchsorted(cal.values, f["filed"].values)
    keep = ["k8_other", "f_offer", "f_insider", "f_other"]
    rows = []
    for j in (0, 1):
        k = i + j
        ok = k < len(cal)
        x = f.loc[ok, ["cik"] + keep].copy()
        x["date"] = cal.values[k[ok]]
        rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    out["cik"] = out["cik"].astype(int)
    return out.groupby(["cik", "date"])[keep].max().reset_index()


def chart_features(d):
    """Per stock series, features known at the close of t-1 for predicting day t (shifted by one row)."""
    d = d.sort_values(["symbol", "seg", "date"]).copy()
    g = d.groupby(["symbol", "seg"], sort=False)
    ac = d["ac"]
    lr = np.log(ac / g["ac"].shift())
    d["_lr"] = lr
    gg = d.groupby(["symbol", "seg"], sort=False)["_lr"]
    feats = {
        "r1": lr, "r5": gg.transform(lambda s: s.rolling(5).sum()), "r20": gg.transform(lambda s: s.rolling(20).sum()),
        "r60": gg.transform(lambda s: s.rolling(60).sum()),
        "hi20": ac / g["ac"].transform(lambda s: s.rolling(20).max()) - 1,
        "lo20": ac / g["ac"].transform(lambda s: s.rolling(20).min()) - 1,
        "vol20": gg.transform(lambda s: s.rolling(20).std()), "rng1": (d["h"] - d["l"]) / d["c"], "vr1": d["vr"],
    }
    for k, v in feats.items():
        d[k] = v
    for k in list(feats):
        d[k] = d.groupby(["symbol", "seg"], sort=False)[k].shift()
    return d.drop(columns=["_lr"])


def oos_r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / (y ** 2).sum())


def rank_ic(df, pred, target):
    v = [g[pred].rank().corr(g[target].rank()) for _, g in df.groupby("date") if len(g) > 30]
    return float(np.nanmean(v)) if v else None


# ---------- job ----------

def year_frame(w1, year, sectors):
    f = [("date", ">=", pd.Timestamp(f"{year - 1}-08-01")), ("date", "<=", min(pd.Timestamp(f"{year}-12-31"), S1.LAST))]
    d = W.read_rows(w1, ["symbol", "date", "o", "h", "l", "c", "v", "ao", "ac", "dv20", "seg", "cik"], f)
    d = d.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    d = d[d["v"] > 0]
    g = d.groupby(["symbol", "seg"], sort=False)["v"]
    d["vr"] = d["v"] / g.transform(lambda s: s.shift().rolling(20, min_periods=10).median())
    d = chart_features(d)
    d = S1.prepare(d)
    d = d[d["date"].dt.year == year].copy()
    d["sector"] = [sectors.get(int(c), 12) if pd.notna(c) else 12 for c in d["cik"]]
    d = S1.comove(d)
    d["big"] = big_stock_return(d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    ap.add_argument("--minutes", type=float, default=150)
    a = ap.parse_args()
    t0 = time.monotonic()
    if not S2.ensure_inputs(a.w1, a.cache, t0 + a.minutes * 60):
        raise SystemExit("[sector3] out of time while fetching news; run again")
    ciks = set()
    for p in W.batch_files(a.w1):
        ciks |= {int(c) for c in pd.read_parquet(p, columns=["cik"])["cik"].dropna().unique()}
    sectors = industry.industries(ciks, a.cache).to_dict()
    filings = W.load_filings(a.w1, ["cik", "form", "filed", "items"])
    ff = french.parse(french.fetch("ff5_daily", a.cache)[0], "D").join(
        french.parse(french.fetch("mom_daily", a.cache)[0], "D"), how="inner")
    ff.columns = [c.strip().lower() for c in ff.columns]
    ff = ff.rename(columns={c: "mom" for c in ff.columns if c.startswith("mom") or c == "umd"})
    keep = ["symbol", "date", "sector", "ret", "mkt", "big", "sec", "vr", "cik"] + CHART[:-2]
    frames = []
    for y in S1.YEARS:
        d = year_frame(a.w1, y, sectors)
        cal = pd.DatetimeIndex(sorted(d["date"].unique()))
        e = S2.earn_days(filings[(filings["filed"] >= cal[0] - pd.Timedelta(days=7)) & (filings["filed"] <= cal[-1])], cal)
        e["earn"] = True
        d["cik_i"] = d["cik"].astype("Int64")
        d = d.merge(e.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
        fl = filing_flags(filings[(filings["filed"] >= cal[0] - pd.Timedelta(days=7)) & (filings["filed"] <= cal[-1])], cal)
        d = d.merge(fl.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
        news = pd.concat([W.news_month(a.w1, p) for p in pd.period_range(f"{y - 1}-12", f"{y}-12", freq="M")],
                         ignore_index=True)
        d = d.merge(S2.news_flags(news, cal), on=["symbol", "date"], how="left")
        for c in ["earn", "k8_other", "f_offer", "f_insider", "f_other"] + [f"t_{k}" for k in S2.TYPES]:
            d[c] = d[c].eq(True)
        d["news"] = d["n"].fillna(0) > 0
        d["group"] = event_group(d)
        frames.append(d[keep + ["group"]].astype({c: "float32" for c in CHART[:-2] + ["vr"]}))
        print(f"[sector3] {y}: {len(d)} stock-days, {time.monotonic() - t0:.0f} s", flush=True)
        del d, news
    d = pd.concat(frames, ignore_index=True)
    del frames
    d = d.merge(ff[["smb", "hml", "rmw", "cma", "mom"]], left_on="date", right_index=True, how="left")
    d = d.reset_index(drop=True)
    # previous-session market and sector-mean returns for the chart model (the stock's own sector, all members)
    sm = d.groupby(["date", "sector"])["ret"].mean().rename("secall").reset_index()
    mk = d.groupby("date")["ret"].mean().rename("mktall")
    dates = mk.index.sort_values()
    prev = pd.Series(dates[:-1], index=dates[1:])
    d["pdate"] = d["date"].map(prev)
    d = d.merge(mk.rename("mkt1"), left_on="pdate", right_index=True, how="left")
    d = d.merge(sm.rename(columns={"date": "pdate", "secall": "sec1"}), on=["pdate", "sector"], how="left")
    d = d.reset_index(drop=True)

    # ---- parts 1 and 2: Shapley common part, residual split by events ----
    rows = []
    resid = np.full(len(d), np.nan)
    cols = [c for b in BLOCKS.values() for c in b]
    for sym, x in d.groupby("symbol"):
        x = x.dropna(subset=["ret"] + cols)
        if len(x) < MIN_DAYS:
            continue
        y = x["ret"].to_numpy(float)
        y = np.clip(y, -0.5, 0.5)
        sh = shapley(y, {k: x[v].to_numpy(float) for k, v in BLOCKS.items()})
        res = residuals(y, x[cols].to_numpy(float))
        resid[x.index.to_numpy()] = res
        rows.append({"symbol": sym, "code": int(x["sector"].mode().iloc[0]), "var": float(y.var()) * len(y),
                     **{k: sh[k] for k in BLOCKS}, "full": sh["full"]})
    d["resid"] = resid
    st = pd.DataFrame(rows)
    print(f"[sector3] {len(st)} stocks fitted, {time.monotonic() - t0:.0f} s", flush=True)
    fit = d.dropna(subset=["resid"])
    fit = fit.assign(r2e=fit["resid"].astype(float) ** 2, big8=fit["ret"].abs() >= S1.BIG[1])

    def breakdown(s, f):
        tot = s["var"].sum()
        common = {k: float((s[k] * s["var"]).sum() / tot) for k in BLOCKS}
        idio_tot = f["r2e"].sum()
        idio_share = 1 - sum(common.values())
        g = f.groupby("group")
        ev = {k: {"share_of_days": float((f["group"] == k).mean()),
                  "share_of_variance": float(idio_share * g["r2e"].sum().get(k, 0.0) / idio_tot),
                  "share_of_stock_specific": float(g["r2e"].sum().get(k, 0.0) / idio_tot),
                  "share_of_8pct_days": float(((f["group"] == k) & f["big8"]).sum() / max(f["big8"].sum(), 1))}
              for k in EVENTS}
        return {"stocks": len(s), "stock_days": len(f), "common": common,
                "stock_specific": float(idio_share), "median_r2": {k: float(s[k].median()) for k in BLOCKS},
                "median_full_r2": float(s["full"].median()), "events": ev,
                "coincident_volume_r2": float(np.corrcoef(np.log(f["vr"].clip(0.05, 50).fillna(1)),
                                                          f["resid"].abs())[0, 1] ** 2)}
    out = {"all": breakdown(st, fit), "sectors": {}}
    for sec, s in st.groupby("code"):
        out["sectors"][industry.NAMES.get(int(sec), f"code {int(sec)}")] = breakdown(s, fit[fit["symbol"].isin(s["symbol"])])

    # ---- part 3: how concentrated market moves are ----
    m = mk.loc[mk.index.year.isin(list(S1.YEARS))]
    sq = (m - m.mean()) ** 2
    out["market_days"] = {f"top_{q}pct_days_share": float(sq[sq >= sq.quantile(1 - q / 100)].sum() / sq.sum())
                          for q in (1, 5, 10)}

    # ---- part 4: charts ----
    c = d.dropna(subset=CHART + ["ret", "mkt"]).copy()
    c["y"] = (c["ret"] - c["mkt"]).clip(-0.5, 0.5)
    tr, te = c[c["date"].dt.year <= 2021], c[c["date"].dt.year >= 2022].copy()
    import lightgbm as lgb
    mdl = lgb.LGBMRegressor(**LGB).fit(tr[CHART].astype("float32"), tr["y"])
    te["p"] = mdl.predict(te[CHART].astype("float32"))
    ch = {"all": {"rows": len(te), "oos_r2": oos_r2(te["y"].to_numpy(), te["p"].to_numpy()),
                  "rank_ic": rank_ic(te, "p", "y")},
          "importance": dict(zip(CHART, [int(v) for v in mdl.booster_.feature_importance("gain")], strict=True))}
    for sec, x in te.groupby("sector"):
        ch[industry.NAMES.get(int(sec), f"code {int(sec)}")] = {"rows": len(x), "oos_r2": oos_r2(x["y"].to_numpy(), x["p"].to_numpy()),
                                         "rank_ic": rank_ic(x, "p", "y")}
    out["charts"] = ch
    out["note"] = ("2019-2023, PICK main universe ($5+, $5M/day), FF12 from current SIC codes. Common part: per-stock "
                   "Shapley R2 of market (equal-weight), big stocks (top 10 by dollar volume), sector (leave-one-out) "
                   "and style (FF SMB/HML/RMW/CMA/MOM), variance-weighted. Stock-specific part split by the first "
                   "event that applies that day. Charts: LightGBM on t-1 chart features, fit 2019-21, scored 2022-23 "
                   "on market-adjusted next-day returns. Descriptive only.")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage3.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(Path(a.out) / "SECTOR1_stage3.json")
    print(json.dumps({"all": out["all"]["common"], "stock_specific": out["all"]["stock_specific"],
                      "charts": ch["all"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
