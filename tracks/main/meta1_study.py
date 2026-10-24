"""META1 (research/prereg/META1.md): a take-or-skip model for R1's picks.

Stage "data": bars (all and raw, Alpaca SIP, delisted kept) for every R1 pick symbol in ~500-symbol batches,
reduced at once to the 18 inputs per pick and cached per batch; then CIKs (Form 3/4/5 ticker map, as R1)
and 8-K counts from EDGAR's quarterly indexes. Stage "fit": the walk-forward, the primary test and the
reported-alongside numbers. Each stage resumes from its cache.

Usage: python -m tracks.main.meta1_study --stage data|fit|all --cache <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main import weekly_panel
from tracks.main.longterm import sec_data
from tracks.main.longterm.prices import segment_ids
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

PICKS = Path("research/results/R1/R1_picks.csv")
SEED = 20261024
COST = 0.002                        # R1's main round trip
BARS_START, BARS_END = "2015-12-01", "2025-09-30"
BATCH = 500
TEST_YEARS = range(2019, 2026)
EMBARGO_WEEKS = 2
LAGS, BLOCK, N_BOOT = 4, 4, 10_000
INPUTS = ["ret5", "rank5", "ret1", "ret20", "ret60", "ret250x20", "vol20", "vol5_20", "drop_z", "dist_hi52",
          "dist_ma200", "log_dv20", "log_mcap", "vol_day", "n_8k", "spy5", "spy20", "n_picks"]
PARAMS = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100,
          "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "random_state": SEED, "verbose": -1}


def picks():
    p = pd.read_csv(PICKS, parse_dates=["date"])
    p = p[p["role"] == "pick"].drop(columns="role").reset_index(drop=True)
    p["n_picks"] = p.groupby("date")["symbol"].transform("size")
    p["rank5"] = p.groupby("date")["ret5"].rank(pct=True)
    return p


def stock_inputs(d, dates):
    """Inputs at each date in `dates` from one symbol's aligned frame d (all-adjusted o/c, raw rc/v), using
    bars up to and including that date, within its current series. Also fwd5 recomputed (entry next open,
    exit 5th close) for the data check. Returns {date: dict}."""
    seg = segment_ids(d.index)
    c, o, rc, v = (d[k].to_numpy(float) for k in ("c", "o", "rc", "v"))
    lr = np.r_[np.nan, np.diff(np.log(c))]
    out = {}
    for t in dates:
        i = d.index.searchsorted(t)
        if i >= len(d) or d.index[i] != t:
            continue
        s0 = int(np.searchsorted(seg, seg[i]))    # first bar of the current series
        n = i - s0 + 1
        if n < 21:
            continue

        def ret(k):
            return float(c[i] / c[i - k] - 1) if n > k else np.nan
        r = lr[i - 19:i + 1]
        v20 = float(np.std(r, ddof=1))
        r5 = lr[i - 4:i + 1]
        row = {"ret1": ret(1), "ret20": ret(20), "ret60": ret(60),
               "ret250x20": float(c[i - 20] / c[i - 250] - 1) if n > 250 else np.nan,
               "vol20": v20, "vol5_20": float(np.std(r5, ddof=1)) / v20 if v20 > 0 else np.nan,
               "drop_z": float(np.log(c[i] / c[i - 5])) / (v20 * np.sqrt(5)) if v20 > 0 and n > 5 else np.nan,
               "dist_hi52": float(c[i] / c[max(s0, i - 251):i + 1].max() - 1),
               "dist_ma200": float(c[i] / c[i - 199:i + 1].mean() - 1) if n >= 200 else np.nan,
               "log_dv20": float(np.log(np.median(rc[i - 19:i + 1] * v[i - 19:i + 1]))),
               "vol_day": float(v[i] / np.median(v[i - 19:i + 1])),
               "fwd5_check": float(c[i + 5] / o[i + 1] - 1) if i + 5 < len(d) and seg[i + 5] == seg[i] else np.nan}
        out[t] = row
    return out


def stage_data(cache):
    p = picks()
    syms = sorted(p["symbol"].unique())
    bdir = cache / "inputs"
    bdir.mkdir(parents=True, exist_ok=True)
    by_sym = p.groupby("symbol")["date"].apply(list).to_dict()
    end = alpaca.sip_end(BARS_END)
    for b in range(0, len(syms), BATCH):
        f = bdir / f"in_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        part = syms[b:b + BATCH]
        h = alpaca_headers()
        fa = to_frames(fetch_daily_bars(part, BARS_START, end, adjustment="all", headers=h))
        fr = to_frames(fetch_daily_bars(part, BARS_START, end, adjustment="raw", headers=h))
        rows = []
        for s in part:
            if s not in fa or s not in fr:
                continue
            d = pd.DataFrame({"o": fa[s]["o"], "c": fa[s]["c"]}).join(
                fr[s][["c", "v"]].rename(columns={"c": "rc"}), how="inner")
            d = d[(d["v"] > 0) & (d["c"] > 0) & (d["rc"] > 0)].sort_index()
            for t, r in stock_inputs(d, by_sym[s]).items():
                rows.append(dict(r, symbol=s, date=t))
        df = pd.DataFrame(rows)
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(f)
        print(f"[meta1] batch {b // BATCH + 1}/{-(-len(syms) // BATCH)}: {len(df)} pick rows", flush=True)
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, end, adjustment="all", headers=alpaca_headers()))["SPY"]
    tmap = sec_data.dera_ticker_map(2015, pd.Period("2025Q3"), cache / "dera")
    eightk = weekly_panel.filings({"8-K", "8-K/A"}, cache / "edgar")
    return spy, tmap, eightk


def assemble(p, inputs, spy, tmap, eightk):
    """The pick table with all 18 inputs. 8-Ks count when filed from the 5th session back to t."""
    x = p.merge(inputs, on=["symbol", "date"], how="left")
    x["log_mcap"] = np.log(x["mcap"])
    cal = spy.index
    sc = spy["c"]
    for k in (5, 20):
        x[f"spy{k}"] = [float(sc.iloc[i] / sc.iloc[i - k] - 1) if (i := cal.searchsorted(t)) < len(cal) and
                        cal[i] == t and i >= k else np.nan for t in x["date"]]
    x["cik"] = sec_data.map_ciks(x[["symbol", "date"]], tmap)
    first = {t: cal[max(0, cal.searchsorted(t) - 4)] for t in x["date"].unique()}
    e = eightk.groupby("cik")["filed"].apply(lambda s: np.sort(s.to_numpy()))
    cnt = []
    for ck, t in zip(x["cik"], x["date"]):
        if ck != ck or int(ck) not in e.index:
            cnt.append(0 if ck == ck else np.nan)
            continue
        f = e[int(ck)]
        cnt.append(int(np.searchsorted(f, np.datetime64(t), "right") - np.searchsorted(f, np.datetime64(first[t]), "left")))
    x["n_8k"] = cnt
    x["label"] = (x["fwd5"] > x.groupby("date")["fwd5"].transform("mean")).astype(int)
    return x


def qa(x):
    q = {"picks": int(len(x)), "with_fwd5": int(x["fwd5"].notna().sum()),
         "with_all_price_inputs": int(x[[c for c in INPUTS if c not in ("n_8k",)]].notna().all(axis=1).sum()),
         "with_cik": int(x["cik"].notna().sum()),
         "inputs": {c: {k: (None if not np.isfinite(v) else float(v)) for k, v in
                        x[c].describe(percentiles=[0.01, 0.5, 0.99])[["count", "min", "1%", "50%", "99%", "max"]].items()}
                    for c in INPUTS}}
    d = (x["fwd5"] - x["fwd5_check"]).abs()
    q["fwd5_vs_our_bars"] = {"compared": int(d.notna().sum()), "within_0.5pct": int((d <= 0.005).sum()),
                             "median_abs_diff": float(d.median())}
    top = x.reindex(x["fwd5"].abs().sort_values(ascending=False).index).head(20)
    q["largest_fwd5"] = [{"symbol": r.symbol, "date": f"{r.date:%Y-%m-%d}", "fwd5": float(r.fwd5),
                          "fwd5_our_bars": None if r.fwd5_check != r.fwd5_check else float(r.fwd5_check)}
                         for r in top.itertuples()]
    return q


def walk_forward(x, inputs=INPUTS):
    """Out-of-sample probabilities and take flags per test year (expanding window, 2-week embargo)."""
    import lightgbm as lgb
    x = x.dropna(subset=["fwd5"]).copy()
    weeks = np.sort(x["date"].unique())
    x["prob"], x["take"], x["model_year"] = np.nan, np.nan, np.nan
    imp = {}
    for y in TEST_YEARS:
        test = x["date"].dt.year == y
        if not test.any():
            continue
        start = x.loc[test, "date"].min()
        cut = weeks[max(0, np.searchsorted(weeks, np.datetime64(start)) - EMBARGO_WEEKS)]
        tr = x["date"] < cut
        m = lgb.LGBMClassifier(**PARAMS).fit(x.loc[tr, inputs], x.loc[tr, "label"])
        thr = float(np.median(m.predict_proba(x.loc[tr, inputs])[:, 1]))
        pr = m.predict_proba(x.loc[test, inputs])[:, 1]
        x.loc[test, "prob"], x.loc[test, "take"], x.loc[test, "model_year"] = pr, (pr >= thr).astype(float), y
        imp[str(y)] = {"train_rows": int(tr.sum()), "threshold": thr,
                       "gain": dict(zip(inputs, map(float, m.booster_.feature_importance("gain"))))}
    return x[x["prob"].notna()], imp


def weekly_diff(t, col="take"):
    """Per week: mean fwd5 of flagged picks minus mean of all picks (0 when nothing is flagged)."""
    out = {}
    for d, g in t.groupby("date"):
        k = g[g[col] == 1]
        out[d] = float(k["fwd5"].mean() - g["fwd5"].mean()) if len(k) else 0.0
    return pd.Series(out).sort_index()


def summary(s):
    m, se, tt = nw_tstat(s.to_numpy(), LAGS)
    lo, hi = block_bootstrap_ci(s.to_numpy(), BLOCK, N_BOOT, SEED)
    return {"weeks": int(len(s)), "mean": m, "se": se, "t_nw": tt, "ci95": [lo, hi]}


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def evaluate(t, imp):
    t = t.copy()
    t["week_idx"] = t["date"].rank(method="dense").astype(int)
    ctrl = []
    for d, g in t.groupby("date"):
        k = int(g["take"].sum())
        idx = np.random.default_rng(SEED + int(g["week_idx"].iloc[0])).choice(len(g), size=k, replace=False) if k else []
        ctrl += list(g.index[np.sort(idx)])
    t["ctrl"] = 0
    t.loc[ctrl, "ctrl"] = 1
    prim = summary(weekly_diff(t))
    taken = t[t["take"] == 1]
    prim["taken_mean_after_costs"] = float((taken.groupby("date")["fwd5"].mean() - COST).mean())
    prim["all_mean_after_costs"] = float((t.groupby("date")["fwd5"].mean() - COST).mean())
    prim["auc_oos"] = auc(t["label"], t["prob"])
    prim["taken_share"] = float(t["take"].mean())
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_nw"] >= 1.5 and prim["taken_mean_after_costs"] > 0)
    t["top5"] = (t.groupby("date")["prob"].rank(ascending=False, method="first") <= 5).astype(int)
    per_year = {}
    for y, g in t.groupby(t["date"].dt.year):
        s = weekly_diff(g)
        per_year[str(y)] = {"weeks": int(len(s)), "mean": float(s.mean()), "auc": auc(g["label"], g["prob"])}
    small = t["mcap"] < 2e9
    return {"PRIMARY": prim, "random_control_minus_all": summary(weekly_diff(t, "ctrl")),
            "alongside": {"top5_minus_all": summary(weekly_diff(t, "top5")), "per_year": per_year,
                          "mcap_300m_2b": summary(weekly_diff(t[small])),
                          "mcap_2b_plus": summary(weekly_diff(t[~small])), "importance": imp}}


def run(stage, cache, out):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spy, tmap, eightk = stage_data(cache)
    if stage == "data":
        return
    inputs = pd.concat([pd.read_parquet(f) for f in sorted((cache / "inputs").glob("in_*.parquet"))],
                       ignore_index=True)
    x = assemble(picks(), inputs, spy, tmap, eightk)
    q = qa(x)
    print("QA", json.dumps({k: v for k, v in q.items() if k not in ("inputs", "largest_fwd5")}), flush=True)
    t, imp = walk_forward(x)
    res = {"idea": "META1", "prereg": "research/prereg/META1.md", "primary_picks": str(PICKS), "qa": q}
    res.update(evaluate(t, imp))
    (out / "META1_result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    t[["date", "symbol", "fwd5", "label", "prob", "take", "model_year"]].to_csv(out / "META1_oos.csv", index=False)
    print("PRIMARY", json.dumps(res["PRIMARY"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "fit", "all"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/META1")
    a = ap.parse_args()
    run(a.stage, a.cache, a.out)


if __name__ == "__main__":
    main()
