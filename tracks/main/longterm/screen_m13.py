"""M13 historical screen (research/prereg/M13.md): hold every stock above a rising 150-day average, vs the
universe, 1 month, costs scaled by turnover.

Same month-end panel as M3 (dates, universe, entry/exit); bars are fetched in ~500-symbol batches and reduced
at once to month-end rows, cached per batch. Downloads stop at --budget-min and resume on the next run.

Usage: python -m tracks.main.longterm.screen_m13 --cache <dir> [--out <dir>] [--budget-min 290]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main import m3_screen as m3
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import align, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals, eligible
from tracks.main.longterm.screen import month_ends
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

ROOT = Path(__file__).resolve().parents[3]
FIRST_SIGNAL, LAST_SIGNAL, LABEL_END = m3.FIRST_SIGNAL, m3.LAST_SIGNAL, m3.LABEL_END
BARS_START, BARS_END = m3.BARS_START, m3.BARS_END
COST, SEED, LAG = 0.002, 20261016, 21
MAS = (150, 200, 20)


def ma_rule(c, nb, i, n):
    """(close above its n-day average, that average above its value LAG bars earlier, distance) at bar i,
    using only bars of the current series; NaNs when the series is shorter than n + LAG bars."""
    if nb[i] < n + LAG:
        return np.nan, np.nan, np.nan
    ma = c[i - n + 1:i + 1].mean()
    ma_lag = c[i - LAG - n + 1:i - LAG + 1].mean()
    return float(c[i] > ma), float(ma > ma_lag), float(c[i] / ma - 1)


def symbol_rows(sym, d, dates, ends1, ends3):
    seg = segment_ids(d.index)
    nb = pd.Series(seg, index=d.index).groupby(seg).cumcount().to_numpy() + 1
    c = d["c"].to_numpy()
    dv = d["rc"].to_numpy() * d["v"].to_numpy()
    mr = m3.monthly_returns(d, seg)
    pos = {t: k for k, t in enumerate(d.index)}
    out = []
    for t in dates:
        i = pos.get(t)
        if i is None or nb[i] < 20:
            continue
        row = {"symbol": sym, "date": t, "price": d["rc"].iloc[i], "dv20": float(np.median(dv[i - 19:i + 1]))}
        for n in MAS:
            above, rising, dist = ma_rule(c, nb, i, n)
            row[f"above{n}"], row[f"rising{n}"], row[f"dist{n}"] = above, rising, dist
        r = mr.reindex(pd.period_range(end=t.to_period("M"), periods=12, freq="M")).to_numpy()
        row["mom"] = float(np.prod(1 + r[:-1]) - 1) if np.isfinite(r[:-1]).all() else np.nan
        row["ret1"] = m3.forward(d, seg, i, ends1[t]) if t in ends1 else np.nan
        row["ret3"] = m3.forward(d, seg, i, ends3[t]) if t in ends3 else np.nan
        out.append(row)
    return out


def add_rules(uni):
    u = uni.copy()
    for n in MAS:
        both = ((u[f"above{n}"] == 1) & (u[f"rising{n}"] == 1)).astype(float)
        u[f"allow{n}"] = both.where(u[f"above{n}"].notna() & u[f"rising{n}"].notna())
    u["above150_only"] = u["above150"]
    return u


def book(df, col, pick, seed=None):
    """Per date: mean `col` of the held set minus the universe mean, minus COST x turnover (share of the held
    names not held at the previous date; the first date counts 100%). `pick(g)` returns the held rows; with
    `seed`, a same-size random set from the universe instead (the control)."""
    rows, prev = [], set()
    for t, g in df.groupby("date", sort=True):
        held = pick(g)
        if seed is not None:
            rng = np.random.default_rng(seed + t.toordinal())
            held = g.iloc[rng.choice(len(g), size=min(len(held), len(g)), replace=False)]
        names = set(held["symbol"])
        if not names or not held[col].notna().any():
            prev = names
            continue
        turnover = len(names - prev) / len(names)
        prev = names
        rows.append({"date": t, "excess": held[col].mean() - g[col].mean() - COST * turnover,
                     "gross": held[col].mean() - g[col].mean(), "turnover": turnover, "n": int(held[col].notna().sum()),
                     "universe": len(g), "loss20_held": float((held[col] <= -0.2).mean()),
                     "loss20_universe": float((g[col] <= -0.2).mean())})
    return pd.DataFrame(rows)


def allowed(rule):
    return lambda g: g[g[rule] == 1]


def summary(ex, lags):
    x = ex["excess"].to_numpy()
    m, se, t = nw_tstat(x, lags)
    lo, hi = block_bootstrap_ci(x, 6, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_dates": int(np.isfinite(x).sum()),
            "hit_rate": float((ex["excess"] > 0).mean()), "mean_gross": float(ex["gross"].mean()),
            "mean_turnover": float(ex["turnover"].mean()), "mean_n_held": float(ex["n"].mean()),
            "worst_month": float(ex["excess"].min())}


def spread(df, col, rule):
    """Allowed minus not-allowed mean `col` per date (no costs)."""
    r = df.dropna(subset=[rule, col]).groupby("date").apply(
        lambda g: g.loc[g[rule] == 1, col].mean() - g.loc[g[rule] == 0, col].mean(), include_groups=False)
    x = r.dropna().to_numpy()
    m, se, t = nw_tstat(x, 3)
    return {"mean": m, "t_hac": t, "n_dates": int(len(x))}


def fama_macbeth(df, rule):
    """Per date: OLS of ret1 on [1, allowed dummy, 12-1 momentum percentile]; mean coefficients with HAC t."""
    co = []
    for _, g in df.dropna(subset=[rule, "mom", "ret1"]).groupby("date", sort=True):
        if len(g) < 30:
            continue
        X = np.c_[np.ones(len(g)), g[rule].to_numpy(float), g["mom"].rank(pct=True).to_numpy()]
        co.append(np.linalg.lstsq(X, g["ret1"].to_numpy(), rcond=None)[0])
    co = np.array(co)
    out = {}
    for k, name in ((1, "allowed"), (2, "mom_pct")):
        m, se, t = nw_tstat(co[:, k], 3)
        out[name] = {"mean": m, "t_hac": t}
    out["n_dates"] = int(len(co))
    return out


def qa_ma(sample, cache):
    """Recompute MA150 for sample (symbol, date) rows from freshly fetched all-adjusted bars with pandas."""
    out = []
    for _, r in sample.iterrows():
        fr = to_frames(fetch_daily_bars([r["symbol"]], str((r["date"] - pd.Timedelta(days=400)).date()),
                                        str(r["date"].date()), adjustment="all", headers=alpaca_headers()))
        d = fr.get(r["symbol"])
        if d is None or len(d) < 150:
            out.append({"symbol": r["symbol"], "date": str(r["date"].date()), "ok": None})
            continue
        d = d[(d["v"] > 0) & (d["c"] > 0)]
        ma = d["c"].rolling(150).mean().iloc[-1]
        out.append({"symbol": r["symbol"], "date": str(r["date"].date()), "dist_panel": r["dist150"],
                    "dist_check": float(d["c"].iloc[-1] / ma - 1),
                    "ok": bool(abs(d["c"].iloc[-1] / ma - 1 - r["dist150"]) < 0.01)})
    return out


def prepare(cache, deadline):
    cache.mkdir(parents=True, exist_ok=True)
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    cal = spy.index
    dates = month_ends(cal, FIRST_SIGNAL, LAST_SIGNAL)
    me_all = month_ends(cal, FIRST_SIGNAL, pd.Timestamp(BARS_END))
    nxt = {t: me_all[k + 1] for k, t in enumerate(me_all[:-1])}
    ends1 = {t: nxt[t] for t in dates if nxt[t] < LABEL_END}
    ends3 = {t: me_all[me_all.index(t) + 3] for t in dates if me_all.index(t) + 3 < len(me_all)
             and me_all[me_all.index(t) + 3] < LABEL_END}
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST_SIGNAL]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    funnel = {"common_named_in_form345": len(syms)}
    (cache / "bars").mkdir(exist_ok=True)
    for b in range(0, len(syms), 500):
        f, fr_path = cache / "bars" / f"m13_{b // 500:04d}.parquet", cache / "bars" / f"m13rtab_{b // 500:04d}.parquet"
        if f.exists() and fr_path.exists():
            continue
        if time.monotonic() > deadline:
            print("PANEL INCOMPLETE: budget reached; rerun to continue")
            return None
        fr = bars_batch.fetch_three(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in syms[b:b + 500]:
            if all(s in fr[a] for a in ("all", "raw", "split")):
                d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
                if len(d):
                    rows += symbol_rows(s, d, dates, ends1, ends3)
                    rts.append(split_points(s, d))
        rt = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
        for df, path in ((rt, fr_path), (pd.DataFrame(rows), f)):
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
        print(f"[bars] batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(rows)} rows")
    feat = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("m13_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("m13rtab_*.parquet"))], ignore_index=True)
    funnel["symbol_dates"] = int(len(feat))
    feat["cik"] = sec_data.map_ciks(feat[["symbol", "date"]], tmap)
    feat = feat.dropna(subset=["cik"]).astype({"cik": "int64"})
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, feat["cik"].unique(), cache / "facts")
    uni = add_rules(eligible(attach_fundamentals(feat, facts, rtab)))
    funnel.update({"eligible_stock_months": int(len(uni)), "with_ma150_rule": int(uni["allow150"].notna().sum()),
                   "dates": int(uni["date"].nunique())})
    print("FUNNEL", json.dumps(funnel))
    return {"uni": uni, "ends1": ends1, "ends3": ends3, "funnel": funnel}


def run(cache, out, budget_min=290):
    p = prepare(cache, time.monotonic() + budget_min * 60)
    if p is None:
        return None
    uni, ends1, ends3, funnel = p["uni"], p["ends1"], p["ends3"], p["funnel"]
    s1 = uni[uni["date"].isin(ends1.keys())]
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    rep = m3.replication(s1, ff5, umd)
    ex1 = book(s1, "ret1", allowed("allow150"))
    prim = summary(ex1, 3)
    p2, both = stats.idea_counts(ROOT / "research" / "IDEAS.md")
    alpha = stats.factor_alpha(ex1["excess"], stats.month_after(ex1["date"]), ff5, umd, 3)
    prim["factor_alpha"] = alpha
    prim["deflated_sharpe"] = stats.deflated_sharpe(ex1["excess"], both)
    prim["passes"] = stats.screen_pass(prim, alpha) and rep["passes"]
    prim["loss20_held"] = float(ex1["loss20_held"].mean())
    prim["loss20_universe"] = float(ex1["loss20_universe"].mean())
    res = {"idea": "M13", "prereg": "research/prereg/M13.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_sha256": sha5, "umd_sha256": shau, "replication": rep, "funnel": funnel, "primary": prim,
           "control": summary(book(s1, "ret1", allowed("allow150"), seed=SEED), 3)}
    s3 = uni[uni["date"].isin(ends3.keys())]
    top25 = lambda g: g[g["allow150"] == 1].nlargest(25, "dist150")
    per_year = ex1.assign(year=ex1["date"].dt.year).groupby("year")
    res["exploratory"] = {
        "ma200": summary(book(s1, "ret1", allowed("allow200")), 3),
        "above150_without_rising": summary(book(s1, "ret1", allowed("above150_only")), 3),
        "ma20": summary(book(s1, "ret1", allowed("allow20")), 3),
        "allowed_minus_not_allowed": spread(s1, "ret1", "allow150"),
        "hold_3m": summary(book(s3, "ret3", allowed("allow150")), 5),
        "per_year": per_year["excess"].mean().to_dict(),
        "share_allowed_by_year": s1.groupby(s1["date"].dt.year)["allow150"].mean().to_dict(),
        "loss20_held_vs_universe": {"held": prim["loss20_held"], "universe": prim["loss20_universe"]},
        "worst_month": {"held_minus_universe": float(ex1["excess"].min()),
                        "date": str(ex1.loc[ex1["excess"].idxmin(), "date"].date())},
        "top25_by_distance": summary(book(s1, "ret1", top25), 3),
        "fama_macbeth_with_momentum": fama_macbeth(s1, "allow150")}
    held = s1[s1["allow150"] == 1]
    res["qa_largest_held_returns"] = held.nlargest(20, "ret1")[["date", "symbol", "ret1"]].to_dict("records")
    rng = np.random.default_rng(SEED)
    h2 = held.dropna(subset=["dist150"])
    smp = h2.iloc[rng.choice(len(h2), min(5, len(h2)), replace=False)]
    res["qa_ma150_recomputed"] = qa_ma(smp, cache)
    out.mkdir(parents=True, exist_ok=True)
    (out / "M13_screen.json").write_text(json.dumps(res, indent=1, default=str))
    ex1.to_csv(out / "M13_monthly.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1, default=str))
    print("CONTROL", json.dumps(res["control"], indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "M13"))
    ap.add_argument("--budget-min", type=int, default=290)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out), a.budget_min)


if __name__ == "__main__":
    main()
