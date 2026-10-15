"""M4 historical screen (research/prereg/M4.md). Runs on GitHub Actions (needs Alpaca and SEC access).

Usage: python -m tracks.main.longterm.screen --cache <dir> --out <dir>
Writes <out>/M4_screen.json (every number reported) and <out>/M4_calibration.json (labels for the
dashboard). Prints the funnel first, then the primary result, then the exploratory tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import horizon_ends
from tracks.main.longterm.score import FAMILIES, TOP_N, attach_fundamentals, composite, eligible
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

FIRST_SIGNAL = pd.Timestamp("2015-12-31")
LAST_SIGNAL = pd.Timestamp("2025-03-31")
LABEL_END = pd.Timestamp("2025-10-01")      # phase-2 holdout: no window may reach it
BARS_START = "2014-06-01"
BARS_END = "2025-09-30"
COST = 0.002                                 # config/risk.yaml main.round_trip_cost
SEED = 20261012
BUCKETS = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]


def month_ends(calendar, first, last):
    s = pd.Series(pd.DatetimeIndex(calendar))
    me = s.groupby(s.dt.to_period("M")).max()
    return [d for d in me if first <= d <= last]


def spy_returns(spy, ends):
    """SPY forward return per horizon and t, same entry/exit rule as the stocks."""
    out = {}
    idx = spy.index
    for name, m in ends.items():
        r = {}
        for t, end in m.items():
            i = idx.get_loc(t)
            x = idx.searchsorted(end, side="right") - 1
            r[t] = spy["c"].iloc[x] / spy["o"].iloc[i + 1] - 1
        out[name] = r
    return out


def excess_series(scored, col, n, cost, rng_seed=None):
    """Per date: mean `col` of the top n (or n random names when rng_seed is set) minus the universe
    mean, minus cost. Picks without a return (no next-day bar) are dropped, not replaced, and counted."""
    rows = []
    for t, g in scored.groupby("date", sort=True):
        uni = g[col].mean()
        if rng_seed is None:
            top = g.nsmallest(n, "rank")
        else:
            rng = np.random.default_rng(rng_seed + t.toordinal())
            top = g.iloc[rng.choice(len(g), size=min(n, len(g)), replace=False)]
        k = top[col].notna().sum()
        rows.append({"date": t, "pick_mean": top[col].mean(), "uni_mean": uni,
                     "excess": top[col].mean() - uni - cost, "n_picks": int(k), "n_universe": int(g[col].notna().sum())})
    return pd.DataFrame(rows)


def summarize(ex, lags, block=6):
    m, se, t = nw_tstat(ex["excess"], lags)
    lo, hi = block_bootstrap_ci(ex["excess"].to_numpy(), block, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_dates": int(ex["excess"].notna().sum()),
            "hit_rate": float((ex["excess"] > 0).mean()),
            "mean_pick_return": float(ex["pick_mean"].mean()), "mean_universe_return": float(ex["uni_mean"].mean())}


def calibration(scored, spy6):
    """By composite percentile bucket: n, median and quartiles of 6-month return, share beating SPY and
    the universe. Exploratory; used only to label the dashboard."""
    d = scored[scored["ret6"].notna()].copy()
    d["spy6"] = d["date"].map(spy6)
    d["uni6"] = d.groupby("date")["ret6"].transform("mean")
    d["bucket"] = pd.cut(d["score_pct"], BUCKETS, include_lowest=True)
    out = []
    for b, g in d.groupby("bucket", observed=True):
        out.append({"lo": b.left, "hi": b.right, "n": int(len(g)), "n_dates": int(g["date"].nunique()),
                    "median": float(g["ret6"].median()), "p25": float(g["ret6"].quantile(.25)),
                    "p75": float(g["ret6"].quantile(.75)), "mean": float(g["ret6"].mean()),
                    "beat_spy": float((g["ret6"] > g["spy6"]).mean()),
                    "beat_universe": float((g["ret6"] > g["uni6"]).mean())})
    top = d[d["rank"] <= TOP_N]
    out_top = {"n": int(len(top)), "median": float(top["ret6"].median()), "p25": float(top["ret6"].quantile(.25)),
               "p75": float(top["ret6"].quantile(.75)), "mean": float(top["ret6"].mean()),
               "beat_spy": float((top["ret6"] > top["spy6"]).mean()),
               "beat_universe": float((top["ret6"] > top["uni6"]).mean()),
               "spy_median": float(pd.Series(spy6).median())}
    return {"buckets": out, "top20": out_top}


def prepare(cache):
    """Shared by M4 and M5: signal dates, SPY returns, and the eligible universe with price features and
    fundamentals. Returns (uni, ends, spy_r, funnel)."""
    cache = Path(cache)
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all",
                                     headers=alpaca_headers()))["SPY"]
    dates = month_ends(spy.index, FIRST_SIGNAL, LAST_SIGNAL)
    ends = horizon_ends(spy.index, dates, LABEL_END)
    spy_r = spy_returns(spy, ends)
    print(f"signal dates {len(dates)} ({dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}); with 6m label {len(ends['ret6'])}, "
          f"12m label {len(ends['ret12'])}")

    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2013, pd.Period(LAST_SIGNAL, "Q"), cache / "dera")
    tmap = tmap[tmap["filed"] <= LAST_SIGNAL]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    funnel = {"alpaca_assets": int(len(assets)), "alpaca_common": int(common["base_symbol"].nunique()),
              "common_named_in_form345": len(syms)}
    print(f"funnel: {funnel}")

    feat, rtab, n_bars = bars_batch.run_batches(syms, BARS_START, alpaca.sip_end(BARS_END), dates, ends,
                                                cache / "bars")
    funnel["symbols_with_bars"] = n_bars
    funnel["symbol_dates_with_lookback"] = int(len(feat))
    feat["cik"] = sec_data.map_ciks(feat[["symbol", "date"]], tmap)
    funnel["unmapped_symbol_dates"] = int(feat["cik"].isna().sum())
    feat = feat.dropna(subset=["cik"]).astype({"cik": "int64"})

    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, feat["cik"].unique(), cache / "facts")
    full = attach_fundamentals(feat, facts, rtab)
    funnel["with_mcap"] = int(full["mcap"].notna().sum())
    uni = eligible(full)
    funnel["eligible_price_mcap_dv"] = int(len(uni))
    return uni, ends, spy_r, funnel


def run(cache, out):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"idea": "M4", "prereg": "research/prereg/M4.md", "phase2_count": 1, "main_count": 1,
           "phase1_plus_phase2": 16}
    uni, ends, spy_r, funnel = prepare(cache)
    scored = composite(uni)
    funnel["with_4_families"] = int(len(scored))
    per_date = scored.groupby("date").size()
    funnel["universe_per_date"] = {"min": int(per_date.min()), "median": float(per_date.median()),
                                   "max": int(per_date.max())}
    funnel["family_coverage"] = {f: float(scored[f].notna().mean()) for f in FAMILIES}
    res["funnel"] = funnel
    print("FUNNEL", json.dumps(funnel, indent=1))

    # ---- primary test (pre-registered) ----
    s6 = scored[scored["date"].isin(ends["ret6"].keys())]
    ex = excess_series(s6, "ret6", TOP_N, COST)
    ctrl = excess_series(s6, "ret6", TOP_N, COST, rng_seed=SEED)
    prim = summarize(ex, 5)
    prim["passes"] = bool(prim["mean"] > 0 and prim["t_hac"] >= 2.0 and prim["ci95"][0] > 0)
    prim["picks_without_return"] = int((TOP_N - ex["n_picks"]).sum())
    res["primary"] = prim
    res["control"] = summarize(ctrl, 5)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))

    # ---- QA ----
    r6 = scored["ret6"].dropna()
    res["qa"] = {"ret6_percentiles": {str(q): float(r6.quantile(q)) for q in (0, .001, .01, .25, .5, .75, .99, .999, 1)},
                 "universe_mean_ret6": float(ex["uni_mean"].mean()),
                 "spy_mean_ret6": float(np.mean(list(spy_r["ret6"].values()))),
                 "truncated_share": float(scored.loc[scored["ret6"].notna(), "ret6_trunc"].mean()),
                 "largest_ret6": scored.nlargest(20, "ret6")[["symbol", "date", "ret6", "price", "ret6_trunc"]]
                 .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
                 "smallest_ret6": scored.nsmallest(10, "ret6")[["symbol", "date", "ret6", "price", "ret6_trunc"]]
                 .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records")}
    print("QA", json.dumps(res["qa"], indent=1, default=str))

    # ---- exploratory (listed in the pre-registration) ----
    exp = {}
    s12 = scored[scored["date"].isin(ends["ret12"].keys())]
    exp["horizon_12m"] = summarize(excess_series(s12, "ret12", TOP_N, COST), 11)
    exp["top50"] = summarize(excess_series(s6, "ret6", 50, COST), 5)
    for f in FAMILIES:
        g = s6[s6[f].notna()].copy()
        g["rank"] = g.groupby("date")[f].rank(ascending=False, method="first")
        exp[f"alone_{f}"] = summarize(excess_series(g, "ret6", TOP_N, COST), 5)
    vs_spy = ex.assign(excess=ex["pick_mean"] - ex["date"].map(spy_r["ret6"]) - COST)
    exp["vs_spy"] = summarize(vs_spy, 5)
    exp["per_year"] = {str(y): float(g["excess"].mean()) for y, g in ex.groupby(ex["date"].dt.year)}
    s6d = s6[s6["ret6"].notna()].copy()
    s6d["decile"] = np.ceil(s6d.groupby("date")["score"].rank(pct=True, method="first") * 10).astype(int)
    s6d["uni"] = s6d.groupby("date")["ret6"].transform("mean")
    exp["deciles_mean_excess"] = {int(k): float(v) for k, v in
                                  (s6d["ret6"] - s6d["uni"]).groupby(s6d["decile"]).mean().items()}
    res["exploratory"] = exp
    print("EXPLORATORY", json.dumps(exp, indent=1))

    cal = calibration(s6, spy_r["ret6"])
    cal.update({"source": "M4 screen, signal dates 2015-12..2025-03, 6-month windows ending before 2025-10-01",
                "primary_passes": prim["passes"], "primary_mean_excess": prim["mean"], "primary_t": prim["t_hac"]})
    res["calibration"] = cal
    print("CALIBRATION", json.dumps(cal, indent=1, default=str))
    (out / "M4_screen.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "M4_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
    ex.to_csv(out / "M4_primary_series.csv", index=False)
    pk = s6[s6["rank"] <= TOP_N][["date", "rank", "symbol", "cik", "score", "ret6"] + FAMILIES]
    pk.to_csv(out / "M4_screen_picks.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
