"""NR1 (research/prereg/NR1.md): big stocks that fell hard against their sector with no news, bought at the next
opening auction and sold at the 5th closing auction.

Stage "events" builds, from W1's cache (`tracks/news/w1_data.py`), the daily top-500-by-market-cap universe,
each stock's 5-day return against its SPDR sector ETF, the no-news and low-turnover filters, the picks and their
same-day random controls, and writes the 20-pick hand check (research/checks/NR1.csv). Nothing in "events" reads
a return after day t. Stage "measure" waits for that check (>= 18 right).

Market cap on t = cover-page shares outstanding (latest 10-K/10-Q filed by t) x all-adjusted close on t x the raw
to all-adjusted price ratio on the share date, i.e. shares carried forward through splits (dividends move the
ratio by about their yield, which doesn't change a top-500 ranking). Rows below $5M of median dollar volume are
left out before ranking (no top-500 company trades that little); disclosed in the coverage file.

Usage: python -m tracks.main.nr1_study --stage events --cache cache/w1 --out research/results/NR1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core import alpaca
from tracks.main import ch_data
from tracks.main.longterm import sec_data
from tracks.main.longterm.fundamentals import as_of, share_records
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import w1_data as D
from tracks.penny import pd1_study as P

SEED, CHECK_N, CHECK_PASS = 20261102, 20, 18
TOP, MAX_PICKS, CUT = 500, 5, -0.03
MIN_PRICE, MIN_DV20 = 5.0, 5e6
BACK, MED_LO, MED_HI = 5, 64, 5        # 5-day return; median volume over t-64 .. t-5
HOLD, LAGS, BLOCK = 5, 5, 5
COST, COST_SPREAD = 0.0005, 0.0041
FIRST_T, HOLDOUT = pd.Timestamp("2017-01-03"), pd.Timestamp("2025-10-01")
TRIALS, MIN_TRADES, MIN_DAYS = 115, 300, 100
CHECK = Path("research/checks/NR1.csv")
ROW_COLS = ["symbol", "date", "c", "v", "ac", "dv20", "cik", "seg"]


def ndir(cache):
    return D.w1dir(cache) / "nr1"


def _atomic(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


def symbol_features(g, cal_pos):
    """Per symbol (rows sorted by date): 5-day all-adjusted return, mean volume t-4..t, median volume t-64..t-5,
    all inside one series segment and only where the bar 5 back is the calendar's t-5 (no missing days)."""
    g = g.sort_values("date")
    seg = g["seg"].to_numpy()
    same = pd.Series(seg).groupby(seg).cumcount().to_numpy()
    ac, v = g["ac"].to_numpy(float), g["v"].to_numpy(float)
    pos = cal_pos.reindex(g["date"]).to_numpy()
    ret5 = np.full(len(g), np.nan)
    ok = same >= BACK
    k = np.flatnonzero(ok)
    good = k[(pos[k] - pos[k - BACK]) == BACK]
    ret5[good] = ac[good] / ac[good - BACK] - 1
    vs = pd.Series(v)
    grp = vs.groupby(seg)
    vol5 = grp.transform(lambda x: x.rolling(BACK).mean()).to_numpy()
    med = grp.transform(lambda x: x.shift(MED_HI).rolling(MED_LO - MED_HI + 1).median()).to_numpy()
    return g.assign(ret5=ret5, vol5=vol5, volmed=med, k=g["ac"] / g["c"])


def stage_rows(cache, cal):
    """Candidate rows ($5+, dv20 >= $5M, with a CIK) with their features, one parquet per W1 batch."""
    cal_pos = pd.Series(np.arange(len(cal)), index=pd.DatetimeIndex(cal["date"]))
    d = ndir(cache) / "rows"
    for i, p in enumerate(D.batch_files(cache)):
        f = d / f"r_{i:04d}.parquet"
        if f.exists():
            continue
        r = pd.read_parquet(p, columns=ROW_COLS, filters=[("date", ">=", pd.Timestamp("2016-06-01")),
                                                          ("date", "<", HOLDOUT)])
        r = r[r["cik"].notna()]
        parts = [symbol_features(g, cal_pos) for _, g in r.groupby("symbol")]
        x = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ROW_COLS)
        keep = (x["c"] >= MIN_PRICE) & (x["dv20"] >= MIN_DV20) & (x["date"] >= FIRST_T - pd.Timedelta(days=10))
        _atomic(x[keep].astype({"cik": "int64"}), f)
        print(f"[nr1 rows] batch {i + 1}: {int(keep.sum())} rows")
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("r_*.parquet"))], ignore_index=True)


def market_caps(cache, rows):
    """mcap per row from cover-page shares (point in time by filing date)."""
    zp = ndir(cache) / "companyfacts.zip"
    fdir = ndir(cache) / "facts"
    if not any(fdir.glob("facts_*.parquet")):
        if not zp.exists():
            sec_data.download_companyfacts(zp)
        sec_data.facts_for(zp, rows["cik"].unique(), fdir)
        zp.unlink(missing_ok=True)   # 1.3 GB; the extracted facts are kept
    facts = pd.concat([pd.read_parquet(p) for p in sorted(fdir.glob("facts_*.parquet"))], ignore_index=True)
    sh = share_records(facts)
    m = as_of(rows[["cik", "date"]], sh)
    # raw/all ratio on the share date, from the same symbol's rows (last row on or before sh_end)
    key = rows[["symbol"]].assign(sh_end=pd.to_datetime(m["sh_end"].to_numpy()), _i=np.arange(len(rows)))
    ref = rows[["symbol", "date", "k"]].sort_values("date")
    j = pd.merge_asof(key.dropna(subset=["sh_end"]).sort_values("sh_end"), ref, left_on="sh_end", right_on="date",
                      by="symbol", direction="backward", tolerance=pd.Timedelta(days=10))
    k_sh = np.full(len(rows), np.nan)
    k_sh[j["_i"].to_numpy()] = j["k"].to_numpy()
    return m["shares"].to_numpy(float) * rows["ac"].to_numpy(float) / k_sh


def etf_ret5(cal):
    raw = fetch_daily_bars(ch_data.ETFS, "2016-06-01", alpaca.sip_end("2025-10-15"), adjustment="all",
                           headers=alpaca_headers())
    fr = to_frames(raw)
    days = pd.DatetimeIndex(cal["date"])
    out = {}
    for s, g in fr.items():
        c = g["c"].reindex(days)
        out[s] = c / c.shift(BACK) - 1
    return pd.DataFrame(out)


def news_window(keys, news_days, filings, cal):
    """True where the symbol had a Benzinga article, or the company an item 2.02 8-K, on the ET calendar dates
    from session t-4 through t."""
    days = pd.DatetimeIndex(cal["date"])
    pos = days.searchsorted(keys["date"].to_numpy())
    lo = days[np.maximum(pos - (BACK - 1), 0)]
    k = keys.assign(_lo=lo, _i=np.arange(len(keys)))
    hit = np.zeros(len(keys), dtype=bool)
    nd = news_days.rename(columns={"date": "_d"})
    fd = filings.assign(_d=filings["accepted_et"].dt.normalize())[["cik", "_d"]]
    for days_tab, col in ((nd, "symbol"), (fd, "cik")):
        j = k[[col, "_lo", "date", "_i"]].merge(days_tab, on=col)
        hit[j.loc[(j["_d"] >= j["_lo"]) & (j["_d"] <= j["date"]), "_i"].unique()] = True
    return hit


def stage_filings(cache, ciks):
    d = ndir(cache) / "subs"
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), P.BATCH):
        f = d / f"e_{b // P.BATCH:04d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + P.BATCH] for r in P.earnings_rows(c)]
        _atomic(pd.DataFrame(rows, columns=P.COLS), f)
        print(f"[nr1 filings] batch {b // P.BATCH + 1}/{-(-len(ciks) // P.BATCH)}: {len(rows)} earnings 8-Ks")
    f = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("e_*.parquet"))], ignore_index=True)
    f["accepted_et"] = pd.to_datetime(f["accepted_et"])
    return f


def choose(day, mask, col="resid"):
    """The MAX_PICKS lowest `col` below CUT among rows passing `mask` (ties by symbol)."""
    g = day[mask & (day[col] < CUT)].sort_values([col, "symbol"])
    return g.head(MAX_PICKS)


def build(uni, cal):
    """Primary picks, the reported-alongside variants, and one same-day random control per primary pick."""
    days = pd.DatetimeIndex(cal["date"])
    picks, ctrls = [], []
    variants = {"all_losers": lambda d: np.ones(len(d), bool), "no_news_only": lambda d: ~d["news"].to_numpy(),
                "low_turnover_only": lambda d: d["low_turn"].to_numpy(),
                "market_relative": lambda d: (~d["news"] & d["low_turn"]).to_numpy()}
    for t, day in uni.groupby("date"):
        i = days.searchsorted(t)
        if t < FIRST_T or i + HOLD >= len(days) or days[i + HOLD] >= HOLDOUT:
            continue
        entry, exit_ = days[i + 1], days[i + HOLD]
        base = {"t": t, "entry": entry, "exit": exit_}
        day = day.reset_index(drop=True)
        p = choose(day, (~day["news"] & day["low_turn"]).to_numpy())
        for r in p.itertuples():
            picks.append(base | {"variant": "primary", "symbol": r.symbol, "cik": r.cik, "resid": r.resid,
                                 "ret5": r.ret5, "etf": r.etf, "etf_ret5": r.etf_ret5, "vol5": r.vol5,
                                 "volmed": r.volmed, "mcap": r.mcap})
        for name, fn in variants.items():
            col = "resid_mkt" if name == "market_relative" else "resid"
            for r in choose(day, fn(day), col).itertuples():
                picks.append(base | {"variant": name, "symbol": r.symbol, "cik": r.cik, "resid": getattr(r, col)})
        if len(p):
            pool = day[~day["symbol"].isin(p["symbol"])].sort_values("symbol").reset_index(drop=True)
            k = np.random.default_rng(SEED + t.toordinal()).choice(len(pool), size=len(p), replace=False)
            for r in pool.iloc[np.sort(k)].itertuples():
                ctrls.append(base | {"symbol": r.symbol, "cik": r.cik})
    return pd.DataFrame(picks), pd.DataFrame(ctrls)


def stage_events(cache, out):
    cal = D.calendar()
    rows = stage_rows(cache, cal)
    rows["mcap"] = market_caps(cache, rows)
    rows = rows[rows["mcap"].notna() & (rows["date"] >= FIRST_T)]
    rows = rows.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])
    rows["rank"] = rows.groupby("date")["mcap"].rank(ascending=False, method="first")
    uni = rows[rows["rank"] <= TOP].copy()
    ciks = sorted(uni["cik"].unique().tolist())
    sic = industry.sic_codes(ciks, ndir(cache))
    uni["etf"] = [ch_data.sector_etf(sic.get(int(c))) for c in uni["cik"]]
    er = etf_ret5(cal)
    e = er.stack().rename("er").reset_index()
    e.columns = ["date", "etf", "er"]
    uni = uni.merge(e, on=["date", "etf"], how="left").merge(
        er["SPY"].rename("spy_ret5").reset_index(names="date"), on="date", how="left")
    uni["etf"] = np.where(uni["er"].isna(), "SPY", uni["etf"])        # XLC before 2018-06: SPY
    uni["etf_ret5"] = uni["er"].fillna(uni["spy_ret5"])
    uni["resid"] = uni["ret5"] - uni["etf_ret5"]
    uni["resid_mkt"] = uni["ret5"] - uni["spy_ret5"]
    fil = stage_filings(cache, ciks)
    uni["news"] = news_window(uni[["symbol", "cik", "date"]], D.news_symbol_days(cache), fil, cal)
    uni["low_turn"] = uni["vol5"] < uni["volmed"]
    picks, ctrls = build(uni, cal)
    for name, df in (("picks", picks), ("controls", ctrls)):
        _atomic(df, ndir(cache) / f"{name}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    prim = picks[picks["variant"] == "primary"].reset_index(drop=True)
    if not CHECK.exists() and len(prim):
        k = np.sort(np.random.default_rng(SEED).choice(len(prim), size=min(CHECK_N, len(prim)), replace=False))
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        prim.iloc[k][["t", "symbol", "cik", "ret5", "etf", "etf_ret5", "resid", "vol5", "volmed", "mcap"]].assign(
            t=lambda x: x["t"].dt.strftime("%Y-%m-%d"), ok="", note="").to_csv(CHECK, index=False)
    days_n = uni["date"].nunique()
    cov = {"universe_days": int(days_n), "universe_rows": int(len(uni)),
           "median_universe_size": float(uni.groupby("date").size().median()),
           "rows_with_ret5": int(uni["ret5"].notna().sum()), "news_share": float(uni["news"].mean()),
           "low_turnover_share": float(uni["low_turn"].mean()),
           "etf_counts": {k: int(v) for k, v in uni["etf"].value_counts().items()},
           "primary_picks": int(len(prim)), "primary_days": int(prim["t"].nunique()) if len(prim) else 0,
           "picks_per_year": {str(y): int(n) for y, n in prim["t"].dt.year.value_counts().sort_index().items()}
           if len(prim) else {},
           "variant_picks": {k: int(v) for k, v in picks["variant"].value_counts().items()},
           "controls": int(len(ctrls)), "earnings_8k": int(len(fil)),
           "prefilter": f"raw close >= ${MIN_PRICE:.0f} and 20-day median dollar volume >= ${MIN_DV20 / 1e6:.0f}M "
                        "before the market-cap ranking"}
    (out / "NR1_coverage.json").write_text(json.dumps(cov, indent=1))
    print("NR1 FUNNEL", json.dumps(cov))


# ---------- measure ----------

def returns(cache, rows):
    out = np.full(len(rows), np.nan)
    rows = rows.reset_index(drop=True)
    idx = {s: np.flatnonzero(rows["symbol"].to_numpy() == s) for s in rows["symbol"].unique()}
    for bars in P.per_batch(cache, set(rows["symbol"])):
        for s, g in bars.items():
            for i in idx.get(s, []):
                out[i] = P.trade_ret(g, rows.at[i, "entry"], rows.at[i, "exit"])
    return rows.assign(r=out)


def daily(p, c, cost):
    p = p.dropna(subset=["r"]).assign(net=lambda x: x["r"] - cost)
    c = c.dropna(subset=["r"]).assign(net=lambda x: x["r"] - cost)
    w = p.groupby("t").agg(mean=("net", "mean"), size=("net", "size"), entry=("entry", "first"),
                           exit=("exit", "first"))
    w = w.join(c.groupby("t")["net"].mean().rename("ctrl"), how="inner")
    return w.assign(diff=w["mean"] - w["ctrl"])


def test(w, fd, cal):
    m, se, t = nw_tstat(w["diff"], LAGS)
    lo, hi = block_bootstrap_ci(w["diff"], BLOCK, 10_000, SEED)
    pm = float(w["mean"].mean())
    days = pd.DatetimeIndex(cal["date"])
    wins = [days[(days >= a) & (days <= b)] for a, b in zip(w["entry"], w["exit"])]
    F = stats.window_factors(fd, wins)
    a = stats.ols_hac(w["diff"].to_numpy(float), F, LAGS)
    return {"days": int(len(w)), "trades": int(w["size"].sum()), "mean_minus_control": m, "se_hac": se,
            "t_hac": t, "ci95": [lo, hi], "picks_mean_net": pm, "control_mean_net": float(w["ctrl"].mean()),
            "ff5_mom_alpha": a}


def stage_measure(cache, out):
    n_ok = check_right()
    res = {"idea": "NR1", "prereg": "research/prereg/NR1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok is None:
        raise SystemExit("[nr1] fill in research/checks/NR1.csv (ok = y/n) first")
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "NR1_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    cal = D.calendar()
    picks = returns(cache, pd.read_parquet(ndir(cache) / "picks.parquet"))
    ctrls = returns(cache, pd.read_parquet(ndir(cache) / "controls.parquet"))
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fd = ffd[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(umdd, how="inner")
    fd.index = pd.DatetimeIndex(fd.index)
    prim_p = picks[picks["variant"] == "primary"]
    w = daily(prim_p, ctrls, COST)
    prim = test(w, fd, cal)
    a = prim["ff5_mom_alpha"]
    prim["passes_rule12"] = bool(prim["mean_minus_control"] > 0 and prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0
                                 and prim["picks_mean_net"] > 0 and prim["trades"] >= MIN_TRADES
                                 and prim["days"] >= MIN_DAYS)
    prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
    prim["deflated_sharpe"] = stats.deflated_sharpe(w["diff"], TRIALS)
    prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
    prim["near_miss_rule19"] = bool(not prim["passes"] and prim["t_hac"] >= 1.5 and prim["picks_mean_net"] > 0)
    nets = prim_p.dropna(subset=["r"])["r"] - COST
    alongside = {
        "at_0.41pct_cost": test(daily(prim_p, ctrls, COST_SPREAD), fd, cal),
        **{v: test(daily(picks[picks["variant"] == v], ctrls, COST), fd, cal)
           for v in ("all_losers", "no_news_only", "low_turnover_only", "market_relative")},
        "by_year": {str(y): {"days": int(len(g)), "mean_minus_control": float(g["diff"].mean())}
                    for y, g in w.groupby(w.index.year)},
        "best_5pct_trade_mean": float(nets[nets >= nets.quantile(0.95)].mean()),
        "worst_5pct_trade_mean": float(nets[nets <= nets.quantile(0.05)].mean()),
    }
    allw = daily(picks[picks["variant"] == "all_losers"], ctrls, COST)
    both = w[["mean"]].join(allw[["mean"]].rename(columns={"mean": "all"}), how="inner")
    alongside["quick_kill_filtered_minus_all_losers"] = float((both["mean"] - both["all"]).mean())
    alongside["quick_kill_rule"] = "kill below +0.1% a week (one 5-day trade = one week)"
    p2, both_n = stats.idea_counts()
    res |= {"phase2_count": p2, "phase1_plus_phase2": both_n, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"days": int(prim_p["t"].nunique()), "picks": int(len(prim_p)),
                       "picks_with_trade": int(prim_p["r"].notna().sum()),
                       "controls_with_trade": int(ctrls["r"].notna().sum())},
            "primary": prim, "reported_alongside": alongside}
    (out / "NR1_result.json").write_text(json.dumps(res, indent=1, default=str))
    w.to_csv(out / "NR1_daily.csv")
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def check_right():
    if not CHECK.exists():
        return None
    ok = pd.read_csv(CHECK)["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/NR1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    (stage_events if a.stage == "events" else stage_measure)(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
