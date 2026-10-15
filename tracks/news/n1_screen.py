"""N1 historical screen (research/prereg/N1.md): two LLMs label headlines; short when both say NEG.

Three stages, each resumable, run by .github/workflows/n1-screen.yml:
  data     headlines, Alpaca assets, SEC ticker map and facts, daily bars -> universe and prices (cache)
  labels   both models label every in-universe headline of the window (stops at a daily quota; rerun
           daily until done), then writes 100 random labels to research/results/N1/label_sample.json
  returns  only after research/results/N1/label_check.md is committed (the hand check): trades, primary
           test, control and exploratory tables -> research/results/N1/N1_screen.json

Usage: python -m tracks.news.n1_screen --stage data|labels|returns --cache <dir> --out <dir>
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
from tracks.main.longterm import bars_batch, news_history, sec_data
from tracks.main.longterm.prices import align, segment_ids, split_points
from tracks.main.longterm.score import attach_fundamentals, eligible
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import llm, lm

ET = "America/New_York"
NEWS_FIRST, NEWS_LAST = "2025-01", "2025-09"
LAST_HEADLINE_DAY = pd.Timestamp("2025-09-24")       # every 3-day exit ends before 2025-10-01
BARS_START, BARS_END = "2023-11-01", "2025-09-30"
PANEL_FIRST = pd.Timestamp("2025-01-02")
MIN_SERIES_BARS = 253
ENTRY_LAG = pd.Timedelta(minutes=30)
HOLD = 3                                              # trading days, entry day = day 1
COST = 0.002 + 0.03 * HOLD / 252                      # config/risk.yaml news: round trip + borrow
SEED = 20261015
MIN_TRADES = 200
SAMPLE_N = 100
MODELS_FILE = Path("research/results/N1/models.json")
GAPS = {"gemini": 6.5, "groq": 2.5}                   # seconds between requests (free-tier per-minute limits)


# ---------- pure helpers (tested) ----------

def window_start(models):
    """First day of the month after the later of the two models' training cutoffs."""
    last = max(pd.Timestamp(m["cutoff"]) for m in models.values())
    return (last + pd.offsets.MonthBegin(1)).normalize()


def headline_rows(news, start, last_day):
    """Single-symbol headlines created in [start, last_day] (ET dates); exact repeats of the same headline
    for the same symbol within 3 days count once."""
    n = news.copy()
    n = n[n["symbols"].fillna("").ne("") & ~n["symbols"].fillna("").str.contains(",")]
    n = n[n["headline"].fillna("").str.strip().ne("")]
    n["t"] = pd.to_datetime(n["created_at"], utc=True).dt.tz_convert(ET)
    day = n["t"].dt.tz_localize(None).dt.normalize()
    n = n[(day >= start) & (day <= last_day)].sort_values("t")
    n["symbol"] = n["symbols"]
    n["hnorm"] = n["headline"].str.lower().str.split().str.join(" ")
    keep, last_seen = [], {}
    for i, s, h, t in zip(n.index, n["symbol"], n["hnorm"], n["t"]):
        k = (s, h)
        if k in last_seen and t - last_seen[k] <= pd.Timedelta(days=3):
            continue
        last_seen[k] = t
        keep.append(i)
    n = n.loc[keep]
    n["key"] = n["id"].astype(str) + "|" + n["symbol"]
    return n[["key", "id", "symbol", "headline", "t"]].reset_index(drop=True)


def timing(t_et, cal):
    """(prev trading day before the headline's date, entry day, exit day) for an ET timestamp; entry is the
    first open (09:30 ET) at least 30 minutes after t; exit is the close of trading day HOLD."""
    cal = pd.DatetimeIndex(cal)
    d = t_et.tz_localize(None).normalize()
    p = cal[cal.searchsorted(d) - 1] if cal.searchsorted(d) > 0 else pd.NaT
    ready = (t_et + ENTRY_LAG).tz_localize(None)
    k = cal.searchsorted(ready.normalize())
    if k < len(cal) and cal[k] == ready.normalize() and ready > cal[k] + pd.Timedelta(hours=9, minutes=30):
        k += 1
    if k + HOLD - 1 >= len(cal):
        return p, pd.NaT, pd.NaT
    return p, cal[k], cal[k + HOLD - 1]


def ret_between(px, symbol, entry, exit_):
    """Adjusted open(entry) -> close(last bar <= exit in the same series); NaN without an entry bar."""
    g = px.get(symbol)
    if g is None or entry not in g.index:
        return np.nan
    seg = g.at[entry, "seg"]
    w = g[(g.index >= entry) & (g.index <= exit_) & (g["seg"] == seg)]
    return w["c"].iloc[-1] / g.at[entry, "o"] - 1


def decide(day_labels):
    """Per (symbol, entry day): 'short' if any headline is NEG by both, 'long' if any is POS by both,
    None when both happen (skipped) or neither."""
    neg = any(a == b == "NEG" for a, b in day_labels)
    pos = any(a == b == "POS" for a, b in day_labels)
    if neg and pos:
        return None
    return "short" if neg else "long" if pos else None


def panel_rows(symbol, d):
    """Daily rows from PANEL_FIRST for one aligned bar frame: adjusted o/c, raw close, series id, bars so
    far in the series and the median 20-day raw dollar volume (never across a series break)."""
    seg = segment_ids(d.index)
    dv = pd.Series(d["rc"].to_numpy() * d["v"].to_numpy(), index=d.index)
    d = d.assign(symbol=symbol, seg=seg, nbars=dv.groupby(seg).cumcount().to_numpy() + 1,
                 dv20=dv.groupby(seg).transform(lambda x: x.rolling(20).median()).to_numpy())
    return d[d.index >= PANEL_FIRST].rename_axis("date").reset_index()


def daily_stats(trades, col):
    per_day = trades.groupby("entry")[col].mean().sort_index()
    m, se, t = nw_tstat(per_day.to_numpy(), 2)
    lo, hi = block_bootstrap_ci(per_day.to_numpy(), 5, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_trades": int(trades[col].notna().sum()),
            "n_days": int(per_day.notna().sum()), "hit_rate": float((trades[col] > 0).mean())}


def dictionary_baseline(sig, col):
    """N1 amendment (dictionary baseline): the same short trade on Loughran-McDonald NEG headlines, the
    per-day difference LLM minus dictionary on days both trade, and the trades only one of them takes.
    Reported beside the primary test; it does not change the decision."""
    ok = sig[sig[col].notna()]
    llm_s, lm_s = ok[ok["side"] == "short"], ok[ok["lm_side"] == "short"]
    out = {"dictionary_short": daily_stats(lm_s, col) if len(lm_s) > 2 else None,
           "llm_only": daily_stats(llm_s[llm_s["lm_side"] != "short"], col) if (llm_s["lm_side"] != "short").sum() > 2 else None,
           "dictionary_only": daily_stats(lm_s[lm_s["side"] != "short"], col) if (lm_s["side"] != "short").sum() > 2 else None,
           "overlap_trades": int(((ok["side"] == "short") & (ok["lm_side"] == "short")).sum())}
    a, b = llm_s.groupby("entry")[col].mean(), lm_s.groupby("entry")[col].mean()
    d = (a - b).dropna()
    if len(d) > 2:
        m, se, t = nw_tstat(d.to_numpy(), 2)
        out["llm_minus_dictionary_per_day"] = {"mean": m, "t_hac": t, "n_days": int(len(d))}
    return out


# ---------- stages ----------

def stage_data(cache):
    news_history.fetch_range(NEWS_FIRST, NEWS_LAST, cache / "news")
    h = alpaca_headers()
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all", headers=h))["SPY"]
    spy[["o", "c"]].to_parquet(cache / "spy.parquet")
    assets = bars_batch.assets_all()
    common = assets[assets["excl_reason"].isna()]
    tmap = sec_data.dera_ticker_map(2021, pd.Period("2025Q3"), cache / "dera")
    tmap = tmap[tmap["filed"] <= pd.Timestamp(BARS_END)]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    print(f"symbols: {len(syms)} common stocks named in Forms 3/4/5")
    (cache / "bars").mkdir(parents=True, exist_ok=True)
    for b in range(0, len(syms), 500):
        f = cache / "bars" / f"panel_{b // 500:04d}.parquet"
        if f.exists():
            continue
        fr = bars_batch.fetch_three(syms[b:b + 500], BARS_START, alpaca.sip_end(BARS_END))
        rows, rts = [], []
        for s in syms[b:b + 500]:
            if not all(s in fr[a] for a in ("all", "raw", "split")):
                continue
            d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
            if d.empty:
                continue
            rows.append(panel_rows(s, d))
            rts.append(split_points(s, d))
        part = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        pd.concat(rts, ignore_index=True).to_parquet(cache / "bars" / f"rtab_{b // 500:04d}.parquet")
        tmp = f.with_suffix(".tmp")
        part.to_parquet(tmp)
        tmp.replace(f)
        print(f"[bars] batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(part)} rows")
    panel = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("panel_*.parquet"))], ignore_index=True)
    rtab = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("rtab_*.parquet"))], ignore_index=True)
    panel[["symbol", "date", "o", "c", "seg"]].to_parquet(cache / "px.parquet")
    cand = panel[(panel["nbars"] >= MIN_SERIES_BARS) & (panel["rc"] >= 5) & (panel["dv20"] >= 2e6)]
    cand = cand.rename(columns={"rc": "price"})[["symbol", "date", "price", "dv20"]].reset_index(drop=True)
    cand["cik"] = sec_data.map_ciks(cand[["symbol", "date"]], tmap)
    cand = cand.dropna(subset=["cik"]).astype({"cik": "int64"})
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, cand["cik"].unique(), cache / "facts")
    uni = eligible(attach_fundamentals(cand, facts, rtab))
    uni[["date", "symbol", "cik", "mcap"]].to_parquet(cache / "universe.parquet")
    print(f"universe: {len(uni)} stock-days, {uni['symbol'].nunique()} stocks, {uni['date'].nunique()} days")


def items_in_universe(cache, models):
    news = news_history.load_range(NEWS_FIRST, NEWS_LAST, cache / "news")
    rows = headline_rows(news, window_start(models), LAST_HEADLINE_DAY)
    cal = pd.read_parquet(cache / "spy.parquet").index
    tm = [timing(t, cal) for t in rows["t"]]
    rows["prev"], rows["entry"], rows["exit"] = zip(*tm) if tm else ([], [], [])
    uni = pd.read_parquet(cache / "universe.parquet")
    ok = set(zip(uni["date"], uni["symbol"]))
    inu = rows[[(p, s) in ok for p, s in zip(rows["prev"], rows["symbol"])]].reset_index(drop=True)
    funnel = {"single_symbol_headlines_in_window": int(len(rows)), "in_universe": int(len(inu))}
    return inu, funnel


def stage_labels(cache, out, models, budget_min):
    inu, funnel = items_in_universe(cache, models)
    print(f"funnel: {funnel}")
    deadline = time.monotonic() + budget_min * 60
    items = list(zip(inu["key"], inu["symbol"], inu["headline"]))
    done = all([llm.label_all(items, p, models[p]["model"], cache / f"labels_{p}.jsonl", GAPS[p], deadline)
                for p in ("gemini", "groq")])
    if not done:
        print("LABELS INCOMPLETE: rerun on the next day (cached labels are kept)")
        return
    la, lb = llm.load_cache(cache / "labels_gemini.jsonl"), llm.load_cache(cache / "labels_groq.jsonl")
    inu["gemini"], inu["groq"] = inu["key"].map(la), inu["key"].map(lb)
    s = inu.sample(min(SAMPLE_N, len(inu)), random_state=SEED)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_sample.json").write_text(json.dumps({
        "funnel": funnel,
        "agreement": float((inu["gemini"] == inu["groq"]).mean()),
        "counts": {p: inu[p].value_counts().to_dict() for p in ("gemini", "groq")},
        "both_neg": int(((inu["gemini"] == "NEG") & (inu["groq"] == "NEG")).sum()),
        "both_pos": int(((inu["gemini"] == "POS") & (inu["groq"] == "POS")).sum()),
        "sample": [{"symbol": r.symbol, "t": f"{r.t:%Y-%m-%d %H:%M}", "headline": r.headline,
                    "gemini": r.gemini, "groq": r.groq} for r in s.itertuples()]}, indent=1))
    print(f"labels complete; sample written to {out / 'label_sample.json'} for the hand check")


def stage_returns(cache, out, models):
    if not (out / "label_check.md").exists():
        raise SystemExit("hand check missing: commit research/results/N1/label_check.md first")
    inu, funnel = items_in_universe(cache, models)
    la, lb = llm.load_cache(cache / "labels_gemini.jsonl"), llm.load_cache(cache / "labels_groq.jsonl")
    inu["gemini"], inu["groq"] = inu["key"].map(la), inu["key"].map(lb)
    assert inu["gemini"].notna().all() and inu["groq"].notna().all(), "labels incomplete"
    inu["lm"] = inu["headline"].map(lm.label)
    pxf = pd.read_parquet(cache / "px.parquet")
    px = {s: g.set_index("date") for s, g in pxf.groupby("symbol")}
    spy = pd.read_parquet(cache / "spy.parquet")
    cal = spy.index
    uni = pd.read_parquet(cache / "universe.parquet")
    by_day = uni.groupby("date")["symbol"].apply(list).to_dict()

    def exits(entry, k):
        return cal[cal.searchsorted(entry) + k - 1]

    def spy_r(entry, ex):
        return spy.at[ex, "c"] / spy.at[entry, "o"] - 1

    sig = []
    for (s, e), g in inu.dropna(subset=["entry"]).groupby(["symbol", "entry"]):
        side = decide(list(zip(g["gemini"], g["groq"])))
        g_neg = (g["gemini"] == "NEG").any()
        q_neg = (g["groq"] == "NEG").any()
        t0 = g["t"].min()
        inside = t0.weekday() < 5 and pd.Timedelta(hours=9, minutes=30) <= t0 - t0.normalize() < pd.Timedelta(hours=16)
        sig.append({"symbol": s, "entry": e, "prev": g["prev"].iloc[0], "side": side, "gemini_neg": bool(g_neg),
                    "groq_neg": bool(q_neg), "inside_hours": bool(inside), "month": f"{e:%Y-%m}",
                    "lm_side": decide(list(zip(g["lm"], g["lm"])))})
    sig = pd.DataFrame(sig)
    for k in (1, HOLD, 5):
        ex = sig["entry"].map(lambda e: exits(e, k) if cal.searchsorted(e) + k - 1 < len(cal) else pd.NaT)
        r = [ret_between(px, s, e, x) - spy_r(e, x) if pd.notna(x) else np.nan
             for s, e, x in zip(sig["symbol"], sig["entry"], ex)]
        sig[f"xr{k}"] = r
    cost = {1: 0.002 + 0.03 / 252, HOLD: COST, 5: 0.002 + 0.03 * 5 / 252}
    for k in (1, HOLD, 5):
        sig[f"short{k}"] = -sig[f"xr{k}"] - cost[k]
    shorts = sig[(sig["side"] == "short") & sig[f"short{HOLD}"].notna()]

    rows = []
    for e, g in shorts.groupby("entry"):
        pool = [s for s in by_day.get(g["prev"].iloc[0], [])]
        rng = np.random.default_rng(SEED + e.toordinal())
        pick = rng.choice(len(pool), size=min(len(g), len(pool)), replace=False) if pool else []
        x = exits(e, HOLD)
        for i in pick:
            rr = ret_between(px, pool[i], e, x)
            rows.append({"entry": e, "symbol": pool[i], "ctrl": -(rr - spy_r(e, x)) - COST})
    ctrl = pd.DataFrame(rows, columns=["entry", "symbol", "ctrl"]).dropna()

    prim = daily_stats(shorts, f"short{HOLD}")
    per_day = shorts.groupby("entry")[f"short{HOLD}"].mean().sort_index()
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fac = ffd.join(umdd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]]
    wins = [cal[cal.searchsorted(e):cal.searchsorted(e) + HOLD] for e in per_day.index]
    alpha = stats.ols_hac(per_day.to_numpy(), stats.window_factors(fac, wins), 2)
    alpha["factors"] = list(fac.columns) + ["(summed over each trade's 3 days, close to close)"]
    p2, both = stats.idea_counts()
    prim["factor_alpha"] = alpha
    prim["deflated_sharpe"] = stats.deflated_sharpe(per_day.to_numpy(), both)
    prim["passes"] = bool(prim["n_trades"] >= MIN_TRADES) and stats.screen_pass(prim, alpha)
    res = {"idea": "N1", "prereg": "research/prereg/N1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_daily_sha256": sha5, "umd_daily_sha256": shau, "models": models, "window_start": f"{window_start(models):%Y-%m-%d}",
           "funnel": {**funnel, "stock_days_with_signal": int(sig["side"].notna().sum()),
                      "short_trades": int(len(shorts)), "skipped_conflicting": int(sig["side"].isna().sum())},
           "primary": prim, "control": daily_stats(ctrl, "ctrl") if len(ctrl) else None}
    longs = sig[(sig["side"] == "long") & sig[f"xr{HOLD}"].notna()].assign(lr=lambda d: d[f"xr{HOLD}"] - 0.002)
    one = sig[sig[f"short{HOLD}"].notna()]
    res["exploratory"] = {
        "long_leg": daily_stats(longs, "lr") if len(longs) else None,
        "gemini_alone": daily_stats(one[one["gemini_neg"]], f"short{HOLD}"),
        "groq_alone": daily_stats(one[one["groq_neg"]], f"short{HOLD}"),
        "outside_hours": daily_stats(shorts[~shorts["inside_hours"]], f"short{HOLD}"),
        "inside_hours": daily_stats(shorts[shorts["inside_hours"]], f"short{HOLD}") if shorts["inside_hours"].any() else None,
        "hold_1": daily_stats(sig[(sig["side"] == "short") & sig["short1"].notna()], "short1"),
        "hold_5": daily_stats(sig[(sig["side"] == "short") & sig["short5"].notna()], "short5"),
        "per_month": shorts.groupby("month")[f"short{HOLD}"].agg(["mean", "count"]).reset_index().to_dict("records")}
    res["dictionary_baseline"] = dictionary_baseline(sig, f"short{HOLD}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "N1_screen.json").write_text(json.dumps(res, indent=1, default=str))
    shorts.sort_values(f"short{HOLD}").to_csv(out / "N1_trades.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(res["control"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "labels", "returns"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=50)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    models = json.loads(MODELS_FILE.read_text())["models"]
    if a.stage == "data":
        stage_data(cache)
    elif a.stage == "labels":
        stage_labels(cache, out, models, a.budget_min)
    else:
        stage_returns(cache, out, models)


if __name__ == "__main__":
    main()
