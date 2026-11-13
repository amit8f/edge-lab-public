"""DIR-GATE (research/prereg/DIRECTION.md): NX3's walk-forward long picks kept only when the headline is novel
(TF-IDF cosine < 0.8 against the same stock's headlines of the prior 20 sessions) and carries neither NX1's
earnings flag nor its FDA flag. No fitted model: the vocabulary is fitted on 2019-2022 headlines only.

Reads NX3's trades (cache/nx1/nx3/trades_v2) and the Benzinga months in NX1's cache (headline text stays in the
cache; only summary numbers are written to research/results/DIR/DIRGATE.json).

Usage: python -m tracks.news.dir_gate --cache cache/nx1 --out research/results/DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import w1_data as W

IDEA = 2                                   # DIR-GATE is test 2 of DIRECTION.md (control seed yyyymmdd + 2)
NOVEL = 0.8
LOOKBACK = 20                              # sessions
VOCAB_YEARS = (2019, 2022)
HOLDOUT = pd.Timestamp("2025-10-01")
MIN_TRADES, PASS_GAIN = 500, 0.001
BOOT, BOOT_SEED = 2000, 0


# ---------- pure helpers (tested) ----------

def max_similarity(vec, event_texts, prior_texts):
    """Highest cosine between any of the event's headlines and any prior headline (0 when there is none)."""
    if not len(event_texts) or not len(prior_texts):
        return 0.0
    a, b = vec.transform(list(event_texts)), vec.transform(list(prior_texts))
    return float((a @ b.T).max()) if a.nnz and b.nnz else 0.0


def gated(picks):
    """The registered gate: novel AND no earnings flag AND no FDA flag."""
    return picks[(picks["sim"] < NOVEL) & (picks["f_earnings"] == 0) & (picks["f_fda"] == 0)]


def control_net(day_events, k, d):
    """k of the day's real NX3 events at random, taken long; seed yyyymmdd + idea number."""
    rng = np.random.default_rng(int(pd.Timestamp(d).strftime("%Y%m%d")) + IDEA)
    e = day_events[np.isfinite(day_events["net_long"])]
    if len(e) < k or k == 0:
        return np.nan
    return float(e["net_long"].to_numpy()[rng.choice(len(e), size=k, replace=False)].mean())


def boot_pooled_diff(a, b, n=BOOT, seed=BOOT_SEED):
    """Pooled per-trade mean of a minus mean of b, days resampled together (a, b: frames with d and net)."""
    days = np.array(sorted(set(a["d"]) | set(b["d"])))
    ga = a.groupby("d")["net"].agg(["sum", "size"]).reindex(days, fill_value=0)
    gb = b.groupby("d")["net"].agg(["sum", "size"]).reindex(days, fill_value=0)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(days), len(days))
        sa, na, sb, nb = ga["sum"].to_numpy()[j].sum(), ga["size"].to_numpy()[j].sum(), \
            gb["sum"].to_numpy()[j].sum(), gb["size"].to_numpy()[j].sum()
        out[i] = sa / na - sb / nb if na and nb else np.nan
    est = a["net"].mean() - b["net"].mean()
    return {"mean": float(est), "lo": float(np.nanquantile(out, 0.025)), "hi": float(np.nanquantile(out, 0.975))}


def boot_days(x, n=BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "days": len(x)}
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)),
            "days": len(x)}


def report(longs, events):
    """longs: NX3 long picks (d, symbol, net, sim, f_*); events: every real event of the test days (d, net_long)."""
    g = gated(longs)
    gain = boot_pooled_diff(g, longs)
    by_day = events.groupby("d")
    ctl = []
    for d, gd in g.groupby("d"):
        c = control_net(by_day.get_group(d), len(gd), d) if d in by_day.groups else np.nan
        ctl.append(gd["net"].mean() - c)
    vs_ctl = boot_days(ctl)
    own = boot_days(g.groupby("d")["net"].mean().to_numpy())
    n = len(g)
    passed = (n >= MIN_TRADES and gain["mean"] >= PASS_GAIN and gain["lo"] > 0 and g["net"].mean() > 0
              and vs_ctl["lo"] is not None and vs_ctl["lo"] > 0)
    flags = {k: {"n": int((longs[f"f_{k}"] == 1).sum()), "mean_net": float(longs.loc[longs[f"f_{k}"] == 1, "net"].mean())
                 if (longs[f"f_{k}"] == 1).any() else None} for k in N.FLAGS if f"f_{k}" in longs}
    dec = pd.qcut(longs["sim"].rank(method="first"), 10, labels=False) if len(longs) >= 10 else None
    deciles = [] if dec is None else [{"decile": int(q) + 1, "sim_max": float(s["sim"].max()), "n": len(s),
                                       "mean_net": float(s["net"].mean())} for q, s in longs.groupby(dec)]
    return {"verdict": "too few" if n < MIN_TRADES else "passed" if passed else "failed",
            "rule": "gated longs: n >= 500, mean net minus all NX3 longs >= +0.10%/trade with day-clustered 95% lower "
                    "bound > 0, own mean net > 0, and gated minus same-day random control lower bound > 0",
            "gated_trades": n, "all_longs": len(longs), "gated_share": float(n / len(longs)) if len(longs) else None,
            "gated_mean_net": float(g["net"].mean()) if n else None, "all_longs_mean_net": float(longs["net"].mean()),
            "gated_hit_rate": float((g["net"] > 0).mean()) if n else None,
            "all_longs_hit_rate": float((longs["net"] > 0).mean()),
            "gain_over_all_longs": gain, "gated_minus_control_per_day": vs_ctl, "gated_own_per_day": own,
            "novel_share": float((longs["sim"] < NOVEL).mean()),
            "per_year": {int(y): {"n": len(s), "mean_net": float(s["net"].mean())}
                         for y, s in g.groupby(pd.DatetimeIndex(g["d"]).year)},
            "flags_all_longs": flags, "novelty_deciles_all_longs": deciles}


# ---------- stage (disk) ----------

def nx3_longs(cache):
    """NX3's walk-forward (3 + 3 a day, same models and seeds as the registered NX3 history): long picks and every
    real test event with its long net."""
    d3 = X.nx3dir(cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    longs, events = [], []
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = ev[ev["d"] <= cut_d]
        te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01")) & (ev["d"] < HOLDOUT)]
        if tr.empty or te.empty:
            continue
        _, pk, real, _ = X.evaluate(te, X.Model(tr), X.SEED + yr)
        lo = pk[pk["side"] == 1]
        longs.append(lo.assign(net=N.net(lo).to_numpy(float)))
        events.append(real.assign(side=1).pipe(lambda r: r.assign(net_long=N.net(r).to_numpy(float)))[["d", "symbol", "net_long"]])
        print(f"[dirgate] {yr}: {len(lo)} NX3 longs")
    return pd.concat(longs, ignore_index=True), pd.concat(events, ignore_index=True)


def news_texts(cache, symbols, first, last):
    """Single-symbol headlines (symbol, t ET, normalized text) of the given symbols, plus the 2019-2022 texts."""
    keep, vocab = [], []
    for per in pd.period_range(first, last, freq="M"):
        n = W.news_month(cache, per)
        n = n[n["symbols"].fillna("").ne("") & ~n["symbols"].fillna("").str.contains(",")]
        n = n[n["headline"].fillna("").str.strip().ne("")]
        h = n["headline"].map(N.norm)
        if VOCAB_YEARS[0] <= per.year <= VOCAB_YEARS[1]:
            vocab.append(h.drop_duplicates())
        m = n["symbols"].isin(symbols)
        keep.append(pd.DataFrame({"symbol": n.loc[m, "symbols"].to_numpy(), "t": W.et_time(n.loc[m, "created_at"]).to_numpy(),
                                  "text": h[m].to_numpy()}))
    return pd.concat(keep, ignore_index=True).sort_values("t"), pd.concat(vocab, ignore_index=True).drop_duplicates()


def similarities(longs, texts, vec, ses):
    s = ses.set_index("date")
    pct = pd.DatetimeIndex(ses["prev_close_t"])
    by = {k: g for k, g in texts.groupby("symbol")}
    out = np.zeros(len(longs))
    for i, (sym, d, t) in enumerate(zip(longs["symbol"], longs["d"], longs["t"])):
        g = by.get(sym)
        if g is None:
            continue
        k = ses.index[ses["date"] == d][0]
        start, pc = pct[max(k - LOOKBACK, 0)], s.loc[d, "prev_close_t"]
        tt = g["t"].to_numpy()
        ev_txt = g["text"].to_numpy()[(tt > np.datetime64(pc)) & (tt <= np.datetime64(t))]
        pr_txt = g["text"].to_numpy()[(tt > np.datetime64(start)) & (tt <= np.datetime64(pc))]
        out[i] = max_similarity(vec, ev_txt, pr_txt)
    return out


def run(cache, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    longs, events = nx3_longs(cache)
    ses = X.sessions(X.calendar(cache)).reset_index(drop=True)
    first = (longs["d"].min() - pd.Timedelta(days=45)).to_period("M")
    texts, vocab = news_texts(cache, set(longs["symbol"]), first, pd.Period("2025-09", "M"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, dtype=np.float32).fit(vocab.to_numpy())
    print(f"[dirgate] vocabulary {len(vec.vocabulary_)} terms from {len(vocab)} headlines; {len(texts)} pick-stock headlines")
    longs = longs.assign(sim=similarities(longs, texts, vec, ses))
    rep = report(longs, events)
    rep |= {"idea_count": "145 (160 with phase 1)", "period": f"{longs['d'].min():%Y-%m-%d}..{longs['d'].max():%Y-%m-%d}",
            "vocabulary": {"years": list(VOCAB_YEARS), "headlines": len(vocab), "terms": len(vec.vocabulary_),
                           "ngrams": [1, 2], "min_df": 2},
            "novelty_note": "a pick's similarity is the highest cosine between any of its event headlines (previous "
                            "close to the decision time) and any of the stock's headlines in the 20 sessions before"}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "DIRGATE.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/DIR")
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
