"""META2 (research/prereg/META2.md): META1's take-or-skip model for R1's picks, plus 8 inputs on the news and
short selling behind each drop.

Stage "data": per ~500-symbol batch, META1's inputs plus 20-day mean raw volume and a recent-split flag per
pick (Alpaca SIP, all and raw); then the sources (FINRA short interest, Reg SHO short volume, Benzinga news),
read from B0's cache when it is complete, else fetched the same way into this cache; then the 8 new inputs
per pick year. Stage "fit": the walk-forward, the primary test and the reported-alongside numbers. Each
stage resumes from its cache.

Usage: python -m tracks.main.meta2_study --stage data|fit|all --cache <dir> [--b0-src <dir>] --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main import meta1_study as M1
from tracks.main import weekly_panel
from tracks.main.longterm import news_history, sec_data
from tracks.main.longterm.prices import segment_ids
from tracks.news import lm
from tracks.penny import b0_study as B0

NEW = ["news5", "neg5", "pos5", "news60", "dtc", "si_chg", "shvr5", "shvr_chg"]
INPUTS = M1.INPUTS + NEW
NEW_ONLY = ["ret5"] + NEW
YEARS = range(2017, 2026)
BASE, WIN = 60, 5


def extra_inputs(d, dates):
    """20-day mean raw volume (shares) and a split in the last 30 bars (raw over adjusted close moved by
    more than 20% in a day), at each date, within the current series."""
    seg = segment_ids(d.index)
    v = d["v"].to_numpy(float)
    f = (d["rc"] / d["c"]).to_numpy(float)
    jump = np.r_[False, np.abs(np.diff(np.log(f))) > np.log(1.2)]
    out = {}
    for t in dates:
        i = d.index.searchsorted(t)
        if i >= len(d) or d.index[i] != t:
            continue
        s0 = int(np.searchsorted(seg, seg[i]))
        if i - s0 + 1 < 20:
            continue
        out[t] = {"avgvol20": float(v[i - 19:i + 1].mean()), "split30": bool(jump[max(s0 + 1, i - 29):i + 1].any())}
    return out


def stage_bars(cache, p):
    syms = sorted(p["symbol"].unique())
    bdir = cache / "inputs"
    bdir.mkdir(parents=True, exist_ok=True)
    by_sym = p.groupby("symbol")["date"].apply(list).to_dict()
    end = alpaca.sip_end(M1.BARS_END)
    for b in range(0, len(syms), M1.BATCH):
        f = bdir / f"in_{b // M1.BATCH:04d}.parquet"
        if f.exists():
            continue
        part = syms[b:b + M1.BATCH]
        h = alpaca_headers()
        fa = to_frames(fetch_daily_bars(part, M1.BARS_START, end, adjustment="all", headers=h))
        fr = to_frames(fetch_daily_bars(part, M1.BARS_START, end, adjustment="raw", headers=h))
        rows = []
        for s in part:
            if s not in fa or s not in fr:
                continue
            d = pd.DataFrame({"o": fa[s]["o"], "c": fa[s]["c"]}).join(
                fr[s][["c", "v"]].rename(columns={"c": "rc"}), how="inner")
            d = d[(d["v"] > 0) & (d["c"] > 0) & (d["rc"] > 0)].sort_index()
            ex = extra_inputs(d, by_sym[s])
            for t, r in M1.stock_inputs(d, by_sym[s]).items():
                rows.append(dict(r, **ex.get(t, {}), symbol=s, date=t))
        B0.atomic_parquet(pd.DataFrame(rows), f)
        print(f"[meta2] batch {b // M1.BATCH + 1}/{-(-len(syms) // M1.BATCH)}: {len(rows)} pick rows", flush=True)


def sources(cache, b0_src, deadline):
    """The folder holding si/, regsho/ and news/ for 2015-11 .. 2025-09, or None when the budget ran out."""
    if b0_src and (Path(b0_src) / "complete.json").exists():
        return Path(b0_src)
    src = cache / "src"
    cal = pd.DatetimeIndex(B0.calendar()["date"])
    ok = (B0.src_short_interest(src, deadline) and B0.src_regsho(src, deadline, cal)
          and news_history.fetch_range("2015-12", B0.LAST_MONTH, src / "news", deadline=deadline))
    return src if ok else None


def feat_si(d, si, cal):
    """dtc and si_chg at each row of d (date, symbol, avgvol20, split30): the latest short interest usable by
    t (8 trading days after settlement, at most 45 days old), over 20-day mean volume; and its log change
    from the settlement before it. Also the settlement date used, for the QA."""
    n = len(d)
    if not len(si):
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    si = si.sort_values(["symbol", "settlement"]).copy()
    si["prev_qty"] = si.groupby("symbol")["short_qty"].shift(1)
    pos = cal.searchsorted(si["settlement"]) + B0.SI_LAG
    si = si[pos < len(cal)].assign(avail=cal[pos[pos < len(cal)]])
    m = B0.asof(d[["date", "symbol"]], si[["symbol", "avail", "settlement", "short_qty", "prev_qty"]], "date",
                "avail", "symbol", 366)
    fresh = ((m["date"] - m["settlement"]).dt.days <= B0.SI_MAX_AGE).to_numpy()
    q = np.where(fresh, m["short_qty"], np.nan)
    split = d["split30"].fillna(True).astype(bool).to_numpy()
    av = d["avgvol20"].to_numpy(float)
    dtc = np.where(split | ~(av > 0), np.nan, q / np.where(av > 0, av, np.nan))
    pq = m["prev_qty"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        chg = np.where(fresh & (pq > 0) & (q > 0), np.log(q / pq), np.nan)
    return dtc, chg, np.where(fresh, m["settlement"].to_numpy(), np.datetime64("NaT"))


def feat_shvr(d, rs, cal):
    """shvr5 (mean ratio over the 5 sessions to t, >= 3 days) and shvr_chg (shvr5 minus the mean over the 60
    sessions before those 5, >= 20 days)."""
    if not len(rs):
        return np.full(len(d), np.nan), np.full(len(d), np.nan)
    rs = rs[(rs["total"] > 0) & rs["symbol"].isin(set(d["symbol"]))]
    p = rs.assign(r=rs["short"] / rs["total"]).pivot_table(index="date", columns="symbol", values="r")
    p = p.reindex(cal[(cal >= p.index.min()) & (cal <= p.index.max())])
    m5 = p.rolling(WIN, min_periods=3).mean()
    base = p.shift(WIN).rolling(BASE, min_periods=20).mean()
    f = pd.DataFrame({"shvr5": m5.stack(), "base": base.stack()}).rename_axis(["date", "symbol"]).reset_index()
    x = d[["date", "symbol"]].merge(f, on=["date", "symbol"], how="left")
    return x["shvr5"].to_numpy(), (x["shvr5"] - x["base"]).to_numpy()


def article_sessions(news, caldf, symbols):
    """One row per (article, tagged pick symbol): session index (first close at or after creation) and tone."""
    closes = B0.session_closes(caldf)
    n = news.dropna(subset=["symbols"])
    n = n.assign(symbol=n["symbols"].str.split(",")).explode("symbol")
    n = n[n["symbol"].isin(symbols)]
    sess = closes.searchsorted(pd.to_datetime(n["created_at"], utc=True), side="left")
    n = n.assign(sess=sess)[sess < len(caldf)]
    tone = {h: lm.label(h) for h in n["headline"].dropna().unique()}
    return n.assign(tone=n["headline"].map(tone).fillna("NEU"))[["id", "symbol", "sess", "tone", "created_at"]]


def feat_news(d, arts, caldf):
    """news5, neg5, pos5 (articles in the 5 sessions to t) and news60 (the 60 sessions before those)."""
    cal = pd.DatetimeIndex(caldf["date"])
    x = pd.DataFrame({"sess": cal.get_indexer(d["date"]), "symbol": d["symbol"].to_numpy()})
    if not len(arts):
        z = np.zeros(len(d))
        return z, z.copy(), z.copy(), z.copy()
    lo, hi = int(arts["sess"].min()), max(int(arts["sess"].max()), int(x["sess"].max()))
    out = {}
    for name, a in (("all", arts), ("neg", arts[arts["tone"] == "NEG"]), ("pos", arts[arts["tone"] == "POS"])):
        cnt = a.groupby(["sess", "symbol"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(index=range(lo, hi + 1), columns=sorted(set(arts["symbol"])), fill_value=0)
        out[name] = cnt.rolling(WIN, min_periods=1).sum()
        if name == "all":
            out["base"] = cnt.shift(WIN).rolling(BASE, min_periods=1).sum()

    def pick(frame):
        s = frame.stack().rename("n").rename_axis(["sess", "symbol"]).reset_index()
        return x.merge(s, on=["sess", "symbol"], how="left")["n"].fillna(0.0).to_numpy()
    return pick(out["all"]), pick(out["neg"]), pick(out["pos"]), pick(out["base"])


def stage_features(cache, src, inputs):
    caldf = B0.calendar()
    cal = pd.DatetimeIndex(caldf["date"])
    fdir = cache / "feat"
    for y in YEARS:
        f = fdir / f"feat_{y}.parquet"
        if f.exists():
            continue
        d = inputs[inputs["date"].dt.year == y][["date", "symbol", "avgvol20", "split30"]].reset_index(drop=True)
        ms = [m for m in pd.period_range(f"{y - 1}-09", f"{y}-12", freq="M") if str(m) <= B0.LAST_MONTH]
        si = B0.load_parts(src / "si", "si_{}.parquet", ms)
        d["dtc"], d["si_chg"], d["si_settlement"] = feat_si(d, si, cal)
        d["shvr5"], d["shvr_chg"] = feat_shvr(d, B0.load_parts(src / "regsho", "regsho_{}.parquet", ms), cal)
        news = news_history.load_range(str(ms[0]), str(ms[-1]), src / "news", columns=["id", "created_at", "headline", "symbols"])
        arts = article_sessions(news, caldf, set(d["symbol"]))
        d["news5"], d["neg5"], d["pos5"], d["news60"] = feat_news(d, arts, caldf)
        B0.atomic_parquet(d.drop(columns=["avgvol20", "split30"]), f)
        B0.atomic_parquet(arts.drop(columns="tone"), fdir / f"arts_{y}.parquet")
        print(f"[meta2] features {y}: {len(d)} picks, {int((d['news5'] > 0).sum())} with news, "
              f"{int(d['dtc'].notna().sum())} with days to cover, {int(d['shvr5'].notna().sum())} with short volume",
              flush=True)


def qa_new(x, cache, caldf):
    cal = pd.DatetimeIndex(caldf["date"])
    closes = B0.session_closes(caldf)
    q = {"coverage_by_year": {}, "inputs": {}}
    for y, g in x.groupby(x["date"].dt.year):
        q["coverage_by_year"][str(y)] = {"picks": int(len(g)), "any_news": float((g["news5"] > 0).mean()),
                                         **{c: float(g[c].notna().mean()) for c in ("dtc", "si_chg", "shvr5", "shvr_chg")}}
    for c in NEW:
        s = x[c].describe(percentiles=[0.01, 0.5, 0.99])[["count", "min", "1%", "50%", "99%", "max"]]
        q["inputs"][c] = {k: (None if not np.isfinite(v) else float(v)) for k, v in s.items()}
    ti = cal.get_indexer(x["date"])
    si = x["si_settlement"].notna()
    lag = ti[si.to_numpy()] - cal.searchsorted(x.loc[si, "si_settlement"])
    q["si_lag_sessions"] = {"rows": int(si.sum()), "min": int(lag.min()) if len(lag) else None,
                            "below_8": int((lag < B0.SI_LAG).sum())}
    rec = []
    for r in x.sample(10, random_state=M1.SEED).itertuples():
        a = pd.read_parquet(cache / "feat" / f"arts_{r.date.year}.parquet")
        a = a[a["symbol"] == r.symbol]
        i = cal.get_loc(r.date)
        t = pd.to_datetime(a["created_at"], utc=True)
        n = int(((t > closes[i - WIN]) & (t <= closes[i])).sum())
        rec.append({"symbol": r.symbol, "date": f"{r.date:%Y-%m-%d}", "news5": float(r.news5), "recount": n})
    q["news5_recount"] = rec
    return q


def splits(t):
    """Weekly mean fwd5 differences: picks with news minus without, high minus low shvr_chg (week median)."""
    out = {}
    for name, flag in (("news_minus_no_news", t["news5"] > 0),
                       ("shvr_chg_high_minus_low", t["shvr_chg"] > t.groupby("date")["shvr_chg"].transform("median"))):
        ok = t["shvr_chg"].notna() if name.startswith("shvr") else pd.Series(True, index=t.index)
        g = t[ok].assign(f=flag[ok])
        w = g.groupby(["date", "f"])["fwd5"].mean().unstack().dropna()
        out[name] = M1.summary(w[True] - w[False]) if True in w and False in w and len(w) > 10 else None
    return out


def diagnose(x):
    """POST-HOC, after META2 failed (Amit, 2026-10-24: find out why the model picks wrong). Not a test:
    (1) overfit: in-sample vs out-of-sample AUC per model year; (2) label noise: how much of a pick's 5-day
    return is its week's average, and the spread left; (3) each input's rank correlation with the excess
    return (fwd5 minus the week mean), per year, and how often its sign flips; (4) out-of-sample probability
    deciles against excess return; (5) how taken and skipped picks differ, input by input; (6) the same
    walk-forward with a plain logistic regression."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    x = x.copy()
    x["excess"] = x["fwd5"] - x.groupby("date")["fwd5"].transform("mean")
    weeks = np.sort(x["date"].unique())
    out = {"overfit": {}, "logistic": {}}
    oos = []
    for y in M1.TEST_YEARS:
        test = x["date"].dt.year == y
        if not test.any():
            continue
        cut = weeks[max(0, np.searchsorted(weeks, np.datetime64(x.loc[test, "date"].min())) - M1.EMBARGO_WEEKS)]
        tr = x["date"] < cut
        m = lgb.LGBMClassifier(**M1.PARAMS).fit(x.loc[tr, INPUTS], x.loc[tr, "label"])
        p_tr = m.predict_proba(x.loc[tr, INPUTS])[:, 1]
        p_te = m.predict_proba(x.loc[test, INPUTS])[:, 1]
        out["overfit"][str(y)] = {"auc_train": float(roc_auc_score(x.loc[tr, "label"], p_tr)),
                                  "auc_test": float(roc_auc_score(x.loc[test, "label"], p_te))}
        oos.append(x.loc[test, ["date", "excess", "label"] + INPUTS].assign(prob=p_te, take=p_te >= np.median(p_tr)))
        med = x.loc[tr, INPUTS].median()
        mu, sd = x.loc[tr, INPUTS].fillna(med).mean(), x.loc[tr, INPUTS].fillna(med).std().replace(0, 1)
        z = lambda f: ((f.fillna(med) - mu) / sd).clip(-5, 5)
        lr = LogisticRegression(C=0.1, max_iter=2000).fit(z(x.loc[tr, INPUTS]), x.loc[tr, "label"])
        out["logistic"][str(y)] = float(roc_auc_score(x.loc[test, "label"], lr.predict_proba(z(x.loc[test, INPUTS]))[:, 1]))
    o = pd.concat(oos, ignore_index=True)
    wk = x.groupby("date")["fwd5"].mean()
    out["label_noise"] = {"sd_fwd5": float(x["fwd5"].std()), "sd_week_mean": float(wk.std()),
                          "sd_excess": float(x["excess"].std()),
                          "share_of_variance_from_week": float(wk.reindex(x["date"]).var() / x["fwd5"].var())}
    o["dec"] = pd.qcut(o["prob"], 10, labels=False, duplicates="drop")
    out["prob_deciles"] = {int(k): {"n": int(len(g)), "mean_excess": float(g["excess"].mean()), "hit": float(g["label"].mean())}
                           for k, g in o.groupby("dec")}
    corr = {}
    for c in INPUTS:
        per = {}
        for y, g in x.groupby(x["date"].dt.year):
            g = g[[c, "excess"]].dropna()
            if len(g) > 100 and g[c].nunique() > 1:
                per[str(y)] = float(spearmanr(g[c], g["excess"]).statistic)
        v = np.array(list(per.values()))
        corr[c] = {"by_year": per, "mean": float(v.mean()) if len(v) else None,
                   "years_pos": int((v > 0).sum()), "years_neg": int((v < 0).sum())}
    out["input_vs_excess"] = corr
    tk, sk = o[o["take"]], o[~o["take"]]
    sd = o[INPUTS].std().replace(0, 1)
    out["taken_minus_skipped_in_sd"] = {c: float((tk[c].mean() - sk[c].mean()) / sd[c]) for c in INPUTS}
    out["taken_minus_skipped_excess"] = float(tk["excess"].mean() - sk["excess"].mean())
    return out


def run(stage, cache, out, b0_src=None, budget_min=300):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_min * 60
    p = M1.picks()
    stage_bars(cache, p)
    end = alpaca.sip_end(M1.BARS_END)
    spy = to_frames(fetch_daily_bars(["SPY"], M1.BARS_START, end, adjustment="all", headers=alpaca_headers()))["SPY"]
    tmap = sec_data.dera_ticker_map(2015, pd.Period("2025Q3"), cache / "dera")
    eightk = weekly_panel.filings({"8-K", "8-K/A"}, cache / "edgar")
    inputs = pd.concat([pd.read_parquet(f) for f in sorted((cache / "inputs").glob("in_*.parquet"))], ignore_index=True)
    src = sources(cache, b0_src, deadline)
    if src is None:
        print("[meta2] sources incomplete (budget); run again", flush=True)
        return
    print(f"[meta2] sources from {src}", flush=True)
    stage_features(cache, src, inputs)
    if stage == "data":
        return
    feat = pd.concat([pd.read_parquet(cache / "feat" / f"feat_{y}.parquet") for y in YEARS], ignore_index=True)
    x = M1.assemble(p, inputs.drop(columns=["avgvol20", "split30"]), spy, tmap, eightk)
    x = x.merge(feat, on=["symbol", "date"], how="left")
    caldf = B0.calendar()
    if stage == "diag":
        d = diagnose(x.dropna(subset=["fwd5"]).reset_index(drop=True))
        (out / "META2_diagnosis.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
        print("DIAG", json.dumps({k: d[k] for k in ("overfit", "label_noise", "prob_deciles", "logistic")}, default=str), flush=True)
        return
    q = M1.qa(x)
    q.update(qa_new(x, cache, caldf))
    print("QA", json.dumps({k: q[k] for k in ("picks", "with_fwd5", "si_lag_sessions", "news5_recount")}, default=str), flush=True)
    t, imp = M1.walk_forward(x, INPUTS)
    res = {"idea": "META2", "prereg": "research/prereg/META2.md", "primary_picks": str(M1.PICKS), "qa": q}
    res.update(M1.evaluate(t, imp))
    t2, imp2 = M1.walk_forward(x, NEW_ONLY)
    e2 = M1.evaluate(t2, imp2)
    res["alongside"]["new_inputs_only"] = {"PRIMARY": e2["PRIMARY"], "importance": imp2}
    res["alongside"]["splits"] = splits(x.dropna(subset=["fwd5"]))
    (out / "META2_result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    t[["date", "symbol", "fwd5", "label", "prob", "take", "model_year"]].to_csv(out / "META2_oos.csv", index=False)
    print("PRIMARY", json.dumps(res["PRIMARY"]), flush=True)
    print("NEW ONLY", json.dumps(e2["PRIMARY"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "fit", "all", "diag"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--b0-src", default=None)
    ap.add_argument("--budget-min", type=float, default=300)
    ap.add_argument("--out", default="research/results/META2")
    a = ap.parse_args()
    run(a.stage, a.cache, a.out, a.b0_src, a.budget_min)


if __name__ == "__main__":
    main()
