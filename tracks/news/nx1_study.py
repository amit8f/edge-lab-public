"""NX1 (NEWS-XGB) history and training (research/prereg/NX1.md).

Stages, each resumable from its cache and run in order by .github/workflows/nx1-study.yml:
  data    daily bars (W1's shared 500-symbol batches), SPY, Benzinga headline months, the trading calendar
  tone    per month: Loughran-McDonald counts, FinBERT probabilities and the 12 type flags of every
          single-symbol headline (no text is kept; a hash of the normalized headline is, for the 3-day dedup)
  events  nightly news events (symbol, batch day d) with their 32 inputs, label and costs, plus placebo
          events (quiet stocks with every news input at its no-news value), one file per bar batch
  train   walk-forward report 2019..2025-09 and the version 1 model on 2016-10 .. 2025-09-24
          -> research/results/NX1/ (summary numbers and the model only)

Usage: python -m tracks.news.nx1_study --stage all|data|tone|events|train --cache <dir> --out <dir>
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core import alpaca
from adaptive import costs
from tracks.main.longterm.stats import nw_tstat
from tracks.news import lm
from tracks.news import w1_data as W

ET = "America/New_York"
FIRST_DAY, LAST_DAY = pd.Timestamp("2016-10-03"), pd.Timestamp("2025-09-24")
LAST_EXIT = pd.Timestamp("2025-09-30")             # nothing from the 2025-10-01 holdout is read
TEST_YEARS = range(2019, 2026)
PURGE = 5
HOLD = 3
CUTOFF = pd.Timedelta(hours=22)                    # a batch closes at 10:00pm ET
MIN_PRICE, MIN_DV20 = 1.0, 1e6
BETA_WIN, BETA_MIN = 250, 120
NOVELTY_DAYS = 30
QUIET_BACK = 3
PICKS = 3
NOTIONAL = 500.0
EDGE = 0.20
SEED = 20261022
FINBERT = "ProsusAI/finbert"
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=50, objective="multi:softprob", tree_method="hist", random_state=SEED)
CLASSES = ("short", "none", "long")                # label 0, 1, 2

FLAGS = {
    "earnings": r"\bearnings\b|\beps\b|quarterly results|\bbeats?\b|\bmiss(es)?\b",
    "guidance": r"guidance|outlook|forecast|(raises|lowers|cuts|boosts) (fy|full[- ]year)|reaffirms",
    "analyst_up": r"upgrades?\b|raises (price )?target|(initiates|initiated).{0,30}\b(buy|outperform|overweight)",
    "analyst_down": r"downgrades?\b|(lowers|cuts) (price )?target",
    "offering": r"offering|private placement|registered direct|at[- ]the[- ]market|prices .{0,40}shares|warrants?",
    "merger": r"merger|\bacquir|acquisition|buyout|takeover|tender offer",
    "fda": r"\bfda\b|clinical|\btrials?\b|\bphase [123i]|approval|pdufa|\bcrl\b",
    "contract": r"contract|\bawards?\b|awarded|partnership|collaboration|agreement|\border\b",
    "legal": r"\bsec\b|lawsuit|\bsues?\b|\bsued\b|investigation|subpoena|settle|class action|\bprobe\b|fraud",
    "bankruptcy": r"bankrupt|chapter 11|delist|going concern|noncompliance|non-compliance|deficiency",
    "dividend": r"dividend|buyback|repurchase",
    "executive": r"\bceo\b|\bcfo\b|chief .{0,20}officer|resign|appoint|steps down",
}
_FLAGS = {k: re.compile(v, re.I) for k, v in FLAGS.items()}
NEWS_INPUTS = (["n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos",
                "fb_neu", "head_len"] + [f"f_{k}" for k in FLAGS] + ["time_bucket"])
STOCK_INPUTS = ["r1", "r5", "r20", "vol20", "log_dv", "log_px"]
MARKET_INPUTS = ["spy_r1", "spy_r5"]
INPUTS = NEWS_INPUTS + ["novelty"] + STOCK_INPUTS + MARKET_INPUTS
assert len(INPUTS) == 32


def nx1dir(cache):
    return Path(cache) / "nx1"


def atomic_parquet(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


# ---------- pure helpers (tested) ----------

def norm(headline):
    return " ".join(str(headline or "").lower().split())


def flag_row(headline):
    return {f"f_{k}": int(bool(rx.search(headline or ""))) for k, rx in _FLAGS.items()}


def headline_rows(news, finbert):
    """One month of articles -> per single-symbol headline: id, symbol, ET time, headline hash and length,
    LM negative/positive counts, FinBERT (pos, neg, neu) and the flags. `finbert(texts)` returns an n x 3
    array of (positive, negative, neutral) probabilities; each distinct normalized headline is scored once."""
    n = news[news["symbols"].fillna("").ne("") & ~news["symbols"].fillna("").str.contains(",")]
    n = n[n["headline"].fillna("").str.strip().ne("")]
    cols = ["id", "symbol", "t", "hkey", "head_len", "lm_neg", "lm_pos", "fb_pos", "fb_neg", "fb_neu",
            *[f"f_{k}" for k in FLAGS]]
    if n.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"id": n["id"].to_numpy(), "symbol": n["symbols"].to_numpy(),
                        "t": W.et_time(n["created_at"]).to_numpy()})
    h = n["headline"].map(norm).to_numpy()
    out["hkey"] = [hashlib.sha1(x.encode()).hexdigest()[:16] for x in h]
    out["head_len"] = [len(x) for x in h]
    neg, pos = lm.lists()
    toks = [[w.upper() for w in lm.TOKEN.findall(x)] for x in h]
    out["lm_neg"] = [sum(w in neg for w in t) for t in toks]
    out["lm_pos"] = [sum(w in pos for w in t) for t in toks]
    uniq, inv = np.unique(h, return_inverse=True)
    p = np.asarray(finbert(list(uniq)), dtype=float).reshape(len(uniq), 3)[inv]
    out["fb_pos"], out["fb_neg"], out["fb_neu"] = p[:, 0], p[:, 1], p[:, 2]
    fl = pd.DataFrame([flag_row(x) for x in h])
    return pd.concat([out, fl], axis=1)[cols]


def batch_days(t, cal_dates):
    """The batch day of each ET headline time: the first trading day d with t <= d 22:00 (NaT past the end)."""
    cut = pd.DatetimeIndex(cal_dates) + CUTOFF
    k = cut.searchsorted(pd.DatetimeIndex(t), side="left")
    d = pd.DatetimeIndex(cal_dates)
    return pd.DatetimeIndex([d[i] if i < len(d) else pd.NaT for i in k])


def dedup(rows, last=None):
    """Drops a headline when the same headline for the same symbol was seen in the 3 days before it. `last`
    carries the last-seen times from earlier (older) months when rows come one month at a time."""
    rows = rows.sort_values("t", kind="stable")
    last = {} if last is None else last
    keep = []
    for i, s, h, t in zip(rows.index, rows["symbol"], rows["hkey"], rows["t"]):
        k = (s, h)
        if k in last and t - last[k] <= pd.Timedelta(days=3):
            continue
        last[k] = t
        keep.append(i)
    return rows.loc[keep]


def news_events(rows, cal):
    """Deduplicated headline rows -> one row per (symbol, batch day d) with the 23 news inputs. Time bucket of
    the last headline: 0 before d's open, 1 in d's session, 2 after d's close."""
    r = rows.copy()
    r["d"] = batch_days(r["t"], cal["date"])
    r = r[r["d"].notna()]
    c = cal.set_index("date")
    op = r["d"] + pd.to_timedelta(c.loc[r["d"], "open"].to_numpy() + ":00")
    cl = r["d"] + pd.to_timedelta(c.loc[r["d"], "close"].to_numpy() + ":00")
    r["bucket"] = np.where(r["t"] < op.to_numpy(), 0, np.where(r["t"] <= cl.to_numpy(), 1, 2))
    g = r.sort_values("t").groupby(["symbol", "d"])
    e = g.agg(n_articles=("id", "size"), lm_neg=("lm_neg", "sum"), lm_pos=("lm_pos", "sum"),
              fb_pos=("fb_pos", "mean"), fb_neg=("fb_neg", "mean"), fb_max_neg=("fb_neg", "max"),
              fb_max_pos=("fb_pos", "max"), fb_neu=("fb_neu", "mean"), head_len=("head_len", "mean"),
              time_bucket=("bucket", "last"), **{f"f_{k}": (f"f_{k}", "max") for k in FLAGS})
    e["lm_net"] = e["lm_pos"] - e["lm_neg"]
    return e.reset_index()


def placebo_news():
    """Every news input at its no-news value (no article, so counts, tone, length and flags are 0)."""
    return {k: 0.0 for k in NEWS_INPUTS}


def novelty(keys, news_days):
    """Days with any Benzinga article naming the symbol in the 30 calendar days before d."""
    out = np.zeros(len(keys), dtype=float)
    by = {s: np.sort(g["date"].to_numpy()) for s, g in news_days.groupby("symbol")}
    for s, idx in keys.groupby("symbol").groups.items():
        days = by.get(s)
        if days is None:
            continue
        d = keys.loc[idx, "d"].to_numpy()
        out[keys.index.get_indexer(idx)] = (np.searchsorted(days, d, side="left")
                                            - np.searchsorted(days, d - np.timedelta64(NOVELTY_DAYS, "D"), "left"))
    return out


def stock_panel(rows, spy, cal):
    """One symbol's daily rows (W1 columns) -> per day d: the 6 stock inputs, beta, the NX1 universe flag,
    and the trade from the next session's open to the close of the third session (entry day = 1):
    all-adjusted return, SPY's return over the same bars, the raw entry open and the calendar days held.
    Nothing crosses a >15-day gap; the entry must be the next market session."""
    out = []
    caldates = pd.DatetimeIndex(cal["date"])
    sp = spy.set_index("date")
    for _, g in rows.sort_values("date").groupby("seg", sort=False):
        g = g.set_index("date")
        ac = g["ac"]
        lr = np.log(ac).diff()
        f = pd.DataFrame(index=g.index)
        f["r1"], f["r5"], f["r20"] = ac.pct_change(1), ac.pct_change(5), ac.pct_change(20)
        f["vol20"] = lr.rolling(20).std()
        f["log_dv"] = np.log(g["dv20"])
        f["log_px"] = np.log(g["c"])
        m = np.log(sp["ac"]).diff().reindex(g.index)
        cov = lr.rolling(BETA_WIN, min_periods=BETA_MIN).cov(m)
        var = m.rolling(BETA_WIN, min_periods=BETA_MIN).var()
        f["beta"] = (cov / var).fillna(1.0)
        f["uni"] = (g["c"] >= MIN_PRICE) & (g["dv20"] >= MIN_DV20) & f["vol20"].notna()
        pos = caldates.searchsorted(g.index)
        nxt = pd.DatetimeIndex([caldates[i + 1] if i + 1 < len(caldates) else pd.NaT for i in pos])
        ex_cal = pd.DatetimeIndex([caldates[i + HOLD] if i + HOLD < len(caldates) else pd.NaT for i in pos])
        dates = g.index.to_numpy()
        j_ent = np.arange(len(g)) + 1
        ok = (j_ent < len(g)) & (dates[np.minimum(j_ent, len(g) - 1)] == nxt.to_numpy())
        j_ex = np.searchsorted(dates, ex_cal.to_numpy(), side="right") - 1
        ok &= ~pd.isna(ex_cal) & (j_ex >= j_ent)
        je, jx = np.minimum(j_ent, len(g) - 1), np.clip(j_ex, 0, len(g) - 1)
        f["entry"] = np.where(ok, dates[je], np.datetime64("NaT"))
        f["exit"] = np.where(ok, dates[jx], np.datetime64("NaT"))
        f["ret"] = np.where(ok, g["ac"].to_numpy()[jx] / g["ao"].to_numpy()[je] - 1, np.nan)
        f["price"] = np.where(ok, g["o"].to_numpy()[je], np.nan)
        out.append(f)
    if not out:
        return pd.DataFrame()
    p = pd.concat(out).rename_axis("d").reset_index()
    sret = sp["ac"].reindex(p["exit"]).to_numpy() / sp["ao"].reindex(p["entry"]).to_numpy() - 1
    p["spy_ret"] = sret
    p["mret"] = p["ret"] - p["beta"] * p["spy_ret"]
    p["days"] = (p["exit"] - p["entry"]).dt.days + 1
    return p


def spy_inputs(spy):
    s = spy.set_index("date")["ac"]
    return pd.DataFrame({"d": s.index, "spy_r1": s.pct_change(1).to_numpy(), "spy_r5": s.pct_change(5).to_numpy()})


def cost_frac(price, short, days):
    """Round-trip cost as a share of a $500 position: adaptive/costs.py fills at the entry price on both sides
    (main spread at $5+, penny below), plus borrow on shorts for the calendar days held."""
    price, short, days = np.asarray(price, float), np.asarray(short, bool), np.asarray(days, float)
    out = np.full(len(price), np.nan)
    for i, (p, s, d) in enumerate(zip(price, short, days)):
        if not np.isfinite(p) or p <= 0:
            continue
        book = "main" if p >= 5 else "penny"
        sh = max(1, int(NOTIONAL // p))
        c = costs.fill_cost(book, sh, p, False) + costs.fill_cost(book, sh, p, True)
        if s:
            c += costs.borrow_cost(book, sh * p, d)
        out[i] = c / (sh * p)
    return out


def edges(mret):
    m = np.asarray(mret, float)
    m = m[np.isfinite(m)]
    return float(np.quantile(m, EDGE)), float(np.quantile(m, 1 - EDGE))


def classes(mret, lo, hi):
    m = np.asarray(mret, float)
    return np.where(m <= lo, 0, np.where(m >= hi, 2, 1))


def weights(y):
    y = np.asarray(y)
    n = np.bincount(y, minlength=3).astype(float)
    return (len(y) / (3 * np.maximum(n, 1)))[y]


def fit(X, y, params=None):
    from xgboost import XGBClassifier
    m = XGBClassifier(**(params or XGB))
    m.fit(X, y, sample_weight=weights(y))
    return m


def picks(night, rng):
    """One night's scored real events (p_short, p_long, argmax 'cls') -> (picks, control). Fired events
    (most likely class long or short) are ranked by p_long - p_short: the top PICKS longs and top PICKS shorts.
    Control: the same number of events drawn at random from the whole night, each taking its slot's side."""
    s = night.assign(score=night["p_long"] - night["p_short"])
    lo = s[s["cls"] == 2].nlargest(PICKS, "score").assign(side=1)
    sh = s[s["cls"] == 0].nsmallest(PICKS, "score").assign(side=-1)
    p = pd.concat([lo, sh])
    if p.empty or len(night) < len(p):
        return p, p.iloc[0:0]
    c = night.iloc[rng.choice(len(night), size=len(p), replace=False)].assign(side=p["side"].to_numpy())
    return p, c


def net(t, col="ret"):
    return t["side"] * t[col] - cost_frac(t["price"], t["side"] < 0, t["days"])


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(d / se) if se > 0 else float("nan")


def evaluate(scored, seed):
    """Picks against control and placebo against real, on scored events (kind real/placebo, d, p_*, cls,
    ret, mret, price, days)."""
    real, plac = scored[scored["kind"] == "real"], scored[scored["kind"] == "placebo"]
    rng = np.random.default_rng(seed)
    diffs, n_p = [], 0
    for _, night in real.groupby("d", sort=True):
        p, c = picks(night, rng)
        if p.empty or c.empty:
            continue
        diffs.append(net(p).mean() - net(c).mean())
        n_p += len(p)
    m, se, t = nw_tstat(diffs, 5) if len(diffs) > 2 else (float("nan"),) * 3
    fr = lambda x: float((x["cls"] != 1).mean()) if len(x) else float("nan")
    fire = lambda x: x[x["cls"] != 1].assign(side=lambda z: np.where(z["p_long"] >= z["p_short"], 1, -1))
    fr_r, fp = fire(real), fire(plac)
    d, wt = welch(net(fr_r, "mret"), net(fp, "mret")) if len(fr_r) and len(fp) else (float("nan"),) * 2
    return {"nights": len(diffs), "pick_trades": n_p, "pick_minus_control": {"mean": m, "se_nw": se, "t_nw": t},
            "placebo": {"real_fire_rate": fr(real), "placebo_fire_rate": fr(plac),
                        "fire_ratio_ok": bool(fr(plac) <= fr(real) / 2) if len(plac) else None,
                        "real_minus_placebo_fired": d, "welch_t": wt,
                        "n_real_fired": int(len(fr_r)), "n_placebo_fired": int(len(fp))}}


def score(model, ev):
    p = model.predict_proba(ev[INPUTS].to_numpy(float))
    return ev.assign(p_short=p[:, 0], p_none=p[:, 1], p_long=p[:, 2], cls=p.argmax(1))


def walk_forward(ev, cal, params=None, years=TEST_YEARS):
    """Per test year: fit on real events of earlier years (last PURGE sessions dropped), score that year."""
    caldates = pd.DatetimeIndex(cal["date"])
    out = {}
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        k = caldates.searchsorted(start)
        cut = caldates[max(k - PURGE - 1, 0)]
        tr = ev[(ev["kind"] == "real") & (ev["d"] <= cut)]
        te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        lo, hi = edges(tr["mret"])
        model = fit(tr[INPUTS].to_numpy(float), classes(tr["mret"], lo, hi), params)
        out[str(yr)] = {"train_events": int(len(tr)), "edges": [lo, hi], **evaluate(score(model, te), SEED + yr)}
    return out


# ---------- stages (network / disk) ----------

def stage_data(cache, deadline):
    root = nx1dir(cache)
    if not (root / "cal.parquet").exists():
        atomic_parquet(W.calendar(), root / "cal.parquet")
    if not (root / "spy.parquet").exists():
        h = alpaca_headers()
        end = alpaca.sip_end(W.BARS_END)
        raw = to_frames(fetch_daily_bars(["SPY"], W.BARS_START, end, adjustment="raw", headers=h))["SPY"]
        adj = to_frames(fetch_daily_bars(["SPY"], W.BARS_START, end, adjustment="all", headers=h))["SPY"]
        atomic_parquet(adj[["o", "c"]].rename(columns={"o": "ao", "c": "ac"}).join(raw[["c"]])
                       .rename_axis("date").reset_index(), root / "spy.parquet")
    W.stage_bars(cache)
    return W.stage_news(cache, deadline)


def finbert_scorer(meta_file):
    """FinBERT on CPU, pinned to the commit resolved on the first run (logged in tone/meta.json)."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification as Model, AutoTokenizer as Tok
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if not meta:
        from huggingface_hub import model_info
        meta = {"model": FINBERT, "revision": model_info(FINBERT).sha, "torch": torch.__version__,
                "transformers": transformers.__version__}
    tok = Tok.from_pretrained(FINBERT, revision=meta["revision"])
    model = Model.from_pretrained(FINBERT, revision=meta["revision"]).eval()
    if not meta_file.exists():
        meta_file.write_text(json.dumps(meta | {"labels": model.config.id2label}, indent=1, default=str))
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    order = [lab["positive"], lab["negative"], lab["neutral"]]
    torch.set_num_threads(4)

    def run(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=64, return_tensors="pt")
                out.append(torch.softmax(model(**b).logits, -1)[:, order].numpy())
        return np.vstack(out) if out else np.zeros((0, 3))
    return run


def tone_months(shard=0, shards=1):
    """The months the tone stage scores; shard i of n takes every n-th one (parallel jobs)."""
    ms = [p for p in pd.period_range(W.NEWS_FIRST, W.NEWS_LAST, freq="M") if p >= pd.Period("2016-08")]
    return ms[shard::shards]


def stage_tone(cache, deadline, finbert=None, shard=0, shards=1):
    d = nx1dir(cache) / "tone"
    d.mkdir(parents=True, exist_ok=True)
    for per in tone_months(shard, shards):
        f = d / f"tone_{per}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        finbert = finbert or finbert_scorer(d / "meta.json")
        rows = headline_rows(W.news_month(cache, per), finbert)
        atomic_parquet(rows, f)
        print(f"[nx1 tone] {per}: {len(rows)} headlines")
    return True


def all_news_events(cache):
    f = nx1dir(cache) / "news_events.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cal = pd.read_parquet(nx1dir(cache) / "cal.parquet")
    files = sorted((nx1dir(cache) / "tone").glob("tone_*.parquet"))
    # One month at a time (all months at once ran the runner out of memory). A headline's batch day is in
    # its own month or the next, so rows whose day falls in the next month wait for that month's rows.
    last, carry, parts = {}, None, []
    for x in files:
        rows = dedup(pd.read_parquet(x), last)
        if len(rows):
            cut = rows["t"].max() - pd.Timedelta(days=3)
            last = {k: t for k, t in last.items() if t >= cut}
        mon = pd.Period(x.stem.removeprefix("tone_"), freq="M")
        rows = rows if carry is None else pd.concat([carry, rows], ignore_index=True)
        d = batch_days(rows["t"], cal["date"])
        later = np.asarray(d.notna() & (d.to_period("M") > mon))
        carry = rows[later]
        if (~later).any():
            parts.append(news_events(rows[~later], cal))
    if carry is not None and len(carry):
        parts.append(news_events(carry, cal))
    e = pd.concat(parts, ignore_index=True)
    e = e[(e["d"] >= FIRST_DAY) & (e["d"] <= LAST_DAY)]
    atomic_parquet(e, f)
    return e


def batch_events(rows, news_ev, nd, spy, cal, b):
    """One bar batch's real and placebo events with inputs, label and cost columns. On history the placebo
    stocks are drawn inside each 500-symbol batch, as many per night as that batch has real events."""
    pan = [stock_panel(g, spy, cal) for _, g in rows.groupby("symbol")]
    syms = [s for s, _ in rows.groupby("symbol")]
    pan = pd.concat([p.assign(symbol=s) for p, s in zip(pan, syms) if len(p)], ignore_index=True)
    pan = pan[pan["uni"] & pan["entry"].notna() & (pan["exit"] <= LAST_EXIT)
              & (pan["d"] >= FIRST_DAY) & (pan["d"] <= LAST_DAY)]
    pan = pan.merge(spy_inputs(spy), on="d", how="left")
    real = pan.merge(news_ev, on=["symbol", "d"], how="inner").assign(kind="real")
    cik = rows.drop_duplicates(["symbol", "date"]).rename(columns={"date": "d"})[["symbol", "d", "cik"]]
    cand = pan.merge(cik, on=["symbol", "d"], how="left")
    keys = cand[["symbol", "cik", "d"]].rename(columns={"d": "date"})
    quiet = W.quiet_mask(keys, nd, pd.DataFrame({"cik": pd.Series(dtype=float), "date": pd.Series(dtype="datetime64[ns]")}),
                         back=QUIET_BACK, cal=cal["date"])
    cand = cand[quiet].drop(columns="cik")
    rng = np.random.default_rng(SEED + b)
    need = real.groupby("d").size()
    plac = []
    for d, g in cand[cand["d"].isin(need.index)].groupby("d"):
        k = min(int(need[d]), len(g))
        plac.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    plac = pd.concat(plac, ignore_index=True) if plac else cand.iloc[0:0]
    plac = plac.assign(kind="placebo", **placebo_news())
    ev = pd.concat([real, plac], ignore_index=True)
    ev["novelty"] = novelty(ev[["symbol", "d"]].reset_index(drop=True), nd)
    return ev[["symbol", "d", "kind", *INPUTS, "beta", "entry", "exit", "ret", "spy_ret", "mret", "price", "days"]]


def stage_events(cache, deadline):
    root = nx1dir(cache)
    news_ev = all_news_events(cache)
    nd = W.news_symbol_days(cache)
    spy = pd.read_parquet(root / "spy.parquet")
    cal = pd.read_parquet(root / "cal.parquet")
    files = W.batch_files(cache)
    for b, p in enumerate(files):
        f = root / "events" / f"e_{b:04d}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        rows = pd.read_parquet(p, columns=["symbol", "date", "o", "c", "ao", "ac", "dv20", "seg", "cik"])
        ev = batch_events(rows, news_ev, nd, spy, cal, b)
        atomic_parquet(ev, f)
        print(f"[nx1 events] batch {b + 1}/{len(files)}: {int((ev['kind'] == 'real').sum())} real, "
              f"{int((ev['kind'] == 'placebo').sum())} placebo")
    return True


def stage_train(cache, out):
    root = nx1dir(cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((root / "events").glob("e_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"])].reset_index(drop=True)
    cal = pd.read_parquet(root / "cal.parquet")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wf = walk_forward(ev, cal)
    tr = ev[(ev["kind"] == "real") & (ev["d"] <= LAST_DAY) & (ev["exit"] <= LAST_EXIT)]
    lo, hi = edges(tr["mret"])
    y = classes(tr["mret"], lo, hi)
    model = fit(tr[INPUTS].to_numpy(float), y)
    mfile = out / "nx1_v1.json"
    model.get_booster().save_model(mfile)
    gain = model.get_booster().get_score(importance_type="gain")
    imp = {INPUTS[int(k[1:])]: v for k, v in gain.items()}
    tone_meta = json.loads((root / "tone" / "meta.json").read_text())
    import xgboost
    meta = {"version": 1, "inputs": INPUTS, "classes": CLASSES, "edges": [lo, hi], "params": XGB,
            "train_first": str(tr["d"].min().date()), "train_last": str(tr["d"].max().date()),
            "train_events": int(len(tr)), "class_counts": np.bincount(y, minlength=3).tolist(),
            "finbert": tone_meta, "xgboost": xgboost.__version__,
            "model_sha256": hashlib.sha256(mfile.read_bytes()).hexdigest()}
    (out / "nx1_v1_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    rep = {"walk_forward": wf, "events": {"real": int((ev["kind"] == "real").sum()),
                                           "placebo": int((ev["kind"] == "placebo").sum())},
           "importance_gain": dict(sorted(imp.items(), key=lambda x: -x[1]))}
    (out / "NX1_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({y_: v["pick_minus_control"] | {"placebo": v["placebo"]} for y_, v in wf.items()},
                     indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "data", "tone", "events", "train"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/NX1")
    ap.add_argument("--budget-min", type=float, default=290)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    stages = ["data", "tone", "events", "train"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "data" and not stage_data(a.cache, deadline):
            print("[nx1] time budget used during data; rerun to continue")
            return
        if s == "tone" and not stage_tone(a.cache, deadline, shard=a.shard, shards=a.shards):
            print("[nx1] time budget used during tone; rerun to continue")
            return
        if s == "events" and not stage_events(a.cache, deadline):
            print("[nx1] time budget used during events; rerun to continue")
            return
        if s == "train":
            stage_train(a.cache, a.out)


if __name__ == "__main__":
    main()
