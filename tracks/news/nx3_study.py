"""NX3 (fast NEWS-XGB: entry 1 minute after the news) history, gate and model (research/prereg/NX3.md).

Reads NX1's cache (calendar, SPY, W1 daily bar batches, tone files: scores and hashes only, never headline
text) and adds 1-minute SIP bars of the event days. Stages, each resumable:
  events   per bar batch: real events (symbol, day d, decision time t, news inputs) and placebo events
  minutes  per month: 1-minute bars of that month's event stocks and SPY -> entry, label and the 5 intraday
           inputs; one file per month
  train    walk-forward 2019..2025-09, the gate and, when it passes, the version 1 models
           -> research/results/NX3/ (summary numbers and models only)

Usage: python -m tracks.news.nx3_study --stage all|events|minutes|train --cache <dir> --out <dir>
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main.longterm.stats import nw_tstat
from tracks.news import nx1_study as N
from tracks.news import w1_data as W

ET = "America/New_York"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
FIRST_DAY, LAST_DAY = N.FIRST_DAY, pd.Timestamp("2025-09-23")
WAIT = pd.Timedelta(minutes=1)
MAX_WAIT = pd.Timedelta(minutes=10)
LAST_T = pd.Timedelta(minutes=2)          # no decision later than 2 minutes before the close
HOLD_SCALE = np.sqrt(1.5)
SEED = 20261024
Z_CLIP, FIRE_Q, GATE_T, PICKS = 6.0, 0.90, 2.0, 3
SYM_CHUNK = 200
INTRADAY = ["mins_open", "r_open", "r_react", "rv30", "spy_open"]
INPUTS = N.INPUTS + INTRADAY
BASE_INPUTS = N.STOCK_INPUTS + N.MARKET_INPUTS + INTRADAY
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=50, objective="reg:pseudohubererror", huber_slope=1.0, tree_method="hist",
           random_state=SEED)
GROUPS = {"news": N.NEWS_INPUTS, "novelty": ["novelty"], "stock": N.STOCK_INPUTS, "market": N.MARKET_INPUTS,
          "intraday": INTRADAY}
assert len(INPUTS) == 37


def nx3dir(cache):
    return Path(cache) / "nx3"


# ---------- pure helpers (tested) ----------

def sessions(cal):
    """Calendar with open and close timestamps (naive ET) and the previous session's close."""
    c = cal[["date"]].copy()
    c["open_t"] = c["date"] + pd.to_timedelta(cal["open"].astype(str) + ":00")
    c["close_t"] = c["date"] + pd.to_timedelta(cal["close"].astype(str) + ":00")
    c["prev_close_t"] = c["close_t"].shift(1)
    c["prev"] = c["date"].shift(1)
    c["next"] = c["date"].shift(-1)
    return c.dropna(subset=["prev", "next"]).reset_index(drop=True)


def session_events(rows, ses):
    """Deduplicated headline rows (symbol, t, scores, flags) -> per (symbol, d): decision time t (the first
    headline after the previous close, or the open if it came before), and the 23 news inputs over the
    headlines from the previous close up to t. Headlines later than 2 minutes before d's close are dropped."""
    r = rows.sort_values("t").copy()
    k = pd.DatetimeIndex(ses["prev_close_t"]).searchsorted(pd.DatetimeIndex(r["t"]), side="left") - 1
    ok = k >= 0
    r, k = r[ok], k[ok]
    s = ses.iloc[k].reset_index(drop=True)
    r = r.reset_index(drop=True)
    r["d"], r["open_t"], r["close_t"] = s["date"].to_numpy(), s["open_t"].to_numpy(), s["close_t"].to_numpy()
    r = r[r["t"] > s["prev_close_t"].to_numpy()]
    r = r[r["t"] <= r["close_t"] - LAST_T]
    if r.empty:
        return pd.DataFrame(columns=["symbol", "d", "t", *N.NEWS_INPUTS])
    first = r.groupby(["symbol", "d"])["t"].transform("min")
    r["t_dec"] = np.maximum(first, r["open_t"])
    r = r[r["t"] <= r["t_dec"]]
    r["bucket"] = (r["t"] >= r["open_t"]).astype(int)
    g = r.groupby(["symbol", "d"])
    e = g.agg(t=("t_dec", "first"), n_articles=("id", "size"), lm_neg=("lm_neg", "sum"), lm_pos=("lm_pos", "sum"),
              fb_pos=("fb_pos", "mean"), fb_neg=("fb_neg", "mean"), fb_max_neg=("fb_neg", "max"),
              fb_max_pos=("fb_pos", "max"), fb_neu=("fb_neu", "mean"), head_len=("head_len", "mean"),
              time_bucket=("bucket", "last"), **{f"f_{k}": (f"f_{k}", "max") for k in N.FLAGS})
    e["lm_net"] = e["lm_pos"] - e["lm_neg"]
    return e.reset_index()[["symbol", "d", "t", *N.NEWS_INPUTS]]


def trade(bars, t, open_t, prev_ac, exit_ac, dv20, prev_c):
    """One event from the stock's 1-minute bars of day d (columns t = bar start, o, c, v; adjustment=all):
    entry, label inputs and the 4 stock intraday inputs, or None when there is no entry bar in time."""
    b = bars[bars["t"] >= t + WAIT]
    if b.empty or b["t"].iloc[0] > t + MAX_WAIT:
        return None
    entry = float(b["o"].iloc[0])
    before = bars[bars["t"] + pd.Timedelta(minutes=1) <= t]
    if t <= open_t or before.empty:
        first = bars[bars["t"] >= open_t]
        px_t = float(first["o"].iloc[0]) if len(first) else np.nan
    else:
        px_t = float(before["c"].iloc[-1])
    if t <= open_t:
        rv = np.nan
    else:
        w = before[before["t"] >= t - pd.Timedelta(minutes=30)]
        base = dv20 / prev_c * 30 / 390 if dv20 > 0 and prev_c > 0 else np.nan
        rv = float(w["v"].sum()) / base if base and np.isfinite(base) else np.nan
    return {"entry_t": b["t"].iloc[0], "entry": entry, "ret": exit_ac / entry - 1 if entry > 0 else np.nan,
            "mins_open": (t - open_t).total_seconds() / 60, "r_open": px_t / prev_ac - 1,
            "r_react": entry / px_t - 1, "rv30": rv}


def spy_at(bars, t, open_t, wait=True):
    """SPY's price at the decision time t, and at the entry bar when `wait` (the same rule as the stock)."""
    before = bars[bars["t"] + pd.Timedelta(minutes=1) <= t]
    first = bars[bars["t"] >= open_t]
    px_t = float(before["c"].iloc[-1]) if (t > open_t and len(before)) else (
        float(first["o"].iloc[0]) if len(first) else np.nan)
    b = bars[bars["t"] >= t + WAIT]
    return px_t, (float(b["o"].iloc[0]) if len(b) else np.nan)


def zlabel(ev):
    v = ev["vol20"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ev["mret"].to_numpy(float) / (v * HOLD_SCALE)
    z[~(v > 0)] = np.nan
    return np.clip(z, -Z_CLIP, Z_CLIP)


def no_news(X):
    X0 = X.copy()
    nv = N.placebo_news()
    for i, k in enumerate(INPUTS):
        if k in nv:
            X0[:, i] = nv[k]
    return X0


def fit(X, y):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB)
    m.fit(X, y)
    return m


class Model:
    """NX2's design on NX3's inputs: baseline on placebo events, news model on real z - B, e = N(x) - N(x0)."""

    def __init__(self, tr):
        tr = tr.assign(z=zlabel(tr))
        tr = tr[np.isfinite(tr["z"])]
        pl, re = tr[tr["kind"] == "placebo"], tr[tr["kind"] == "real"]
        self.base = fit(pl[BASE_INPUTS].to_numpy(float), pl["z"].to_numpy(float))
        X = re[INPUTS].to_numpy(float)
        self.news = fit(X, re["z"].to_numpy(float) - self.base.predict(re[BASE_INPUTS].to_numpy(float)))
        self.cut = float(np.quantile(np.abs(self.effect(X)), FIRE_Q))
        self.n_base, self.n_news = int(len(pl)), int(len(re))

    def effect(self, X):
        return self.news.predict(X) - self.news.predict(no_news(X))

    def fires(self, e):
        return (np.abs(e) >= self.cut) & (np.abs(e) > 0)


def picks(day, rng, col="e"):
    """Fired events in time order: the first PICKS with e > 0 go long, the first PICKS with e < 0 short.
    Control: as many of the day's real events at random, same sides."""
    f = day[day["fired"]].sort_values("t", kind="stable")
    p = pd.concat([f[f[col] > 0].head(PICKS).assign(side=1), f[f[col] < 0].head(PICKS).assign(side=-1)])
    if p.empty or len(day) < len(p):
        return p, p.iloc[0:0]
    c = day.iloc[rng.choice(len(day), size=len(p), replace=False)].assign(side=p["side"].to_numpy())
    return p, c


def shuffled(day, model, rng):
    X = day[INPUTS].to_numpy(float)
    idx = [INPUTS.index(k) for k in N.NEWS_INPUTS]
    X[:, idx] = X[rng.permutation(len(X))][:, idx]
    e = model.effect(X)
    return day.assign(e=e, fired=model.fires(e))


def tstat(x):
    m, se, t = nw_tstat(x, 5) if np.isfinite(x).sum() > 2 else (float("nan"),) * 3
    return {"mean": m, "se_nw": se, "t_nw": t, "days": int(np.isfinite(x).sum())}


def importance(model):
    gain = model.news.get_booster().get_score(importance_type="gain")
    by = {INPUTS[int(k[1:])]: v for k, v in gain.items()}
    return {g: float(sum(by.get(k, 0.0) for k in ks)) for g, ks in GROUPS.items()}


def evaluate(te, model, seed):
    real = te[te["kind"] == "real"]
    e = model.effect(real[INPUTS].to_numpy(float))
    real = real.assign(e=e, fired=model.fires(e))
    plac = te[te["kind"] == "placebo"]
    pe = model.effect(plac[INPUTS].to_numpy(float)) if len(plac) else np.zeros(0)
    rng = np.random.default_rng(seed)
    rows, pk = [], []
    for d, day in real.groupby("d", sort=True):
        p, c = picks(day, rng)
        if p.empty or c.empty:
            continue
        s, _ = picks(shuffled(day, model, rng), rng)
        pn = N.net(p).mean()
        rows.append({"d": d, "pick_minus_control": pn - N.net(c).mean(),
                     "pick_minus_shuffled": pn - N.net(s).mean() if len(s) else np.nan})
        pk.append(p.assign(vpct=day["vol20"].rank(pct=True).loc[p.index]))
    return pd.DataFrame(rows), (pd.concat(pk) if pk else pd.DataFrame(columns=["vpct", "mins_open"])), real, pe


def walk_forward(ev, cal, years=N.TEST_YEARS):
    caldates = pd.DatetimeIndex(cal["date"])
    out, days = {}, []
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = ev[ev["d"] <= cut]
        te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        model = Model(tr)
        n, pk, real, pe = evaluate(te, model, SEED + yr)
        days.append(n)
        out[str(yr)] = {"train_placebo": model.n_base, "train_real": model.n_news, "fire_cut": model.cut,
                        "real_fire_rate": float(real["fired"].mean()),
                        "placebo_fire_rate": float(model.fires(pe).mean()) if len(pe) else None,
                        "pick_minus_control": tstat(n["pick_minus_control"].to_numpy(float)) if len(n) else None,
                        "pick_minus_shuffled": tstat(n["pick_minus_shuffled"].to_numpy(float)) if len(n) else None,
                        "picks_median_vol20_pct": float(pk["vpct"].median()) if len(pk) else None,
                        "picks_share_before_open": float((pk["mins_open"] <= 0).mean()) if len(pk) else None,
                        "importance_gain_by_group": importance(model)}
    alln = pd.concat(days, ignore_index=True) if days else pd.DataFrame(
        columns=["pick_minus_control", "pick_minus_shuffled"])
    pooled = {k: tstat(alln[k].to_numpy(float)) for k in ("pick_minus_control", "pick_minus_shuffled")}
    gate = {k: bool(v["mean"] > 0 and v["t_nw"] >= GATE_T) for k, v in pooled.items()}
    return out, pooled, gate


# ---------- stages (network / disk) ----------

def all_session_events(cache, ses):
    f = nx3dir(cache) / "session_events.parquet"
    if f.exists():
        return pd.read_parquet(f)
    files = sorted((N.nx1dir(cache) / "tone").glob("tone_*.parquet"))
    last, prev, parts = {}, None, []
    for x in files:
        rows = N.dedup(pd.read_parquet(x), last)
        if len(rows):
            cut = rows["t"].max() - pd.Timedelta(days=3)
            last = {k: t for k, t in last.items() if t >= cut}
        mon = pd.Period(x.stem.removeprefix("tone_"), freq="M")
        both = rows if prev is None else pd.concat([prev, rows], ignore_index=True)
        e = session_events(both, ses)
        parts.append(e[e["d"].dt.to_period("M") == mon])
        prev = rows[rows["t"] >= rows["t"].max() - pd.Timedelta(days=5)] if len(rows) else None
    e = pd.concat(parts, ignore_index=True)
    e = e[(e["d"] >= FIRST_DAY) & (e["d"] <= LAST_DAY)]
    N.atomic_parquet(e, f)
    return e


def batch_events(rows, sev, nd, spy, cal, ses, b):
    """One bar batch's real and placebo events: inputs at the previous session's close, t, and what the
    minutes stage needs (previous adjusted and raw close, dv20, the next session's adjusted close)."""
    pan = []
    for s, g in rows.groupby("symbol"):
        p = N.stock_panel(g, spy, cal)
        if len(p):
            gi = g.drop_duplicates("date").set_index("date")
            pan.append(p[["d", *N.STOCK_INPUTS, "uni"]].assign(
                symbol=s, prev_ac=gi["ac"].reindex(p["d"]).to_numpy(), prev_c=gi["c"].reindex(p["d"]).to_numpy(),
                dv20=gi["dv20"].reindex(p["d"]).to_numpy()))
    if not pan:
        return pd.DataFrame()
    pan = pd.concat(pan, ignore_index=True)
    pan = pan[pan["uni"]].rename(columns={"d": "prev"})
    pan = pan.merge(N.spy_inputs(spy).rename(columns={"d": "prev"}), on="prev", how="left")
    pan = pan.merge(ses[["date", "prev", "next", "open_t"]], on="prev", how="inner").rename(columns={"date": "d"})
    acs = rows[["symbol", "date", "ac"]].rename(columns={"date": "next", "ac": "exit_ac"})
    pan = pan.merge(acs, on=["symbol", "next"], how="inner")
    pan = pan[(pan["d"] >= FIRST_DAY) & (pan["d"] <= LAST_DAY)]
    real = pan.merge(sev, on=["symbol", "d"], how="inner").assign(kind="real")
    cik = rows.drop_duplicates(["symbol", "date"]).rename(columns={"date": "d"})[["symbol", "d", "cik"]]
    cand = pan.merge(cik, on=["symbol", "d"], how="left")
    keys = cand[["symbol", "cik", "d"]].rename(columns={"d": "date"})
    cand = cand[N.quiet_news(keys.reset_index(drop=True), nd, N.QUIET_BACK, ses["date"])].drop(columns="cik")
    rng = np.random.default_rng(SEED + b)
    plac = []
    for d, g in real.groupby("d"):
        c = cand[(cand["d"] == d) & ~cand["symbol"].isin(g["symbol"])]
        k = min(len(g), len(c))
        if k:
            pick = c.iloc[rng.choice(len(c), size=k, replace=False)]
            plac.append(pick.assign(t=g["t"].to_numpy()[rng.permutation(len(g))[:k]]))
    plac = pd.concat(plac, ignore_index=True) if plac else cand.iloc[0:0].assign(t=pd.NaT)
    plac = plac.assign(kind="placebo", **N.placebo_news())
    ev = pd.concat([real, plac], ignore_index=True)
    ev["novelty"] = N.novelty(ev[["symbol", "d"]].reset_index(drop=True), nd)
    return ev


def stage_events(cache, deadline):
    root, out = N.nx1dir(cache), nx3dir(cache) / "events"
    cal = pd.read_parquet(root / "cal.parquet")
    ses = sessions(cal)
    sev = all_session_events(cache, ses)
    nd = W.news_symbol_days(cache)
    spy = pd.read_parquet(root / "spy.parquet")
    files = W.batch_files(cache)
    for b, p in enumerate(files):
        f = out / f"e_{b:04d}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        rows = pd.read_parquet(p, columns=["symbol", "date", "o", "c", "ao", "ac", "dv20", "seg", "cik"])
        ev = batch_events(rows, sev, nd, spy, cal, ses, b)
        N.atomic_parquet(ev, f)
        print(f"[nx3 events] batch {b + 1}/{len(files)}: {int((ev.get('kind') == 'real').sum())} real")
    return True


def day_bars(symbols, day, open_t, close_t, headers):
    """{symbol: 1-minute SIP bars (t naive ET bar start, o, c, v), adjustment=all} for one session."""
    s = pd.Timestamp(open_t, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = pd.Timestamp(close_t, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for i in range(0, len(symbols), SYM_CHUNK):
        tok = None
        while True:
            p = {"symbols": ",".join(symbols[i:i + SYM_CHUNK]), "timeframe": "1Min", "start": s, "end": e,
                 "feed": "sip", "adjustment": "all", "limit": 10000}
            if tok:
                p["page_token"] = tok
            j = alpaca.get(BARS_URL, p, headers=headers)
            for sym, bs in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend((b["t"], b["o"], b["c"], b["v"]) for b in bs)
            tok = j.get("next_page_token")
            if not tok:
                break
    res = {}
    for sym, rows in out.items():
        d = pd.DataFrame(rows, columns=["t", "o", "c", "v"])
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        res[sym] = d.sort_values("t").reset_index(drop=True)
    return res


def month_trades(ev, ses, spy_ac, headers):
    s = ses.set_index("date")
    out = []
    for d, g in ev.groupby("d"):
        o, c = s.loc[d, "open_t"], s.loc[d, "close_t"]
        bars = day_bars(sorted(set(g["symbol"]) | {"SPY"}), d, o, c, headers)
        spyb = bars.get("SPY")
        if spyb is None or spyb.empty:
            continue
        sp_prev, sp_exit = spy_ac.get(s.loc[d, "prev"]), spy_ac.get(s.loc[d, "next"])
        for r in g.itertuples(index=False):
            b = bars.get(r.symbol)
            if b is None or b.empty:
                continue
            tr = trade(b, r.t, o, r.prev_ac, r.exit_ac, r.dv20, r.prev_c)
            if tr is None:
                continue
            spx_t, spx_e = spy_at(spyb, r.t, o)
            row = r._asdict() | tr
            row["spy_open"] = spx_t / sp_prev - 1 if sp_prev else np.nan
            row["spy_ret"] = sp_exit / spx_e - 1 if sp_exit and spx_e else np.nan
            out.append(row)
    t = pd.DataFrame(out)
    if len(t):
        t["mret"] = t["ret"] - t["spy_ret"]
        t["price"] = t["prev_c"]
        t["days"] = (t["next"] - t["d"]).dt.days + 1
    return t


def stage_minutes(cache, deadline):
    root, d3 = N.nx1dir(cache), nx3dir(cache)
    ses = sessions(pd.read_parquet(root / "cal.parquet"))
    spy = pd.read_parquet(root / "spy.parquet")
    spy_ac = dict(zip(spy["date"], spy["ac"]))
    cols = ["symbol", "d", "t", "kind", *N.INPUTS, "prev_ac", "prev_c", "dv20", "exit_ac", "next"]
    files = sorted((d3 / "events").glob("e_*.parquet"))
    months = pd.period_range(FIRST_DAY, LAST_DAY, freq="M")
    h = alpaca_headers()
    for m in months:
        f = d3 / "trades" / f"t_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        lo, hi = m.start_time, m.end_time
        ev = pd.concat([pd.read_parquet(x, columns=cols, filters=[("d", ">=", lo), ("d", "<=", hi)]) for x in files],
                       ignore_index=True)
        t = month_trades(ev, ses, spy_ac, h) if len(ev) else pd.DataFrame()
        N.atomic_parquet(t, f)
        print(f"[nx3 minutes] {m}: {len(ev)} events, {len(t)} with an entry")
    return True


def stage_train(cache, out):
    d3 = nx3dir(cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / "trades").glob("t_*.parquet"))], ignore_index=True)
    ev = ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)
    cal = pd.read_parquet(N.nx1dir(cache) / "cal.parquet")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wf, pooled, gate = walk_forward(ev, cal)
    rep = {"walk_forward": wf, "pooled": pooled, "gate": gate, "gate_passed": all(gate.values()),
           "events": {"real": int((ev["kind"] == "real").sum()), "placebo": int((ev["kind"] == "placebo").sum())}}
    if rep["gate_passed"]:
        import xgboost
        m = Model(ev)
        files = {"base": out / "nx3_v1_base.json", "news": out / "nx3_v1_news.json"}
        m.base.get_booster().save_model(files["base"])
        m.news.get_booster().save_model(files["news"])
        meta = {"version": 1, "inputs": INPUTS, "base_inputs": BASE_INPUTS, "fire_cut": m.cut, "params": XGB,
                "train_first": str(ev["d"].min().date()), "train_last": str(ev["d"].max().date()),
                "xgboost": xgboost.__version__, "importance_gain_by_group": importance(m),
                "sha256": {k: hashlib.sha256(f.read_bytes()).hexdigest() for k, f in files.items()}}
        (out / "nx3_v1_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "NX3_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"pooled": pooled, "gate": gate}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "events", "minutes", "train"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/NX3")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    for s in (["events", "minutes", "train"] if a.stage == "all" else [a.stage]):
        if s == "events" and not stage_events(a.cache, deadline):
            print("[nx3] time budget used during events; rerun to continue")
            return
        if s == "minutes" and not stage_minutes(a.cache, deadline):
            print("[nx3] time budget used during minutes; rerun to continue")
            return
        if s == "train":
            stage_train(a.cache, a.out)


if __name__ == "__main__":
    main()
