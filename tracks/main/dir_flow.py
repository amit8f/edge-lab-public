"""DIR-FLOW and DIR-OFI (research/prereg/DIRECTION.md, DIR-FLOW and Amendment 2): BOOM direction from order flow.

Stage dirflow_data (resumable, day by day, time-budgeted): for each 2022-01..2025-09 session with BOOM2 picks, Alpaca
SIP trades 4:00-9:35 ET and quotes 4:00-9:51 ET for that day's 20 picks. Only a per pick summary is kept
(cache nx3/dirflow/days/<date>.parquet); no trade or quote is stored.
- DIR-FLOW input: Lee-Ready signed-volume imbalance (buy - sell) / (buy + sell), 4:00-9:35, each trade signed against
  the quote in force 1 second earlier (tick rule at the midpoint or without a quote), opening cross prints excluded.
- DIR-OFI input: Cont-Kukanov-Stoikov order flow imbalance over NBBO updates 9:30:00-9:50:59, divided by the window's
  mean (bid size + ask size) / 2; crossed, locked and zero-size quotes skipped; also its 9:30-9:40 and 9:40-9:51 halves.
Stage dirflow: both rules on the 9:51 entry with +-4% barriers (BOOM2's lab951/exit951), their baselines, random-side
controls and the registered pass tests. Writes research/results/DIR/DIRFLOW.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main import boom2_study as B2
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades"
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
FIRST, LAST = pd.Timestamp("2022-01-01"), pd.Timestamp("2025-09-30")
SEED = 20261113
MIN_CALLS = 1000
N_BOOT = 2000
# Condition table, fixed before any result (CTA/UTP codes as Alpaca reports them). Excluded trades:
EXCLUDE = {"4": "derivatively priced", "B": "average price", "W": "average price", "Z": "sold out of sequence",
           "U": "extended hours sold out of sequence", "P": "prior reference price", "O": "opening print",
           "Q": "market center official open", "M": "market center official close", "9": "corrected close",
           "H": "price variation", "V": "contingent", "7": "qualified contingent"}
ODD_LOT = "I"
LARGE_SHARES, LARGE_USD = 10_000, 200_000
LAG = pd.Timedelta(seconds=1)


def ddir(cache):
    return X.nx3dir(cache) / "dirflow"


# ---------- pure helpers (tested) ----------

def clean_trades(t):
    """Trades frame (t, p, s, x, i, c: list of conditions) -> deduplicated on exchange and id, excluded conditions
    dropped, sorted by time."""
    if t.empty:
        return t
    t = t.drop_duplicates(["x", "i"])
    bad = t["c"].apply(lambda cs: any(c in EXCLUDE for c in (cs or [])))
    return t[~bad & (t["s"] > 0) & (t["p"] > 0)].sort_values("t", kind="stable").reset_index(drop=True)


def good_quotes(q):
    """Quotes (t, bp, bs, ap, as) with a positive, uncrossed, unlocked spread and non-zero sizes, sorted."""
    if q.empty:
        return q
    ok = (q["bp"] > 0) & (q["ap"] > q["bp"]) & (q["bs"] > 0) & (q["as"] > 0)
    return q[ok].sort_values("t", kind="stable").reset_index(drop=True)


def lee_ready(t, q, lag=LAG):
    """Sign of each trade: +1 buy, -1 sell, by the quote in force `lag` earlier (above / below its midpoint), else the
    tick rule (last different price); 0 when neither decides."""
    if t.empty:
        return np.zeros(0)
    sign = np.zeros(len(t))
    if len(q):
        mid = ((q["bp"] + q["ap"]) / 2).to_numpy(float)
        k = np.searchsorted(q["t"].to_numpy(), (t["t"] - lag).to_numpy(), side="right") - 1
        has = k >= 0
        m = np.where(has, mid[np.clip(k, 0, None)], np.nan)
        p = t["p"].to_numpy(float)
        sign = np.where(has & (p > m), 1.0, np.where(has & (p < m), -1.0, 0.0))
    p = t["p"].to_numpy(float)
    tick, prev_tick = np.zeros(len(p)), 0.0
    for j in range(1, len(p)):
        if p[j] != p[j - 1]:
            prev_tick = 1.0 if p[j] > p[j - 1] else -1.0
        tick[j] = prev_tick
    return np.where(sign != 0, sign, tick)


def imbalance(t, sign):
    v = t["s"].to_numpy(float) if len(t) else np.zeros(0)
    buy, sell = v[sign > 0].sum(), v[sign < 0].sum()
    return (buy - sell) / (buy + sell) if buy + sell > 0 else np.nan


def ofi(q):
    """Cont-Kukanov-Stoikov OFI over consecutive quotes, divided by the mean depth (bs + as) / 2; NaN if < 2 quotes."""
    if len(q) < 2:
        return np.nan
    b, bs, a, as_ = (q[c].to_numpy(float) for c in ("bp", "bs", "ap", "as"))
    e = ((b[1:] >= b[:-1]) * bs[1:] - (b[1:] <= b[:-1]) * bs[:-1]
         - (a[1:] <= a[:-1]) * as_[1:] + (a[1:] >= a[:-1]) * as_[:-1])
    depth = ((bs + as_) / 2).mean()
    return float(e.sum() / depth) if depth > 0 else np.nan


def pick_summary(trades, quotes, day):
    """One pick's trades and quotes (naive ET times) -> its summary row."""
    t0, t935 = day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=9, minutes=35)
    o, o40, o51 = day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=9, minutes=40), day + pd.Timedelta(hours=9, minutes=51)
    t = clean_trades(trades)
    t = t[(t["t"] >= t0) & (t["t"] < t935)] if len(t) else t
    q = good_quotes(quotes)
    s = lee_ready(t, q[q["t"] < t935] if len(q) else q)
    v = t["s"].to_numpy(float) if len(t) else np.zeros(0)
    usd = v * (t["p"].to_numpy(float) if len(t) else np.zeros(0))
    raw = trades.drop_duplicates(["x", "i"]) if len(trades) else trades
    odd = raw["c"].apply(lambda cs: ODD_LOT in (cs or [])).mean() if len(raw) else np.nan
    w = q[(q["t"] >= o) & (q["t"] < o51)] if len(q) else q
    return {"flow_imb": imbalance(t, s), "n_trades": len(t), "n_unsigned": int((s == 0).sum()),
            "odd_lot_share": float(odd) if np.isfinite(odd) else np.nan,
            "large_share": float(v[(v >= LARGE_SHARES) | (usd >= LARGE_USD)].sum() / v.sum()) if v.sum() > 0 else np.nan,
            "d_share": float(v[(t["x"] == "D").to_numpy()].sum() / v.sum()) if v.sum() > 0 else np.nan,
            "ofi": ofi(w), "ofi_h1": ofi(w[w["t"] < o40]) if len(w) else np.nan,
            "ofi_h2": ofi(w[w["t"] >= o40]) if len(w) else np.nan, "n_quotes": len(w)}


def balanced_accuracy(side, lab):
    """Mean of the up and down hit rates over calls (side != 0) with lab in {1, -1}."""
    side, lab = np.asarray(side), np.asarray(lab)
    m = (side != 0) & np.isin(lab, [1, -1])
    if not m.any():
        return np.nan
    up, dn = m & (lab == 1), m & (lab == -1)
    r = [np.mean(side[k] == lab[k]) for k in (up, dn) if k.any()]
    return float(np.mean(r))


def boot_lower(days, diff, seed=SEED, n=N_BOOT):
    """5th percentile of the mean per-day difference, resampling whole days."""
    d = pd.Series(np.asarray(diff, float)).groupby(np.asarray(days)).mean().dropna().to_numpy()
    if len(d) < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    return float(np.quantile(rng.choice(d, (n, len(d))).mean(axis=1), 0.05))


# ---------- data ----------

def _frames(rows, cols):
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["t"] = pd.to_datetime(d["t"], utc=True, format="ISO8601").dt.tz_convert(X.ET).dt.tz_localize(None)
    return d


def fetch(url, key, symbols, start, end, h):
    s, e = (pd.Timestamp(x, tz=X.ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") for x in (start, end))
    out, tok = {}, None
    while True:
        p = {"symbols": ",".join(symbols), "start": s, "end": e, "feed": "sip", "limit": 10000}
        if tok:
            p["page_token"] = tok
        j = alpaca.get(url, p, headers=h)
        for sym, rows in (j.get(key) or {}).items():
            out.setdefault(sym, []).extend(rows)
        tok = j.get("next_page_token")
        time.sleep(0.3)
        if not tok:
            return out


def stage_data(cache, deadline):
    picks = B2.load_all(cache)
    picks = picks[(picks["nxt"] >= FIRST) & (picks["nxt"] <= LAST)]
    root = ddir(cache) / "days"
    root.mkdir(parents=True, exist_ok=True)
    h = alpaca_headers()
    days = sorted(picks["nxt"].unique())
    done = 0
    for d in days:
        d = pd.Timestamp(d)
        f = root / f"{d:%Y-%m-%d}.parquet"
        if f.exists():
            done += 1
            continue
        if time.monotonic() > deadline:
            print(f"[dirflow data] time budget used; {done} of {len(days)} days done")
            return False
        syms = sorted(picks.loc[picks["nxt"] == d, "symbol"])
        rows = []
        for s in syms:                                   # one stock at a time keeps memory small
            tr = fetch(TRADES_URL, "trades", [s], d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=9, minutes=35), h)
            qu = fetch(QUOTES_URL, "quotes", [s], d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=9, minutes=51), h)
            t = _frames(tr.get(s, []), ["t", "p", "s", "x", "i", "c"])
            q = _frames(qu.get(s, []), ["t", "bp", "bs", "ap", "as"])
            rows.append({"symbol": s, "nxt": d, **pick_summary(t, q, d)})
        N.atomic_parquet(pd.DataFrame(rows), f)
        done += 1
        if done % 20 == 0:
            print(f"[dirflow data] {done} of {len(days)} days", flush=True)
    print(f"[dirflow data] all {len(days)} days done")
    return True


# ---------- test ----------

def arm(df, side, base_side, ctrl, name):
    lab = df["lab951"].to_numpy(int)
    price = df["e951"].to_numpy(float)
    net = B2.net(side, df["e951"], df["exit951"], price)
    cnet = B2.net(ctrl, df["e951"], df["exit951"], price)
    m = (side != 0) & np.isfinite(net)
    ba, bb = balanced_accuracy(side[m], lab[m]), balanced_accuracy(base_side[m], lab[m])
    diff = net[m] - cnet[m]
    out = {"arm": name, "calls": int(m.sum()), "balanced_accuracy": ba, "baseline_balanced_accuracy": bb,
           "mean_net": float(np.nanmean(net[m])) if m.any() else None,
           "minus_control": float(np.nanmean(diff)) if m.any() else None,
           "minus_control_boot_lower": boot_lower(df.loc[m, "nxt"], diff)}
    return out


def run(cache, out="research/results/DIR"):
    picks = B2.load_all(cache)
    picks = picks[(picks["nxt"] >= FIRST) & (picks["nxt"] <= LAST) & (picks["lab"] != 2)]
    fl = pd.concat([pd.read_parquet(f) for f in sorted((ddir(cache) / "days").glob("*.parquet"))], ignore_index=True)
    df = picks.merge(fl, on=["symbol", "nxt"], how="inner").reset_index(drop=True)
    df = df[np.isfinite(df["e951"].to_numpy(float))].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    ctrl = rng.choice([-1.0, 1.0], len(df))
    gap = np.sign(df["pm_gap"].to_numpy(float))
    prev_c = df["c"].to_numpy(float)
    early = np.sign(df["e951"].to_numpy(float) / prev_c - 1)
    flow = np.sign(np.nan_to_num(df["flow_imb"].to_numpy(float)))
    of = np.sign(np.nan_to_num(df["ofi"].to_numpy(float)))
    rep = {"picks": len(df), "days": int(df["nxt"].nunique()),
           "DIR-FLOW": arm(df, flow, gap, ctrl, "DIR-FLOW"), "DIR-OFI": arm(df, of, early, ctrl, "DIR-OFI")}
    f, o = rep["DIR-FLOW"], rep["DIR-OFI"]
    rep["DIR-FLOW"]["pass"] = bool(f["calls"] >= MIN_CALLS and f["balanced_accuracy"] >= f["baseline_balanced_accuracy"] + 0.03
                                   and (f["minus_control_boot_lower"] or -1) > 0)
    rep["DIR-OFI"]["pass"] = bool(o["calls"] >= MIN_CALLS and o["balanced_accuracy"] >= o["baseline_balanced_accuracy"] + 0.03
                                  and (o["minus_control_boot_lower"] or -1) > 0 and (o["mean_net"] or -1) > 0)
    rep["reported"] = {
        "ofi_halves": {k: arm(df, np.sign(np.nan_to_num(df[k].to_numpy(float))), early, ctrl, k) for k in ("ofi_h1", "ofi_h2")},
        "deciles": {c: [{"decile": int(k), "up_first_of_touched": float((g.loc[g["lab951"] != 0, "lab951"] == 1).mean())
                         if (g["lab951"] != 0).any() else None, "n": len(g)}
                        for k, g in df.groupby(pd.qcut(df[c].rank(method="first"), 10, labels=False))]
                    for c in ("flow_imb", "ofi", "large_share", "odd_lot_share", "d_share") if df[c].notna().sum() > 100},
        "quotes_per_pick_median": float(df["n_quotes"].median()), "trades_per_pick_median": float(df["n_trades"].median())}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "DIRFLOW.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep
