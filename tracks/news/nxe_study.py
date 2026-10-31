"""NXE and NXF (research/prereg/NXE.md, NXF.md): other exits and filters on NX3's own walk-forward trades.

Stages (each resumes from its state folder):
  picks    rerun NX3's walk-forward (same models, seeds and draws) and keep, per test year, its picks and
           controls, every fired event of the test year and of the training years (scored by that year's
           model), and the test year's real events (for NXF's controls).
  daily    ATR and the next 20 sessions' prices of every kept trade from W1's daily bars in NX1's cache,
           restated in day d's raw prices (NX3's entries are raw minute opens).
  minutes  day d's raw 1-minute SIP bars after each kept trade's entry: high, low and close after entry and
           the entry bar's volume; the path to 15:55 for picks and controls (the DAY arm).
  rv       the median volume of the entry minute over the 20 sessions before d (NXF-RV), fired test events.
  filings  8-K items and acceptance times of the companies behind fired test longs (NXF-NO8K).
  measure  every arm and filter, the registered tests and the reports.

Usage: python -m tracks.news.nxe_study --stage picks --cache cache/nx1 --state cache/nxe --out research/results/NXE
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, sec
from core.env import alpaca_headers
from tracks.main.longterm.stats import nw_tstat
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X3
from tracks.news import w1_data as W

ET = X3.ET
ATR_N = 14
HOLDS = {"H2": 2, "H3": 3, "H5": 5, "H20": 20}
HOLD_STOP = 3.0                       # H5, H20: close-checked stop in ATRs
DAY_STOP, DAY_TARGET = 1.5, 3.0       # DAY and EXA1
FRAME = 5                             # EXA1 / EXQ: sessions after d
DAY_LAST = pd.Timedelta(hours=15, minutes=55)
Q_STOP, Q_TARGET, Q_CLIP = 0.80, 0.50, (0.5, 6.0)
EXQ_LGBM = dict(n_estimators=200, max_depth=3, num_leaves=7, min_child_samples=200, learning_rate=0.05,
                deterministic=True, force_row_wise=True, verbose=-1, random_state=X3.SEED)
PASS_T = 1.5
TRAIL_K, TRAIL_ARM, TRAIL_C_MAX = 1.5, 1.0, 2.0   # NXT: trail distance, profit (ATRs) that arms it, cap on c
HOLDOUT = pd.Timestamp("2025-10-01")
FILTER_SEED = {"NOV": 101, "RV": 202, "NO8K": 303}
BAD_8K = {"1.03", "2.06", "3.01", "4.01", "4.02"}
NOV_MAX, RV_MULT, RV_DAYS, BAD_8K_SESSIONS = 2, 2.0, 20, 5
SUBS = "https://data.sec.gov/submissions/"
KEEP = ["symbol", "d", "t", "entry_t", "entry", "e", "prev_c", "price", "days", "ret", "next", "novelty",
        "rv30", "mins_open"]


# ---------- pure helpers (tested) ----------

def hold_exit(entry, side, atr, closes, dates, k, stop=None):
    """Exit at the close of session k (closes[0] is d's close), or earlier at the first close that is `stop`
    ATRs or more against the trade. (price, date) or None when a needed close is missing."""
    if stop is not None and not atr > 0:
        return None
    for i in range(0, k + 1):
        c = closes[i] if i < len(closes) else np.nan
        if not np.isfinite(c):
            return None
        if stop is not None and side * (c - entry) <= -stop * atr:
            return c, dates[i]
    return closes[k], dates[k]


def path_exit(entry, side, stop_px, target_px, highs, lows, closes):
    """Intraday exit on bars after the entry: the stop, the target (a bar reaching both counts as the stop),
    else the last bar's close. (price, reason)."""
    for h, lo in zip(highs, lows):
        hit_stop = lo <= stop_px if side > 0 else h >= stop_px
        hit_tgt = h >= target_px if side > 0 else lo <= target_px
        if hit_stop:
            return stop_px, "stop"
        if hit_tgt:
            return target_px, "target"
    return (closes[-1], "time") if len(closes) else (np.nan, "none")


def frame_exit(entry, side, stop_px, target_px, sessions):
    """EXA1 / EXQ on sessions [(open, high, low, close, date)], the first being d after the entry (its open
    unused). A session opening past the stop fills at its open; a session reaching both levels counts as
    the stop; the target fills at its level; else the last session's close. (price, date, reason)."""
    for i, (o, h, lo, c, dt) in enumerate(sessions):
        if not all(np.isfinite(x) for x in (h, lo, c)):
            return np.nan, dt, "missing"
        if i > 0 and np.isfinite(o) and (o <= stop_px if side > 0 else o >= stop_px):
            return o, dt, "stop"
        if (lo <= stop_px) if side > 0 else (h >= stop_px):
            return stop_px, dt, "stop"
        if (h >= target_px) if side > 0 else (lo <= target_px):
            return target_px, dt, "target"
    return sessions[-1][3], sessions[-1][4], "time"


def trail_exit(entry, side, atr, k, sessions):
    """NXT on sessions [(open, high, low, close, date)], the first being d after the entry (its open
    unused). The stop starts k ATRs away; each session is checked against the stop before its best price
    moves it; once the best is 1 ATR in profit the stop goes to max(stop, entry, best - k ATR), never back.
    (price, date, reason)."""
    stop, best, why = entry - side * k * atr, entry, "stop"
    for i, (o, h, lo, c, dt) in enumerate(sessions):
        if not all(np.isfinite(x) for x in (h, lo, c)):
            return np.nan, dt, "missing"
        if i > 0 and np.isfinite(o) and (o <= stop if side > 0 else o >= stop):
            return o, dt, why
        if (lo <= stop) if side > 0 else (h >= stop):
            return stop, dt, why
        best = max(best, h) if side > 0 else min(best, lo)
        if side * (best - entry) >= TRAIL_ARM * atr:
            new = max(entry, best - k * atr) if side > 0 else min(entry, best + k * atr)
            if side * (new - stop) > 0:
                stop, why = new, ("breakeven" if new == entry else "trail")
    return sessions[-1][3], sessions[-1][4], "time"


def excursions(entry, side, atr, sessions):
    """(adverse, favourable) excursion over the frame, in ATRs, both >= 0."""
    hs = np.array([s[1] for s in sessions], float)
    ls = np.array([s[2] for s in sessions], float)
    if not (np.isfinite(hs).all() and np.isfinite(ls).all()) or not atr > 0:
        return np.nan, np.nan
    up, down = (hs.max() - entry) / atr, (entry - ls.min()) / atr
    return (max(down, 0.0), max(up, 0.0)) if side > 0 else (max(up, 0.0), max(down, 0.0))


def atr(h, l, c, n=ATR_N):
    """Average true range of the last n sessions (arrays oldest first, the last one is d - 1)."""
    if len(c) < n + 1:
        return np.nan
    h, l, c = np.asarray(h[-n:], float), np.asarray(l[-n:], float), np.asarray(c, float)
    pc = c[-n - 1:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(np.mean(tr))


def bad_8k(filings, cik, d, t, sessions_before):
    """True when the company filed an 8-K with a bad-news item dated in the sessions before d, or on d and
    accepted before t. `filings`: rows cik, filed, accepted_et, items (set)."""
    f = filings.get(int(cik)) if pd.notna(cik) else None
    if f is None:
        return False
    for filed, acc, items in f:
        if not (items & BAD_8K):
            continue
        if sessions_before[0] <= filed < d or (filed == d and acc < t):
            return True
    return False


def daily_diff(rows, col="net"):
    """Per day: mean of picks minus mean of controls (rows: d, role, col); days with both."""
    g = rows.groupby(["d", "role"])[col].mean().unstack()
    if "pick" not in g or "control" not in g:
        return pd.Series(dtype=float)
    return (g["pick"] - g["control"]).dropna()


def tstat(x, lags):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 3:
        return {"mean": float("nan"), "t_nw": float("nan"), "days": int(np.isfinite(x).sum())}
    m, se, t = nw_tstat(x, max(1, lags))
    return {"mean": float(m), "se_nw": float(se), "t_nw": float(t), "days": int(np.isfinite(x).sum())}


# ---------- stage: picks ----------

def load_ev(cache):
    d3 = X3.nx3dir(cache)
    ev = pd.concat([pd.read_parquet(f) for f in sorted((d3 / X3.TRADES).glob("t_*.parquet"))], ignore_index=True)
    return ev[np.isfinite(ev["mret"]) & (ev["next"] <= N.LAST_EXIT)].reset_index(drop=True)


def year_picks(ev, cal, yr):
    """NX3's test year yr exactly as its walk-forward ran it, keeping what NXE and NXF need."""
    caldates = pd.DatetimeIndex(cal["date"])
    start = pd.Timestamp(f"{yr}-01-01")
    cut = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
    tr = ev[ev["d"] <= cut]
    te = ev[(ev["d"] >= start) & (ev["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
    if tr.empty or te.empty:
        return None
    model = X3.Model(tr)
    real = te[te["kind"] == "real"]
    e = model.effect(real[X3.INPUTS].to_numpy(float))
    real = real.assign(e=e, fired=model.fires(e))
    rng = np.random.default_rng(X3.SEED + yr)
    out = []
    for d, day in real.groupby("d", sort=True):           # NX3's evaluate(), draw for draw
        p, c = X3.picks(day, rng)
        if p.empty or c.empty:
            continue
        X3.picks(X3.shuffled(day, model, rng), rng)
        out += [p.assign(role="pick"), c.assign(role="control")]
    tp = pd.concat(out, ignore_index=True)
    fired = real[real["fired"]].assign(side=lambda x: np.sign(x["e"]).astype(int), role="fired_test")
    trr = tr[tr["kind"] == "real"]
    te_ = model.effect(trr[X3.INPUTS].to_numpy(float))
    trf = trr.assign(e=te_, fired=model.fires(te_))
    trf = trf[trf["fired"]].assign(side=lambda x: np.sign(x["e"]).astype(int), role="fired_train")
    cols = KEEP + X3.INPUTS + ["side", "role"]
    cols = list(dict.fromkeys(cols))
    rows = pd.concat([tp, fired, trf], ignore_index=True)[cols].assign(year=yr, cut=model.cut)
    real_small = real[KEEP + ["fired"]].assign(year=yr, cut=model.cut)
    return rows, real_small


def stage_picks(ev, cal, state, deadline):
    state.mkdir(parents=True, exist_ok=True)
    for yr in N.TEST_YEARS:
        if (state / f"y{yr}_trades.parquet").exists():
            continue
        if time.monotonic() > deadline:
            return False
        t0 = time.monotonic()
        res = year_picks(ev, cal, yr)
        if res is None:
            continue
        N.atomic_parquet(res[1], state / f"y{yr}_real.parquet")
        N.atomic_parquet(res[0], state / f"y{yr}_trades.parquet")
        print(f"[nxe picks] {yr}: {res[0]['role'].value_counts().to_dict()} in {time.monotonic() - t0:.0f}s",
              flush=True)
    return True


def trades(state):
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(state).glob("y*_trades.parquet"))], ignore_index=True)


# ---------- stage: daily ----------

def stage_daily(cache, cal, state):
    f = state / "daily.parquet"
    if f.exists():
        return True
    if not W.batch_files(cache):
        raise SystemExit(f"[nxe daily] no W1 daily bars under {W.w1dir(cache)}")
    t = trades(state)[["symbol", "d"]].drop_duplicates()
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    days_of = t.groupby("symbol")["d"].apply(list).to_dict()
    syms = sorted(days_of)
    for i in range(0, len(syms), 500):
        chunk = syms[i:i + 500]
        rows = W.read_rows(cache, ["symbol", "date", "o", "h", "l", "c", "ao", "ac", "cik"],
                           [("symbol", "in", chunk)])
        for s, g in rows.groupby("symbol"):
            g = g.drop_duplicates("date").set_index("date").sort_index()
            k = (g["ac"] / g["c"]).where(g["c"] > 0)
            adj = pd.DataFrame({"o": g["ao"], "h": g["h"] * k, "l": g["l"] * k, "c": g["ac"], "cik": g["cik"]})
            for d in days_of.get(s, []):
                if d not in adj.index or not g.loc[d, "c"] > 0:
                    continue
                f_d = adj.loc[d, "c"] / g.loc[d, "c"]                  # all-adjusted -> day d raw prices
                j = caldates.searchsorted(d)
                prev = caldates[max(j - ATR_N - 1, 0):j]
                nxt = caldates[j:j + 21]
                hist = adj.reindex(prev)
                fwd = adj.reindex(nxt)
                rec = {"symbol": s, "d": d, "cik": adj.loc[d, "cik"],
                       "atr": atr(hist["h"].to_numpy() / f_d, hist["l"].to_numpy() / f_d, hist["c"].to_numpy() / f_d)}
                for n_, (dt, r) in enumerate(fwd.iterrows()):
                    rec[f"date{n_}"] = dt
                    for col in ("o", "h", "l", "c") if n_ <= FRAME else ("c",):
                        rec[f"{col}{n_}"] = r[col] / f_d
                out.append(rec)
        print(f"[nxe daily] {min(i + 500, len(syms))}/{len(syms)} symbols", flush=True)
    N.atomic_parquet(pd.DataFrame(out), f)
    return True


# ---------- stage: minutes ----------

def bars_1min(symbols, start, end, headers):
    """{symbol: DataFrame(t naive ET, o, h, l, c, v)}: raw 1-minute SIP bars from start to end (naive ET)."""
    s, e = (pd.Timestamp(x, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") for x in (start, end))
    out = {}
    for i in range(0, len(symbols), X3.SYM_CHUNK):
        tok = None
        while True:
            p = {"symbols": ",".join(symbols[i:i + X3.SYM_CHUNK]), "timeframe": "1Min", "start": s, "end": e,
                 "feed": "sip", "adjustment": "raw", "limit": 10000}
            if tok:
                p["page_token"] = tok
            j = alpaca.get(X3.BARS_URL, p, headers=headers)
            for sym, bs in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bs)
            tok = j.get("next_page_token")
            if not tok:
                break
    res = {}
    for sym, rows in out.items():
        d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
        d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        res[sym] = d.sort_values("t").reset_index(drop=True)
    return res


def after_entry(b, entry_t, day, keep_path):
    a = b[b["t"] >= entry_t]
    rec = {"post_h": float(a["h"].max()) if len(a) else np.nan, "post_l": float(a["l"].min()) if len(a) else np.nan,
           "post_c": float(a["c"].iloc[-1]) if len(a) else np.nan,
           "v_entry": float(a["v"].iloc[0]) if len(a) and a["t"].iloc[0] == entry_t else 0.0}
    if keep_path:
        p = a[a["t"] <= day + DAY_LAST]
        rec |= {"path_h": p["h"].astype("float32").tolist(), "path_l": p["l"].astype("float32").tolist(),
                "path_c": p["c"].astype("float32").tolist()}
    return rec


def stage_minutes(cal, state, deadline):
    t = trades(state)
    t["is_pc"] = t["role"].isin(["pick", "control"])
    need = t.groupby(["symbol", "d", "entry_t"], as_index=False)["is_pc"].max()
    ses = X3.sessions(cal).set_index("date")
    h = alpaca_headers()
    md = state / "minutes"
    md.mkdir(exist_ok=True)
    for m, gm in need.groupby(need["d"].dt.to_period("M")):
        f = md / f"m_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        out = []
        for d, g in gm.groupby("d"):
            bars = bars_1min(sorted(g["symbol"].unique()), ses.loc[d, "open_t"], ses.loc[d, "close_t"], h)
            for r in g.itertuples(index=False):
                b = bars.get(r.symbol)
                if b is None or b.empty:
                    continue
                out.append({"symbol": r.symbol, "d": d, "entry_t": r.entry_t} |
                           after_entry(b, r.entry_t, d, bool(r.is_pc)))
        N.atomic_parquet(pd.DataFrame(out), f)
        print(f"[nxe minutes] {m}: {len(gm)} trades, {len(out)} with bars", flush=True)
    return True


def minutes(state):
    return pd.concat([pd.read_parquet(f) for f in sorted((state / "minutes").glob("m_*.parquet"))], ignore_index=True)


# ---------- stage: rv ----------

def stage_rv(cal, state, deadline):
    t = trades(state)
    t = t[t["role"] == "fired_test"][["symbol", "d", "entry_t"]].drop_duplicates()
    caldates = pd.DatetimeIndex(cal["date"])
    h = alpaca_headers()
    rd = state / "rv"
    rd.mkdir(exist_ok=True)
    for m, gm in t.groupby(t["d"].dt.to_period("M")):
        f = rd / f"r_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        want = {}                                        # (date, minute of day) -> symbols
        for r in gm.itertuples(index=False):
            j = caldates.searchsorted(r.d)
            mod = r.entry_t - r.d
            for pd_ in caldates[max(j - RV_DAYS, 0):j]:
                want.setdefault((pd_, mod), set()).add(r.symbol)
        vol = {}
        for (pd_, mod), syms in sorted(want.items()):
            b = bars_1min(sorted(syms), pd_ + mod, pd_ + mod + pd.Timedelta(minutes=1), h)
            for s in syms:
                x = b.get(s)
                vol[(s, pd_, mod)] = float(x.loc[x["t"] == pd_ + mod, "v"].sum()) if x is not None else 0.0
        out = []
        for r in gm.itertuples(index=False):
            j = caldates.searchsorted(r.d)
            mod = r.entry_t - r.d
            v = [vol.get((r.symbol, pd_, mod), 0.0) for pd_ in caldates[max(j - RV_DAYS, 0):j]]
            out.append({"symbol": r.symbol, "d": r.d, "med20": float(np.median(v)) if v else np.nan})
        N.atomic_parquet(pd.DataFrame(out), f)
        print(f"[nxe rv] {m}: {len(gm)} events, {len(want)} minute requests", flush=True)
    return True


# ---------- stage: filings ----------

def filings_8k(cik, get=sec.get, first=pd.Timestamp("2018-12-01"), last=pd.Timestamp("2025-09-30")):
    """8-K rows (cik, filed, accepted_et naive New York, items) of one company, filed first .. last."""
    r = get(f"{SUBS}CIK{int(cik):010d}.json")
    if r is None:
        return []
    j = r.json()
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        if pd.Timestamp(f.get("filingTo", "1900-01-01")) >= first:
            rr = get(SUBS + f["name"])
            if rr is not None:
                blocks.append(rr.json())
    out = []
    for b in blocks:
        n = len(b.get("form", []))
        for form, filed, items, at in zip(b.get("form", []), b.get("filingDate", []), b.get("items", [""] * n),
                                          b.get("acceptanceDateTime", [""] * n)):
            if form != "8-K" or not at or not first <= pd.Timestamp(filed) <= last:
                continue
            out.append({"cik": int(cik), "filed": pd.Timestamp(filed),
                        "accepted_et": pd.Timestamp(at[:19], tz="UTC").tz_convert(ET).tz_localize(None),
                        "items": ",".join(i.strip() for i in (items or "").split(",") if i.strip())})
    return out


def stage_filings(state, deadline):
    f = state / "filings.parquet"
    if f.exists():
        return True
    t = trades(state)
    t = t[(t["role"] == "fired_test") & (t["side"] > 0)][["symbol", "d"]].drop_duplicates()
    dl = pd.read_parquet(state / "daily.parquet", columns=["symbol", "d", "cik"])
    ciks = sorted({int(c) for c in t.merge(dl, on=["symbol", "d"])["cik"].dropna()})
    part = state / "filings_part.parquet"
    rows = pd.read_parquet(part) if part.exists() else pd.DataFrame(columns=["cik", "filed", "accepted_et", "items"])
    done = set(rows["cik"].astype(int)) if len(rows) else set()
    todo = [c for c in ciks if c not in done]
    new = []
    for i, c in enumerate(todo):
        if time.monotonic() > deadline:
            break
        got = filings_8k(c)
        new += got or [{"cik": c, "filed": pd.NaT, "accepted_et": pd.NaT, "items": ""}]
        if (i + 1) % 200 == 0:
            rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True)
            new = []
            N.atomic_parquet(rows, part)
            print(f"[nxe filings] {len(set(rows['cik']))}/{len(ciks)} companies", flush=True)
    rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True) if new else rows
    N.atomic_parquet(rows, part)
    if len(set(rows["cik"].astype(int))) < len(ciks):
        return False
    N.atomic_parquet(rows, f)
    return True


# ---------- stage: measure ----------

def frame_sessions(r):
    """Day d after the entry, then sessions 1..FRAME from the daily table (row r merged)."""
    s = [(np.nan, r["post_h"], r["post_l"], r["post_c"], r["d"])]
    for n_ in range(1, FRAME + 1):
        s.append((r.get(f"o{n_}", np.nan), r.get(f"h{n_}", np.nan), r.get(f"l{n_}", np.nan),
                  r.get(f"c{n_}", np.nan), r.get(f"date{n_}", pd.NaT)))
    return s


def costed(t, ret, exit_date):
    t = t.assign(ret=ret, days=(pd.to_datetime(exit_date) - t["d"]).dt.days + 1)
    ok = np.isfinite(t["ret"]) & (pd.to_datetime(exit_date) < HOLDOUT)
    t = t[ok.to_numpy()]
    return t.assign(net=N.net(t))


def arm_trades(t, arm, levels=None):
    """Per trade of t (picks, controls or fired, merged with daily and minutes): the arm's net."""
    px, dates, why = [], [], []
    for r in t.to_dict("records"):
        e, s, a = r["entry"], r["side"], r["atr"]
        if arm in HOLDS:
            closes = [r.get(f"c{n_}", np.nan) for n_ in range(0, HOLDS[arm] + 1)]
            ds = [r.get(f"date{n_}", pd.NaT) for n_ in range(0, HOLDS[arm] + 1)]
            closes[0] = r["post_c"] if np.isfinite(r.get("post_c", np.nan)) else closes[0]
            x = hold_exit(e, s, a, closes, ds, HOLDS[arm], HOLD_STOP if arm in ("H5", "H20") else None)
            px.append(x[0] if x else np.nan), dates.append(x[1] if x else pd.NaT), why.append("time")
        elif arm == "DAY":
            sp, tp = e - s * DAY_STOP * a, e + s * DAY_TARGET * a
            ph = r.get("path_h")
            if ph is None or not len(ph) or not a > 0:
                px.append(np.nan), dates.append(r["d"]), why.append("none")
                continue
            p, w = path_exit(e, s, sp, tp, ph, r["path_l"], r["path_c"])
            px.append(p), dates.append(r["d"]), why.append(w)
        elif arm in ("NXT-TR", "NXT-TC"):
            if not a > 0:
                px.append(np.nan), dates.append(pd.NaT), why.append("none")
                continue
            c = max(abs(r["e"]) / r["cut"], 1.0) if r.get("cut", 0) > 0 and np.isfinite(r.get("e", np.nan)) else 1.0
            k = TRAIL_K if arm == "NXT-TR" else TRAIL_K * min(c, TRAIL_C_MAX)
            p, dt, w = trail_exit(e, s, a, k, frame_sessions(r))
            px.append(p), dates.append(dt), why.append(w)
        else:
            k_s, k_t = (DAY_STOP, DAY_TARGET) if levels is None else levels[r["_i"]]
            if not a > 0:
                px.append(np.nan), dates.append(pd.NaT), why.append("none")
                continue
            p, dt, w = frame_exit(e, s, e - s * k_s * a, e + s * k_t * a, frame_sessions(r))
            px.append(p), dates.append(dt), why.append(w)
    px = np.asarray(px, float)
    out = costed(t.assign(why=why), px / t["entry"].to_numpy(float) - 1, dates)
    return out


def exq_fit(train):
    from lightgbm import LGBMRegressor
    X = exq_X(train)
    adv, fav = zip(*[excursions(r["entry"], r["side"], r["atr"], frame_sessions(r)) for r in train.to_dict("records")])
    adv, fav = np.asarray(adv, float), np.asarray(fav, float)
    ok = np.isfinite(adv) & np.isfinite(fav)
    ms = LGBMRegressor(objective="quantile", alpha=Q_STOP, **EXQ_LGBM).fit(X[ok], adv[ok])
    mt = LGBMRegressor(objective="quantile", alpha=Q_TARGET, **EXQ_LGBM).fit(X[ok], fav[ok])
    return ms, mt


def exq_X(t):
    return np.column_stack([t[X3.INPUTS].to_numpy(float), t["side"].to_numpy(float),
                            (t["e"] / t["cut"]).to_numpy(float), (t["atr"] / t["entry"]).to_numpy(float)])


def exq_levels(models, t):
    ms, mt = models
    X = exq_X(t)
    return list(zip(np.clip(ms.predict(X), *Q_CLIP), np.clip(mt.predict(X), *Q_CLIP)))


def summary(rows, lags):
    dd = daily_diff(rows)
    return {"pick_minus_control": tstat(dd.to_numpy(), lags), "trades": int((rows["role"] == "pick").sum()),
            "pick_mean_net": float(rows.loc[rows["role"] == "pick", "net"].mean()),
            "control_mean_net": float(rows.loc[rows["role"] == "control", "net"].mean())}


def paired(a, b, lags, key=("symbol", "d")):
    """Daily mean of a's picks' net minus b's on the same picks."""
    m = a[a["role"] == "pick"].merge(b[b["role"] == "pick"], on=list(key), suffixes=("", "_b"))
    if m.empty:
        return tstat([], lags)
    return tstat((m["net"] - m["net_b"]).groupby(m["d"]).mean().to_numpy(), lags)


def details(rows):
    p = rows[rows["role"] == "pick"]
    if p.empty:
        return {}
    by = lambda m: {str(k): {"n": int(len(g)), "mean_net": float(g["net"].mean())} for k, g in p.groupby(m)}
    return {"worst_1pct_net": float(p["net"].quantile(0.01)),
            "worst_month": float(p.groupby(p["d"].dt.to_period("M"))["net"].mean().min()),
            "exit_reasons": p["why"].value_counts(normalize=True).round(4).to_dict() if "why" in p else {},
            "by_side": by(np.where(p["side"] > 0, "long", "short")),
            "by_price": by(np.where(p["price"] >= 5, "$5+", "under $5")), "by_year": by(p["d"].dt.year)}


def measure_nxe(t, state):
    pc = t[t["role"].isin(["pick", "control"])]
    h1 = pc.assign(net=N.net(pc), why="time")
    h1 = h1[(h1["next"] < HOLDOUT).to_numpy()]
    rep = {"H1 (NX3 itself)": summary(h1, 1)}
    arms = {}
    for arm, lags in [("H2", 2), ("H3", 3), ("H5", 5), ("H20", 20), ("DAY", 1), ("EXA1", FRAME),
                      ("NXT-TR", FRAME), ("NXT-TC", FRAME)]:
        arms[arm] = (arm_trades(pc, arm), lags)
    # EXQ: per test year, models fitted on that year's model's fired training events
    exq, ins, outs = [], [], []
    for yr, g in pc.groupby("year"):
        tr = t[(t["role"] == "fired_train") & (t["year"] == yr)].reset_index(drop=True)
        tr = tr[(tr["atr"] > 0) & tr["post_h"].notna()].reset_index(drop=True)
        te = t[(t["role"] == "fired_test") & (t["year"] == yr) & (t["atr"] > 0) & t["post_h"].notna()].reset_index(drop=True)
        if len(tr) < 500:
            continue
        models = exq_fit(tr)
        g = g[(g["atr"] > 0)].reset_index(drop=True).assign(_i=lambda x: range(len(x)))
        exq.append(arm_trades(g, "EXQ", exq_levels(models, g)))
        for sample, store in ((tr, ins), (te, outs)):
            s_ = sample.assign(_i=range(len(sample)))
            q, a = arm_trades(s_, "EXQ", exq_levels(models, s_)), arm_trades(s_, "EXA1")
            m = q.merge(a, on=["symbol", "d"], suffixes=("", "_a"))
            store.append((m["net"] - m["net_a"]).to_numpy())
    if exq:
        arms["EXQ"] = (pd.concat(exq, ignore_index=True), FRAME)
    for arm, (rows, lags) in arms.items():
        rep[arm] = summary(rows, lags) | {"minus_H1_same_picks": paired(rows, h1, lags)} | details(rows)
        pm = rep[arm]["pick_minus_control"]
        rep[arm]["verdict"] = "passed" if pm["mean"] > 0 and pm["t_nw"] >= PASS_T else "failed"
    if "EXQ" in arms:
        i_m = float(np.nanmean(np.concatenate(ins))) if ins else float("nan")
        o_m = float(np.nanmean(np.concatenate(outs))) if outs else float("nan")
        overfit = not (o_m > 0) or i_m > 2 * o_m
        vs = paired(arms["EXQ"][0], arms["EXA1"][0], FRAME)
        rep["EXQ"] |= {"rule21_insample_minus_EXA1": i_m, "rule21_outsample_minus_EXA1": o_m, "rule21_overfit": overfit,
                       "minus_EXA1_same_picks": vs}
        if overfit or not (vs["mean"] > 0 and vs["t_nw"] >= PASS_T):
            rep["EXQ"]["verdict"] = "failed"
    for arm in ("NXT-TR", "NXT-TC"):    # NXT.md: also reported against EXA1, and TC against TR
        rep[arm]["minus_EXA1_same_picks"] = paired(arms[arm][0], arms["EXA1"][0], FRAME)
    rep["NXT-TC"]["minus_NXT-TR_same_picks"] = paired(arms["NXT-TC"][0], arms["NXT-TR"][0], FRAME)
    return rep


def measure_nxf(state, t, filings):
    real = pd.concat([pd.read_parquet(f) for f in sorted(state.glob("y*_real.parquet"))], ignore_index=True)
    mins = minutes(state)[["symbol", "d", "v_entry"]]
    rv = pd.concat([pd.read_parquet(f) for f in sorted((state / "rv").glob("r_*.parquet"))], ignore_index=True)
    dl = pd.read_parquet(state / "daily.parquet", columns=["symbol", "d", "cik"])
    real = real.merge(mins, on=["symbol", "d"], how="left").merge(rv, on=["symbol", "d"], how="left") \
               .merge(dl, on=["symbol", "d"], how="left")
    fl = {}
    for r in filings.dropna(subset=["filed"]).itertuples(index=False):
        fl.setdefault(int(r.cik), []).append((r.filed, r.accepted_et, set(str(r.items).split(","))))
    cal = pd.DatetimeIndex(sorted(real["d"].unique()))
    nx3 = t[t["role"] == "pick"].assign(net=lambda x: N.net(x))
    rep = {}
    for name, seed in FILTER_SEED.items():
        if name == "NOV":
            ok = real["novelty"] <= NOV_MAX
        elif name == "RV":
            ok = real["v_entry"].fillna(0) >= RV_MULT * real["med20"].fillna(0)
        else:
            ok = pd.Series([not (e > 0 and bad_8k(fl, c, d, tt, cal[max(cal.searchsorted(d) - BAD_8K_SESSIONS, 0):]))
                            for e, c, d, tt in zip(real["e"], real["cik"], real["d"], real["t"])], index=real.index)
        rows = []
        for yr, gy in real.assign(fired=real["fired"] & ok.to_numpy()).groupby("year"):
            rng = np.random.default_rng(X3.SEED + yr + seed)
            for d, day in gy.groupby("d", sort=True):
                p, c = X3.picks(day, rng)
                if p.empty or c.empty:
                    continue
                rows += [p.assign(role="pick"), c.assign(role="control")]
        rows = pd.concat(rows, ignore_index=True)
        rows = rows[(rows["next"] < HOLDOUT).to_numpy()]
        rows = rows.assign(net=N.net(rows), why="time")
        rep[name] = summary(rows, 5) | {"filtered_picks_minus_NX3_picks": tstat(
            (rows[rows["role"] == "pick"].groupby("d")["net"].mean() - nx3.groupby("d")["net"].mean()).dropna().to_numpy(), 5),
            "share_of_fired_kept": float(ok[real["fired"]].mean())} | details(rows)
        pm = rep[name]["pick_minus_control"]
        rep[name]["verdict"] = "passed" if pm["mean"] > 0 and pm["t_nw"] >= PASS_T else "failed"
    return rep


def stage_measure(state, out):
    t = trades(state)
    t = t.merge(pd.read_parquet(state / "daily.parquet"), on=["symbol", "d"], how="left")
    t = t.merge(minutes(state).drop(columns=["v_entry"]), on=["symbol", "d", "entry_t"], how="left")
    pc = t[t["role"].isin(["pick", "control"])]
    # data QA: NX3's cached next-close return vs the daily table's close 1 on day d's basis
    qa = (pc["c1"] / pc["entry"] - 1 - pc["ret"]).abs()
    nxe = measure_nxe(t, state)
    wf = Path("research/results/NX3/NX3_walkforward.json")
    nx3_pooled = json.loads(wf.read_text())["pooled"]["pick_minus_control"] if wf.exists() else None
    nxe["qa"] = {"next_close_matches_cache_share": float((qa < 1e-3).mean()), "trades": int(len(pc)),
                 "with_atr": float((pc["atr"] > 0).mean()), "with_minutes": float(pc["post_h"].notna().mean()),
                 "nx3_walkforward_pick_minus_control": nx3_pooled}       # H1 above must reproduce it
    nxf = measure_nxf(state, t, pd.read_parquet(state / "filings.parquet"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "NXE_result.json").write_text(json.dumps(nxe, indent=1, default=str))
    (out.parent / "NXF").mkdir(exist_ok=True)
    (out.parent / "NXF" / "NXF_result.json").write_text(json.dumps(nxf, indent=1, default=str))
    show = {k: (v.get("verdict"), {kk: round(vv, 5) if isinstance(vv, float) else vv
                                   for kk, vv in v["pick_minus_control"].items()}) for k, v in (nxe | nxf).items()
            if isinstance(v, dict) and "pick_minus_control" in v}
    print(json.dumps(show | {"qa": nxe["qa"]}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "picks", "daily", "minutes", "rv", "filings", "measure"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--state", default="cache/nxe")
    ap.add_argument("--out", default="research/results/NXE")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    state, cal = Path(a.state), X3.calendar(a.cache)
    for s in (["picks", "daily", "minutes", "rv", "filings", "measure"] if a.stage == "all" else [a.stage]):
        if s == "picks":
            if not stage_picks(load_ev(a.cache), cal, state, deadline):
                return print("[nxe] time budget used in picks; rerun to continue")
        elif s == "daily":
            stage_daily(a.cache, cal, state)
        elif s == "minutes" and not stage_minutes(cal, state, deadline):
            return print("[nxe] time budget used in minutes; rerun to continue")
        elif s == "rv" and not stage_rv(cal, state, deadline):
            return print("[nxe] time budget used in rv; rerun to continue")
        elif s == "filings" and not stage_filings(state, deadline):
            return print("[nxe] time budget used in filings; rerun to continue")
        elif s == "measure":
            stage_measure(state, Path(a.out))


if __name__ == "__main__":
    main()
