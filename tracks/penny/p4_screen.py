"""P4 historical screen (research/prereg/P4.md): insider cluster buys in penny stocks with a fixed -20%
stop-loss and +40% take-profit, held up to 21 trading days, against a same-day, same-volatility-decile
random penny stock traded with identical rules. Penny track.

Data (all point-in-time):
- Insider purchases: SEC DERA Form 3/4/5 data sets (tracks.main.longterm.sec_data.dera_owner_buys): original
  Forms 4 and 5, non-derivative code P (acquired), reporting owner an officer, director or 10% owner.
  The filing date F is the DERA FILING_DATE (EDGAR keeps the same date for filings accepted until 10pm ET).
- Ticker -> CIK: the DERA ticker map (sec_data.dera_ticker_map / map_ciks), so a stock is in the universe
  only once it is named in a Form 3/4/5 filing by that date.
- Bars: Alpaca SIP daily, adjustment=raw (price band, dollar volume, the reverse-split test) and
  adjustment=all (returns, entry/stop/take-profit/exit path), delisted symbols kept, zero-volume bars
  dropped (core.bars.to_frames), a >15-calendar-day gap ends a series.
- Filings for the exclusions: each candidate issuer's EDGAR submissions record
  (tracks.news.l1_study.submissions_rows: forms, filing dates, 8-K items).

Exclusion coverage (also written into the result JSON and REPORT.md):
- +30% in 5 trading days ending F: applied (all-adjusted closes).
- Top decile of the day's penny universe by largest one-day return over the 21 trading days ending F
  (MAX): applied.
- Reverse split in the 90 days ending F: applied from the bars, not from a filing. There is no SEC form type
  that marks a reverse split (it appears in 8-K item 5.03 / proxy text together with unrelated charter
  changes), so the split is detected where it happens: the raw/all-adjusted close ratio jumps by >= 1.5x
  between consecutive bars (the p26 E2 / A9 rule in tracks/penny/frozen/rules.py without its volume check).
- Offering: any 424B* prospectus filed in the 90 days ending F is applied. "S-1/S-3 effective" is applied
  through the SEC's EFFECT notices in the issuer's submissions record; an EFFECT notice does not name the
  registration form, so it also fires for other registrations that the SEC declares effective (S-4, S-11,
  F-1, F-3, 10-12B, ...). S-8s and automatic shelf registrations get no EFFECT notice; an automatic shelf
  is still caught when its 424B prospectus is filed.
- Delisting notice: 8-K (or 8-K/A) with item 3.01 in the 90 days ending F: applied (submissions items).

Stages run in one invocation and resume from the cache (atomic parquet writes):
  sources  DERA owner purchases and ticker map, Alpaca assets and calendar -> <cache>/sec, <cache>/meta
  bars     per ~500-symbol batch: penny-universe rows with every exit variant's outcome -> <cache>/rows
  filings  per ~500-CIK batch: submissions records of the candidate issuers -> <cache>/filings
  report   signals, exclusions, trades, controls, statistics -> <out>/

Usage: python -m tracks.penny.p4_screen --cache <dir> --out <dir> [--budget-min 300]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats

# ------------------------------------------------------------------ pre-registered constants (P4.md)
SIGNAL_START = pd.Timestamp("2017-01-01")
HOLDOUT = pd.Timestamp("2025-10-01")        # no exit window may reach this day
BARS_START, BARS_END, CAL_END = "2015-11-01", "2025-09-30", "2025-12-31"
DERA_START_YEAR = 2013                      # three prior years of purchases before the first 2017 signal
MIN_PRICE, MAX_PRICE, MIN_DV20 = 1.0, 5.0, 1e6
LISTED = {"NYSE", "NASDAQ", "AMEX"}         # NYSE, Nasdaq, NYSE American; no OTC (and no ARCA/BATS)
GAP_DAYS = 15
WINDOW_DAYS, MIN_OWNERS, MIN_VALUE = 10, 2, 25_000.0
RUNUP_DAYS, RUNUP_MAX = 5, 0.30
MAX_DAYS, MAX_DECILE = 21, 0.9              # top decile: percentile rank > 0.9 within the day's universe
FILING_LOOKBACK = 90
ENTRY_LIMIT = 0.01
HOLD = 21
EXITS = {"base": (-0.20, 0.40), "wide": (-0.25, 0.50), "hold": (None, None)}   # base is the primary
COST, COST_C0 = 0.02, 0.0065
VOL_DAYS = 60
SEED = 20261019
NW_LAGS, BLOCK, N_BOOT = 4, 8, 10_000
ALPHA_LAGS = 3
N_TRIALS = 46                               # P4.md: deflated Sharpe with 46 trials
MAX_STOCK_SHARE = 0.20
MIN_TRADES = 100
COUNTS = {"phase2_count": 31, "penny_count": 5, "phase1_plus_phase2": 46}
BATCH = 500

EXCLUSIONS_APPLIED = {
    "runup_30pct_5d": "applied (all-adjusted closes, 5 trading days ending F)",
    "max_top_decile": "applied (largest one-day all-adjusted return over the 21 trading days ending F, top "
                      "decile of that day's penny universe)",
    "reverse_split_90d": "applied from bars, not from a filing: raw/all-adjusted close ratio jumps >= 1.5x "
                         "between consecutive bars in the 90 days ending F (no SEC form type marks a reverse split)",
    "offering_424b_90d": "applied (any 424B* form in the issuer's submissions record, 90 days ending F)",
    "offering_s1_s3_effective_90d": "approximated: SEC EFFECT notices in the 90 days ending F; the notice does "
                                    "not name the form, so it also fires for S-4/S-11/F-1/F-3/10-12B etc.; S-8 "
                                    "and automatic shelves get no EFFECT notice (their 424B prospectus is caught)",
    "delisting_notice_8k_301_90d": "applied (8-K or 8-K/A with item 3.01 in the submissions record)",
}

ROW_COLS = ["date", "symbol", "close", "dv20", "vol60", "ret5", "max21", "rsplit90", "has_entry", "skip1",
            "entry_day", "exit_day"] + [f"{k}_{v}" for k in EXITS for v in ("gross", "why", "xday")]


def atomic_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ insider signal (pure)

def routine_keys(buys):
    """{(cik, owner, year, month): earliest filing date} over all purchases; the month is the trade date's
    (the filing date's when the trade date is missing)."""
    t = buys["tdate"].fillna(buys["filed"])
    k = pd.DataFrame({"cik": buys["cik"].to_numpy(), "owner": buys["owner"].to_numpy(), "y": t.dt.year.to_numpy(),
                      "m": t.dt.month.to_numpy(), "filed": buys["filed"].to_numpy()})
    k = k.groupby(["cik", "owner", "y", "m"])["filed"].min()
    return dict(zip(k.index, k.to_numpy()))


def is_opportunistic(cik, owner, tdate, asof, keys):
    """P4.md signal rule 3 (Cohen, Malloy & Pomorski 2012) as registered: the purchase is opportunistic
    unless the same owner has a code-P purchase of the same issuer in the same calendar month in EACH of the
    three prior years (only purchases filed by `asof` count)."""
    t = pd.Timestamp(tdate)
    for k in (1, 2, 3):
        f = keys.get((int(cik), int(owner), t.year - k, t.month))
        if f is None or pd.Timestamp(f) > pd.Timestamp(asof):
            return True
    return False


def clusters(buys, start=SIGNAL_START, end=None, window=WINDOW_DAYS):
    """One row per (cik, F) where F is a filing date of at least one purchase, F >= start: the purchases
    filed in the `window` calendar days ending F (F-window+1 .. F): number of distinct owners, total value
    (each transaction once), number of distinct opportunistic owners."""
    cols = ["cik", "date", "n_owners", "value", "n_opp"]
    if not len(buys):
        return pd.DataFrame(columns=cols)
    b = buys.copy()
    b["filed"] = pd.to_datetime(b["filed"]).astype("datetime64[ns]")
    b["tdate"] = pd.to_datetime(b["tdate"]).astype("datetime64[ns]")
    keys = routine_keys(b)
    out = []
    for cik, g in b.sort_values("filed").groupby("cik", sort=True):
        f = g["filed"].to_numpy()
        for F in np.unique(f):
            F = pd.Timestamp(F)
            if F < start or (end is not None and F > end):
                continue
            lo = np.searchsorted(f, np.datetime64(F - pd.Timedelta(days=window - 1)), side="left")
            hi = np.searchsorted(f, np.datetime64(F), side="right")
            w = g.iloc[lo:hi]
            opp = {o for o, td, fd in zip(w["owner"], w["tdate"], w["filed"])
                   if is_opportunistic(cik, o, fd if pd.isna(td) else td, F, keys)}
            out.append((int(cik), F, int(w["owner"].nunique()),
                        float(w.drop_duplicates(["accn", "sk"])["value"].sum()), len(opp)))
    return pd.DataFrame(out, columns=cols)


def qualifies(c, min_owners=MIN_OWNERS, exact=False):
    """Signal rules 1-3 on clusters() rows; exact=True asks for exactly `min_owners` owners."""
    n = (c["n_owners"] == min_owners) if exact else (c["n_owners"] >= min_owners)
    return n & (c["value"] >= MIN_VALUE) & (c["n_opp"] >= 1)


# ------------------------------------------------------------------ trade path (pure)

def simulate(o, h, l, c, stop, tp):
    """Long trade entered at o[0] (the entry day's open). Each day's low and high are checked against the
    stop and take-profit; from the second day an open beyond a level fills at that open; a day touching
    both counts the stop first. Without a hit, the exit is the last close given. Returns (gross, reason,
    index of the exit bar). Reasons: stop, stop_gap, tp, tp_gap, end."""
    p0 = float(o[0])
    S = p0 * (1 + stop) if stop is not None else None
    T = p0 * (1 + tp) if tp is not None else None
    for i in range(len(o)):
        if i > 0:
            if S is not None and o[i] <= S:
                return o[i] / p0 - 1, "stop_gap", i
            if T is not None and o[i] >= T:
                return o[i] / p0 - 1, "tp_gap", i
        if S is not None and l[i] <= S:
            return S / p0 - 1, "stop", i
        if T is not None and h[i] >= T:
            return T / p0 - 1, "tp", i
    return c[-1] / p0 - 1, "end", len(o) - 1


def reverse_split_days(raw, adj, ratio=1.5):
    """Dates where the raw / all-adjusted close ratio rose by >= `ratio` from the previous common bar."""
    idx = raw.index.intersection(adj.index)
    f = raw.loc[idx, "c"] / adj.loc[idx, "c"]
    k = f / f.shift(1)
    return idx[(k >= ratio).to_numpy()]


def symbol_rows(sym, raw, adj, cal, start=SIGNAL_START, holdout=HOLDOUT):
    """Penny-universe rows of one symbol (ROW_COLS) with each exit variant's outcome. raw/adj: frames
    indexed by date with o, h, l, c, v (zero-volume bars dropped); cal: DatetimeIndex of trading days.
    A row's day F is in the universe when 1 <= raw close < 5, the median 20-bar raw dollar volume >= $1M and
    60 bars of volatility exist, all inside one series (a >15-day gap ends it). The entry day E is the
    next trading day; the exit window is E .. the 21st trading day after E, and it must end before the
    holdout. Prices on the path are all-adjusted."""
    if raw is None or adj is None:
        return pd.DataFrame(columns=ROW_COLS)
    rs = reverse_split_days(raw, adj)
    idx = raw.index.intersection(adj.index)
    if len(idx) < VOL_DAYS + 1:
        return pd.DataFrame(columns=ROW_COLS)
    raw, adj = raw.loc[idx], adj.loc[idx]
    seg = (idx.to_series().diff().dt.days > GAP_DAYS).cumsum().to_numpy()
    out = []
    for s in np.unique(seg):
        r, a = raw[seg == s], adj[seg == s]
        if len(r) < VOL_DAYS + 1:
            continue
        d = pd.DataFrame(index=r.index)
        d["close"] = r["c"]
        d["dv20"] = (r["c"] * r["v"]).rolling(20).median()
        lr = np.log(a["c"]).diff()
        d["vol60"] = lr.rolling(VOL_DAYS).std()
        d["ret5"] = a["c"] / a["c"].shift(RUNUP_DAYS) - 1
        d["max21"] = a["c"].pct_change().rolling(MAX_DAYS).max()
        ok = ((d["close"] >= MIN_PRICE) & (d["close"] < MAX_PRICE) & (d["dv20"] >= MIN_DV20) & d["vol60"].notna()
              & (d.index >= start)).to_numpy()
        dates = r.index
        ao, ah, al, ac = (a[k].to_numpy(dtype=float) for k in ("o", "h", "l", "c"))
        for j in np.flatnonzero(ok):
            F = dates[j]
            p = cal.searchsorted(F)
            if p >= len(cal) or cal[p] != F or p + 1 + HOLD >= len(cal):
                continue
            E, X = cal[p + 1], cal[p + 1 + HOLD]
            if X >= holdout:
                continue
            row = {"date": F, "symbol": sym, "close": float(d["close"].iat[j]), "dv20": float(d["dv20"].iat[j]),
                   "vol60": float(d["vol60"].iat[j]), "ret5": float(d["ret5"].iat[j]),
                   "max21": float(d["max21"].iat[j]),
                   "rsplit90": bool(((rs > F - pd.Timedelta(days=FILING_LOOKBACK)) & (rs <= F)).any()),
                   "has_entry": False, "skip1": False, "entry_day": E, "exit_day": X}
            for k in EXITS:
                row[f"{k}_gross"], row[f"{k}_why"], row[f"{k}_xday"] = np.nan, None, pd.NaT
            if j + 1 < len(dates) and dates[j + 1] == E:
                row["has_entry"] = True
                if ao[j + 1] > ac[j] * (1 + ENTRY_LIMIT):
                    row["skip1"] = True
                else:
                    hi = np.searchsorted(dates, X, side="right")
                    sl = slice(j + 1, hi)
                    ended = dates[hi - 1] != X
                    for k, (stp, tp) in EXITS.items():
                        g, why, i = simulate(ao[sl], ah[sl], al[sl], ac[sl], stp, tp)
                        if why == "end":
                            why = "last_close" if ended else "day21"
                        row[f"{k}_gross"], row[f"{k}_why"], row[f"{k}_xday"] = float(g), why, dates[j + 1 + i]
            out.append(row)
    return pd.DataFrame(out, columns=ROW_COLS)


def car_for(adj, spy_ret, entry, n_days=HOLD + 1):
    """Market-model CAR over the entry day and the 21 trading days after it, estimation days -250..-30
    (L1's layout), on all-adjusted close-to-close returns. (car, scar, last 100 estimation residuals) or
    None."""
    r = adj["c"].pct_change().reindex(spy_ret.index)
    if entry not in r.index:
        return None
    i = r.index.get_loc(entry)
    if i < 251:
        return None
    est, win = slice(i - 250, i - 29), slice(i, i + n_days)
    ra, ma = r.to_numpy(), spy_ret.to_numpy()
    if np.isfinite(ra[est]).sum() < 100:
        return None
    car, scar, res = stats.market_model_car(ra, ma, est, win)
    return car, scar, (list(res[-100:]) if len(res) >= 100 else None)


# ------------------------------------------------------------------ universe, exclusions, trades (pure)

def universe(rows):
    """One row per (date, company): the listing with the highest dollar volume; adds the day's
    volatility decile (1..10) and MAX percentile rank."""
    u = rows.dropna(subset=["cik"]).sort_values(["date", "cik", "dv20", "symbol"], ascending=[True, True, False, True])
    u = u.drop_duplicates(["date", "cik"]).reset_index(drop=True)
    g = u.groupby("date")
    u["vol_dec"] = np.ceil(g["vol60"].rank(pct=True, method="first") * 10).clip(1, 10).astype(int)
    u["max_pct"] = g["max21"].rank(pct=True, method="average")
    return u


def filing_flags(sig, filings, lookback=FILING_LOOKBACK):
    """For each signal row (cik, date): offering (424B* or EFFECT) and delisting notice (8-K item 3.01)
    filed in the `lookback` calendar days ending the date."""
    off = np.zeros(len(sig), bool)
    dl = np.zeros(len(sig), bool)
    if len(filings):
        f = filings.assign(filed=pd.to_datetime(filings["filed"]))
        form = f["form"].astype(str).str.upper().str.strip()
        items = f["items"].astype(str)
        f["offer"] = form.str.startswith("424B") | (form == "EFFECT")
        f["delist"] = form.isin(["8-K", "8-K/A"]) & items.str.contains(r"(?:^|[,\s\[\]'\"])3\.01", regex=True)
        by = {c: g for c, g in f.groupby("cik")}
        for i, (c, F) in enumerate(zip(sig["cik"].to_numpy(), pd.to_datetime(sig["date"]))):
            g = by.get(int(c))
            if g is None:
                continue
            w = g[(g["filed"] > F - pd.Timedelta(days=lookback)) & (g["filed"] <= F)]
            off[i], dl[i] = bool(w["offer"].any()), bool(w["delist"].any())
    return off, dl


def exclusion_reason(s):
    """First exclusion that applies to each signal row, in P4.md's order, else None."""
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    for name, m in (("runup_30pct_5d", s["ret5"] > RUNUP_MAX), ("max_top_decile", s["max_pct"] > MAX_DECILE),
                    ("reverse_split_90d", s["rsplit90"].astype(bool)), ("offering_90d", s["offer90"].astype(bool)),
                    ("delisting_notice_90d", s["delist90"].astype(bool))):
        out[out.isna() & m.to_numpy()] = name
    return out


def take_trades(sig, variant="base"):
    """Signals that pass the exclusions, have an entry bar, pass the 1% entry limit, and come after the
    company's previous trade (in this variant) has closed. Returns (trades, funnel)."""
    s = sig.sort_values(["date", "cik"]).reset_index(drop=True)
    fun = {"signals": int(len(s))}
    s = s.assign(excl=exclusion_reason(s))
    for k, v in s["excl"].value_counts().items():
        fun[f"excluded_{k}"] = int(v)
    s = s[s["excl"].isna()]
    fun["no_entry_bar"] = int((~s["has_entry"]).sum())
    fun["skipped_entry_above_1pct"] = int((s["has_entry"] & s["skip1"]).sum())
    s = s[s["has_entry"] & ~s["skip1"] & s[f"{variant}_gross"].notna()]
    keep, last_exit = [], {}
    for i, (c, F, x) in enumerate(zip(s["cik"], s["date"], s[f"{variant}_xday"])):
        prev = last_exit.get(c)
        if prev is not None and F < prev:
            continue
        keep.append(i)
        last_exit[c] = x
    fun["skipped_open_trade"] = int(len(s) - len(keep))
    t = s.iloc[keep].reset_index(drop=True)
    fun["trades"] = int(len(t))
    return t, fun


def pick_controls(trades, uni, signal_ciks, seed=SEED):
    """For each trade, one company from the same day's universe in the same volatility decile that has an
    entry fill (bar on the entry day, within the 1% limit), no qualifying signal that day, and is not
    already a control that day; drawn with default_rng(seed + date ordinal) over candidates sorted by
    symbol, trades taken in symbol order. Returns the control rows aligned with `trades` (NaN where none)."""
    t = trades.reset_index(drop=True)
    out = [None] * len(t)
    cand = uni[uni["has_entry"] & ~uni["skip1"] & uni["base_gross"].notna()]
    by_day = {d: g.sort_values("symbol") for d, g in cand.groupby("date")}
    for d, g in t.groupby("date", sort=True):
        pool = by_day.get(d)
        if pool is None:
            continue
        pool = pool[~pool["cik"].isin(signal_ciks.get(d, set()))]
        rng = np.random.default_rng(seed + pd.Timestamp(d).toordinal())
        used = set()
        for i in g.sort_values("symbol").index:
            p = pool[(pool["vol_dec"] == t.at[i, "vol_dec"]) & ~pool["cik"].isin(used)]
            if not len(p):
                continue
            k = int(rng.integers(0, len(p)))
            out[i] = p.iloc[k]
            used.add(p.iloc[k]["cik"])
    cols = list(uni.columns)
    return pd.DataFrame([r.to_dict() if r is not None else {c: np.nan for c in cols} for r in out], columns=cols)


def paired(trades, ctrl, variant="base"):
    """Trade and control returns side by side (rows with a control only)."""
    p = pd.DataFrame({"date": trades["date"].to_numpy(), "entry_day": pd.to_datetime(trades["entry_day"]).to_numpy(),
                      "symbol": trades["symbol"].to_numpy(), "cik": trades["cik"].to_numpy(),
                      "gross": trades[f"{variant}_gross"].to_numpy(dtype=float),
                      "why": trades[f"{variant}_why"].to_numpy(),
                      "xday": pd.to_datetime(trades[f"{variant}_xday"]).to_numpy(),
                      "exit_day": pd.to_datetime(trades["exit_day"]).to_numpy(),
                      "c_symbol": ctrl["symbol"].to_numpy(), "c_gross": ctrl[f"{variant}_gross"].to_numpy(dtype=float),
                      "c_why": ctrl[f"{variant}_why"].to_numpy(),
                      "c_exit_day": pd.to_datetime(ctrl["exit_day"]).to_numpy()})
    for k in ("car", "scar", "car_res"):
        p[k] = trades[k].to_numpy() if k in trades else np.nan
    return p[p["c_gross"].notna()].reset_index(drop=True)


def holdout_guard(p, holdout=HOLDOUT):
    """Raise when any trade's or control's exit window reaches the holdout."""
    bad = (p["exit_day"] >= holdout) | (p["c_exit_day"] >= holdout) | (p["xday"] >= holdout)
    if bool(bad.any()):
        raise AssertionError(f"{int(bad.sum())} exit windows reach the holdout {holdout:%Y-%m-%d}")


def weekly(p, cost=COST):
    """Weekly series by entry week (W-FRI): mean of (trade net - control net); both pay `cost`."""
    d = p.assign(net=p["gross"] - cost, c_net=p["c_gross"] - cost)
    d["diff"] = d["net"] - d["c_net"]
    d["week"] = pd.to_datetime(d["entry_day"]).dt.to_period("W-FRI")
    w = d.groupby("week").agg(diff=("diff", "mean"), n=("diff", "size"), net=("net", "mean"), c_net=("c_net", "mean"))
    return w.sort_index()


def primary_stats(w, seed=SEED):
    x = w["diff"].to_numpy(dtype=float)
    m, se, t = _nw(x)
    lo, hi = _block_ci(x, seed)
    return {"weeks": int(len(x)), "mean": m, "se_nw": se, "t_hac": t, "ci95": [lo, hi]}


def _nw(x):
    from tracks.main.longterm.stats import nw_tstat
    return nw_tstat(x, NW_LAGS)


def _block_ci(x, seed):
    from tracks.main.longterm.stats import block_bootstrap_ci
    if len(x) < BLOCK:
        return float("nan"), float("nan")
    return block_bootstrap_ci(x, BLOCK, N_BOOT, seed)


def monthly_alpha(p, fac, rf, cost=COST):
    """Rule 11 calendar-time alpha: per entry month, the mean over its trades of (net return - risk-free
    over the holding days) on the mean of the FF5 + momentum factors summed over each trade's holding days
    (entry day .. exit day), Newey-West with 3 lags."""
    d = p.reset_index(drop=True)
    days = fac.index
    wins = [days[(days >= e) & (days <= x)] for e, x in zip(pd.to_datetime(d["entry_day"]), pd.to_datetime(d["xday"]))]
    F = stats.window_factors(fac, wins)
    R = np.array([rf.reindex(w).sum() for w in wins])
    y = d["gross"].to_numpy(dtype=float) - cost - R
    m = pd.to_datetime(d["entry_day"]).dt.to_period("M").to_numpy()
    Y = pd.Series(y).groupby(m).mean()
    X = pd.DataFrame(F, columns=list(fac.columns)).groupby(m).mean().reindex(Y.index)
    out = stats.ols_hac(Y.to_numpy(), X.to_numpy(), ALPHA_LAGS)
    out["factors"] = list(fac.columns) + ["(summed over each trade's holding days, averaged per entry month)"]
    out["months"] = int(len(Y))
    return out


def stock_share(p, cost=COST):
    """Largest single company's share of the trades' total net profit (inf when the total is <= 0)."""
    net = (p["gross"] - cost).groupby(p["cik"]).sum()
    tot = float(net.sum())
    if tot <= 0:
        return float("inf"), None
    return float(net.max() / tot), int(net.idxmax())


def verdict(prim, alpha, share, n_trades):
    if n_trades < MIN_TRADES:
        return "INCONCLUSIVE", False
    ok = stats.screen_pass(prim, alpha) and share <= MAX_STOCK_SHARE
    return ("PASSED" if ok else "FAILED"), bool(ok)


def hit_rates(p):
    why = p["why"].map(lambda w: {"stop_gap": "stop", "tp_gap": "tp"}.get(w, w))
    cw = p["c_why"].map(lambda w: {"stop_gap": "stop", "tp_gap": "tp"}.get(w, w))
    return {"trades": {k: float(v) for k, v in why.value_counts(normalize=True).items()},
            "controls": {k: float(v) for k, v in cw.value_counts(normalize=True).items()},
            "gap_fills_trades": int(p["why"].isin(["stop_gap", "tp_gap"]).sum())}


def summary(p, cost=COST, seed=SEED):
    if not len(p):
        return {"trades": 0}
    w = weekly(p, cost)
    r = primary_stats(w, seed) if len(w) > NW_LAGS + 1 else {"weeks": int(len(w))}
    r.update({"trades": int(len(p)), "trade_net_mean": float((p["gross"] - cost).mean()),
              "control_net_mean": float((p["c_gross"] - cost).mean())})
    return r


# ------------------------------------------------------------------ stages (network)

def stage_sources(cache):
    from core import alpaca
    from tracks.main.longterm import bars_batch, sec_data
    meta = cache / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    cf, af = meta / "calendar.parquet", meta / "assets.parquet"
    if not cf.exists():
        atomic_parquet(alpaca.trading_calendar(BARS_START, CAL_END)[["date"]], cf)
    if not af.exists():
        a = bars_batch.assets_all()
        atomic_parquet(a[["symbol", "base_symbol", "exchange", "excl_reason"]].copy(), af)
    end_q = pd.Period(BARS_END, "Q")
    tmap = sec_data.dera_ticker_map(DERA_START_YEAR, end_q, cache / "sec")
    buys = sec_data.dera_owner_buys(DERA_START_YEAR, end_q, cache / "sec")
    return pd.DatetimeIndex(pd.read_parquet(cf)["date"]), pd.read_parquet(af), tmap, buys


def stage_bars(cache, syms, cal, tmap, cand, deadline):
    """Per 500-symbol batch: universe rows (with CIK and, on candidate (cik, date) rows, the market-model
    CAR inputs). Returns all rows, or None when the budget ran out first."""
    from core import alpaca
    from core.bars import fetch_daily_bars, to_frames
    from core.env import alpaca_headers
    from tracks.main.longterm import sec_data
    h = alpaca_headers()
    end = alpaca.sip_end(BARS_END)
    spy_f = cache / "meta" / "spy.parquet"
    if not spy_f.exists():
        atomic_parquet(to_frames(fetch_daily_bars(["SPY"], BARS_START, end, adjustment="all", headers=h))["SPY"], spy_f)
    spy_ret = pd.read_parquet(spy_f)["c"].pct_change()
    parts = []
    n = -(-len(syms) // BATCH)
    for b in range(0, len(syms), BATCH):
        f = cache / "rows" / f"rows_{b // BATCH:04d}.parquet"
        if not f.exists():
            if time.monotonic() > deadline:
                return None
            batch = syms[b:b + BATCH]
            raw = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="raw", headers=h))
            adj = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="all", headers=h))
            rows = [symbol_rows(s, raw.get(s), adj.get(s), cal) for s in batch]
            rows = [r for r in rows if len(r)]
            d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ROW_COLS)
            d["cik"] = sec_data.map_ciks(d[["symbol", "date"]], tmap) if len(d) else np.array([], float)
            car = np.full(len(d), np.nan)
            scar = np.full(len(d), np.nan)
            res = [None] * len(d)
            for i in np.flatnonzero([(c, t) in cand for c, t in zip(d["cik"].fillna(-1).astype("int64"), d["date"])]):
                x = car_for(adj[d.at[i, "symbol"]], spy_ret, d.at[i, "entry_day"])
                if x is not None:
                    car[i], scar[i], res[i] = x
            d["car"], d["scar"], d["car_res"] = car, scar, res
            atomic_parquet(d, f)
        parts.append(pd.read_parquet(f))
        print(f"[p4] bars batch {b // BATCH + 1}/{n}")
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def stage_filings(cache, ciks, deadline):
    from tracks.news.l1_study import submissions_rows
    ciks = sorted({int(c) for c in ciks})
    parts = []
    first, last = pd.Timestamp(BARS_START), pd.Timestamp(BARS_END)
    for b in range(0, len(ciks), BATCH):
        f = cache / "filings" / f"filings_{b // BATCH:04d}.parquet"
        lst = cache / "filings" / f"filings_{b // BATCH:04d}.json"
        if not f.exists() or not lst.exists() or json.loads(lst.read_text()) != ciks[b:b + BATCH]:
            if time.monotonic() > deadline:
                return None
            rows = [r for c in ciks[b:b + BATCH] for r in submissions_rows(c, first=first, last=last)]
            d = pd.DataFrame(rows, columns=["cik", "accession", "form", "filed", "items"])
            d["items"] = d["items"].map(lambda x: ",".join(x) if isinstance(x, list) else "")
            atomic_parquet(d, f)
            tmp = lst.with_suffix(".tmp")
            tmp.write_text(json.dumps(ciks[b:b + BATCH]))
            tmp.replace(lst)
        parts.append(pd.read_parquet(f))
        print(f"[p4] filings batch {b // BATCH + 1}/{-(-len(ciks) // BATCH)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cik", "accession", "form", "filed", "items"])


def analyse(uni, cl, filings, fac, rf):
    """Everything after the data: signals, trades, controls, primary test, exploratory. Pure given inputs."""
    cl = cl.assign(date=pd.to_datetime(cl["date"]))
    sig_all = uni.merge(cl, on=["cik", "date"], how="inner")
    off, dl = filing_flags(sig_all, filings)
    sig_all = sig_all.assign(offer90=off, delist90=dl)
    q = qualifies(sig_all)
    signal_ciks = {d: set(g["cik"]) for d, g in sig_all[q].groupby("date")}
    res = {}
    trades, fun = take_trades(sig_all[q], "base")
    fun = {"universe_rows": int(len(uni)), "universe_days": int(uni["date"].nunique()),
           "universe_median_per_day": float(uni.groupby("date").size().median()) if len(uni) else 0.0,
           "cluster_rows_in_universe": int(len(sig_all)), **fun}
    ctrl = pick_controls(trades, uni, signal_ciks)
    p = paired(trades, ctrl, "base")
    fun["trades_without_control"] = int(len(trades) - len(p))
    holdout_guard(p)
    w = weekly(p)
    prim = summary(p)
    a = monthly_alpha(p, fac, rf) if len(p) > 10 else {"alpha": float("nan"), "t": float("nan")}
    share, top = stock_share(p) if len(p) else (float("inf"), None)
    prim.update({"factor_alpha": a, "max_stock_share_of_net_profit": share, "max_share_cik": top,
                 "deflated_sharpe": stats.deflated_sharpe(w["diff"].to_numpy(), N_TRIALS) if len(w) > 2 else None,
                 "spread_c0_reported_beside": COST_C0})
    if "mean" not in prim:
        prim.update({"mean": float("nan"), "t_hac": float("nan"), "ci95": [float("nan")] * 2})
    v, ok = verdict(prim, a, share, len(p))
    prim["verdict"], prim["passes"] = v, ok
    res["PRIMARY"], res["funnel"] = prim, fun

    ex = {"label": "EXPLORATORY (pre-listed in P4.md; none is a finding)"}
    ex["hit_rates"] = hit_rates(p) if len(p) else {}
    for var in ("hold", "wide"):
        tv, fv = take_trades(sig_all[q], var)
        pv = paired(tv, pick_controls(tv, uni, signal_ciks), var)
        holdout_guard(pv)
        ex[{"hold": "hold21_no_stop_no_tp", "wide": "stop25_tp50"}[var]] = {**summary(pv), "funnel": fv}
    q1 = qualifies(sig_all, 1, exact=True)
    t1, f1 = take_trades(sig_all[q1], "base")
    p1 = paired(t1, pick_controls(t1, uni, {d: set(g["cik"]) for d, g in sig_all[q1 | q].groupby("date")}), "base")
    holdout_guard(p1)
    ex["one_insider"] = {**summary(p1), "funnel": f1}
    if len(p):
        d = p.assign(diff=p["gross"] - p["c_gross"], net=p["gross"] - COST)
        yr = d.groupby(pd.to_datetime(d["entry_day"]).dt.year)
        ex["by_year"] = {int(y): {"n": int(len(g)), "diff_mean": float(g["diff"].mean()),
                                  "net_mean": float(g["net"].mean())} for y, g in yr}
        ex["median_trade_net"] = float(d["net"].median())
        ex["median_diff"] = float(d["diff"].median())
        ex["without_best_trade"] = summary(p.drop(index=d["diff"].idxmax()))
        ex["cost_0_65pct"] = {**summary(p, COST_C0),
                              "note": "the difference series is the same (trade and control pay the same cost)"}
        sc = p[np.isfinite(p["scar"].to_numpy(dtype=float))]
        rs = [np.asarray(r, float) for r in sc["car_res"] if r is not None and len(r) == 100]
        if len(sc) > 2:
            ex["car_market_model_kp"] = {**stats.bmp_kp(sc["scar"].to_numpy(dtype=float),
                                                        stats.mean_residual_corr(rs) if len(rs) > 1 else 0.0),
                                         "car_mean": float(sc["car"].mean()),
                                         "window": "entry day .. 21st trading day after, close to close; est -250..-30"}
    res["EXPLORATORY"] = ex
    return res, p, w


def write_report(out, res):
    pr = res["PRIMARY"]
    fun = res["funnel"]
    a = pr.get("factor_alpha", {})
    lines = [
        "# P4 report: insider cluster buys in penny stocks, stop -20% / take-profit +40% (penny track)", "",
        f"Pre-registration: `research/prereg/P4.md`. Phase-2 count {COUNTS['phase2_count']} (penny "
        f"{COUNTS['penny_count']}); phase 1 + phase 2 = {COUNTS['phase1_plus_phase2']}.", "",
        f"## Verdict: {pr['verdict']}", "",
        "| | Mean weekly (trade - control), net 2.0% | Newey-West t (4 lags) | 95% block CI (8 weeks) | Weeks | Trades |",
        "|---|---|---|---|---|---|",
        f"| **Primary** | {pr['mean']:.4f} | {pr['t_hac']:.2f} | {pr['ci95'][0]:.4f} .. {pr['ci95'][1]:.4f} | "
        f"{pr.get('weeks', 0)} | {pr.get('trades', 0)} |", "",
        f"FF5 + momentum monthly alpha: {a.get('alpha', float('nan')):.4f} (t {a.get('t', float('nan')):.2f}). "
        f"Largest single stock's share of net profit: {pr['max_stock_share_of_net_profit']:.2f}. "
        f"Deflated Sharpe (46 trials): {json.dumps(pr.get('deflated_sharpe'))}. Measured penny spread (C0) "
        f"{COST_C0:.2%} beside the 2.0% cost.", "",
        "Pass rule: mean > 0, t >= 3.0, CI lower bound > 0, alpha > 0 with t >= 2.0, no stock > 20% of net "
        "profit; fewer than 100 trades is inconclusive.", "",
        "## Funnel", "", "```", json.dumps(fun, indent=1), "```", "",
        "## Exclusions: what could be applied from our data", "",
    ] + [f"- `{k}`: {v}" for k, v in EXCLUSIONS_APPLIED.items()] + [
        "",
        "## Implementation notes", "",
        "- Opportunistic (rule 3) is coded as registered: a buyer is opportunistic unless that owner bought the "
        "same issuer in the same calendar month (trade date) in each of the three prior years. Cohen, Malloy & "
        "Pomorski also leave owners without three years of history unclassified; P4.md does not, so they count "
        "as opportunistic here.",
        "- Owners are distinct reporting-owner CIKs on original Forms 4 and 5 (amendments ignored); a joint "
        "filing's owners count separately, and each transaction's value is counted once.",
        "- Entry limit compared on all-adjusted prices (equal to raw except across a split or ex-dividend day "
        "between F and the entry day). Paths use all-adjusted open/high/low/close.",
        "- Controls are drawn among same-day, same-decile companies whose entry would fill, so every trade with "
        "a candidate gets a traded control. Weeks with no trade are not in the series.",
        "- Exchange (no OTC) comes from Alpaca's current asset record, as in P3.", "",
        "## Exploratory (pre-listed; none is a finding)", "", "```",
        json.dumps(res["EXPLORATORY"], indent=1, default=str), "```", "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(cache, out, budget_min=300):
    cache, out = Path(cache), Path(out)
    deadline = time.monotonic() + budget_min * 60
    cal, assets, tmap, buys = stage_sources(cache)
    cl = clusters(buys)
    cand = set(zip(cl["cik"].astype("int64"), pd.to_datetime(cl["date"])))
    common = assets[assets["excl_reason"].isna() & assets["exchange"].isin(LISTED)]
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    print(f"[p4] {len(syms)} listed common symbols named in Form 3/4/5; {len(cl)} (cik, filing date) rows")
    rows = stage_bars(cache, syms, cal, tmap, cand, deadline)
    if rows is None:
        print("BUDGET USED: rerun the workflow to continue (cached batches are kept)")
        return None
    uni = universe(rows.astype({"date": "datetime64[ns]"}))
    sig = uni.merge(cl.assign(date=pd.to_datetime(cl["date"])), on=["cik", "date"])
    need = sig.loc[(sig["value"] >= MIN_VALUE) & (sig["n_opp"] >= 1), "cik"].unique()
    filings = stage_filings(cache, need, deadline)
    if filings is None:
        print("BUDGET USED: rerun the workflow to continue (cached batches are kept)")
        return None
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fac = ffd.join(umdd, how="inner")
    rf = fac["RF"]
    fac = fac[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]]
    res, p, w = analyse(uni, cl, filings, fac, rf)
    res = {"idea": "P4", "prereg": "research/prereg/P4.md", **COUNTS, "cost_round_trip": COST,
           "seed": SEED, "holdout": f"{HOLDOUT:%Y-%m-%d}", "ff5_daily_sha256": sha5, "umd_daily_sha256": shau,
           "exclusions_applied": EXCLUSIONS_APPLIED, **res}
    out.mkdir(parents=True, exist_ok=True)
    (out / "P4_result.json").write_text(json.dumps(res, indent=1, default=str))
    p.to_csv(out / "P4_trades.csv", index=False)
    w.reset_index().astype({"week": str}).to_csv(out / "P4_weekly.csv", index=False)
    chk = p.sample(n=min(20, len(p)), random_state=SEED) if len(p) else p
    chk.assign(checked_by="", ok="", note="").to_csv(out / "P4_rows_check.csv", index=False)
    write_report(out, res)
    print(json.dumps(res["PRIMARY"], indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
