"""CH (research/prereg/CH.md): triggers, the 68 chart inputs, the six labels and the image windows, built
month by month from the candidate days and their 5-minute bars (`tracks/main/ch_data.py`). The 5-minute bars
of a month are fetched, used and dropped; only these compact rows are kept, under <cache>/ch/rows/<YYYY-MM>.
Each candidate symbol-day gives at most one row: its first trigger (bar close s, 9:45 .. 30 minutes before the
close). Inputs use bars ending at or before s; entry is the open of the bar starting at s + 15 minutes.

Nothing here fits a model or reads a pick's result; labels are written for training and are read only by the
walk-forward in `tracks/main/ch_study.py`.
"""
import numpy as np
import pandas as pd

from adaptive import costs

SLOT = pd.Timedelta(minutes=5)
FIRST_S, OR_END = pd.Timedelta(hours=9, minutes=45), pd.Timedelta(hours=10)
OPEN = pd.Timedelta(hours=9, minutes=30)
LAG = pd.Timedelta(minutes=15)                    # bars end 16 minutes before now: entry at the bar s + 15
RVOL_X, RVOL_SOFT, VWAP_X = 2.0, 1.5, 3.0
FIRST30 = {"main": 0.02, "penny": 0.04}
NEAR52, GAP = 0.02, 0.03
PAIRS = (1.0, 1.5, 2.0)                           # stop = k x ATR(14), target = 2k x ATR(14)
HOLD = 5
NOTIONAL = 250.0
TRIGGERS = ("rvol", "gap", "hl20", "vwap", "orb", "first30", "near52")
LABELS = [f"{side}_{k}" for side in ("long", "short") for k in PAIRS]
DAILY_N = 60                                       # daily bars in the image window


# ---------- small indicator helpers (numpy, oldest first) ----------

def ema(x, n):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    a, out = 2 / (n + 1), np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(x, n=14):
    x = np.asarray(x, float)
    if len(x) <= n:
        return np.nan
    d = np.diff(x)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for u, w in zip(up[n:], dn[n:]):
        au, ad = (au * (n - 1) + u) / n, (ad * (n - 1) + w) / n
    return 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)


def macd_hist(x):
    x = np.asarray(x, float)
    if len(x) < 35:
        return np.nan
    m = ema(x, 12) - ema(x, 26)
    return float((m - ema(m, 9))[-1] / x[-1])


def atr(h, l, c, n=14):
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    if len(c) < n + 1:
        return np.nan
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    return float(tr[-n:].mean())


def adx(h, l, c, n=14):
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    if len(c) < 2 * n + 1:
        return np.nan
    up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    s = lambda v: ema(v, 2 * n - 1)                # Wilder's smoothing (alpha 1/n)
    pdi, ndi = 100 * s(pdm) / np.maximum(s(tr), 1e-12), 100 * s(ndm) / np.maximum(s(tr), 1e-12)
    dx = 100 * abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12)
    return float(s(dx)[-1])


def volume_profile(h, l, c, v, bins=50):
    """Point of control and 70% value area of the daily bars (volume placed at each day's typical price)."""
    tp = (np.asarray(h) + np.asarray(l) + np.asarray(c)) / 3
    lo, hi = np.min(l), np.max(h)
    if not np.isfinite(lo) or hi <= lo:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.bincount(np.clip(np.digitize(tp, edges) - 1, 0, bins - 1), weights=v, minlength=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    poc = int(vol.argmax())
    a, b, tot = poc, poc, vol[poc]
    while tot < 0.7 * vol.sum() and (a > 0 or b < bins - 1):
        left = vol[a - 1] if a > 0 else -1
        right = vol[b + 1] if b < bins - 1 else -1
        if right >= left:
            b += 1
            tot += vol[b]
        else:
            a -= 1
            tot += vol[a]
    return mid[poc], mid[b], mid[a]


def round_level(p):
    step = 1.0 if p < 10 else 5.0 if p < 100 else 10.0
    return round(p / step) * step


# ---------- volume share curve ----------

def share_curve(bars):
    """Median share of the regular session's volume done by each bar close (slot k = bars 0..k), from
    regular-session 5-minute bars of many symbol-days (columns symbol, date, slot, v)."""
    b = bars.sort_values(["symbol", "date", "slot"])
    tot = b.groupby(["symbol", "date"])["v"].transform("sum")
    b = b.assign(cum=b.groupby(["symbol", "date"])["v"].cumsum() / tot)
    return b.groupby("slot")["cum"].median().reindex(range(78)).ffill().fillna(0).to_numpy()


# ---------- one symbol-day ----------

def session(bars, day, close_t):
    """Regular-session bars of one symbol-day, with slot numbers and bar close times."""
    t0 = day + OPEN
    b = bars[(bars["t"] >= t0) & (bars["t"] < day + close_t)].sort_values("t").copy()
    b["slot"] = ((b["t"] - t0) / SLOT).astype(int)
    b["end"] = b["t"] + SLOT
    return b.reset_index(drop=True)


def first_trigger(b, ctx, curve, track, last_s):
    """Index into `b` of the first bar whose close s fires any trigger, and the trigger's name; (None, None)
    when nothing fires. `ctx` holds the prior-day levels (prev_c, vol20, hi20, lo20, hi252, lo252, gap) on the
    day's raw price basis."""
    if b.empty:
        return None, None
    cumv = b["v"].cumsum().to_numpy()
    pv = (b["vw"].fillna(b["c"]) * b["v"]).cumsum().to_numpy()
    vwap = pv / np.maximum(cumv, 1)
    c = b["c"].to_numpy()
    o0 = b["o"].iloc[0]
    or_mask = (b["end"] <= b["t"].iloc[0].normalize() + OR_END).to_numpy()
    or_h = b["h"][or_mask].max() if or_mask.any() else np.nan
    or_l = b["l"][or_mask].min() if or_mask.any() else np.nan
    bar_avg = ctx["vol20"] / 78
    for i in range(len(b)):
        s = b["end"].iloc[i] - b["t"].iloc[i].normalize()
        if s < FIRST_S:
            continue
        if s > last_s:
            break
        slot = int(b["slot"].iloc[i])
        exp = ctx["vol20"] * max(curve[min(slot, 77)], 1e-6)
        rv = cumv[i] / exp if exp > 0 else np.nan
        p = c[i]
        if rv >= RVOL_X:
            return i, "rvol"
        if abs(o0 / ctx["prev_c"] - 1) >= GAP:
            return i, "gap"
        if p > ctx["hi20"] or p < ctx["lo20"]:
            return i, "hl20"
        if i > 0 and np.sign(c[i] - vwap[i]) != np.sign(c[i - 1] - vwap[i - 1]) and b["v"].iloc[i] >= VWAP_X * bar_avg:
            return i, "vwap"
        if s >= OR_END and np.isfinite(or_h) and (p > or_h or p < or_l) and rv >= RVOL_SOFT:
            return i, "orb"
        if s == OR_END and abs(p / ctx["prev_c"] - 1) >= FIRST30[track]:
            return i, "first30"
        if (p >= (1 - NEAR52) * ctx["hi252"] or p <= (1 + NEAR52) * ctx["lo252"]) and rv >= RVOL_SOFT:
            return i, "near52"
    return None, None


def features(b, i, daily, ctx, mkt, curve, trig):
    """The 68 inputs at bar i's close. `daily` is the prior sessions' daily bars on today's raw basis (columns
    h, l, c, v, oldest first, ending yesterday); `mkt` holds spy_ret_today, spy_ret5, spy_above_vwap,
    sector_ret_today, vix, vix_chg5, dow."""
    d = b.iloc[: i + 1]
    c5, h5, l5, v5 = (d[k].to_numpy(float) for k in ("c", "h", "l", "v"))
    p = c5[-1]
    cumv = v5.cumsum()
    vwap_t = (d["vw"].fillna(d["c"]).to_numpy() * v5).cumsum() / np.maximum(cumv, 1)
    vw = vwap_t[-1]
    sd = np.sqrt(np.average((c5 - vw) ** 2, weights=np.maximum(v5, 1))) if len(c5) > 1 else np.nan
    dc, dh, dl, dv = (daily[k].to_numpy(float) for k in ("c", "h", "l", "v"))
    closes = np.append(dc, p)
    A = atr(dh, dl, dc)
    f = {}
    rel = lambda x: p / x - 1 if x and np.isfinite(x) and x > 0 else np.nan
    # trend
    f["vwap_dist"] = rel(vw)
    f["vwap_z"] = (p - vw) / sd if sd and sd > 0 else np.nan
    f["vwap_pctb"] = (p - (vw - 2 * sd)) / (4 * sd) if sd and sd > 0 else np.nan
    f["ema9_5m"], f["ema20_5m"] = rel(ema(c5, 9)[-1]), rel(ema(c5, 20)[-1])
    sma = lambda n: closes[-n:].mean() if len(closes) >= n else np.nan
    f["sma20"], f["sma50"], f["sma200"] = rel(sma(20)), rel(sma(50)), rel(sma(200))
    s50 = pd.Series(dc).rolling(50).mean().to_numpy()
    f["sma50_slope20"] = s50[-1] / s50[-21] - 1 if len(s50) > 70 and s50[-21] > 0 else np.nan
    f["adx14"] = adx(dh, dl, dc)
    # momentum
    f["rsi_5m"], f["rsi_d"] = rsi(c5), rsi(closes)
    f["macdh_5m"], f["macdh_d"] = macd_hist(c5), macd_hist(closes)
    f["ret_30m"] = p / c5[-7] - 1 if len(c5) >= 7 else p / d["o"].iloc[0] - 1
    f["ret_today"] = rel(ctx["prev_c"])
    f["ret_1d"] = dc[-1] / dc[-2] - 1 if len(dc) >= 2 else np.nan
    f["ret_5d"] = dc[-1] / dc[-6] - 1 if len(dc) >= 6 else np.nan
    f["ret_20d"] = dc[-1] / dc[-21] - 1 if len(dc) >= 21 else np.nan
    # volume
    slot = int(d["slot"].iloc[-1])
    exp = ctx["vol20"] * max(curve[min(slot, 77)], 1e-6)
    f["rvol"] = cumv[-1] / exp if exp > 0 else np.nan
    f["lastbar_rel"] = v5[-1] / (ctx["vol20"] / 78) if ctx["vol20"] > 0 else np.nan
    sgn = np.sign(np.diff(dc[-21:])) if len(dc) >= 21 else np.array([])
    obv = np.cumsum(sgn * dv[-20:]) if len(sgn) == 20 else None
    f["obv_slope20"] = np.polyfit(np.arange(20), obv, 1)[0] / max(dv[-20:].mean(), 1) if obv is not None else np.nan
    if len(dc) >= 20:
        mfm = ((dc - dl) - (dh - dc)) / np.maximum(dh - dl, 1e-12)
        f["cmf20"] = (mfm[-20:] * dv[-20:]).sum() / max(dv[-20:].sum(), 1)
        poc, vah, val = volume_profile(dh[-20:], dl[-20:], dc[-20:], dv[-20:])
    else:
        f["cmf20"], poc, vah, val = np.nan, np.nan, np.nan, np.nan
    up = (np.diff(np.append(d["o"].iloc[0], c5)) > 0)
    f["upvol_share"] = v5[up].sum() / max(v5.sum(), 1)
    f["poc_dist"], f["vah_dist"], f["val_dist"] = rel(poc), rel(vah), rel(val)
    f["log_dv20"] = np.log(max(ctx["prev_dv20"], 1))
    f["vol_trend20"] = np.polyfit(np.arange(20), np.log(np.maximum(dv[-20:], 1)), 1)[0] if len(dv) >= 20 else np.nan
    # volatility
    f["atr_pct"] = A / p if np.isfinite(A) else np.nan
    f["atr5m_pct"] = atr(h5, l5, c5) / p if len(c5) > 14 else np.nan
    bb = closes[-20:]
    bm, bs = bb.mean(), bb.std()
    f["bb_width_d"] = 4 * bs / bm if len(bb) == 20 and bm > 0 else np.nan
    f["bb_pctb_d"] = (p - (bm - 2 * bs)) / (4 * bs) if len(bb) == 20 and bs > 0 else np.nan
    b5 = c5[-20:]
    f["bb_pctb_5m"] = (p - (b5.mean() - 2 * b5.std())) / (4 * b5.std()) if len(b5) == 20 and b5.std() > 0 else np.nan
    f["range_atr"] = (h5.max() - l5.min()) / A if np.isfinite(A) and A > 0 else np.nan
    f["rv20"] = np.std(np.diff(np.log(dc[-21:]))) * np.sqrt(252) if len(dc) >= 21 else np.nan
    # levels
    f["d_prev_h"], f["d_prev_l"], f["d_prev_c"] = rel(dh[-1]), rel(dl[-1]), rel(dc[-1])
    or_m = (d["end"] <= d["t"].iloc[0].normalize() + OR_END).to_numpy()
    full_or = d["end"].iloc[-1] - d["t"].iloc[-1].normalize() >= OR_END
    f["d_or_h"] = rel(h5[or_m].max()) if full_or and or_m.any() else np.nan
    f["d_or_l"] = rel(l5[or_m].min()) if full_or and or_m.any() else np.nan
    f["d_52h"], f["d_52l"] = rel(ctx["hi252"]), rel(ctx["lo252"])
    P = (dh[-1] + dl[-1] + dc[-1]) / 3
    piv = np.array([P, 2 * P - dl[-1], 2 * P - dh[-1]])
    f["d_pivot"] = rel(piv[np.abs(piv - p).argmin()])
    if len(dh) >= 60:
        sh, sl = dh[-60:].max(), dl[-60:].min()
        fib = sl + (sh - sl) * np.array([0.382, 0.5, 0.618])
        f["d_fib"] = rel(fib[np.abs(fib - p).argmin()])
    else:
        f["d_fib"] = np.nan
    o0 = d["o"].iloc[0]
    f["gap"] = o0 / ctx["prev_c"] - 1
    g = o0 - ctx["prev_c"]
    filled = (o0 - l5.min()) / g if g > 0 else (h5.max() - o0) / -g if g < 0 else np.nan
    f["gap_filled"] = min(max(filled, 0.0), 1.0) if np.isfinite(filled) else np.nan
    f["open_in_yrange"] = (o0 - dl[-1]) / (dh[-1] - dl[-1]) if dh[-1] > dl[-1] else np.nan
    f["round_dist"] = rel(round_level(p))
    f["days_since_52h"] = float(len(dh[-252:]) - 1 - np.argmax(dh[-252:])) if len(dh) else np.nan
    # candles
    ob, cb = d["o"].to_numpy(float), c5
    f["c_engulf_5m"] = (np.sign(cb[-1] - ob[-1]) if len(cb) >= 2 and np.sign(cb[-1] - ob[-1]) != np.sign(cb[-2] - ob[-2])
                        and abs(cb[-1] - ob[-1]) > abs(cb[-2] - ob[-2]) else 0.0)
    yo = daily["o"].to_numpy(float)[-1] if "o" in daily else dc[-2] if len(dc) >= 2 else np.nan
    body, rng = abs(dc[-1] - yo), dh[-1] - dl[-1]
    lw, uw = min(dc[-1], yo) - dl[-1], dh[-1] - max(dc[-1], yo)
    f["c_hammer_d"] = 1.0 if rng > 0 and lw > 2 * body and uw < body else -1.0 if rng > 0 and uw > 2 * body and lw < body else 0.0
    f["c_doji_d"] = float(rng > 0 and body <= 0.1 * rng)
    f["c_inside_d"] = float(len(dh) >= 2 and dh[-1] < dh[-2] and dl[-1] > dl[-2])
    f["c_outside_d"] = float(len(dh) >= 2 and dh[-1] > dh[-2] and dl[-1] < dl[-2])
    f["c_run3_5m"] = (1.0 if len(cb) >= 3 and (cb[-3:] > ob[-3:]).all() else -1.0 if len(cb) >= 3 and (cb[-3:] < ob[-3:]).all()
                      else 0.0)
    f["c_gapgo"] = float(abs(f["gap"]) >= GAP and (f["gap_filled"] == 0))
    tr = h5.max() - l5.min()
    f["c_wick"] = ((h5.max() - max(o0, p)) - (min(o0, p) - l5.min())) / tr if tr > 0 else 0.0
    # market and context
    for k in ("spy_ret_today", "spy_ret5", "spy_above_vwap", "sector_ret_today", "vix", "vix_chg5", "dow"):
        f[k] = mkt.get(k, np.nan)
    f["minutes"] = (d["end"].iloc[-1] - d["t"].iloc[-1].normalize() - OPEN) / pd.Timedelta(minutes=1)
    f["trigger"] = float(TRIGGERS.index(trig))
    f["log_price"] = np.log(p)
    return f


def cost_frac(track, price, short, days):
    sh = max(1, int(NOTIONAL // price))
    c = costs.fill_cost(track, sh, price, False) + costs.fill_cost(track, sh, price, True)
    if short:
        c += costs.borrow_cost(track, sh * price, days)
    return c / (sh * price)


def labels(after5, entry, A, later, track, day):
    """The six net returns. `after5`: the entry day's 5-minute bars from the entry bar on (o, h, l, c);
    `later`: the next HOLD-1 sessions' daily bars on the entry day's raw basis (date, h, l, c), possibly fewer
    when the series ends. A barrier touched inside a bar counts at the barrier; a bar or day touching both
    counts as the stop. Exit at the last close otherwise."""
    out = {}
    if not np.isfinite(A) or A <= 0 or entry <= 0 or after5.empty:
        return {k: np.nan for k in LABELS}
    path = [(after5["h"].to_numpy(float), after5["l"].to_numpy(float))]
    path += [(np.array([r.h]), np.array([r.l])) for r in later.itertuples()]
    last_c = later["c"].iloc[-1] if len(later) else after5["c"].iloc[-1]
    last_d = later["date"].iloc[-1] if len(later) else day
    for side in ("long", "short"):
        sgn = 1 if side == "long" else -1
        for k in PAIRS:
            stop, tgt = entry - sgn * k * A, entry + sgn * 2 * k * A
            ex, ex_d = None, last_d
            for j, (hh, ll) in enumerate(path):
                hit_s = (ll <= stop) if sgn > 0 else (hh >= stop)
                hit_t = (hh >= tgt) if sgn > 0 else (ll <= tgt)
                if hit_s.any() or hit_t.any():
                    fs = np.argmax(hit_s) if hit_s.any() else 10 ** 9
                    ft = np.argmax(hit_t) if hit_t.any() else 10 ** 9
                    ex = stop if fs <= ft else tgt
                    ex_d = day if j == 0 else later["date"].iloc[j - 1]
                    break
            ex = last_c if ex is None else ex
            days = max(1, (ex_d - day).days)
            out[f"{side}_{k}"] = sgn * (ex / entry - 1) - cost_frac(track, entry, sgn < 0, days)
    return out
