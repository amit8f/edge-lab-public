"""S1: the 5-minute VWAP snap-back scalp from Alex Carter's video, historical screen on stocks and crypto
(research/prereg/S1.md). Writes funnel, pipeline check, trades and the primary result to --out.

    python -m tracks.main.s1_vwap --out state/s1 [--market stocks|crypto|both]
"""
import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.stats import deflated_sharpe, factor_alpha, fetch_french, screen_pass, FF5, UMD
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

SEED = 20261017
N_TRIALS = 38
STOCK_YEARS = range(2018, 2026)
STOCK_END = pd.Timestamp("2025-09-30")
CRYPTO_START, CRYPTO_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2025-09-30")
TOP = 20
ETFS = ["SPY", "QQQ"]
COST = {"stocks": 0.002, "crypto": 0.001}
ALT_COST = {"stocks": 0.0002, "crypto": 0.005}
MAX_BARS = 12
MIN_TRADES = 200
ET = "America/New_York"


# ---------- indicators and rules ----------

def indicators(df):
    """df: o h l c v, a 'session' key, sorted by time. Adds vwap, sd, k (StochRSI K), rvol, n (bar number
    in the session, from 1) and last (True on a session's final bar)."""
    d = df.copy()
    tp = (d["h"] + d["l"] + d["c"]) / 3
    g = d.groupby("session", sort=False)
    cv = g["v"].cumsum()
    d["vwap"] = (tp * d["v"]).groupby(d["session"], sort=False).cumsum() / cv
    m2 = (tp * tp * d["v"]).groupby(d["session"], sort=False).cumsum() / cv
    d["sd"] = np.sqrt(np.maximum(m2 - d["vwap"] ** 2, 0.0))
    delta = d["c"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.where(dn > 0, 100.0)
    lo, hi = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch = 100 * (rsi - lo) / (hi - lo).replace(0, np.nan)
    d["k"] = stoch.rolling(3).mean()
    d["rvol"] = d["v"] / d["v"].shift(1).rolling(20).mean()
    d["n"] = g.cumcount() + 1
    d["last"] = d["session"].ne(d["session"].shift(-1))
    d.loc[d.index[:50], "k"] = np.nan          # full indicator history before any signal
    return d


def signal_sides(d, band=2.0):
    """+1 long, -1 short, 0 none, per bar (decided at the bar's close)."""
    ok = (d["n"] >= 20) & ~d["last"] & d["k"].notna() & d["rvol"].notna()
    kp = d["k"].shift(1)
    long_ = ok & (d["c"] < d["vwap"] - band * d["sd"]) & (d["k"] < 20) & (d["k"] > kp) & (d["rvol"] > 1.5) & (d["c"] > d["o"])
    short = ok & (d["c"] > d["vwap"] + band * d["sd"]) & (d["k"] > 80) & (d["k"] < kp) & (d["rvol"] > 1.5) & (d["c"] < d["o"])
    return np.where(long_, 1, np.where(short, -1, 0))


def trade_from(o, c, vwap, last, i, side, max_bars=MAX_BARS):
    """Signal at bar i: enter at bar i+1's open, exit per S1. Returns (exit bar, gross return)."""
    e = i + 1
    for j in range(e, min(e + max_bars, len(c))):
        if (side > 0 and c[j] >= vwap[j]) or (side < 0 and c[j] <= vwap[j]) or j == e + max_bars - 1 or last[j]:
            return j, side * (c[j] / o[e] - 1)
    j = len(c) - 1
    return j, side * (c[j] / o[e] - 1)


def simulate(d, sides, cost, max_bars=MAX_BARS, rng=None, shortable=True):
    """Signal trades (one position at a time) and, when rng is given, one random same-session twin each."""
    o, c, vw, last = d["o"].to_numpy(), d["c"].to_numpy(), d["vwap"].to_numpy(), d["last"].to_numpy()
    sess = d["session"].to_numpy()
    eligible = ((d["n"] >= 20) & ~d["last"]).to_numpy()
    by_sess = pd.Series(np.flatnonzero(eligible)).groupby(sess[eligible]).apply(np.asarray).to_dict()
    times = d.index
    out, free = [], 0
    for i in np.flatnonzero(sides):
        if i < free or (sides[i] < 0 and not shortable):
            continue
        j, g = trade_from(o, c, vw, last, i, sides[i], max_bars)
        free = j + 1
        row = {"t": times[i + 1], "session": sess[i], "side": int(sides[i]), "bars": int(j - i), "gross": g,
               "net": g - cost}
        if rng is not None:
            k = int(rng.choice(by_sess[sess[i]]))
            _, cg = trade_from(o, c, vw, last, k, sides[i], max_bars)
            row.update(ctrl_gross=cg, ctrl_net=cg - cost)
        out.append(row)
    return out


# ---------- data: stocks ----------

def _assets():
    from core import alpaca
    from core.instruments import classify
    rows = []
    for st in ("active", "inactive"):
        rows += alpaca.get(f"{alpaca.TRADE}/assets", {"status": st, "asset_class": "us_equity"})
    a = classify(pd.DataFrame(rows))
    return a[a["excl_reason"].isna()]


def stock_universe(log=print):
    """{year: [symbols]}: SPY, QQQ and the top 20 by prior-year median daily raw dollar volume."""
    from core.bars import fetch_daily_bars, to_frames
    from core.env import alpaca_headers
    a = _assets()
    syms = sorted(set(a["symbol"]))
    h, rows = alpaca_headers(), []
    for b in range(0, len(syms), 500):
        fr = to_frames(fetch_daily_bars(syms[b:b + 500], "2017-01-01", "2024-12-31", adjustment="raw", headers=h,
                                        batch_size=100))
        for s, df in fr.items():
            for y, g in df.groupby(df.index.year):
                rows.append((s, y, float((g["c"] * g["v"]).median()), float(g["c"].iloc[-1]), g.index[-1]))
        log(f"[S1] daily bars {min(b + 500, len(syms))}/{len(syms)}")
    r = pd.DataFrame(rows, columns=["symbol", "year", "dv", "close", "last"])
    out = {}
    for y in STOCK_YEARS:
        p = r[r["year"] == y - 1]
        p = p[(p["last"] == p["last"].max()) & (p["close"] >= 5) & ~p["symbol"].isin(ETFS)]
        out[y] = ETFS + p.sort_values(["dv", "symbol"], ascending=[False, True])["symbol"].head(TOP).tolist()
    return out, len(syms)


def _pages(url, params):
    from core import alpaca
    rows, tok = [], None
    while True:
        p = dict(params, limit=10000, **({"page_token": tok} if tok else {}))
        j = alpaca.get(url, p)
        b = j.get("bars") or {}
        rows += b if isinstance(b, list) else next(iter(b.values()), [])
        tok = j.get("next_page_token")
        if not tok:
            return rows


def stock_bars(sym, start, end, sessions):
    """Regular-session 5-minute SIP bars (adjustment=all) between two dates, zero volume dropped."""
    rows = _pages(f"https://data.alpaca.markets/v2/stocks/{sym}/bars",
                  {"timeframe": "5Min", "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}T23:59:59Z",
                   "adjustment": "all", "feed": "sip"})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["t"] = pd.to_datetime(d["t"], utc=True)
    d = d[d["v"] > 0].set_index("t").sort_index()
    day = d.index.tz_convert(ET).normalize().tz_localize(None)
    o = sessions["open"].reindex(day).to_numpy()
    cl = sessions["close"].reindex(day).to_numpy()
    keep = (d.index >= o) & (d.index + pd.Timedelta(minutes=5) <= cl)
    d = d[keep]
    d["session"] = d.index.tz_convert(ET).strftime("%Y-%m-%d")
    return d[["o", "h", "l", "c", "v", "session"]]


def calendar(start, end):
    from core.alpaca import trading_calendar
    cal = trading_calendar(start.date(), end.date())
    def ts(dt, hm):
        return pd.Timestamp(f"{dt:%Y-%m-%d} {hm}", tz=ET).tz_convert("UTC")
    return pd.DataFrame({"open": [ts(d, o) for d, o in zip(cal["date"], cal["open"])],
                         "close": [ts(d, c) for d, c in zip(cal["date"], cal["close"])]},
                        index=pd.DatetimeIndex(cal["date"]))


def pipeline_check(spy5, sessions):
    """S1 rule 14 substitute: 5-minute bars vs daily bars for SPY."""
    from core.bars import fetch_daily_bars, to_frames
    from core.env import alpaca_headers
    daily = to_frames(fetch_daily_bars(["SPY"], "2018-01-01", f"{STOCK_END:%Y-%m-%d}", adjustment="all",
                                       headers=alpaca_headers()))["SPY"]
    g = spy5.groupby("session").agg(c=("c", "last"), v=("v", "sum"))
    g.index = pd.to_datetime(g.index)
    j = g.join(daily[["c", "v"]], rsuffix="_d", how="inner")
    close_ok = float((abs(j["c"] / j["c_d"] - 1) <= 0.0025).mean())
    vol_ok = float(((j["v"] / j["v_d"]).between(0.6, 1.05)).mean())
    return {"sessions": int(len(j)), "close_match_share": close_ok, "volume_match_share": vol_ok,
            "pass": bool(close_ok >= 0.99 and vol_ok >= 0.95)}


# ---------- data: crypto ----------

def binance_month(sym, ym):
    import requests
    url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/5m/{sym}-5m-{ym}.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6),
                        names=["t", "o", "h", "l", "c", "v"])
    d = d[pd.to_numeric(d["t"], errors="coerce").notna()].astype(float)
    unit = "us" if d["t"].iloc[0] > 1e14 else "ms"            # Binance switched spot files to microseconds in 2025
    d.index = pd.to_datetime(d.pop("t").astype("int64"), unit=unit, utc=True)
    return d


def crypto_bars(sym, log=print):
    months = pd.period_range(CRYPTO_START, CRYPTO_END, freq="M")
    try:
        d = pd.concat([binance_month(sym, f"{m}") for m in months])
        src = "binance"
    except Exception as e:  # unreachable archive: Alpaca fallback (pre-registered)
        log(f"[S1] Binance archive failed for {sym} ({type(e).__name__}); using Alpaca")
        pair = sym.replace("USDT", "/USD")
        rows = _pages("https://data.alpaca.markets/v1beta3/crypto/us/bars",
                      {"symbols": pair, "timeframe": "5Min", "start": f"{CRYPTO_START:%Y-%m-%d}",
                       "end": f"{CRYPTO_END:%Y-%m-%d}T23:59:59Z"})
        d = pd.DataFrame(rows)
        d.index = pd.to_datetime(d.pop("t"), utc=True)
        d = d[["o", "h", "l", "c", "v"]]
        src = "alpaca"
    d = d[(d["v"] > 0) & (d.index < CRYPTO_END + pd.Timedelta(days=1))].sort_index()
    d = d[~d.index.duplicated()]
    d["session"] = d.index.strftime("%Y-%m-%d")
    return d, src


# ---------- statistics ----------

def daily_series(trades):
    t = pd.DataFrame(trades)
    t["date"] = pd.to_datetime(t["session"])
    g = t.groupby("date")
    return pd.DataFrame({"net": g["net"].mean(), "ctrl": g["ctrl_net"].mean(), "n": g.size()}).sort_index()


def primary(trades, market, ff=None):
    if len(trades) < MIN_TRADES:
        return {"verdict": "no conclusion", "trades": len(trades)}
    dd = daily_series(trades)
    x = (dd["net"] - dd["ctrl"]).to_numpy()
    m, se, t = nw_tstat(x, 5)
    lo, hi = block_bootstrap_ci(x, 20, 10_000, SEED)
    prim = {"mean": m, "se": se, "t_hac": t, "ci95": [lo, hi], "days": int(len(x)), "trades": len(trades),
            "mean_net_trade": float(np.mean([r["net"] for r in trades])),
            "mean_gross_trade": float(np.mean([r["gross"] for r in trades])),
            "mean_ctrl_net_trade": float(np.mean([r["ctrl_net"] for r in trades])),
            "hit_rate": float(np.mean([r["net"] > 0 for r in trades]))}
    prim["dsr"] = deflated_sharpe(dd["net"].to_numpy(), N_TRIALS)
    if market == "stocks" and ff is not None:
        mon = dd["net"].groupby(dd.index.to_period("M")).sum()
        alpha = factor_alpha(mon.to_numpy(), mon.index, ff[0], ff[1], 3)
        prim["alpha"] = alpha
        prim["pass"] = screen_pass(prim, alpha)
    else:
        prim["pass"] = bool(m > 0 and t >= 3.0 and lo > 0)
    prim["verdict"] = "PASS" if prim["pass"] else "FAIL"
    return prim


def exploratory(series, market):
    """{name: {trades, mean_net, mean_gross}} for the pre-listed exploratory cells (never findings)."""
    cells = {"band1": dict(band=1.0), "exit6": dict(max_bars=6), "exit24": dict(max_bars=24)}
    out = {}
    for name, kw in cells.items():
        tr = []
        for d, sh in series:
            tr += simulate(d, signal_sides(d, kw.get("band", 2.0)), COST[market], kw.get("max_bars", MAX_BARS),
                           shortable=sh)
        out[name] = {"trades": len(tr), "mean_net": float(np.mean([r["net"] for r in tr])) if tr else None,
                     "mean_gross": float(np.mean([r["gross"] for r in tr])) if tr else None}
    return out


def splits(trades, market):
    t = pd.DataFrame(trades)
    t["year"] = pd.to_datetime(t["session"]).dt.year
    f = lambda g: {"trades": int(len(g)), "mean_net": float(g["net"].mean()), "mean_gross": float(g["gross"].mean())}
    return {"by_side": {("long" if k > 0 else "short"): f(g) for k, g in t.groupby("side")},
            "by_year": {int(k): f(g) for k, g in t.groupby("year")},
            "by_symbol": {k: f(g) for k, g in t.groupby("symbol")} if "symbol" in t else {},
            "alt_cost_mean_net": float((t["gross"] - ALT_COST[market]).mean()),
            "mean_bars_held": float(t["bars"].mean())}


# ---------- run ----------

def _atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_stocks(out, log=print):
    from core.alpaca import active_assets
    uni, n_cand = stock_universe(log)
    sessions = calendar(pd.Timestamp("2018-01-01"), STOCK_END)
    etb = set(active_assets().query("easy_to_borrow == True")["symbol"]) | set(ETFS)
    rng = np.random.default_rng(SEED)
    trades, series, funnel = [], [], {"candidates": n_cand, "universe": {str(y): s for y, s in uni.items()},
                                      "bars": {}, "signals": {}}
    check = None
    for sym in sorted({s for v in uni.values() for s in v}):
        years = [y for y, v in uni.items() if sym in v]
        for y in years:
            start, end = pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y}-12-31"), STOCK_END)
            d = stock_bars(sym, start, end, sessions)
            if d is None or len(d) < 100:
                funnel["bars"][f"{sym} {y}"] = 0
                continue
            if sym == "SPY":
                check = d if check is None else pd.concat([check, d])
            d = indicators(d)
            sd = signal_sides(d)
            funnel["bars"][f"{sym} {y}"] = int(len(d))
            funnel["signals"][f"{sym} {y}"] = int((sd != 0).sum())
            tr = simulate(d, sd, COST["stocks"], rng=rng, shortable=sym in etb)
            for r in tr:
                r["symbol"] = sym
            trades += tr
            series.append((d, sym in etb))
        log(f"[S1] {sym}: {len(trades)} trades so far")
    pc = pipeline_check(check, sessions)
    _atomic(out / "stocks_funnel.json", dict(funnel, pipeline_check=pc))
    if not pc["pass"]:
        _atomic(out / "stocks_result.json", {"verdict": "stopped: pipeline check failed", "pipeline_check": pc})
        return
    ff5, h1 = fetch_french(FF5)
    umd, h2 = fetch_french(UMD)
    res = primary(trades, "stocks", (ff5, umd))
    res.update(french_sha256={FF5: h1, UMD: h2}, pipeline_check=pc, splits=splits(trades, "stocks"),
               exploratory=exploratory(series, "stocks"))
    _atomic(out / "stocks_result.json", res)
    pd.DataFrame(trades).sample(min(50, len(trades)), random_state=SEED).to_csv(out / "stocks_trade_sample.csv")


def run_crypto(out, log=print):
    rng = np.random.default_rng(SEED + 1)
    trades, series, funnel = [], [], {"bars": {}, "signals": {}, "source": {}}
    for sym in ("BTCUSDT", "ETHUSDT"):
        d, src = crypto_bars(sym, log)
        d = indicators(d)
        sd = signal_sides(d)
        funnel["bars"][sym], funnel["signals"][sym], funnel["source"][sym] = int(len(d)), int((sd != 0).sum()), src
        tr = simulate(d, sd, COST["crypto"], rng=rng)
        for r in tr:
            r["symbol"] = sym
        trades += tr
        series.append((d, True))
        log(f"[S1] {sym}: {len(tr)} trades ({src})")
    _atomic(out / "crypto_funnel.json", funnel)
    res = primary(trades, "crypto")
    res.update(splits=splits(trades, "crypto"), exploratory=exploratory(series, "crypto"))
    _atomic(out / "crypto_result.json", res)
    pd.DataFrame(trades).sample(min(50, len(trades)), random_state=SEED).to_csv(out / "crypto_trade_sample.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="state/s1")
    ap.add_argument("--market", choices=["stocks", "crypto", "both"], default="both")
    a = ap.parse_args()
    out = Path(a.out)
    if a.market in ("crypto", "both"):
        run_crypto(out, lambda m: print(m, flush=True))
    if a.market in ("stocks", "both"):
        run_stocks(out, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
