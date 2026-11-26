"""SECTOR1 stage 4 (research/prereg/SECTOR1.md, Amendment 3): what moves the market and the sector ETFs on a day, and
which early signs (pre-market, first half hour, other ETFs, big-stock earnings nights) see it coming. 2019-2023 only.
Descriptive. Writes <out>/SECTOR1_stage4.json.

    python -m tracks.main.sector4_study --w1 cache/w1 --out research/results/SECTOR1 --cache cache/sector1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.env import alpaca_headers
from tracks.main import sector1_study as S1
from tracks.main import sector2_study as S2
from tracks.main import sector3_study as S3
from tracks.news import w1_data as W

INDEX = ["SPY", "QQQ", "IWM"]
SECTORS = ["XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLB", "XLP", "XLY", "XLRE", "XLC"]
MACRO = ["USO", "TLT", "UUP", "HYG", "GLD", "VIXY"]
BIG = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
BIG_CIK = [320193, 789019, 1045810, 1018724, 1652044, 1326801, 1318605]
TARGETS = INDEX + SECTORS
SYMBOLS = INDEX + SECTORS + MACRO + BIG
BLOCKS = {"spy": ["r_SPY"], "big_stocks": ["r_BIG"], "rates": ["r_TLT"], "oil": ["r_USO"],
          "dollar_gold": ["r_UUP", "r_GLD"], "credit_fear": ["r_HYG", "r_VIXY"]}
COSTS = (0.0002, 0.002)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
URL = "https://data.alpaca.markets/v2/stocks/bars"


# ---------- pure helpers (tested) ----------

def daily_points(b):
    """5-minute bars of one symbol (t = bar start, UTC; o, c) -> one row per ET date with pre (last close before 09:30),
    open (first regular open), p1000 (close of the 09:55 bar), p1530 (close of the 15:25 bar), close (close of the
    15:55 bar) and rv (sum of squared regular-session 5-minute log returns, open to close)."""
    b = b.copy()
    b["et"] = pd.to_datetime(b["t"], utc=True).dt.tz_convert("America/New_York")
    b["date"] = b["et"].dt.tz_localize(None).dt.normalize()
    hm = b["et"].dt.hour * 60 + b["et"].dt.minute
    b["hm"] = hm
    out = []
    for d, g in b.groupby("date"):
        reg = g[(g["hm"] >= 570) & (g["hm"] < 960)].sort_values("hm")
        pre = g[g["hm"] < 570]
        if reg.empty:
            continue
        at = dict(zip(reg["hm"], reg["c"], strict=True))
        px = np.r_[reg["o"].iloc[0], reg["c"].to_numpy(float)]
        out.append({"date": d, "pre": float(pre.sort_values("hm")["c"].iloc[-1]) if len(pre) else np.nan,
                    "open": float(reg["o"].iloc[0]), "p1000": at.get(595, np.nan), "p1530": at.get(925, np.nan),
                    "close": at.get(955, np.nan), "rv": float((np.diff(np.log(px)) ** 2).sum())})
    return pd.DataFrame(out)


def returns(p):
    """Daily points of one symbol -> the return pieces, with pc = the previous day's close."""
    p = p.sort_values("date").copy()
    p["pc"] = p["close"].shift()
    return pd.DataFrame({"date": p["date"], "r": p["close"] / p["pc"] - 1, "on": p["open"] / p["pc"] - 1,
                         "pm": p["pre"] / p["pc"] - 1, "oc": p["close"] / p["open"] - 1,
                         "f30": p["p1000"] / p["open"] - 1, "rest": p["close"] / p["p1000"] - 1,
                         "l30": p["close"] / p["p1530"] - 1, "lrv": np.log(p["rv"].where(p["rv"] > 0))})


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return None
    e = x - x.mean()
    v = (e @ e) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else None


def ridge_oos(X_tr, y_tr, X_te, alphas=ALPHAS, folds=5):
    """Standardise on train, pick alpha by time-ordered CV on train, refit on all train, predict test."""
    from sklearn.linear_model import Ridge
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    n = len(y_tr)
    cuts = np.linspace(n // 2, n, folds + 1).astype(int)
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        err = 0.0
        for i in range(folds):
            lo, hi = cuts[i], cuts[i + 1]
            m = Ridge(alpha=a).fit(Z_tr[:lo], y_tr[:lo])
            err += ((y_tr[lo:hi] - m.predict(Z_tr[lo:hi])) ** 2).sum()
        if err < best:
            best, best_a = err, a
    m = Ridge(alpha=best_a).fit(Z_tr, y_tr)
    return m.predict(Z_te), best_a


def score_direction(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    pnl = np.sign(p) * y
    out = {"days": len(y), "oos_r2": float(1 - ((y - p) ** 2).sum() / (y ** 2).sum()),
           "corr": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else None,
           "hit_rate": float((np.sign(p) == np.sign(y))[y != 0].mean())}
    for c in COSTS:
        net = pnl - c
        out[f"net_per_day_cost_{c * 100:.2f}pct"] = float(net.mean())
        out[f"t_nw_cost_{c * 100:.2f}pct"] = nw_t(net)
    return out


# ---------- job ----------

def fetch(sym, year, cache, h):
    f = Path(cache) / "etf5m" / f"{sym}_{year}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, tok = [], None
    while True:
        prm = {"symbols": sym, "timeframe": "5Min", "start": f"{year}-01-01T00:00:00Z", "end": f"{year}-12-31T23:59:59Z",
               "feed": "sip", "adjustment": "all", "limit": 10000}
        if tok:
            prm["page_token"] = tok
        for k in range(8):
            r = requests.get(URL, headers=h, params=prm, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
            break
        js = r.json()
        rows += js.get("bars", {}).get(sym, []) or []
        tok = js.get("next_page_token")
        if not tok:
            break
        time.sleep(0.3)
    d = pd.DataFrame(rows)
    if len(d):
        d = d[d["v"] > 0][["t", "o", "c", "v"]]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    d.to_parquet(tmp)
    tmp.replace(f)
    return d


def build(cache):
    h = alpaca_headers()
    per = {}
    for s in SYMBOLS:
        p = pd.concat([daily_points(b) for y in S1.YEARS if len(b := fetch(s, y, cache, h))], ignore_index=True)
        per[s] = returns(p).set_index("date")
        print(f"[sector4] {s}: {len(per[s])} days", flush=True)
    wide = pd.concat({s: v for s, v in per.items()}, axis=1)
    wide.columns = [f"{c}_{s}" for s, c in wide.columns]
    for k in ("r", "on", "f30"):
        wide[f"{k}_BIG"] = wide[[f"{k}_{s}" for s in BIG]].mean(axis=1)
    return wide.sort_index()


def earnings_nights(w1, cache, dates):
    S2.ensure_inputs(w1, cache, time.monotonic() + 3600)
    f = W.load_filings(w1, ["cik", "form", "filed", "items"])
    f = f[f["cik"].isin(BIG_CIK)]
    cal = pd.DatetimeIndex(dates)
    e = S2.earn_days(f, cal)
    return pd.Series(cal.isin(e["date"]), index=cal)


def part_a(w):
    out = {}
    for t in TARGETS:
        x = w.dropna(subset=[f"r_{t}"] + [c for b in BLOCKS.values() for c in b])
        blocks = {k: x[v].to_numpy(float) for k, v in BLOCKS.items() if not (t == "SPY" and k == "spy")}
        y = x[f"r_{t}"].to_numpy(float)
        sh = S3.shapley(y, blocks)
        sq = (y - y.mean()) ** 2
        on = x[f"on_{t}"].to_numpy(float)
        out[t] = {"days": len(y), **{k: float(v) for k, v in sh.items()},
                  "big_earnings_days_share_of_days": float(x["earn"].mean()),
                  "big_earnings_days_share_of_sq": float(sq[x["earn"].to_numpy(bool)].sum() / sq.sum()),
                  "overnight_share_of_sq": float(np.nansum(on ** 2) / (np.nansum(on ** 2) + np.nansum(x[f"oc_{t}"] ** 2)))}
    return out


def split(df):
    y = df.index.year
    return df[y <= 2021], df[y >= 2022]


def part_b(w):
    w = w.copy()
    out = {"size": {}, "premarket": {}, "momentum": {}, "leadlag": {}}
    for t in TARGETS:
        x = pd.DataFrame(index=w.index)
        x["y"] = w[f"lrv_{t}"]
        x["l1"] = w[f"lrv_{t}"].shift()
        x["l5"] = w[f"lrv_{t}"].rolling(5).mean().shift()
        x["l22"] = w[f"lrv_{t}"].rolling(22).mean().shift()
        for s in (t, "SPY", "VIXY"):
            x[f"apm_{s}"] = w[f"pm_{s}"].abs()
        x["earn"] = w["earn"].astype(float)
        x = x.loc[:, ~x.columns.duplicated()].dropna()
        tr, te = split(x)
        f = [c for c in x.columns if c != "y"]
        p, a = ridge_oos(tr[f].to_numpy(float), tr["y"].to_numpy(float), te[f].to_numpy(float))
        y = te["y"].to_numpy(float)
        base = tr["y"].mean()
        top = y >= np.quantile(y, 0.8)
        ptop = p >= np.quantile(p, 0.8)
        out["size"][t] = {"days": len(y), "alpha": a,
                          "oos_r2_vs_train_mean": float(1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum()),
                          "top20_caught": float((top & ptop).sum() / top.sum()),
                          "corr": float(np.corrcoef(y, p)[0, 1])}

        def direction(feats, target, key, t=t):
            x = pd.concat([w[feats], w[target].rename("y")], axis=1).dropna()
            tr, te = split(x)
            p, a = ridge_oos(tr[feats].to_numpy(float), tr["y"].to_numpy(float), te[feats].to_numpy(float))
            out[key][t] = {"alpha": a, **score_direction(te["y"].to_numpy(float), p)}
        direction(sorted({f"pm_{s}" for s in (t, "SPY", "QQQ", "TLT", "USO", "VIXY")}), f"oc_{t}", "premarket")
        direction(sorted({f"on_{t}", f"f30_{t}", "on_SPY", "f30_SPY"}), f"l30_{t}", "momentum")
        direction([f"f30_{s}" for s in SYMBOLS], f"rest_{t}", "leadlag")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    ap.add_argument("--minutes", type=float, default=150)
    a = ap.parse_args()
    w = build(a.cache)
    w = w[(w.index.year >= S1.YEARS[0]) & (w.index <= S1.LAST)]
    w["earn"] = earnings_nights(a.w1, a.cache, w.index).reindex(w.index).fillna(False).astype(bool)
    print(f"[sector4] {len(w)} days, {int(w['earn'].sum())} big-stock earnings days", flush=True)
    out = {"part_a": part_a(w), "part_b": part_b(w),
           "note": ("2019-2023, Alpaca SIP 5-minute bars adjustment=all. Part A: same-day Shapley R2 of each ETF's "
                    "close-to-close return on SPY, the 7 biggest stocks, TLT, USO, UUP+GLD, HYG+VIXY. Part B: ridge "
                    "fit 2019-2021, scored 2022-2023. Descriptive only.")}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage4.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.replace(Path(a.out) / "SECTOR1_stage4.json")
    print(json.dumps({k: out["part_b"][k].get("SPY") for k in out["part_b"]}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
