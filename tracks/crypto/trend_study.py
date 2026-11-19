"""CRY-TREND history check (research/prereg/CRYPTO.md): hold BTC and ETH, half the book each, only while the
coin's daily close is above its close 28 days earlier. It can only reject.

Usage: python -m tracks.crypto.trend_study --cache cache --out research/results/CRYTREND
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.crypto import cdata as C

COINS = ("BTCUSDT", "ETHUSDT")
LOOKBACK = 28
COSTS = {"primary_0.5pct": 0.005, "futures_0.12pct": 0.0012}      # round trip
PERIODS = {"2018-01..2025-09": ("2018-01-01", "2025-09-30"), "2022-01..2025-09": ("2022-01-01", "2025-09-30")}
TWINS, SEED = 1000, 0


# ---------- pure helpers (tested) ----------

def signal(close, lookback=LOOKBACK):
    """1 on days whose close is above the close `lookback` days earlier (known at that close), else 0."""
    c = np.asarray(close, float)
    s = np.zeros(len(c))
    s[lookback:] = c[lookback:] > c[:-lookback]
    return s


def book(ret, sig, cost):
    """Daily equity of the book. ret[d, k]: coin k's return over day d; sig[d, k]: 1 if coin k is held over day d
    (decided at the previous close). A coin is entered with half the book's value at that moment (or the cash
    left, if less) and sold whole on exit; each side pays cost / 2."""
    n, k = ret.shape
    cash, pos = 1.0, np.zeros(k)
    eq = np.empty(n)
    for d in range(n):
        for j in range(k):
            if pos[j] > 0 and not sig[d, j]:
                cash += pos[j] * (1 - cost / 2)
                pos[j] = 0.0
        total = cash + pos.sum()
        for j in range(k):
            if pos[j] == 0 and sig[d, j]:
                put = min(total / k, cash)
                cash -= put
                pos[j] = put * (1 - cost / 2)
        pos = pos * (1 + ret[d])
        eq[d] = cash + pos.sum()
    return eq


def hold(ret):
    """Half in each coin at the start, never touched."""
    return (np.cumprod(1 + ret, axis=0)).mean(axis=1)


def stats(eq):
    r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    sd = r.std(ddof=1)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    years = len(eq) / 365
    return {"sharpe": float(r.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan"),
            "cagr": float(eq[-1] ** (1 / years) - 1), "max_drawdown": float((1 - np.r_[1.0, eq] / peak).max()),
            "end": float(eq[-1])}


def runs(s):
    """A 0/1 series -> (first value, run lengths)."""
    s = np.asarray(s).astype(int)
    cut = np.flatnonzero(np.diff(s)) + 1
    return int(s[0]), np.diff(np.r_[0, cut, len(s)])


def twin(s, rng):
    """A random-timing copy of s: the same held runs and the same cash runs, each set shuffled, so the number of
    switches and the share of days held are unchanged."""
    first, ln = runs(s)
    a, b = ln[0::2].copy(), ln[1::2].copy()
    rng.shuffle(a)
    rng.shuffle(b)
    out, v = [], first
    for i in range(len(ln)):
        out.append(np.full((a if i % 2 == 0 else b)[i // 2], v))
        v = 1 - v
    return np.concatenate(out)


def round_trips(s):
    return int((np.diff(np.r_[0, np.asarray(s).astype(int)]) == 1).sum())


def evaluate(ret, sig, cost, rng):
    eq = book(ret, sig, cost)
    st, hd = stats(eq), stats(hold(ret))
    tw = np.array([stats(book(ret, np.column_stack([twin(sig[:, j], rng) for j in range(sig.shape[1])]), cost))
                   ["sharpe"] for _ in range(TWINS)])
    return {"rule": st, "hold": hd, "round_trips": [round_trips(sig[:, j]) for j in range(sig.shape[1])],
            "share_of_days_held": [float(sig[:, j].mean()) for j in range(sig.shape[1])],
            "twin_sharpe_percentile": float((tw < st["sharpe"]).mean()), "twin_sharpe_median": float(np.median(tw)),
            "ok": bool(st["sharpe"] > hd["sharpe"] and st["max_drawdown"] < hd["max_drawdown"])}


# ---------- CRY-TREND-VT (research/prereg/CRYTRENDVT.md) ----------

VT_TARGET, VT_WINDOW, MONDAY = 0.50, 20, 0


def realized_vol(ret, window=VT_WINDOW):
    """Annualised standard deviation of the last `window` daily returns, known at each day's close (NaN until full)."""
    r = pd.DataFrame(np.asarray(ret, float)).rolling(window).std(ddof=1).to_numpy()
    return r * np.sqrt(365)


def vt_weight(vol, k, target=VT_TARGET):
    """Share of the book a held coin gets: (1/k) x min(1, target / vol); 0 where the volatility is unknown."""
    v = np.asarray(vol, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.minimum(1.0, target / v) / k
    return np.where(np.isfinite(v) & (v > 0), w, 0.0)


def vt_book(ret, sig, vol, monday, cost):
    """Daily equity of the volatility-sized book. sig[d, k] and vol[d, k] are known at the previous close; monday[d]
    is True when that previous close was a Monday's. A coin is rebalanced to (1/k) x min(1, target / vol) of the
    book's value when it is entered, on Mondays while held, and sold whole on exit; each dollar traded pays cost / 2."""
    n, k = ret.shape
    cash, pos = 1.0, np.zeros(k)
    eq = np.empty(n)
    w = vt_weight(vol, k)
    for d in range(n):
        total = cash + pos.sum()
        for j in range(k):
            want = w[d, j] if sig[d, j] else 0.0
            if pos[j] > 0 and not sig[d, j]:
                trade = -pos[j]
            elif sig[d, j] and (pos[j] == 0 or monday[d]):
                trade = min(want * total, cash + pos[j]) - pos[j]
            else:
                continue
            fee = abs(trade) * cost / 2
            cash -= trade + fee
            pos[j] += trade
        pos = pos * (1 + ret[d])
        eq[d] = cash + pos.sum()
    return eq


def trades_per_year(sig):
    return float(np.abs(np.diff(np.asarray(sig), axis=0)).sum(axis=0).mean() / (len(sig) / 365))


# ---------- stage ----------

def load(cache):
    px = {}
    for s in COINS:
        d = pd.read_parquet(C.cdir(cache) / "spot1d" / f"{s}.parquet")
        px[s] = d.set_index("t")["c"]
    P = pd.DataFrame(px).dropna()
    assert P.index.max() < pd.Timestamp("2025-10-01"), "holdout data read"
    return P


def run(cache, out):
    P = load(cache)
    sig_all = np.column_stack([signal(P[s].to_numpy()) for s in COINS])
    ret_all = P.pct_change().to_numpy()
    res = {"rule": f"hold while close > close {LOOKBACK} days earlier; BTC and ETH, half each", "periods": {},
           "days": {"first": str(P.index[0].date()), "last": str(P.index[-1].date())}}
    for name, (a, b) in PERIODS.items():
        # day d's return is earned with the signal of the close before it
        m = (P.index >= pd.Timestamp(a)) & (P.index <= pd.Timestamp(b))
        idx = np.flatnonzero(m)
        ret, sig = ret_all[idx], sig_all[idx - 1]
        res["periods"][name] = {k: evaluate(ret, sig, c, np.random.default_rng(SEED)) for k, c in COSTS.items()}
    ok = all(p["primary_0.5pct"]["ok"] for p in res["periods"].values())
    res["verdict"] = "not rejected: start the forward book" if ok else "FAILED: rejected by the history check"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CRYTREND.json").write_text(json.dumps(res, indent=1))
    print(f"[cry-trend] {res['verdict']}", flush=True)
    return res


def run_vt(cache, out):
    P = load(cache)
    sig_all = np.column_stack([signal(P[s].to_numpy()) for s in COINS])
    ret_all = P.pct_change().to_numpy()
    vol_all = realized_vol(ret_all)
    mon_all = (P.index.dayofweek == MONDAY)
    res = {"rule": f"CRY-TREND with weight 1/2 x min(1, {VT_TARGET} / {VT_WINDOW}-day vol); rebalanced at entry, "
                   "Mondays, exit", "periods": {}, "days": {"first": str(P.index[0].date()), "last": str(P.index[-1].date())}}
    for name, (a, b) in PERIODS.items():
        m = (P.index >= pd.Timestamp(a)) & (P.index <= pd.Timestamp(b))
        idx = np.flatnonzero(m)
        ret, sig, vol, mon = ret_all[idx], sig_all[idx - 1], vol_all[idx - 1], mon_all[idx - 1]
        res["periods"][name] = {}
        for k, c in COSTS.items():
            vt, base, hd = stats(vt_book(ret, sig, vol, mon, c)), stats(book(ret, sig, c)), stats(hold(ret))
            res["periods"][name][k] = {"vt": vt, "trend": base, "hold": hd,
                                       "ok": bool(vt["sharpe"] > base["sharpe"] and vt["max_drawdown"] < base["max_drawdown"]
                                                  and vt["sharpe"] > hd["sharpe"])}
        w = vt_weight(vol, ret.shape[1])
        res["periods"][name]["avg_invested"] = float((sig * w).sum(axis=1).mean())
        res["periods"][name]["trades_per_year_signal"] = trades_per_year(sig)
        rng = np.random.default_rng(SEED)
        tw = np.array([stats(vt_book(ret, np.column_stack([twin(sig[:, j], rng) for j in range(sig.shape[1])]),
                                     vol, mon, COSTS["primary_0.5pct"]))["sharpe"] for _ in range(TWINS)])
        res["periods"][name]["twin_sharpe_percentile"] = float(
            (tw < res["periods"][name]["primary_0.5pct"]["vt"]["sharpe"]).mean())
    ok = all(p["primary_0.5pct"]["ok"] for p in res["periods"].values())
    res["verdict"] = "not rejected: start the forward book" if ok else "FAILED: rejected by the history check"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CRYTRENDVT.json").write_text(json.dumps(res, indent=1))
    print(f"[cry-trend-vt] {res['verdict']}", flush=True)
    return res


# ---------- CRY-BIG and CRY-MEME (research/prereg/CRYBASKETS.md) ----------

BASKETS = {
    "CRYBIG": {"coins": ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"), "primary": "primary_0.5pct",
               "costs": {"primary_0.5pct": 0.005, "futures_0.12pct": 0.0012},
               "periods": PERIODS},
    "CRYMEME": {"coins": ("DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "WIFUSDT", "FLOKIUSDT"),
                "primary": "primary_1.0pct",
                "costs": {"primary_1.0pct": 0.01, "spot_0.5pct": 0.005, "futures_0.12pct": 0.0012},
                "periods": {"2021-06..2025-09": ("2021-06-01", "2025-09-30"),
                            "2023-06..2025-09": ("2023-06-01", "2025-09-30")}},
}


def basket_frame(P, lookback=LOOKBACK):
    """Wide daily closes (NaN before a coin lists) -> (signal, return, available), each days x coins. A coin is
    available once it has a close `lookback` days back; before that its signal is 0 and its return counts as 0."""
    sig = np.column_stack([(P[c] > P[c].shift(lookback)).to_numpy(float) for c in P.columns])
    avail = np.column_stack([P[c].shift(lookback).notna().to_numpy(float) for c in P.columns])
    ret = P.pct_change(fill_method=None).fillna(0.0).to_numpy()
    return sig, ret, avail


def hold_late(ret, avail):
    """Buy and hold with one slot per coin: a slot sits in cash until the coin is available, then is bought and never
    touched. avail[d, k] is known at the close before day d."""
    return np.cumprod(1 + ret * avail, axis=0).mean(axis=1)


def twin_late(s, avail, rng):
    """twin() on the part of s where the coin is available; zeros before it."""
    s = np.asarray(s).astype(int)
    on = np.flatnonzero(np.asarray(avail) > 0)
    out = np.zeros(len(s), int)
    if len(on):
        out[on[0]:] = twin(s[on[0]:], rng)
    return out


def evaluate_late(ret, sig, avail, cost, rng):
    eq = book(ret, sig, cost)
    st, hd = stats(eq), stats(hold_late(ret, avail))
    k = sig.shape[1]
    tw = np.array([stats(book(ret, np.column_stack([twin_late(sig[:, j], avail[:, j], rng) for j in range(k)]),
                              cost))["sharpe"] for _ in range(TWINS)])
    return {"rule": st, "hold": hd, "round_trips": [round_trips(sig[:, j]) for j in range(k)],
            "share_of_days_held": [float(sig[:, j].mean()) for j in range(k)],
            "days_available": [int(avail[:, j].sum()) for j in range(k)],
            "twin_sharpe_percentile": float((tw < st["sharpe"]).mean()), "twin_sharpe_median": float(np.median(tw)),
            "ok": bool(st["sharpe"] > hd["sharpe"] and st["max_drawdown"] < hd["max_drawdown"])}


def load_wide(cache, coins):
    px = {}
    for s in coins:
        d = pd.read_parquet(C.cdir(cache) / "spot1d" / f"{s}.parquet")
        px[s] = d.set_index("t")["c"]
    P = pd.DataFrame(px).sort_index()
    P = P.loc[P.notna().any(axis=1)]
    assert P.index.max() < pd.Timestamp("2025-10-01"), "holdout data read"
    full = pd.date_range(P.index.min(), P.index.max(), freq="D")
    assert len(full) == len(P), "missing days in the daily closes"
    return P


def run_baskets(cache, out):
    res = {}
    for name, cfg in BASKETS.items():
        P = load_wide(cache, cfg["coins"])
        sig_all, ret_all, av_all = basket_frame(P)
        r = {"coins": list(cfg["coins"]), "first_close": {c: str(P[c].first_valid_index().date()) for c in P.columns},
             "periods": {}}
        for pname, (a, b) in cfg["periods"].items():
            m = (P.index >= pd.Timestamp(a)) & (P.index <= pd.Timestamp(b))
            idx = np.flatnonzero(m)
            ret, sig, av = ret_all[idx], sig_all[idx - 1], av_all[idx - 1]
            r["periods"][pname] = {k: evaluate_late(ret, sig, av, c, np.random.default_rng(SEED))
                                   for k, c in cfg["costs"].items()}
        ok = all(p[cfg["primary"]]["ok"] for p in r["periods"].values())
        r["verdict"] = "not rejected: start the forward book" if ok else "FAILED: rejected by the history check"
        print(f"[{name}] {r['verdict']}", flush=True)
        res[name] = r
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CRYBASKETS.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="research/results/CRYTREND")
    ap.add_argument("--rule", default="plain", choices=["plain", "vt", "baskets"])
    a = ap.parse_args()
    {"plain": run, "vt": run_vt, "baskets": run_baskets}[a.rule](a.cache, a.out)


if __name__ == "__main__":
    main()
