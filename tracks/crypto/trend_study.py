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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="research/results/CRYTREND")
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
