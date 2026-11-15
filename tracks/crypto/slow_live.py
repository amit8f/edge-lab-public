"""CRY-WIDE and CRY-MOM forward books (research/prereg/CRYSLOW.md, Amendment 1). Ledger only: no orders.

Binance's archive publishes a month after it ends, so each run downloads every finished month since the warm-up
start, rebuilds xs_study's rows with its own code, and replays both rules and the equal-weight twin over the days
from START. Each month is logged once; every earlier day is recomputed and must match the ledger.

    python -m tracks.crypto.slow_live --state state
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.crypto import cdata as C
from tracks.crypto import xs_study as X

WARM = "2026-08"                       # three months of candles before the first decision for the 28/60-day inputs
START = pd.Timestamp("2026-11-15")     # first day held (decision at the 2026-11-14 close)
JUDGED = "2027-12-31"
BOOKS = ("CRY-WIDE", "CRY-MOM", "TWIN")
NO_FUND = pd.DataFrame({"t": pd.to_datetime([]), "rate": []})


# ---------- pure helpers (tested) ----------

def books(R):
    """{book: daily frame} over the days held from START."""
    R = R[R["t"] >= START - pd.Timedelta(days=1)].reset_index(drop=True)
    picks = X.mom_picks(R)
    out = {"CRY-WIDE": X.weights_book(R, X.w_wide), "CRY-MOM": X.weights_book(R, X.w_mom(picks)),
           "TWIN": X.weights_book(R, X.w_hold)}
    return {k: v[v.index >= START] for k, v in out.items()}


def rows_of(B):
    return {k: [[str(d.date()), float(r.net), float(r.gross), float(r.fund), float(r.turnover)] for d, r in v.iterrows()]
            for k, v in B.items()}


def same(old, new):
    """Every logged day is reproduced (same day and net, to 1e-9)."""
    for k in BOOKS:
        o, n = old.get(k, []), new[k]
        if len(n) < len(o) or any(a[0] != b[0] or abs(a[1] - b[1]) > 1e-9 for a, b in zip(o, n, strict=False)):
            return False
    return True


def summary(rows):
    out = {"first_day": str(START.date()), "judged": JUDGED, "books": {}}
    for k, r in rows.items():
        net = np.array([x[1] for x in r])
        eq = np.cumprod(1 + net) if len(net) else np.array([1.0])
        out["books"][k] = {"days": len(net), "usd": float(1000 * eq[-1]), "sharpe": X.sharpe(net) if len(net) > 2 else None,
                           "max_drawdown": float((1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max()) if len(net) else 0.0,
                           "last_day": r[-1][0] if r else None}
    return out


# ---------- network ----------

def months_done(today):
    out = []
    for m in C.months(WARM, str(pd.Period(today, freq="M") - 1)):
        if C.fetch(C.kline_url("futures/um", "BTCUSDT", "1d", m)) is None:
            break
        out.append(m)
    return out


def source(today):
    """Rows for every finished month, or None before the first month after START is published."""
    months = months_done(today)
    if not months or months[-1] < str(START.to_period("M")):
        return None
    syms = C.list_symbols()

    def get(s):
        k = [C.parse_klines(r) for r in (C.fetch(C.kline_url("futures/um", s, "1d", m)) for m in months) if r]
        return s, (pd.concat(k, ignore_index=True).sort_values("t").drop_duplicates("t").reset_index(drop=True) if k else None)
    with ThreadPoolExecutor(C.THREADS) as ex:
        bars = {s: b for s, b in ex.map(get, syms) if b is not None}
    vol = pd.DataFrame({s: b.groupby(pd.PeriodIndex(b["t"], freq="M"))["qv"].sum() for s, b in bars.items()}).fillna(0.0)
    uni = {m: v for m, v in X.universe(vol, {s: b["t"].iat[0] for s, b in bars.items()}).items()
           if m >= str(START.to_period("M"))}
    need = sorted({s for v in uni.values() for s in v} | {"BTCUSDT"})
    rows = {}
    for s in need:
        fr = [C.parse_funding(r) for r in (C.fetch(f"{C.BASE}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")
                                           for m in months) if r]
        fund = pd.concat(fr, ignore_index=True).sort_values("t").drop_duplicates("t") if fr else NO_FUND
        rows[s] = X.coin_rows(bars[s], fund)
    return X.assemble(rows, uni), uni


# ---------- job ----------

def run(state, today=None, source=source):
    root = Path(state) / "cryslow"
    led = Ledger(root / "ledger")
    got = source(today or pd.Timestamp.now("UTC").tz_localize(None))
    if got is None:
        print("[cry-slow] no finished month after the start yet", flush=True)
        return
    R, uni = got
    new = rows_of(books(R))
    last = max((r[-1][0] for r in new.values() if r), default=None)
    entries = [e for e in led.entries() if e["kind"] == "month"]
    if last is None or (entries and last <= entries[-1]["date"]):
        print("[cry-slow] nothing new", flush=True)
        return
    old = {k: [] for k in BOOKS}
    for e in entries:
        for k, v in json.loads((led.root / e["file"]).read_text())["rows"].items():
            old[k] += v
    if not same(old, new):
        raise RuntimeError("a logged day is not reproduced from the archive; nothing written")
    add = {k: new[k][len(old[k]):] for k in BOOKS}
    led.write_json("month", last, f"months/{last}.json", {"through": last, "universe": uni, "rows": add,
                                                           "columns": ["day", "net", "gross", "funding", "turnover"]})
    probs = led.verify()
    if probs:
        raise RuntimeError(f"ledger problems: {probs[:3]}")
    tmp = root / "summary.tmp"
    tmp.write_text(json.dumps(summary(new), indent=1))
    tmp.replace(root / "summary.json")
    print(f"[cry-slow] logged through {last}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
