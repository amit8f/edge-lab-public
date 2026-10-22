"""M3 live ranking (SYS1 strategy, research/prereg/M3.md signal): on the last session of each month, score
residual momentum on that night's M4 universe and log the 25 picks, a 25-stock random control and the
universe to the M4 forward ledger (kind "M3_picks"), where adaptive/nightly.py reads them.

Live factors are ETF stand-ins, because Ken French's monthly file runs about two months behind:
mkt = SPY - BIL, smb = IWM - SPY, hml = IWD - IWF, rf = BIL (Alpaca SIP daily bars, adjustment=all).
Same regression, window and score as the screen (tracks/main/m3_screen.py).

Usage: python -m tracks.main.m3_live --state <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.ledger import Ledger
from tracks.main import m3_screen
from tracks.main.longterm import bars_batch
from tracks.main.longterm.prices import align, segment_ids

ETFS = ("SPY", "BIL", "IWM", "IWD", "IWF")
TOP_N, SEED = m3_screen.TOP_N, m3_screen.SEED
LOOKBACK_DAYS = 40 * 31          # 36 regression months plus the first month's start and slack
BATCH = 500


def proxy_factors(frames):
    """Monthly mkt, smb, hml, rf from the ETF stand-ins, indexed by month period."""
    r = {}
    for s in ETFS:
        d = frames[s]
        r[s] = m3_screen.monthly_returns(d, segment_ids(d.index))
    r = pd.DataFrame(r)
    return pd.DataFrame({"mkt": r["SPY"] - r["BIL"], "smb": r["IWM"] - r["SPY"],
                         "hml": r["IWD"] - r["IWF"], "rf": r["BIL"]}).dropna()


def score_rows(fr, symbols, t, ff):
    rows = []
    for s in symbols:
        if all(s in fr[a] for a in ("all", "raw", "split")):
            d = align(fr["all"][s], fr["raw"][s], fr["split"][s])
            if len(d):
                rows += m3_screen.symbol_rows(s, d, [t], {}, {}, ff)
    return rows


def ranking(universe, t, end, fetch=bars_batch.fetch_three):
    start = (t - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    etf = fetch(list(ETFS), start, end)
    ff = proxy_factors({s: align(etf["all"][s], etf["raw"][s], etf["split"][s]) for s in ETFS})
    rows = []
    for b in range(0, len(universe), BATCH):
        part = universe[b:b + BATCH]
        rows += score_rows(fetch(part, start, end), part, t, ff)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ff
    df = df.dropna(subset=["score"]).sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, ff


def run(state, now=None, fetch=bars_batch.fetch_three):
    from tracks.main.longterm.live import last_session
    now = now if now is not None else alpaca.utcnow()
    root = Path(state) / "longterm"
    led = Ledger(root)
    t, is_month_end = last_session(now)
    day = f"{t:%Y-%m-%d}"
    if not is_month_end:
        print(f"{day} is not a month end; M3 logs only then")
        return None
    if led.has("M3_picks", day):
        print(f"M3 already logged for {day}")
        return None
    m4 = led.entry("M4_picks", day)
    if m4 is None:
        raise RuntimeError(f"no M4_picks entry for {day}: run tracks.main.longterm.live first")
    universe = sorted(json.loads((root / m4["file"]).read_text())["universe"])
    scored, ff = ranking(universe, t, alpaca.sip_end(t, now), fetch)
    if ff.empty or ff.index.max() != t.to_period("M"):
        raise RuntimeError(f"stand-in factors end at {ff.index.max() if len(ff) else None}, not {t:%Y-%m}")
    if len(scored) < TOP_N:
        raise RuntimeError(f"only {len(scored)} stocks have an M3 score")
    rng = np.random.default_rng(SEED + t.toordinal())
    ctrl = scored.iloc[rng.choice(len(scored), size=TOP_N, replace=False)]["symbol"].tolist()
    rec = {"idea": "M3", "as_of": day, "factors": "ETF stand-ins: SPY-BIL, IWM-SPY, IWD-IWF, rf BIL",
           "picks": [{"rank": int(r["rank"]), "symbol": r["symbol"], "score": float(r["score"]),
                      "price": float(r["price"])} for _, r in scored.head(TOP_N).iterrows()],
           "control": ctrl, "control_seed": SEED + t.toordinal(), "universe": scored["symbol"].tolist(),
           "universe_in": len(universe)}
    led.write_json("M3_picks", day, f"m3/{day}.json", rec)
    print(f"logged M3 picks for {day}: {len(scored)} of {len(universe)} stocks scored")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    run(a.state)


if __name__ == "__main__":
    main()
