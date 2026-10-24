"""R1 forward shadow book (research/prereg/FWD1.md, PROTOCOL rules 19-20): on the last session of each week,
the R1 picks (research/prereg/R1.md: bottom-decile 5-day losers of the main universe that are still above
their 200-day average, at most 20) on that night's M4 universe, a same-size random control, and the
5-deepest-loser variant. Logged to the M4 forward ledger (kind "R1_picks"); no orders. Returns are read
later from SIP bars: entry at the next session's open, exit at the 5th session's close.

Called from tracks/main/longterm/live.py, which already holds the universe and all-adjusted closes.
"""
import numpy as np
import pandas as pd

from tracks.main import r1_screen
from tracks.main.longterm.prices import segment_ids

TOP_VARIANT = 5


def features(closes, symbols, t):
    """ret5 and above200 at t, computed as in tracks/main/weekly_panel.symbol_rows (all-adjusted closes,
    within the current series only)."""
    rows = []
    for sym in symbols:
        d = closes.get(sym)
        if d is None or t not in d.index:
            continue
        d = d.loc[:t]
        seg = segment_ids(d.index)
        nb = int((seg == seg[-1]).sum())
        if nb < 21:
            continue
        c = d["c"].to_numpy()
        rows.append({"symbol": sym, "date": t, "main": True,
                     "ret5": float(c[-1] / c[-6] - 1) if nb > 5 else np.nan,
                     "above200": bool(c[-1] > c[-200:].mean()) if nb >= 200 else False,
                     "price": float(c[-1])})
    return pd.DataFrame(rows)


def record(closes, universe, t):
    """The R1_picks ledger record for week end t, or None when there are no losers. The picks follow
    r1_screen.pick; the control is drawn with SEED + t's ordinal (the screen used SEED + week index)."""
    f = features(closes, universe, t)
    g = f[f["ret5"].notna()].sort_values(["ret5", "symbol"]).reset_index(drop=True) if len(f) else f
    n = len(g) // 10
    if n < 1:
        return None
    low = g.iloc[:n]
    picks = low[low["above200"]].head(r1_screen.MAX_PICKS)
    if picks.empty:
        return None
    seed = r1_screen.SEED + t.toordinal()
    rest = g[~g["symbol"].isin(picks["symbol"])].sort_values("symbol").reset_index(drop=True)
    k = np.random.default_rng(seed).choice(len(rest), size=min(len(picks), len(rest)), replace=False)
    return {"idea": "R1", "book": "FWD1 near-miss shadow book", "as_of": f"{t:%Y-%m-%d}",
            "picks": [{"rank": i + 1, "symbol": r["symbol"], "ret5": r["ret5"], "price": r["price"]}
                      for i, (_, r) in enumerate(picks.iterrows())],
            "top5": picks["symbol"].head(TOP_VARIANT).tolist(),
            "control": rest.iloc[np.sort(k)]["symbol"].tolist(), "control_seed": seed,
            "universe": g["symbol"].tolist(), "hold_sessions": 5, "cost_round_trip": r1_screen.COST}
