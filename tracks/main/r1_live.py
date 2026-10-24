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


def vol20(closes, sym, t):
    """META3's vol20: sd (ddof 1) of the last 20 daily log returns of all-adjusted closes to t, within the
    current series (21 bars needed), else None."""
    d = closes.get(sym)
    if d is None or t not in d.index:
        return None
    d = d.loc[:t]
    seg = segment_ids(d.index)
    c = d["c"].to_numpy()[seg == seg[-1]]
    if len(c) < 21:
        return None
    return float(np.std(np.diff(np.log(c[-21:])), ddof=1))


def eightk_counts(symbols, t, sessions, index_of, cik_of):
    """META3's n_8k per symbol: 8-K and 8-K/A filings by its CIK with filing date from sessions[-5] through t
    (sessions: trading days up to t). None for every symbol when a daily index isn't published yet; None for
    a symbol with no CIK."""
    days = pd.bdate_range(sessions[-5], t)
    parts = []
    for d in days:
        idx = index_of(d)
        if idx is None:
            return {s: None for s in symbols}
        parts.append(idx[idx["form"].isin({"8-K", "8-K/A"})])
    f = pd.concat(parts) if parts else pd.DataFrame(columns=["cik"])
    cnt = f.groupby("cik").size().to_dict()
    return {s: (int(cnt.get(cik_of[s], 0)) if s in cik_of else None) for s in symbols}


def meta3(picks, closes, t, n8k):
    """research/prereg/META3.md: mean within-week percentile rank of ret5 (ascending), vol20 and n_8k
    (descending); a missing input ranks 0.5; keep when the score is at or above the week's median."""
    df = pd.DataFrame({"symbol": [p["symbol"] for p in picks], "ret5": [p["ret5"] for p in picks]})
    df["vol20"] = [vol20(closes, s, t) for s in df["symbol"]]
    df["n_8k"] = [n8k.get(s) for s in df["symbol"]]
    ranks = []
    for col, asc in (("ret5", True), ("vol20", False), ("n_8k", False)):
        v = pd.to_numeric(df[col], errors="coerce")
        ranks.append(v.rank(pct=True, ascending=asc).fillna(0.5))
    df["score"] = sum(ranks) / 3
    df["keep"] = df["score"] >= df["score"].median()
    return [{"symbol": r.symbol, "vol20": None if pd.isna(r.vol20) else float(r.vol20),
             "n_8k": None if pd.isna(r.n_8k) else int(r.n_8k), "score": float(r.score), "keep": bool(r.keep)}
            for r in df.itertuples()]
