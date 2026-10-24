"""FWD2 forward books I1F and A1F (research/prereg/FWD2.md): ledger records only, no orders. Returns are read
later from SIP bars.

I1F: on the last session of each week, I1's picks (tracks/penny/i1_screen.pick) on that night's panel rows,
held 20 trading days (entry at the next session's open, exit at the close of the 20th session from entry).
A1F: on the last session of each month, penny stocks with a 10-K filed in the 3 months up to t whose
M-score is above -1.78 (tracks/penny/a1_screen), minus the FWD2 exclusion (prior-year sales under $10M or
SGI above 3), shorted only when Alpaca marks them easy to borrow, held 63 trading days with a 35% stop.

Each record carries a same-size random control drawn with the screen's seed plus t's ordinal. Called from
tracks/main/longterm/live.py, which already holds the panel rows, the M4 universe and companyfacts.
"""
import numpy as np
import pandas as pd

from core import industry
from tracks.penny import a1_screen, i1_screen

I1F_HOLD = 20
A1F_MIN_PRIOR_SALES, A1F_MAX_SGI = 1e7, 3.0


def flags(panel, main_symbols):
    """Panel rows (weekly_panel.symbol_rows for t, with cik) -> the I1 screen's main and penny flags. Main is
    the M4 universe (one listing per company, $300M+ cap) with the screen's bar floors on top."""
    r = panel.copy()
    r["main"] = r["symbol"].isin(set(main_symbols)) & (r["price"] >= 5) & (r["dv20"] >= 2e6) & (r["nb"] >= 253)
    r["penny"] = (r["price"] >= 1) & (r["price"] < 5) & (r["dv20"] >= 5e5) & (r["nb"] >= 60)
    return r


def _control(rest, n, seed):
    rest = rest.sort_values("symbol").reset_index(drop=True)
    k = np.random.default_rng(seed).choice(len(rest), size=min(n, len(rest)), replace=False)
    return rest.iloc[np.sort(k)]["symbol"].tolist()


def i1f_record(rows, ind, spy5, t):
    """The I1F_picks record for week end t, or None when no industry moved enough or no penny stock lags.
    rows: flags() output; ind: Series cik -> FF12 code; spy5: SPY's 5-day return to t."""
    r = rows.copy()
    r["ind"] = r["cik"].map(lambda c: ind.get(int(c)) if pd.notna(c) else np.nan)
    mv = i1_screen.moves(r, {t: spy5})
    sel = i1_screen.pick(r, mv, [t])
    if sel.empty:
        return None
    picks = sel[sel["role"] == "pick"]
    pen = r[r["penny"]]
    seed = i1_screen.SEED + t.toordinal()
    top = mv.sort_values(["move", "ind"], ascending=False).head(i1_screen.TOP_IND)
    return {"idea": "I1F", "book": "FWD2 forward book", "as_of": f"{t:%Y-%m-%d}",
            "picks": [{"symbol": p["symbol"], "ind": int(p["ind"]), "ret5": p["ret5"], "gap": p["gap"],
                       "price": p["price"], "dv20": p["dv20"]} for _, p in picks.iterrows()],
            "control": _control(pen[~pen["symbol"].isin(picks["symbol"])], len(picks), seed),
            "control_seed": seed, "universe": sorted(pen["symbol"]),
            "industries": [{"ind": int(m["ind"]), "name": industry.NAMES[int(m["ind"])], "move": m["move"],
                            "n_big": int(m["size"])} for _, m in top.iterrows()],
            "spy_ret5": spy5, "hold_sessions": I1F_HOLD, "cost_round_trip": i1_screen.COST,
            "entry": "next session's open", "exit": f"close of the {I1F_HOLD}th session from entry"}


def a1f_record(rows, scored, t, short_ok, logged_accns):
    """The A1F_picks record for month end t, or None when nothing is flagged. rows: flags() output (penny
    rows with cik); scored: a1_screen.scores() for their CIKs; short_ok: symbols Alpaca marks shortable and
    easy to borrow tonight; logged_accns: 10-Ks already used by an earlier A1F record (a filing enters once)."""
    pen = rows[rows["penny"] & rows["cik"].notna()].astype({"cik": "int64"})
    f = scored[scored["m"] > a1_screen.M_FLAG].copy()
    f = f[(f["filed"] <= t) & (f["filed"] > t - pd.DateOffset(months=3)) & ~f["accn"].isin(set(logged_accns))]
    c = pen.merge(f, on="cik")
    if c.empty:
        return None
    c = c.sort_values("symbol").drop_duplicates("accn").drop_duplicates("cik")
    c["prior_sales"] = c["sales"] / c["sgi"]
    c["why"] = np.where(c["prior_sales"] < A1F_MIN_PRIOR_SALES, "prior-year sales under $10M",
                        np.where(c["sgi"] > A1F_MAX_SGI, "SGI above 3",
                                 np.where(~c["symbol"].isin(short_ok), "not easy to borrow", "")))
    take = c[c["why"] == ""]
    seed = a1_screen.SEED + t.toordinal()
    rest = pen[~pen["symbol"].isin(c["symbol"]) & pen["symbol"].isin(short_ok)]

    def row(x):
        return {"symbol": x["symbol"], "cik": int(x["cik"]), "accn": x["accn"], "filed": f"{x['filed']:%Y-%m-%d}",
                "m": x["m"], "sales": x["sales"], "prior_sales": x["prior_sales"], "sgi": x["sgi"],
                "price": x["price"], "dv20": x["dv20"]}
    return {"idea": "A1F", "book": "FWD2 forward book", "as_of": f"{t:%Y-%m-%d}", "side": "short",
            "picks": [row(x) for _, x in take.iterrows()],
            "skipped": [dict(row(x), why=x["why"]) for _, x in c[c["why"] != ""].iterrows()],
            "accns": sorted(c["accn"]),
            "control": _control(rest, len(take), seed) if len(take) else [], "control_seed": seed,
            "universe": sorted(pen["symbol"]), "control_pool": "penny universe, easy to borrow, not flagged",
            "hold_sessions": a1_screen.HOLD, "stop": a1_screen.STOP, "cost_round_trip": a1_screen.COST,
            "borrow_per_year": a1_screen.BORROW, "entry": "next session's open"}
