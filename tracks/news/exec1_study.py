"""EXEC1 (research/prereg/EXEC1.md): NX3-ALL-HI trades that would enter before T enter at T instead (narrower
spreads), both arms exiting at the closing auction (no exit spread). The closing-auction saving is reported.

    python -m tracks.news.exec1_study --stage bars|run --cache cache/nx1 --out research/results/MODELS5

bars: per month, for trades entering before the last T, the 1-minute SIP bars (raw) 9:30-10:31 and the NBBO at each
T -> cache nx3/exec1/x_<month>.parquet. Needs models5 picks and COST1's quotes.
run: T chosen on 2019-2023, one look 2024-01..2025-09. Summaries only.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import alpaca_headers
from tracks.news import cost1_study as K
from tracks.news import models5_study as M
from tracks.news import nx3_study as X

T_GRID = ("09:45", "10:00", "10:30")
MAX_LAG = pd.Timedelta(minutes=10)
C_SKIP = 0.005


def xdir(cache):
    return X.nx3dir(cache) / "exec1"


# ---------- pure helpers (tested) ----------

def at_time(d, hhmm):
    return pd.Timestamp(d).normalize() + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))


def bar_open_at(bars, t):
    """Open of the first 1-minute bar at or after t (within MAX_LAG), else NaN."""
    if bars is None or len(bars) == 0:
        return np.nan
    b = bars[bars["t"] >= t]
    if b.empty or b["t"].iloc[0] > t + MAX_LAG:
        return np.nan
    return float(b["o"].iloc[0])


def delayed(entry_t, entry, ret, hs_in, d, px, hs_t, hhmm):
    """Arrays (ret, entry half-spread, moved?) for the delay to T: a trade entering before T with a bar at T
    re-enters at px with the quote at T; every other trade is unchanged. ret is on the exit price, which stays."""
    tt = np.array([at_time(x, hhmm) for x in pd.to_datetime(d)], dtype="datetime64[ns]")
    et = pd.to_datetime(entry_t).to_numpy("datetime64[ns]")
    entry, ret, px = np.asarray(entry, float), np.asarray(ret, float), np.asarray(px, float)
    move = (et < tt) & np.isfinite(px) & (px > 0) & np.isfinite(entry)
    r2 = np.where(move, (1 + ret) * entry / np.where(px > 0, px, 1.0) - 1, ret)
    h2 = np.where(move, np.asarray(hs_t, float), np.asarray(hs_in, float))
    return r2, h2, move


def net_moc(side, ret, flat, price, hs_entry):
    """Net with the measured entry half-spread and no exit spread (market-on-close at the official close)."""
    return K.measured_net(np.asarray(side) * np.asarray(ret, float), flat, price, hs_entry,
                          np.zeros(len(np.atleast_1d(hs_entry))))


# ---------- stages ----------

def trades(cache):
    p = pd.read_parquet(M.mdir(cache) / "picks.parquet")
    ev = pd.concat([pd.read_parquet(f, columns=["symbol", "d", "t", "entry_t", "entry"])
                    for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True).drop_duplicates(["symbol", "d", "t"])
    return p.merge(ev, on=["symbol", "d", "t"], how="left")


def stage_bars(cache, deadline):
    p = trades(cache)
    p = p[pd.notna(p["entry_t"])]
    p = p[p["entry_t"] < p["d"].map(lambda x: at_time(x, T_GRID[-1]))]
    ses = X.sessions(X.calendar(cache)).set_index("date")
    h = alpaca_headers()
    out = xdir(cache)
    out.mkdir(parents=True, exist_ok=True)
    for m, gm in p.groupby(p["d"].dt.to_period("M")):
        f = out / f"x_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        rows = []
        for d, g in gm.groupby("d"):
            o = ses.loc[d, "open_t"]
            syms = sorted(set(g["symbol"]))
            bars = X.day_bars(syms, d, o, at_time(d, T_GRID[-1]) + pd.Timedelta(minutes=MAX_LAG.seconds // 60 + 1), h)
            px = {hm: {s: bar_open_at(bars.get(s), at_time(d, hm)) for s in syms} for hm in T_GRID}
            hs = {hm: K.quote_at(syms, at_time(d, hm), h) for hm in T_GRID}
            for r in g.itertuples(index=False):
                rows.append({"symbol": r.symbol, "d": r.d, "t": r.t,
                             **{f"px_{hm}": px[hm][r.symbol] for hm in T_GRID},
                             **{f"hs_{hm}": hs[hm].get(r.symbol, np.nan) for hm in T_GRID}})
        res = pd.DataFrame(rows)
        X.N.atomic_parquet(res, f)
        print(f"[exec1 bars] {m}: {len(res)} early trades, bar at 10:00 {res['px_10:00'].notna().mean():.0%}, "
              f"quote at 10:00 {res['hs_10:00'].notna().mean():.0%}")
    return True


def stage_run(cache, out):
    p = trades(cache)
    qf = sorted(K.cdir(cache).glob("q_*.parquet"))
    xf = sorted(xdir(cache).glob("x_*.parquet"))
    need = set(p.loc[pd.notna(p["entry_t"]), "d"].dt.to_period("M").astype(str))
    if need - {f.stem[2:] for f in qf}:
        raise SystemExit("COST1 quotes missing; run nx3-study stage cost1 first")
    early = p[pd.notna(p["entry_t"]) & (p["entry_t"] < p["d"].map(lambda x: at_time(x, T_GRID[-1])))]
    if set(early["d"].dt.to_period("M").astype(str)) - {f.stem[2:] for f in xf}:
        raise SystemExit("exec1 bars unfinished; run --stage bars again")
    p = p.merge(pd.concat([pd.read_parquet(f) for f in qf], ignore_index=True), on=["symbol", "d", "t"], how="left")
    p = p.merge(pd.concat([pd.read_parquet(f) for f in xf], ignore_index=True), on=["symbol", "d", "t"], how="left")
    p = p[pd.notna(p["entry_t"])].reset_index(drop=True)
    flat = X.N.cost_frac(p["price"], p["side"] < 0, p["days"])
    base = net_moc(p["side"], p["ret"], flat, p["price"], p["hs_in"])
    with_exit = K.measured_net(p["side"] * p["ret"], flat, p["price"], p["hs_in"], p["hs_out"])
    ch, te = p["year"].isin(M.CHOOSE).to_numpy(), p["year"].isin(M.TEST).to_numpy()
    arms = {}
    for hm in T_GRID:
        r2, h2, mv = delayed(p["entry_t"], p["entry"], p["ret"], p["hs_in"], p["d"], p[f"px_{hm}"], p[f"hs_{hm}"], hm)
        arms[hm] = (net_moc(p["side"], r2, flat, p["price"], h2), r2, h2, mv)
    choose = {hm: float(arms[hm][0][ch].mean()) for hm in T_GRID}
    hm = max(T_GRID, key=lambda k: choose[k])
    n2, r2, h2, mv = arms[hm]
    t = p[te].reset_index(drop=True)
    b_t, n_t, mv_t, h_t = base[te], n2[te], mv[te], h2[te]
    w = t["w"].to_numpy(float)
    gain = X.tstat((K.day_sum(t["d"], w * n_t) - K.day_sum(t["d"], w * b_t)).to_numpy(float))
    sr = t["side"].to_numpy() * t["ret"].to_numpy(float)
    sr2 = t["side"].to_numpy() * r2[te]
    pre = (t["entry_t"] <= t["d"].map(lambda x: at_time(x, "09:31"))).to_numpy()
    keep_new = K.keep_mask(h_t, C_SKIP)
    rep = {"rule": "EXEC-10 passes if the sized daily return with the delay minus without it (entry half-spread "
                   "measured, no exit spread) has mean > 0 with NW t >= 2 and the delayed book's own net per trade > 0 "
                   "(2024-01..2025-09)",
           "trades": int(len(p)), "test_trades": int(te.sum()),
           "closing_auction": {"exit_half_spread_mean": float(np.nanmean(p["hs_out"])),
                               "exit_half_spread_median": float(np.nanmedian(p["hs_out"])),
                               "net_per_trade_quote_exit": float(np.mean(with_exit)),
                               "net_per_trade_auction_exit": float(np.mean(base)),
                               "test_quote_exit": float(np.mean(with_exit[te])),
                               "test_auction_exit": float(np.mean(base[te]))},
           "choose_2019_2023": choose, "T": hm,
           "test": {"gain_by_day": gain, "own_net_per_trade": float(n_t.mean()),
                    "base_net_per_trade": float(b_t.mean()),
                    "moved_share": float(mv_t.mean()),
                    "moved": {"trades": int(mv_t.sum()),
                              "gross_before": float(sr[mv_t].mean()) if mv_t.any() else None,
                              "gross_after": float(sr2[mv_t].mean()) if mv_t.any() else None,
                              "half_spread_before": float(np.nanmean(t["hs_in"].to_numpy(float)[mv_t])) if mv_t.any() else None,
                              "half_spread_after": float(np.nanmean(h_t[mv_t])) if mv_t.any() else None},
                    "per_side": {int(s): {"base": float(b_t[t["side"].to_numpy() == s].mean()),
                                          "delayed": float(n_t[t["side"].to_numpy() == s].mean())} for s in (1, -1)},
                    "pre_open_news": {"trades": int(pre.sum()), "base": float(b_t[pre].mean()) if pre.any() else None,
                                      "delayed": float(n_t[pre].mean()) if pre.any() else None},
                    "with_cost1_skip": {"kept_share": float(keep_new.mean()),
                                        "own_net_per_trade": float(n_t[keep_new].mean()) if keep_new.any() else None}}}
    rep["verdict"] = ("passed" if gain["mean"] > 0 and gain["t_nw"] >= 2 and rep["test"]["own_net_per_trade"] > 0
                      else "failed")
    M.write(out, "EXEC1_result.json", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bars", "run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/MODELS5")
    ap.add_argument("--minutes", type=float, default=260)
    a = ap.parse_args()
    cache = Path(a.cache)
    if a.stage == "bars":
        stage_bars(cache, time.monotonic() + 60 * a.minutes)
    else:
        stage_run(cache, a.out)


if __name__ == "__main__":
    main()
