"""COST1 (research/prereg/COST1.md): the quoted spread at each NX3-ALL-HI entry and exit, and skipping the trades
that enter on a wide spread.

    python -m tracks.news.cost1_study --stage quotes|run --cache cache/nx1 --out research/results/MODELS5

quotes: per month of entries, the NBBO as in C0 (core/costs.py) at each trade's entry minute and one minute before
the exit session's close -> cache nx3/cost1/q_<month>.parquet (half-spreads only). Needs models5_study's picks.
run: c chosen on 2019-2023, one look 2024-01..2025-09. Summaries only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive import costs as AC
from core import costs as C
from core.env import alpaca_headers
from tracks.news import models5_study as M
from tracks.news import nx3_study as X

C_GRID = (0.0025, 0.005, 0.01)
CHOOSE = M.CHOOSE
TEST = M.TEST


def cdir(cache):
    return X.nx3dir(cache) / "cost1"


# ---------- pure helpers (tested) ----------

def measured_net(signed_ret, flat_cost, price, hs_in, hs_out):
    """Net with the flat spread swapped for the quoted half-spreads at entry and exit; a trade missing either
    quote keeps the flat cost."""
    flat_spread = np.where(np.asarray(price, float) >= 5, AC.SPREAD_ROUND_TRIP["main"], AC.SPREAD_ROUND_TRIP["penny"])
    hs_in, hs_out = np.asarray(hs_in, float), np.asarray(hs_out, float)
    ok = np.isfinite(hs_in) & np.isfinite(hs_out)
    cost = np.where(ok, np.asarray(flat_cost, float) - flat_spread + hs_in + hs_out, flat_cost)
    return np.asarray(signed_ret, float) - cost


def keep_mask(hs_in, c):
    """COST1's rule: skip a trade whose entry half-spread is above c; no quote = keep."""
    hs_in = np.asarray(hs_in, float)
    return ~(np.isfinite(hs_in) & (hs_in > c))


def quote_at(symbols, t, headers, fetch=C.fetch_quotes):
    """{symbol: half-spread} of the quote in force just before t (2-second window, then 60 seconds), as C0."""
    t = pd.Timestamp(t, tz=X.ET).tz_convert("UTC")
    out, todo = {}, list(symbols)
    for w in C.WINDOWS:
        if not todo:
            break
        got = {}
        for i in range(0, len(todo), 100):
            got.update(fetch(todo[i:i + 100], t - w, t, headers))
        left = []
        for s in todo:
            q = C.as_of(got.get(s, []), t)
            if q is None:
                left.append(s)
            else:
                out[s] = C.half_spread(q)
        todo = left
    return out


# ---------- stages ----------

def stage_quotes(cache, deadline):
    p = pd.read_parquet(M.mdir(cache) / "picks.parquet")
    cols = ["symbol", "d", "t", "entry_t"]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True).drop_duplicates(["symbol", "d", "t"])
    p = p.merge(ev, on=["symbol", "d", "t"], how="left")
    ses = X.sessions(X.calendar(cache)).set_index("date")
    h = alpaca_headers()
    out = cdir(cache)
    for m, gm in p.groupby(p["d"].dt.to_period("M")):
        f = out / f"q_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        hs_in, hs_out = {}, {}
        for et, g in gm.dropna(subset=["entry_t"]).groupby("entry_t"):
            for s, v in quote_at(sorted(set(g["symbol"])), et, h).items():
                hs_in[(s, et)] = v
        for nx, g in gm.groupby("next"):
            for s, v in quote_at(sorted(set(g["symbol"])), ses.loc[nx, "close_t"] - pd.Timedelta(minutes=1), h).items():
                hs_out[(s, nx)] = v
        res = gm[["symbol", "d", "t"]].assign(
            hs_in=[hs_in.get((s, e), np.nan) for s, e in zip(gm["symbol"], gm["entry_t"])],
            hs_out=[hs_out.get((s, n), np.nan) for s, n in zip(gm["symbol"], gm["next"])],
            mins_open=[(e - ses.loc[d, "open_t"]).total_seconds() / 60 if pd.notna(e) else np.nan
                       for e, d in zip(gm["entry_t"], gm["d"])])
        X.N.atomic_parquet(res, f)
        print(f"[cost1 quotes] {m}: {len(res)} trades, entry quote {res['hs_in'].notna().mean():.0%}, "
              f"exit quote {res['hs_out'].notna().mean():.0%}")
    return True


def day_sum(d, x):
    return pd.Series(x).groupby(np.asarray(d)).sum()


def spread_model(cache, p):
    """Reported: LightGBM of the entry half-spread (log) from inputs known before the entry, trained on 2019-2022,
    scored on 2023-2025 by rank correlation."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    ev = pd.concat([pd.read_parquet(f, columns=["symbol", "d", "t", "vol20", "log_dv", "log_px"])
                    for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))], ignore_index=True)
    x = p.merge(ev.drop_duplicates(["symbol", "d", "t"]), on=["symbol", "d", "t"], how="left")
    x = x[np.isfinite(x["hs_in"]) & (x["hs_in"] > 0)]
    cols = ["log_px", "log_dv", "vol20", "mins_open"]
    tr, te = x[x["year"] <= 2022], x[x["year"] >= 2023]
    if len(tr) < 500 or len(te) < 100:
        return None
    m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=7, verbose=-1)
    m.fit(tr[cols], np.log(tr["hs_in"]))
    return {"inputs": cols, "train": int(len(tr)), "test": int(len(te)),
            "spearman_test": float(spearmanr(m.predict(te[cols]), te["hs_in"]).correlation)}


def stage_run(cache, out):
    p = pd.read_parquet(M.mdir(cache) / "picks.parquet")
    files = sorted(cdir(cache).glob("q_*.parquet"))
    missing = set(p["d"].dt.to_period("M").astype(str)) - {f.stem[2:] for f in files}
    if missing:
        raise SystemExit(f"cost1 quotes unfinished: {len(missing)} months missing; run --stage quotes again")
    p = p.merge(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), on=["symbol", "d", "t"], how="left")
    flat = X.N.cost_frac(p["price"], p["side"] < 0, p["days"])
    p["net_m"] = measured_net(p["side"] * p["ret"], flat, p["price"], p["hs_in"], p["hs_out"])
    p["rt"] = p["hs_in"] + p["hs_out"]
    ch, te = p["year"].isin(CHOOSE).to_numpy(), p["year"].isin(TEST).to_numpy()
    choose = {str(c): float(p.loc[ch & keep_mask(p["hs_in"], c), "net_m"].mean()) for c in C_GRID}
    c = float(max(C_GRID, key=lambda k: choose[str(k)]))
    t = p[te].reset_index(drop=True)
    keep = keep_mask(t["hs_in"], c)
    all_day = day_sum(t["d"], t["w"] * t["net_m"])
    kept_day = day_sum(t["d"], np.where(keep, t["w"] * t["net_m"], 0.0))
    gain = X.tstat((kept_day - all_day).to_numpy(float))
    tier = np.where(p["price"] < 5, "penny", np.where(p["price"] < 20, "5-20", "20+"))
    when = pd.cut(p["mins_open"], [-1e9, 0.5, 30, 1e9], labels=["at open", "first 30 min", "later"])
    rep = {"rule": "COST1 passes if the sized daily return with the skip rule minus without it (measured costs) has mean > 0 "
                   "with NW t >= 2 and the rule's own net per trade > 0 (2024-01..2025-09)",
           "trades": int(len(p)), "entry_quote_share": float(p["hs_in"].notna().mean()),
           "exit_quote_share": float(p["hs_out"].notna().mean()),
           "measured_round_trip": {"median": float(p["rt"].median()), "mean": float(p["rt"].mean()),
                                   "p75": float(p["rt"].quantile(0.75)), "flat_main": AC.SPREAD_ROUND_TRIP["main"],
                                   "by_price": {k: float(g.median()) for k, g in p.groupby(tier)["rt"]},
                                   "by_time": {str(k): float(g.median()) for k, g in p.groupby(when, observed=True)["rt"]}},
           "net_per_trade_2019_2025": {"flat": float(p["net"].mean()), "measured": float(p["net_m"].mean())},
           "choose_2019_2023": choose, "c": c,
           "test": {"trades": int(len(t)), "skipped_share": float((~keep).mean()), "gain_by_day": gain,
                    "own_net_per_trade": float(t.loc[keep, "net_m"].mean()),
                    "all_net_per_trade": float(t["net_m"].mean()),
                    "skipped_net_per_trade": float(t.loc[~keep, "net_m"].mean()) if (~keep).any() else None,
                    "per_side_kept": {int(s): float(t.loc[keep & (t["side"] == s).to_numpy(), "net_m"].mean()) for s in (1, -1)}},
           "spread_model": spread_model(cache, p)}
    rep["verdict"] = ("passed" if gain["mean"] > 0 and gain["t_nw"] >= 2 and rep["test"]["own_net_per_trade"] > 0
                      else "failed")
    M.write(out, "COST1_result.json", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["quotes", "run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/MODELS5")
    ap.add_argument("--minutes", type=float, default=260)
    a = ap.parse_args()
    cache = Path(a.cache)
    if a.stage == "quotes":
        stage_quotes(cache, time.monotonic() + 60 * a.minutes)
    else:
        stage_run(cache, a.out)


if __name__ == "__main__":
    main()
