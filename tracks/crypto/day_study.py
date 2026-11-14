"""CRY-DAY: a day-trading model on the most traded USDT perpetuals (research/prereg/CRYPTO.md).

Stage rows: per coin, every 15-minute decision row with its inputs and the net result of a long and of a short
            (stop 1.5 swings, target 3 swings, 16 bars; 0.12% costs plus funding) -> <cache>/crypto/rows/<SYM>.parquet
Stage fit:  train on 2020-04..2023-12, choose learner and bar on 2024, refit on ..2024-12. Never reads 2025.
            -> <out>/CRYDAY_fit.json and models
Stage test: the one test on 2025-01-01..2025-09-30, only if the 2024 gate passed -> <out>/CRYDAY_test.json

Usage: python -m tracks.crypto.day_study --stage rows|fit|test --cache cache --out research/results/CRYDAY
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats as CS
from tracks.crypto import cdata as C
from tracks.main import pick2_study as P2
from tracks.main import pick_study as S
from tracks.main.longterm.stats import nw_tstat

BAR = pd.Timedelta(minutes=15)
H = 16                                    # bars a trade may last
STOP_K, TP_K = 1.5, 3.0                   # in swings
COST = 0.0012                             # 0.05% taker a side + 0.02% slippage
COST_SPOT = 0.005                         # the long-only Alpaca scenario
SWING_WIN, HOUR = 672, 4                  # 7 days of bars; bars in an hour
FIRST = pd.Timestamp("2020-04-01")
VAL0, TEST0, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-10-01")
PURGE = H * BAR
BAR_Q = (0.01, 0.005, 0.002, 0.001)
MAX_OPEN, MIN_VAL, MIN_TEST = 3, 200, 100
TRIALS, SEED, LAGS = 177, 0, 5
OWN = ["r_1", "r_4", "r_16", "r_96", "r_672", "vol_16", "vol_96", "pos_96", "pace", "n_pace", "tb_4", "tb_16",
       "swing", "fund", "fund_mins", "hour", "dow"]
MKT = ["btc_r_1", "btc_r_4", "btc_r_16", "btc_r_96", "btc_vol_16"]
FEATS = OWN + MKT + ["rel_16", "breadth"]
SIDES = ("L", "S")


# ---------- pure helpers (tested) ----------

def regular(df):
    """Candles on a complete 15-minute grid; a missing bar is flat at the last close with no volume (`filled`)."""
    g = pd.date_range(df["t"].iat[0], df["t"].iat[-1], freq=BAR)
    out = df.set_index("t").reindex(g)
    out["filled"] = out["c"].isna()
    out["c"] = out["c"].ffill()
    for c in ("o", "h", "l"):
        out[c] = out[c].fillna(out["c"])
    for c in ("v", "qv", "n", "tbqv"):
        out[c] = out[c].fillna(0.0)
    return out.rename_axis("t").reset_index()


def swing_of(h, lo, c):
    """Median 1-hour range (as a share of the price) over the last 7 days, known at each bar's close."""
    rng = (pd.Series(h).rolling(HOUR).max() - pd.Series(lo).rolling(HOUR).min()) / pd.Series(c)
    return rng.rolling(SWING_WIN, min_periods=SWING_WIN // 2).median().to_numpy()


def simulate(o, h, lo, c, swing, side):
    """For a decision at every bar i: enter at the open of bar i+1, exit at the stop, the target or the close of
    the 16th bar; a bar that touches both counts as the stop. -> (gross return, exit kind, exit bar index)."""
    n = len(c)
    i = np.arange(n)
    e = i + 1
    ok = (e + H - 1 < n) & np.isfinite(swing) & (swing > 0)
    ee = np.where(ok, e, 0)
    entry = o[ee]
    sgn = 1.0 if side == "L" else -1.0
    stop = entry * (1 - sgn * STOP_K * np.where(ok, swing, 0))
    tp = entry * (1 + sgn * TP_K * np.where(ok, swing, 0))
    px = np.full(n, np.nan)
    kind = np.full(n, "", dtype=object)
    xi = np.full(n, -1)
    alive = ok.copy()
    for k in range(H):
        j = np.minimum(ee + k, n - 1)
        oj, hj, lj = o[j], h[j], lo[j]
        if side == "L":
            hit_stop_open, hit_stop = (oj <= stop) & (k > 0), lj <= stop
            hit_tp_open, hit_tp = (oj >= tp) & (k > 0), hj >= tp
        else:
            hit_stop_open, hit_stop = (oj >= stop) & (k > 0), hj >= stop
            hit_tp_open, hit_tp = (oj <= tp) & (k > 0), lj <= tp
        for mask, price, name in ((hit_stop_open, oj, "stop"), (hit_stop, stop, "stop"),
                                  (hit_tp_open, oj, "tp"), (hit_tp, tp, "tp")):
            m = alive & mask
            px[m], xi[m] = price[m], j[m]
            kind[m] = name
            alive &= ~m
    j = np.minimum(ee + H - 1, n - 1)
    px[alive], xi[alive] = c[j][alive], j[alive]
    kind[alive] = "time"
    gross = np.where(ok, sgn * (px / entry - 1), np.nan)
    return gross, kind, xi


def trade_ref(o, h, lo, c, i, swing, side):
    """The same trade, one row at a time (a slow, separate routine used to check `simulate`)."""
    e = i + 1
    entry, sgn = o[e], (1.0 if side == "L" else -1.0)
    stop, tp = entry * (1 - sgn * STOP_K * swing), entry * (1 + sgn * TP_K * swing)
    for j in range(e, e + H):
        lo_side, hi_side = (lo[j], h[j]) if side == "L" else (-h[j], -lo[j])
        s, t_, oo = sgn * stop, sgn * tp, sgn * o[j]
        if j > e and oo <= s:
            return sgn * (o[j] / entry - 1), "stop", j
        if lo_side <= s:
            return sgn * (stop / entry - 1), "stop", j
        if j > e and oo >= t_:
            return sgn * (o[j] / entry - 1), "tp", j
        if hi_side >= t_:
            return sgn * (tp / entry - 1), "tp", j
    return sgn * (c[e + H - 1] / entry - 1), "time", e + H - 1


def funding_cum(t, fund):
    """Cumulative funding rate by bar: a payment at time tau lands on the bar that opens at tau."""
    f = pd.Series(fund["rate"].to_numpy(), index=pd.DatetimeIndex(fund["t"])) if len(fund) else pd.Series(dtype=float)
    return f.reindex(pd.DatetimeIndex(t)).fillna(0.0).cumsum().to_numpy()


def features(df, fund):
    c, lr = df["c"], np.log(df["c"]).diff()
    out = pd.DataFrame({"t": df["t"]})
    for k in (1, 4, 16, 96, 672):
        out[f"r_{k}"] = c / c.shift(k) - 1
    out["vol_16"], out["vol_96"] = lr.rolling(16).std(), lr.rolling(96).std()
    hi, lw = df["h"].rolling(96).max(), df["l"].rolling(96).min()
    out["pos_96"] = (c - lw) / (hi - lw).replace(0, np.nan)
    for name, col in (("pace", "qv"), ("n_pace", "n")):
        out[name] = df[col].rolling(HOUR).sum() / (df[col].rolling(SWING_WIN).mean() * HOUR).replace(0, np.nan)
    for k in (4, 16):
        out[f"tb_{k}"] = df["tbqv"].rolling(k).sum() / df["qv"].rolling(k).sum().replace(0, np.nan)
    out["swing"] = swing_of(df["h"].to_numpy(), df["l"].to_numpy(), c.to_numpy())
    close_t = pd.DatetimeIndex(df["t"]) + BAR                     # what is known at the bar's close
    if len(fund):
        k = np.searchsorted(fund["t"].to_numpy(), close_t.to_numpy(), side="right") - 1
        has = k >= 0
        out["fund"] = np.where(has, fund["rate"].to_numpy()[np.clip(k, 0, None)], np.nan)
        last = fund["t"].to_numpy()[np.clip(k, 0, None)]
        out["fund_mins"] = np.where(has, (close_t.to_numpy() - last) / np.timedelta64(1, "m"), np.nan)
    else:
        out["fund"], out["fund_mins"] = np.nan, np.nan
    out["hour"] = close_t.hour + close_t.minute / 60
    out["dow"] = close_t.dayofweek
    return out


def coin_rows(df, fund, quarters):
    """One coin's candles -> its decision rows (only bars of quarters where the coin is in the universe)."""
    df = regular(df)
    X = features(df, fund)
    o, h, lo, c = (df[x].to_numpy(float) for x in ("o", "h", "l", "c"))
    cf = funding_cum(df["t"], fund)
    n = len(df)
    e = np.minimum(np.arange(n) + 1, n - 1)
    for side in SIDES:
        g, kind, xi = simulate(o, h, lo, c, X["swing"].to_numpy(), side)
        paid = cf[np.clip(xi, 0, None)] - cf[e]                    # paid by a long, received by a short
        X[f"gross_{side}"] = g
        X[f"net_{side}"] = g - COST - (paid if side == "L" else -paid)
        X[f"exit_{side}"] = kind
        X[f"bars_{side}"] = np.where(xi >= 0, xi - e + 1, -1)
    X["entry"] = o[e]
    keep = (np.isin(C.quarter_of(df["t"]), list(quarters)) & ~df["filled"].to_numpy() & (df["t"] >= FIRST).to_numpy()
            & np.isfinite(X["net_L"].to_numpy()))
    return X[keep].reset_index(drop=True)


def take(cands):
    """cands: rows (t, symbol, side, pred, bars) sorted by t then pred (highest first) -> index of the trades the
    book takes: a coin holds one position, the book at most MAX_OPEN; a position frees its slot when it exits."""
    open_, got = {}, []
    for r in cands.itertuples():
        enter = r.t + BAR
        for s in [s for s, rel in open_.items() if rel <= enter]:
            del open_[s]
        if r.symbol in open_ or len(open_) >= MAX_OPEN:
            continue
        open_[r.symbol] = enter + r.bars * BAR
        got.append(r.Index)
    return got


def candidates(R, pred, thr, long_only=False):
    """Rows whose better side clears the bar -> one candidate each, in the book's order."""
    pl, ps = pred["L"], (np.full(len(R), -np.inf) if long_only else pred["S"])
    side = np.where(pl >= ps, "L", "S")
    best = np.maximum(pl, ps)
    m = best >= thr
    c = R.loc[m, ["t", "symbol"]].assign(side=side[m], pred=best[m])
    for col in ("net", "gross", "exit", "bars"):
        c[col] = np.where(c["side"] == "L", R.loc[m, f"{col}_L"], R.loc[m, f"{col}_S"])
    return c.sort_values(["t", "pred"], ascending=[True, False], kind="stable")


def add_control(tk, R, rng):
    """For each trade: the net of a random other universe coin on a random side at the same bar."""
    by_t = {t: g for t, g in R[R["t"].isin(set(tk["t"]))].groupby("t")}
    out = []
    for r in tk.itertuples():
        pool = by_t[r.t]
        pool = pool[pool["symbol"] != r.symbol].sort_values("symbol")
        if not len(pool):
            out.append(np.nan)
            continue
        row = pool.iloc[int(rng.integers(len(pool)))]
        out.append(float(row["net_L"] if rng.integers(2) else row["net_S"]))
    return tk.assign(ctrl=out)


def daily(tk, col, days):
    """Daily book return: each position is a third of the book; days without a trade count as 0."""
    d = pd.to_datetime(tk["t"] + BAR).dt.normalize()
    return (tk[col] / MAX_OPEN).groupby(d).sum().reindex(days, fill_value=0.0)


def summary(tk, days, btc=None):
    n = len(tk)
    if not n:
        return {"trades": 0, "mean_net": float("nan")}
    book = daily(tk, "net", days)
    m, se, t = nw_tstat(book.to_numpy(), LAGS)
    out = {"trades": n, "mean_net": float(tk["net"].mean()), "median_net": float(tk["net"].median()),
           "win_share": float((tk["net"] > 0).mean()), "book_daily": {"mean": m, "se_nw": se, "t_nw": t},
           "book_total": float(book.sum()), "days_with_trades": int((book != 0).sum()),
           "exits": {k: float(v) for k, v in tk["exit"].value_counts(normalize=True).items()},
           "by_side": {k: {"n": len(g), "mean_net": float(g["net"].mean())} for k, g in tk.groupby("side")},
           "by_coin": {k: {"n": len(g), "mean_net": float(g["net"].mean())} for k, g in tk.groupby("symbol")}}
    if "ctrl" in tk:
        diff = daily(tk.assign(d=tk["net"] - tk["ctrl"]).dropna(subset=["d"]), "d", days)
        dm, dse, dt_ = nw_tstat(diff.to_numpy(), LAGS)
        out["minus_control_daily"] = {"mean": dm, "se_nw": dse, "t_nw": dt_}
        out["mean_ctrl_net"] = float(tk["ctrl"].mean())
    if btc is not None:
        a = CS.ols_hac(book.to_numpy(), btc.reindex(days).to_numpy(), LAGS)
        out["alpha_on_btc"] = {"alpha": a["alpha"], "t": a["t"], "beta": a["betas"][0]}
        out["dsr"] = CS.deflated_sharpe(book.to_numpy(), TRIALS)
    return out


def choose(table):
    ok = [(L, q, s) for L, by_q in table.items() for q, s in by_q.items() if s["trades"] >= MIN_VAL]
    if not ok:
        return None
    L, q, s = max(ok, key=lambda x: x[2]["mean_net"])
    return {"learner": L, "q": q, "trades": s["trades"], "mean_net": s["mean_net"]}


def verdict(s):
    """The registered primary test on the 2025 summary."""
    rules = {"mean_net_above_0": bool(s["mean_net"] > 0), "t_3": bool(s["book_daily"]["t_nw"] >= 3),
             "trades_100": bool(s["trades"] >= MIN_TEST),
             "beats_control_t_2": bool(s["minus_control_daily"]["mean"] > 0 and s["minus_control_daily"]["t_nw"] >= 2),
             "alpha_on_btc_t_2": bool(s["alpha_on_btc"]["alpha"] > 0 and s["alpha_on_btc"]["t"] >= 2)}
    return rules, ("PASS" if all(rules.values()) else "FAIL")


# ---------- stages ----------

def read_bars(cache, sym):
    return pd.read_parquet(C.cdir(cache) / "um15" / f"{sym}.parquet")


def stage_rows(cache):
    uni = json.loads((C.cdir(cache) / "universe.json").read_text())
    quarters = {}
    for q, syms in uni.items():
        for s in syms:
            quarters.setdefault(s, set()).add(q)
    btc = features(regular(read_bars(cache, "BTCUSDT")), pd.DataFrame({"t": pd.to_datetime([]), "rate": []}))
    btc = btc[["t", "r_1", "r_4", "r_16", "r_96", "vol_16"]].rename(columns=lambda x: x if x == "t" else f"btc_{x}")
    for s in sorted(quarters):
        f = C.cdir(cache) / "rows" / f"{s}.parquet"
        if f.exists():
            continue
        R = coin_rows(read_bars(cache, s), pd.read_parquet(C.cdir(cache) / "funding" / f"{s}.parquet"), quarters[s])
        R = R.merge(btc, on="t", how="left")
        R["rel_16"] = R["r_16"] - R["btc_r_16"]
        R.insert(0, "symbol", s)
        C.atomic(R, f)
        print(f"[cry-day rows] {s}: {len(R)} rows", flush=True)


def load(cache, lo, hi):
    """Rows with lo <= t < hi, with the universe's average 4-hour return added."""
    parts = []
    for f in sorted((C.cdir(cache) / "rows").glob("*.parquet")):
        parts.append(pd.read_parquet(f, filters=[("t", ">=", lo), ("t", "<", hi)]))
    R = pd.concat(parts, ignore_index=True)
    assert R["t"].max() < hi and R["t"].min() >= lo
    R["breadth"] = R.groupby("t")["r_16"].transform("mean")
    return R.sort_values(["t", "symbol"]).reset_index(drop=True)


def fit_sides(L, R, mask):
    X, sw = R[FEATS].to_numpy(np.float32), R["swing"].to_numpy(float)
    models = {}
    for side in SIDES:
        y = P2.target(R[f"net_{side}"].to_numpy(), sw)
        k = mask & np.isfinite(y)
        m = P2.make(L)
        m.fit(X[k], y[k])
        models[side] = m
    return models


def predict_sides(models, R):
    X, sw = R[FEATS].to_numpy(np.float32), R["swing"].to_numpy(float)
    return {side: P2.predict(models[side], X) * sw for side in SIDES}


def bars_of(pred, mask):
    pooled = np.concatenate([pred[s][mask] for s in SIDES])
    return {str(q): S.top_threshold(pooled, q) for q in BAR_Q}


def days_of(lo, hi):
    return pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D")


def stage_fit(cache, out):
    out = Path(out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    R = load(cache, FIRST, TEST0)
    assert R["t"].max() < TEST0, "2025 rows reached the fit stage"
    t = R["t"]
    tr = (t < VAL0 - PURGE).to_numpy()
    va = ((t >= VAL0) & (t < TEST0 - PURGE)).to_numpy()
    days = days_of(VAL0, TEST0)
    table, thr = {}, {}
    for L in P2.LEARNERS:
        pred = predict_sides(fit_sides(L, R, tr), R)
        thr[L] = bars_of(pred, tr)
        Rv, pv = R[va], {s: pred[s][va] for s in SIDES}
        table[L] = {}
        for q, b in thr[L].items():
            c = candidates(Rv, pv, b)
            table[L][q] = summary(c.loc[take(c)], days)
        print(f"[cry-day fit] {L}: " + ", ".join(f"{q}: {v['trades']}" for q, v in table[L].items()), flush=True)
    ch = choose(table)
    res = {"train": "2020-04..2023-12", "choose": "2024", "refit": "2020-04..2024-12", "features": FEATS,
           "rows": {"train": int(tr.sum()), "choose": int(va.sum())}, "bars_2020_2023": thr, "table_2024": table,
           "chosen": ch, "gate": "failed at validation"}
    if ch and ch["mean_net"] > 0:
        full = (t < TEST0 - PURGE).to_numpy()
        models = fit_sides(ch["learner"], R, full)
        paths = {s: S.save_model(ch["learner"], models[s], str(out / "models" / f"CRYDAY_{s}")) for s in SIDES}
        res["gate"] = "passed: run the 2025 test"
        res["refit"] = {"models": paths, "bar": bars_of(predict_sides(models, R), full)[ch["q"]]}
    S.atomic_json(res, out / "CRYDAY_fit.json")
    print(f"[cry-day fit] chosen {ch}; gate: {res['gate']}", flush=True)
    return res


def btc_daily(cache):
    b = read_bars(cache, "BTCUSDT").set_index("t")["c"].resample("D").last()
    return b.pct_change()


def hand_check(cache, tk, n=20):
    """Recompute a random sample of trades from the raw candles with the slow routine."""
    rows = []
    for r in tk.sample(min(n, len(tk)), random_state=SEED).itertuples():
        df = regular(read_bars(cache, r.symbol))
        i = int(np.flatnonzero(df["t"].to_numpy() == np.datetime64(r.t))[0])
        sw = swing_of(df["h"].to_numpy(), df["l"].to_numpy(), df["c"].to_numpy())[i]
        g, kind, _ = trade_ref(*(df[x].to_numpy(float) for x in ("o", "h", "l", "c")), i, sw, r.side)
        rows.append({"symbol": r.symbol, "t": str(r.t), "side": r.side, "gross": float(r.gross), "gross_recomputed":
                     float(g), "exit": r.exit, "exit_recomputed": kind, "match": bool(abs(g - r.gross) < 1e-9
                                                                                    and kind == r.exit)})
    return rows


def stage_test(cache, out):
    out = Path(out)
    fit = json.loads((out / "CRYDAY_fit.json").read_text())
    if not fit["gate"].startswith("passed"):
        raise SystemExit("the 2024 gate did not pass; 2025 stays unread")
    if (out / "CRYDAY_test.json").exists():
        raise SystemExit("the one test has already been run")
    R = load(cache, TEST0, END)
    models = {s: P2.load_model(fit["refit"]["models"][s]) for s in SIDES}
    pred = predict_sides(models, R)
    days, btc = days_of(TEST0, END), btc_daily(cache)
    c = candidates(R, pred, fit["refit"]["bar"])
    tk = add_control(c.loc[take(c)], R, np.random.default_rng(SEED))
    s = summary(tk, days, btc)
    rules, v = verdict(s) if s["trades"] else ({}, "FAIL")
    lo = candidates(R, pred, fit["refit"]["bar"], long_only=True)
    lo = lo.loc[take(lo)]
    lo = lo.assign(net=lo["gross"] - COST_SPOT)
    checks = hand_check(cache, tk) if len(tk) else []
    res = {"test": "2025-01-01..2025-09-30", "chosen": fit["chosen"], "bar": fit["refit"]["bar"], "primary": s,
           "rules": rules, "verdict": v, "long_only_at_0.5pct": summary(lo, days), "idea_count": TRIALS,
           "hand_check": {"n": len(checks), "all_match": all(x["match"] for x in checks), "trades": checks}}
    S.atomic_json(res, out / "CRYDAY_test.json")
    print(f"[cry-day test] {v}: {s['trades']} trades, mean net {s['mean_net']:.5f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rows", "fit", "test"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="research/results/CRYDAY")
    a = ap.parse_args()
    if a.stage == "rows":
        stage_rows(a.cache)
    elif a.stage == "fit":
        stage_fit(a.cache, a.out)
    else:
        stage_test(a.cache, a.out)
    print(f"[cry-day {a.stage}] done", flush=True)


if __name__ == "__main__":
    main()
