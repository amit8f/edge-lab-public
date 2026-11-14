"""CRY-XS (research/prereg/CRYXS.md): a daily model that ranks the 30 most traded USDT perpetuals; long the top K,
short the bottom K, one day at a time.

Stages (cache under <cache>/crypto):
  data  daily candles of every USDT perpetual -> um1d/<SYM>.parquet; the monthly universe -> xs_universe.json;
        funding of the universe coins -> funding/<SYM>.parquet; the decision rows -> xs_rows.parquet
  fit   fit on 2020-07..2023-12, choose learner and K on 2024, refit -> CRYXS_fit.json + models
  test  the one test on 2025-01..09 -> CRYXS_test.json

Usage: python -m tracks.crypto.xs_study --stage data|fit|test --cache cache --out research/results/CRYXS
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats as CS
from tracks.crypto import cdata as C
from tracks.main import pick2_study as P2
from tracks.main import pick_study as S
from tracks.main.longterm.stats import nw_tstat

TOP_N, MIN_AGE = 30, 60
KS = (3, 5)
COST_SIDE, COST_SPOT_SIDE = 0.0006, 0.0025
CLIP, SCALE = 0.15, 100.0
FIRST, VAL0, TEST0, END = (pd.Timestamp(x) for x in ("2020-07-01", "2024-01-01", "2025-01-01", "2025-10-01"))
PURGE = pd.Timedelta(days=2)
LEARNERS = [*P2.LEARNERS, "Ridge"]
TRIALS, SEED, LAGS = 179, 0, 5
OWN = ["r_1", "r_3", "r_7", "r_14", "r_28", "r_60", "vol_7", "vol_28", "pos_28", "vol_ratio", "tb_1", "tb_7",
       "fund_1", "fund_7", "age"]
FEATS = OWN + [f"rk_{c}" for c in OWN] + ["btc_r_1", "btc_r_7", "btc_r_28", "breadth"]


# ---------- pure helpers (tested) ----------

def daily_grid(df):
    """Daily candles on a complete calendar; a missing day keeps the last close and is flagged `filled`."""
    g = pd.date_range(df["t"].iat[0], df["t"].iat[-1], freq="D")
    out = df.set_index("t").reindex(g)
    out["filled"] = out["c"].isna()
    out["c"] = out["c"].ffill()
    for c in ("o", "h", "l"):
        out[c] = out[c].fillna(out["c"])
    for c in ("qv", "tbqv"):
        out[c] = out[c].fillna(0.0)
    return out.rename_axis("t").reset_index()


def fund_by_day(fund):
    """Funding summed by the UTC day it is paid in. A payment at exactly 00:00 belongs to the day that just ended
    (a position held over that day pays it)."""
    if not len(fund):
        return pd.Series(dtype=float)
    d = (pd.DatetimeIndex(fund["t"]) - pd.Timedelta(seconds=1)).normalize()
    return pd.Series(fund["rate"].to_numpy(), index=d).groupby(level=0).sum()


def coin_rows(df, fund):
    """One coin's daily candles -> a row per decision day d with its inputs (known at d's close), the return from
    the open of d+1 to the open of d+2 (`fwd`) and the funding paid over that day (`fund_fwd`)."""
    df = daily_grid(df)
    c, lr = df["c"], np.log(df["c"]).diff()
    X = pd.DataFrame({"t": df["t"]})
    for k in (1, 3, 7, 14, 28, 60):
        X[f"r_{k}"] = c / c.shift(k) - 1
    X["vol_7"], X["vol_28"] = lr.rolling(7).std(), lr.rolling(28).std()
    hi, lo = df["h"].rolling(28).max(), df["l"].rolling(28).min()
    X["pos_28"] = (c - lo) / (hi - lo).replace(0, np.nan)
    X["vol_ratio"] = df["qv"] / df["qv"].rolling(28).mean().replace(0, np.nan)
    X["tb_1"] = df["tbqv"] / df["qv"].replace(0, np.nan)
    X["tb_7"] = df["tbqv"].rolling(7).sum() / df["qv"].rolling(7).sum().replace(0, np.nan)
    f = fund_by_day(fund).reindex(pd.DatetimeIndex(df["t"]))
    X["fund_1"] = f.to_numpy()
    X["fund_7"] = f.rolling(7, min_periods=1).mean().to_numpy()
    X["age"] = np.minimum(np.arange(len(df)), 365)
    X["qv"] = df["qv"]
    o = df["o"]
    X["fwd"] = o.shift(-2) / o.shift(-1) - 1
    X["fund_fwd"] = f.shift(-1).fillna(0.0).to_numpy()
    bad = df["filled"] | df["filled"].shift(-1, fill_value=True) | df["filled"].shift(-2, fill_value=True)
    X.loc[bad.to_numpy(), "fwd"] = np.nan
    return X


def universe(vol, age, n=TOP_N):
    """{month: [symbols]}: the n eligible coins with the most quote volume in the month before, listed at least
    MIN_AGE days at the month's start. vol: frame of monthly quote volume (index month, columns symbols);
    age: {symbol: first day}."""
    out = {}
    for m in vol.index[1:]:
        prev = vol.loc[m - 1]
        start = m.start_time
        ok = {s: v for s, v in prev.items() if v > 0 and (start - age[s]).days >= MIN_AGE}
        out[str(m)] = C.top_symbols(ok, n)
    return out


def assemble(rows, uni):
    """Coin rows -> the panel of universe rows with ranks, market inputs and the target. rows: {symbol: frame}."""
    parts = []
    for s, X in rows.items():
        months = {m for m, syms in uni.items() if s in syms}
        k = X[pd.PeriodIndex(X["t"], freq="M").astype(str).isin(months)]
        parts.append(k.assign(symbol=s))
    R = pd.concat(parts, ignore_index=True)
    R = R[np.isfinite(R["fwd"]) & (R["t"] >= FIRST)].sort_values(["t", "symbol"]).reset_index(drop=True)
    g = R.groupby("t")
    for c in OWN:
        R[f"rk_{c}"] = g[c].rank(pct=True) - 0.5
    R["breadth"] = g["r_1"].transform("mean")
    btc = rows["BTCUSDT"].set_index("t")[["r_1", "r_7", "r_28"]].add_prefix("btc_")
    R = R.join(btc, on="t")
    R["rel"] = R["fwd"] - g["fwd"].transform("mean")
    return R


def target(R):
    return np.clip(R["rel"].to_numpy(), -CLIP, CLIP) * SCALE


def book(R, pred, k, cost=COST_SIDE, side="both", seed=None):
    """Daily book returns. Each day: long the k highest predictions and short the k lowest, equal weights, half
    the book a side (`side` L or S keeps one side at full weight). With `seed` the coins are drawn at random
    instead. -> frame indexed by entry day: gross, fund, turnover, net (turnover x cost charged)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, []
    for t, g in R.assign(pred=pred).groupby("t", sort=True):
        w = {}
        if len(g) >= 2 * k:
            order = g.sort_values(["pred", "symbol"], ascending=[False, True])
            if rng is not None:
                order = g.sort_values("symbol").iloc[rng.permutation(len(g))]
            per = 1.0 / k if side != "both" else 0.5 / k
            if side in ("both", "L"):
                w.update(dict.fromkeys(order["symbol"].iloc[:k], per))
            if side in ("both", "S"):
                w.update(dict.fromkeys(order["symbol"].iloc[-k:], -per))
        gs = g.set_index("symbol")
        gross = sum(v * gs.at[s, "fwd"] for s, v in w.items())
        fund = -sum(v * gs.at[s, "fund_fwd"] for s, v in w.items())
        turn = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in set(w) | set(prev))
        out.append((t + pd.Timedelta(days=1), gross, fund, turn))
        prev = w
    B = pd.DataFrame(out, columns=["day", "gross", "fund", "turnover"]).set_index("day")
    B["net"] = B["gross"] + B["fund"] - B["turnover"] * cost
    return B


def sharpe(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(365)) if len(x) > 1 and sd > 0 else float("nan")


def summary(B, ctrl=None, btc=None):
    x = B["net"].to_numpy()
    m, se, t = nw_tstat(x, LAGS)
    out = {"days": len(B), "mean_daily_net": m, "se_nw": se, "t_nw": t, "sharpe": sharpe(x), "total_net": float(x.sum()),
           "mean_daily_gross": float(B["gross"].mean()), "mean_daily_funding": float(B["fund"].mean()),
           "mean_daily_turnover": float(B["turnover"].mean()),
           "by_month": {str(k): float(v) for k, v in B["net"].groupby(B.index.to_period("M")).sum().items()}}
    if ctrl is not None:
        d = (B["gross"] + B["fund"] - ctrl["gross"] - ctrl["fund"]).to_numpy()
        dm, dse, dt_ = nw_tstat(d, LAGS)
        out["minus_control"] = {"mean": dm, "se_nw": dse, "t_nw": dt_}
    if btc is not None:
        a = CS.ols_hac(x, btc.reindex(B.index).to_numpy(), LAGS)
        out["alpha_on_btc"] = {"alpha": a["alpha"], "t": a["t"], "beta": a["betas"][0]}
        out["dsr"] = CS.deflated_sharpe(x, TRIALS)
    return out


def choose(table):
    """{learner: {K: summary}} -> the version with the highest daily net Sharpe."""
    ok = [(L, k, s) for L, by in table.items() for k, s in by.items() if np.isfinite(s["sharpe"])]
    if not ok:
        return None
    L, k, s = max(ok, key=lambda x: x[2]["sharpe"])
    return {"learner": L, "k": int(k), "sharpe": s["sharpe"], "mean_daily_net": s["mean_daily_net"]}


def verdict(s):
    mc, al = s["minus_control"], s["alpha_on_btc"]
    rules = {"mean_net_above_0": bool(s["mean_daily_net"] > 0), "t_3": bool(s["t_nw"] >= 3),
             "beats_control_t_2": bool(mc["mean"] > 0 and mc["t_nw"] >= 2),
             "alpha_on_btc_t_2": bool(al["alpha"] > 0 and al["t"] >= 2)}
    return rules, ("PASS" if all(rules.values()) else "FAIL")


def make(name):
    if name != "Ridge":
        return P2.make(name)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def fit(name, R, mask):
    y = target(R)
    m = make(name)
    m.fit(R.loc[mask, FEATS].to_numpy(np.float32), y[mask])
    return m


def predict(m, R):
    return P2.predict(m, R[FEATS].to_numpy(np.float32))


# ---------- stages ----------

def stage_data(cache):
    d = C.cdir(cache)
    syms = C.list_symbols()
    print(f"[cry-xs data] {len(syms)} candidate symbols", flush=True)
    for s in syms:
        try:
            C.download("futures/um", s, "1d", C.FIRST, d / "um1d" / f"{s}.parquet")
        except RuntimeError:
            pass                                    # listed after the study period
    bars = {f.stem: pd.read_parquet(f) for f in sorted((d / "um1d").glob("*.parquet"))}
    vol = pd.DataFrame({s: b.groupby(pd.PeriodIndex(b["t"], freq="M"))["qv"].sum() for s, b in bars.items()}).fillna(0.0)
    uni = universe(vol, {s: b["t"].iat[0] for s, b in bars.items()})
    S.atomic_json(uni, d / "xs_universe.json")
    need = sorted({s for v in uni.values() for s in v} | {"BTCUSDT"})
    print(f"[cry-xs data] {len(uni)} months, {len(need)} coins ever in the universe", flush=True)
    rows = {}
    for s in need:
        f = d / "funding" / f"{s}.parquet"
        if not f.exists():
            with ThreadPoolExecutor(C.THREADS) as ex:
                raws = list(ex.map(lambda m, s=s: C.fetch(
                    f"{C.BASE}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"), C.months(C.FIRST, C.LAST)))
            parts = [C.parse_funding(r) for r in raws if r]
            fr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"t": pd.to_datetime([]), "rate": []})
            C.atomic(fr.sort_values("t").drop_duplicates("t").reset_index(drop=True), f)
        rows[s] = coin_rows(bars[s], pd.read_parquet(f))
    R = assemble(rows, uni)
    assert R["t"].max() < END - pd.Timedelta(days=2), "holdout rows"
    C.atomic(R, d / "xs_rows.parquet")
    print(f"[cry-xs data] {len(R)} rows, {R['t'].min().date()} .. {R['t'].max().date()}, "
          f"{R.groupby('t').size().median():.0f} coins on a median day", flush=True)


def btc_daily(cache):
    b = pd.read_parquet(C.cdir(cache) / "um1d" / "BTCUSDT.parquet").set_index("t")["o"]
    return (b.shift(-1) / b - 1)                    # the return of the day itself, open to next open


def stage_fit(cache, out):
    out = Path(out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(C.cdir(cache) / "xs_rows.parquet", filters=[("t", "<", TEST0 - PURGE)])
    assert R["t"].max() < TEST0, "2025 rows reached the fit stage"
    t = R["t"]
    tr, va = (t < VAL0 - PURGE).to_numpy(), (t >= VAL0).to_numpy()
    table = {}
    for L in LEARNERS:
        pred = predict(fit(L, R, tr), R)
        table[L] = {str(k): summary(book(R[va], pred[va], k)) for k in KS}
        print(f"[cry-xs fit] {L}: " + ", ".join(f"K{k} Sharpe {v['sharpe']:.2f}" for k, v in table[L].items()), flush=True)
    ch = choose(table)
    res = {"train": "2020-07..2023-12", "choose": "2024", "features": FEATS, "rows": {"train": int(tr.sum()), "choose": int(va.sum())},
           "table_2024": table, "chosen": ch, "gate": "failed at validation"}
    if ch and ch["mean_daily_net"] > 0:
        m = fit(ch["learner"], R, np.ones(len(R), bool))
        res["gate"] = "passed: run the 2025 test"
        res["refit"] = {"period": "2020-07..2024-12", "model": S.save_model(ch["learner"], m, str(out / "models" / "CRYXS"))}
    S.atomic_json(res, out / "CRYXS_fit.json")
    print(f"[cry-xs fit] chosen {ch}; gate: {res['gate']}", flush=True)
    return res


def stage_qa(cache, out):
    """Data checks on the 2024 choice year for the chosen version, before the one test (Amendment 1). Reads
    nothing from 2025."""
    out = Path(out)
    ch = json.loads((out / "CRYXS_fit.json").read_text())["chosen"]
    R = pd.read_parquet(C.cdir(cache) / "xs_rows.parquet", filters=[("t", "<", TEST0 - PURGE)]).reset_index(drop=True)
    assert R["t"].max() < TEST0
    tr = (R["t"] < VAL0 - PURGE).to_numpy()
    m = fit(ch["learner"], R, tr)
    V = R[~tr & (R["t"] >= VAL0).to_numpy()].reset_index(drop=True)
    pred, k = predict(m, V), ch["k"]
    B = book(V, pred, k)
    V = V.assign(pred=pred)
    V["rank_hi"] = V.groupby("t")["pred"].rank(ascending=False, method="first")
    V["rank_lo"] = V.groupby("t")["pred"].rank(ascending=True, method="first")
    held = V[(V["rank_hi"] <= k) | (V["rank_lo"] <= k)].assign(side=lambda x: np.where(x["rank_hi"] <= k, "L", "S"))
    held["contrib"] = np.where(held["side"] == "L", 1, -1) * held["fwd"] / (2 * k)
    top = B["net"].abs().sort_values(ascending=False).head(8).index
    nxt = V.sort_values(["symbol", "t"]).groupby("symbol")["fwd"].shift(-1)       # the day after the traded day
    late = V.assign(fwd=nxt.reindex(V.index)).dropna(subset=["fwd"])
    imp = getattr(m, "feature_importances_", None)
    res = {"chosen": ch, "net_total": float(B["net"].sum()),
           "share_of_net_from_top_5_days": float(B["net"].sort_values(ascending=False).head(5).sum() / B["net"].sum()),
           "days_positive": float((B["net"] > 0).mean()),
           "long_side": summary(book(V, pred, k, side="L")), "short_side": summary(book(V, pred, k, side="S")),
           "universe_mean_daily_fwd": float(V.groupby("t")["fwd"].mean().mean()),
           "entry_one_day_late": summary(book(late, late["pred"].to_numpy(), k)),
           "abs_fwd_over_50pct": json.loads(V.loc[V["fwd"].abs() > 0.5, ["t", "symbol", "fwd", "pred"]].assign(
               t=lambda x: x["t"].astype(str)).to_json(orient="records")),
           "biggest_days": {str(d.date()): {"net": float(B.at[d, "net"]), "held": json.loads(
               held.loc[held["t"] == d - pd.Timedelta(days=1), ["symbol", "side", "fwd", "r_1", "pred"]].to_json(orient="records"))}
               for d in top},
           "by_coin_top": {s: float(v) for s, v in held.groupby("symbol")["contrib"].sum().sort_values(ascending=False).head(10).items()},
           "by_coin_bottom": {s: float(v) for s, v in held.groupby("symbol")["contrib"].sum().sort_values().head(5).items()},
           "coins_held": int(held["symbol"].nunique()),
           "feature_importance": dict(sorted(zip(FEATS, map(float, imp), strict=True), key=lambda x: -x[1])[:12]) if imp is not None else None,
           "rows_per_day": {"min": int(V.groupby("t").size().min()), "median": float(V.groupby("t").size().median())},
           "pred_vs_rel_corr": float(np.corrcoef(pred, V["rel"])[0, 1]),
           "r1_vs_rel_corr": float(V[["r_1", "rel"]].corr().iloc[0, 1])}
    S.atomic_json(res, out / "CRYXS_qa.json")
    print(f"[cry-xs qa] top-5-day share {res['share_of_net_from_top_5_days']:.2f}; one day late mean "
          f"{res['entry_one_day_late']['mean_daily_net']:.5f}", flush=True)
    return res


def stage_trace(cache, out):
    """POST-HOC, after the one test: which coins and days made the 2025 result (CRYXS_trace.json). Changes nothing."""
    out = Path(out)
    if not (out / "CRYXS_test.json").exists():
        raise SystemExit("the test has not run")
    fitj = json.loads((out / "CRYXS_fit.json").read_text())
    R = pd.read_parquet(C.cdir(cache) / "xs_rows.parquet", filters=[("t", ">=", TEST0)]).reset_index(drop=True)
    k = fitj["chosen"]["k"]
    R["pred"] = predict(load_model(fitj["refit"]["model"]), R)
    g = R.groupby("t")["pred"]
    R["side"] = np.where(g.rank(ascending=False, method="first") <= k, "L",
                         np.where(g.rank(method="first") <= k, "S", ""))
    H = R[R["side"] != ""].copy()
    sg = np.where(H["side"] == "L", 1.0, -1.0)
    H["gross"] = sg * H["fwd"] / (2 * k)
    H["fund"] = -sg * H["fund_fwd"] / (2 * k)
    H["month"] = H["t"].dt.to_period("M").astype(str)
    by = H.groupby(["month", "symbol", "side"])[["gross", "fund"]].sum().reset_index()
    by["sum"] = by["gross"] + by["fund"]
    days = H.groupby("t")[["gross", "fund"]].sum().assign(sum=lambda x: x["gross"] + x["fund"])
    worst = days["sum"].sort_values().head(10).index
    res = {"worst_coin_months": json.loads(by.sort_values("sum").head(15).to_json(orient="records")),
           "best_coin_months": json.loads(by.sort_values("sum").tail(10).to_json(orient="records")),
           "worst_days": {str(d.date()): json.loads(H.loc[H["t"] == d, ["symbol", "side", "fwd", "fund_fwd", "r_1", "r_7",
                                                                        "fund_1"]].to_json(orient="records")) for d in worst},
           "funding_by_side": {sd: float(H.loc[H["side"] == sd, "fund"].sum()) for sd in ("L", "S")}}
    S.atomic_json(res, out / "CRYXS_trace.json")
    print("[cry-xs trace] written", flush=True)
    return res


FUND_VERSIONS = [(f, k) for f in ("fund_1", "fund_7") for k in KS]


def stage_fund(cache, out):
    """CRY-FUND (research/prereg/CRYFUND.md): long the coins with the most negative funding, short the most positive.
    Version chosen on 2020-07..2024, then the one test on 2025-01..09 -> CRYFUND.json."""
    out = Path(out)
    if (out / "CRYFUND.json").exists():
        raise SystemExit("the one test has already been run")
    R = pd.read_parquet(C.cdir(cache) / "xs_rows.parquet").reset_index(drop=True)
    hist = (R["t"] < TEST0).to_numpy()
    H = R[hist].reset_index(drop=True)
    table = {f"{f} K{k}": summary(book(H, -H[f].fillna(0).to_numpy(), k)) for f, k in FUND_VERSIONS}
    best = max(table, key=lambda v: table[v]["sharpe"] if np.isfinite(table[v]["sharpe"]) else -9)
    res = {"history": "2020-07..2024-12", "table_history": table, "chosen": best, "idea_count": 180,
           "gate": "failed on history"}
    print("[cry-fund] " + ", ".join(f"{v}: Sharpe {s['sharpe']:.2f}" for v, s in table.items()), flush=True)
    if table[best]["mean_daily_net"] > 0:
        res["gate"] = "passed"
        f, k = FUND_VERSIONS[list(table).index(best)]
        T = R[~hist & (R["t"] >= TEST0).to_numpy()].reset_index(drop=True)
        p = -T[f].fillna(0).to_numpy()
        B, ctrl = book(T, p, k), book(T, p, k, seed=SEED)
        s = summary(B, ctrl, btc_daily(cache))
        s["dsr"] = CS.deflated_sharpe(B["net"].to_numpy(), 180)
        rules, v = verdict(s)
        T = T.assign(p=p)
        g = T.groupby("t")["p"]
        held = T[(g.rank(ascending=False, method="first") <= k) | (g.rank(method="first") <= k)]
        res.update({"test": "2025-01-01..2025-09-30", "primary": s, "rules": rules, "verdict": v,
                    "long_side": summary(book(T, p, k, side="L")), "short_side": summary(book(T, p, k, side="S")),
                    "held_days_with_abs_move_over_50pct": int((held["fwd"].abs() > 0.5).sum())})
        print(f"[cry-fund] {best}: {v}, mean daily net {s['mean_daily_net']:.5f}, t {s['t_nw']:.2f}", flush=True)
    S.atomic_json(res, out / "CRYFUND.json")
    return res


def load_model(path):
    if str(path).endswith(".joblib"):
        import joblib
        return joblib.load(path)
    return P2.load_model(path)


def stage_test(cache, out):
    out = Path(out)
    fitj = json.loads((out / "CRYXS_fit.json").read_text())
    if not fitj["gate"].startswith("passed"):
        raise SystemExit("the 2024 gate did not pass; 2025 stays unread")
    if (out / "CRYXS_test.json").exists():
        raise SystemExit("the one test has already been run")
    R = pd.read_parquet(C.cdir(cache) / "xs_rows.parquet", filters=[("t", ">=", TEST0)]).reset_index(drop=True)
    k = fitj["chosen"]["k"]
    pred = predict(load_model(fitj["refit"]["model"]), R)
    B, ctrl = book(R, pred, k), book(R, pred, k, seed=SEED)
    s = summary(B, ctrl, btc_daily(cache))
    rules, v = verdict(s)
    res = {"test": "2025-01-01..2025-09-30", "chosen": fitj["chosen"], "primary": s, "rules": rules, "verdict": v,
           "long_side": summary(book(R, pred, k, side="L")), "short_side": summary(book(R, pred, k, side="S")),
           "long_side_spot_0.25pct": summary(book(R, pred, k, cost=COST_SPOT_SIDE, side="L")), "idea_count": TRIALS}
    S.atomic_json(res, out / "CRYXS_test.json")
    print(f"[cry-xs test] {v}: mean daily net {s['mean_daily_net']:.5f}, t {s['t_nw']:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "fit", "qa", "test", "trace", "fund"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="research/results/CRYXS")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.cache)
    else:
        {"fit": stage_fit, "qa": stage_qa, "test": stage_test, "trace": stage_trace, "fund": stage_fund}[a.stage](a.cache, a.out)


if __name__ == "__main__":
    main()
