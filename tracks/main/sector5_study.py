"""SECTOR1 stage 5 (research/prereg/SECTOR1.md, Amendment 4): signs known the day before (or before the open) that
point to the next day's direction of the market, the 7 biggest stocks, oil, fear, rates, credit, gold and the dollar.
2019-2023 only; fit on 2019-2021, scored once on 2022-2023. Descriptive. Writes <out>/SECTOR1_stage5.json.

    python -m tracks.main.sector5_study --w1 cache/w1 --out research/results/SECTOR1 --cache cache/sector1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import alpaca_headers
from tracks.crypto import cdata as C
from tracks.main import sector1_study as S1
from tracks.main import sector2_study as S2
from tracks.main import sector4_study as S4
from tracks.news import w1_data as W

ET = "America/New_York"
BIG = S4.BIG
CIK = dict(zip(S4.BIG, S4.BIG_CIK, strict=True))
TARGETS = ["SPY", "QQQ", "IWM", *BIG, "XLE", "USO", "VIXY", "TLT", "HYG", "GLD", "UUP"]
SYMBOLS = S4.SYMBOLS
MACRO_CSV = Path(__file__).resolve().parents[2] / "research" / "macro_dates_2019_2023.csv"
POS = r"\b(beat|beats|raise[sd]?|upgrade[sd]?|surge[sd]?|soar[sd]?|jump[sd]?|record|approv\w*|strong|tops|exceed\w*|" \
      r"gain[sd]?|rall\w*|rises?|climb[sd]?)\b"
NEG = r"\b(miss|misses|missed|cut|cuts|downgrade[sd]?|plunge[sd]?|fall|falls|drop[sd]?|slump\w*|lawsuit|recall\w*|" \
      r"probe|investigation|layoffs?|weak\w*|warns?|lower[sd]?|sinks?|tumble[sd]?|fraud|crash\w*|slide[sd]?)\b"
LGB = {"n_estimators": 200, "learning_rate": 0.02, "num_leaves": 7, "min_child_samples": 50, "subsample": 0.8,
       "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 7, "verbose": -1, "n_jobs": 4}
CALENDAR = ["c_mon", "c_fri", "c_tom", "c_opex", "c_preholiday", "c_fomc", "c_fomc_eve", "c_cpi", "c_jobs",
            "c_big_earn", "c_own_earn"]


def cost_of(t):
    return 0.0005 if t in BIG else 0.0002


# ---------- pure helpers (tested) ----------

def after_hours(b):
    """5-minute bars (t UTC, c) -> per ET date the last close of the 16:00-20:00 bars (p2000)."""
    et = pd.to_datetime(b["t"], utc=True).dt.tz_convert(ET)
    hm = et.dt.hour * 60 + et.dt.minute
    x = pd.DataFrame({"date": et.dt.tz_localize(None).dt.normalize(), "hm": hm, "c": b["c"].to_numpy(float)})
    x = x[(x["hm"] >= 960) & (x["hm"] < 1200)].sort_values(["date", "hm"])
    return x.groupby("date")["c"].last().rename("p2000")


def tone(headlines):
    h = pd.Series(headlines).fillna("").str.lower()
    return np.sign(h.str.count(POS) - h.str.count(NEG)).astype(float)


def night_session(ts_et, cal):
    """ET timestamps -> the trading day whose 09:30 open is the first after them, for headlines outside that
    day's regular session; NaT for headlines inside a regular session."""
    ts = pd.DatetimeIndex(ts_et)
    tod = ts.hour * 60 + ts.minute
    cal = pd.DatetimeIndex(cal)
    on_trading_day = pd.DatetimeIndex(ts.normalize()).isin(cal)
    inside = on_trading_day & (tod >= 570) & (tod < 960)
    day = (ts + pd.Timedelta(hours=8)).normalize()
    i = np.searchsorted(cal.values, day.values)
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (i < len(cal)) & ~inside
    out[ok] = cal.values[i[ok]]
    return out


def calendar_flags(dates, macro):
    """Trading dates -> flags describing each date (applied to day t+1 later). macro: (date, event) rows."""
    d = pd.DatetimeIndex(dates)
    s = pd.Series(d, index=d)
    nxt = s.shift(-1)
    gap = (nxt - s).dt.days
    out = pd.DataFrame(index=d)
    out["c_mon"] = d.dayofweek == 0
    out["c_fri"] = d.dayofweek == 4
    first3 = s.groupby([d.year, d.month]).rank(method="first") <= 3
    last = nxt.dt.month.ne(d.month).to_numpy() & nxt.notna().to_numpy()
    out["c_tom"] = first3.to_numpy() | last
    out["c_opex"] = (d.dayofweek == 4) & (d.day >= 15) & (d.day <= 21)
    out["c_preholiday"] = ((d.dayofweek < 4) & (gap > 1)) | ((d.dayofweek == 4) & (gap > 3))
    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"])
    i = np.searchsorted(d.values, m["date"].values)
    m = m[i < len(d)].assign(day=d.values[i[i < len(d)]])
    for ev, col in (("FOMC", "c_fomc"), ("CPI", "c_cpi"), ("JOBS", "c_jobs")):
        out[col] = d.isin(pd.DatetimeIndex(m.loc[m["event"] == ev, "day"]))
    out["c_fomc_eve"] = out["c_fomc"].shift(-1, fill_value=False).astype(bool)
    return out


def sign_rule(x_tr, y_tr, x_te, y_te, cost):
    """One sign alone: slope sign from 2019-2021 on the de-meaned sign; trade sign(slope)*sign(x - mean) in 2022-23."""
    mu = np.nanmean(x_tr)
    b = np.polyfit(x_tr - mu, y_tr, 1)[0]
    d = np.sign(b) * np.sign(x_te - mu)
    net = d * y_te - cost * (d != 0)
    d_tr = np.sign(b) * np.sign(x_tr - mu)
    return {"corr_train": float(np.corrcoef(x_tr, y_tr)[0, 1]), "corr_test": float(np.corrcoef(x_te, y_te)[0, 1]),
            "hit_test": float((np.sign(y_te) == d)[(y_te != 0) & (d != 0)].mean()),
            "net_test": float(net.mean()), "t_test": S4.nw_t(net), "t_train": S4.nw_t(d_tr * y_tr - cost),
            "days_test": len(y_te)}


def tail_rule(x_tr, y_tr, x_te, y_te, cost, q=0.1):
    """The sign's top and bottom 10% (cut on 2019-2021): each tail's 2019-2021 direction, then traded in 2022-23."""
    lo, hi = np.nanquantile(x_tr, q), np.nanquantile(x_tr, 1 - q)
    out = {}
    for name, m_tr, m_te in (("low", x_tr <= lo, x_te <= lo), ("high", x_tr >= hi, x_te >= hi)):
        d = float(np.sign(y_tr[m_tr].mean())) if m_tr.any() else 0.0
        y = y_te[m_te]
        net = d * y - cost
        out[name] = {"cut": float(lo if name == "low" else hi), "dir": d, "mean_train": float(y_tr[m_tr].mean()),
                     "hit_train": float((np.sign(y_tr[m_tr]) == d).mean()), "days_test": int(m_te.sum()),
                     "hit_test": float((np.sign(y) == d).mean()) if len(y) else None,
                     "mean_test": float(y.mean()) if len(y) else None,
                     "net_test": float(net.mean()) if len(y) else None, "t_test": S4.nw_t(net)}
    return out


def flag_rule(f_tr, y_tr, f_te, y_te, cost):
    f_tr, f_te = f_tr.astype(bool), f_te.astype(bool)
    if f_tr.sum() < 5 or f_te.sum() < 5:
        return None
    d = float(np.sign(y_tr[f_tr].mean()))
    net = d * y_te[f_te] - cost

    def diff_t(f, y):
        a, b = y[f], y[~f]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return float(a.mean() - b.mean()), float((a.mean() - b.mean()) / se) if se > 0 else None
    dtr, ttr = diff_t(f_tr, y_tr)
    dte, tte = diff_t(f_te, y_te)
    return {"days_train": int(f_tr.sum()), "days_test": int(f_te.sum()), "mean_train": float(y_tr[f_tr].mean()),
            "mean_test": float(y_te[f_te].mean()), "others_test": float(y_te[~f_te].mean()),
            "diff_train": dtr, "t_diff_train": ttr, "diff_test": dte, "t_diff_test": tte, "dir": d,
            "hit_test": float((np.sign(y_te[f_te]) == d).mean()), "net_test": float(net.mean()),
            "t_test": S4.nw_t(net) if len(net) >= 20 else None}


# ---------- data ----------

def bars_frame(cache):
    """Per symbol: r, on, pm, oc, l30, lrv (stage 4) plus ah (16:00-20:00 after-hours return) and n2c (20:00 price to
    the next close), indexed by ET date."""
    h = alpaca_headers()
    per = {}
    for s in SYMBOLS:
        bars = [b for y in S1.YEARS if len(b := S4.fetch(s, y, cache, h))]
        p = pd.concat([S4.daily_points(b) for b in bars], ignore_index=True)
        rr = S4.returns(p).set_index("date")
        p2000 = pd.concat([after_hours(b) for b in bars]).groupby(level=0).last()
        close = p.set_index("date")["close"]
        rr["ah"] = p2000.reindex(rr.index) / close.reindex(rr.index) - 1
        rr["n2c"] = close.shift(-1).reindex(rr.index) / p2000.reindex(rr.index) - 1
        per[s] = rr
        print(f"[sector5] {s}: {len(rr)} days", flush=True)
    wide = pd.concat(per, axis=1)
    wide.columns = [f"{c}_{s}" for s, c in wide.columns]
    return wide.sort_index()


def btc(cache):
    f = Path(cache) / "btc1h.parquet"
    if not f.exists():
        parts = [C.parse_klines(r) for m in C.months("2018-12", "2023-12")
                 if (r := C.fetch(C.kline_url("spot", "BTCUSDT", "1h", m)))]
        d = pd.concat(parts, ignore_index=True).drop_duplicates("t")
        assert d["t"].max() < pd.Timestamp("2024-01-01")
        C.atomic(d, f)
    d = pd.read_parquet(f)
    return pd.Series(d["c"].to_numpy(float), index=pd.DatetimeIndex(d["t"]) + pd.Timedelta(hours=1))  # close time


def btc_at(px, dates, hour):
    """BTC price at ET clock `hour` on each date (the hourly close that ends then)."""
    ts = (pd.DatetimeIndex(dates) + pd.Timedelta(hours=hour)).tz_localize(ET).tz_convert("UTC").tz_localize(None)
    return pd.Series(px.reindex(ts).to_numpy(), index=pd.DatetimeIndex(dates))


def breadth(w1):
    out = []
    for y in S1.YEARS:
        f = [("date", ">=", pd.Timestamp(f"{y - 1}-11-01")), ("date", "<=", pd.Timestamp(f"{y}-12-31")),
             ("uni", "==", True)]
        d = W.read_rows(w1, ["symbol", "date", "ac"], f).sort_values(["symbol", "date"])
        g = d.groupby("symbol")["ac"]
        d["up"] = g.pct_change() > 0
        roll = g.rolling(20, min_periods=20)
        d["hi"] = d["ac"] >= roll.max().reset_index(level=0, drop=True)
        d["lo"] = d["ac"] <= roll.min().reset_index(level=0, drop=True)
        a = d.groupby("date").agg(up=("up", "mean"), hi=("hi", "mean"), lo=("lo", "mean"), n=("up", "size"))
        out.append(a[a.index.year == y])
        print(f"[sector5] breadth {y}: {len(d)} rows", flush=True)
        del d
    a = pd.concat(out)
    return pd.DataFrame({"b_up": a["up"], "b_hl20": a["hi"] - a["lo"]})


def news(w1, cache, cal):
    S2.ensure_inputs(w1, cache, time.monotonic() + 3600)
    big = set(BIG) | {"GOOG", "FB"}
    rows = []
    for per in pd.period_range("2018-12", "2023-12", freq="M"):
        n = W.news_month(w1, per)
        n = n[n["symbols"].fillna("") != ""]
        ts = W.et_time(n["created_at"])
        syms = n["symbols"].str.split(",")
        isbig = syms.map(lambda v: bool(big & set(v)))
        own = {s: syms.map(lambda v, s=s: s in v or (s == "GOOGL" and "GOOG" in v) or (s == "META" and "FB" in v))
               for s in BIG}
        x = pd.DataFrame({"sess": S2.to_session(ts, cal), "night": night_session(ts, cal),
                          "tone": tone(n["headline"]).to_numpy(), "big": isbig.to_numpy(),
                          **{f"own_{s}": v.to_numpy() for s, v in own.items()}})
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)

    def agg(key, tag):
        g = x.dropna(subset=[key])
        out = pd.DataFrame({f"{tag}_n_all": g.groupby(key).size(),
                            f"{tag}_tone_all": g.groupby(key)["tone"].mean(),
                            f"{tag}_n_big": g[g["big"]].groupby(key).size(),
                            f"{tag}_tone_big": g[g["big"]].groupby(key)["tone"].mean()})
        for s in BIG:
            out[f"{tag}_n_{s}"] = g[g[f"own_{s}"]].groupby(key).size()
        out = out.reindex(cal)
        for c in [c for c in out.columns if "_n_" in c]:
            v = np.log1p(out[c].fillna(0))
            out[c] = v - v.rolling(60, min_periods=20).mean().shift()
        return out
    return pd.concat([agg("sess", "news"), agg("night", "night")], axis=1)


def earnings(w1, cal):
    f = W.load_filings(w1, ["cik", "form", "filed", "items"])
    f = f[f["cik"].isin(S4.BIG_CIK)]
    e = S2.earn_days(f, cal)
    out = pd.DataFrame(index=cal)
    out["c_big_earn"] = cal.isin(e["date"])
    for s, c in CIK.items():
        out[f"earn_{s}"] = cal.isin(e.loc[e["cik"] == c, "date"])
    return out


# ---------- features ----------

def features(w, extra):
    """Family A (known at 16:00 on t) and family B (known by 09:25 on t+1) signs, indexed by t."""
    A = pd.DataFrame(index=w.index)
    spy = w["r_SPY"]
    for s in ("SPY", "QQQ", "IWM"):
        A[f"{s}_today"] = w[f"r_{s}"]
    A["SPY_last30"] = w["l30_SPY"]
    A["SPY_gap"] = w["on_SPY"]
    A["SPY_open_to_close"] = w["oc_SPY"]
    A["SPY_5d"] = spy.rolling(5).sum()
    A["SPY_20d"] = spy.rolling(20).sum()
    px = (1 + spy.fillna(0)).cumprod()
    A["SPY_vs_20d_high"] = px / px.rolling(20).max() - 1
    A["SPY_volatility"] = w["lrv_SPY"]
    A["SPY_volatility_change"] = w["lrv_SPY"] - w["lrv_SPY"].rolling(22).mean()
    A["VIXY_today"] = w["r_VIXY"]
    A["VIXY_5d"] = w["r_VIXY"].rolling(5).sum()
    A["HYG_today"] = w["r_HYG"]
    A["HYG_minus_SPY"] = w["r_HYG"] - spy
    for s in ("TLT", "USO", "UUP", "GLD"):
        A[f"{s}_today"] = w[f"r_{s}"]
    A["TLT_5d"] = w["r_TLT"].rolling(5).sum()
    A["USO_5d"] = w["r_USO"].rolling(5).sum()
    big = w[[f"r_{s}" for s in BIG]]
    A["BIG7_today"] = big.mean(axis=1)
    A["BIG7_minus_SPY"] = A["BIG7_today"] - spy
    A["BIG7_count_up"] = (big > 0).sum(axis=1).where(big.notna().any(axis=1))
    A["IWM_minus_SPY"] = w["r_IWM"] - spy
    A["XLU_minus_XLK"] = w["r_XLU"] - w["r_XLK"]
    A["XLY_minus_XLP"] = w["r_XLY"] - w["r_XLP"]
    for c in ("b_up", "b_hl20", "news_n_all", "news_tone_all", "news_n_big", "news_tone_big", "btc_day"):
        A[c] = extra[c]
    B = pd.DataFrame(index=w.index)
    for s in ("SPY", "QQQ"):
        B[f"{s}_after_hours"] = w[f"ah_{s}"]
    B["BIG7_after_hours"] = w[[f"ah_{s}" for s in BIG]].mean(axis=1)
    for c in ("btc_night", "night_n_all", "night_tone_all", "night_n_big", "night_tone_big"):
        B[c] = extra[c]
    B["SPY_premarket"] = w["pm_SPY"].shift(-1)   # the pre-market of t+1, known by 09:29 on t+1
    return A, B


def own_features(w, extra, t):
    A = pd.DataFrame(index=w.index)
    A["own_today"] = w[f"r_{t}"]
    A["own_5d"] = w[f"r_{t}"].rolling(5).sum()
    A["own_last30"] = w[f"l30_{t}"]
    A["own_volatility_change"] = w[f"lrv_{t}"] - w[f"lrv_{t}"].rolling(22).mean()
    if t in BIG:
        A["own_news"] = extra[f"news_n_{t}"]
    B = pd.DataFrame({"own_after_hours": w[f"ah_{t}"], "own_premarket": w[f"pm_{t}"].shift(-1)}, index=w.index)
    if t in BIG:
        B["own_night_news"] = extra[f"night_n_{t}"]
    return A, B


def split_mask(idx):
    return idx.year <= 2021, idx.year >= 2022


def run_family(X, y, cost):
    """Every sign alone and in its tails."""
    alone, tails = {}, {}
    for c in X.columns:
        z = pd.concat([X[c], y], axis=1).dropna()
        if len(z) < 300:
            continue
        tr, te = split_mask(z.index)
        a = z.to_numpy(float)
        if tr.sum() < 200 or te.sum() < 100 or np.nanstd(a[tr, 0]) == 0:
            continue
        alone[c] = sign_rule(a[tr, 0], a[tr, 1], a[te, 0], a[te, 1], cost)
        tails[c] = tail_rule(a[tr, 0], a[tr, 1], a[te, 0], a[te, 1], cost)
    return alone, tails


def combined(X, y, cost):
    X = X.loc[:, X.isna().mean() < 0.2]
    z = pd.concat([X, y.rename("y")], axis=1).dropna()
    tr, te = split_mask(z.index)
    f = [c for c in z.columns if c != "y"]
    Xtr, Xte = z.loc[tr, f].to_numpy(float), z.loc[te, f].to_numpy(float)
    ytr, yte = z.loc[tr, "y"].to_numpy(float), z.loc[te, "y"].to_numpy(float)
    out = {}
    p, a = S4.ridge_oos(Xtr, ytr, Xte)
    import lightgbm as lgb
    q = lgb.LGBMRegressor(**LGB).fit(Xtr, ytr).predict(Xte)
    for name, pr in (("ridge", p), ("lightgbm", q)):
        d = np.sign(pr)
        net = d * yte - cost
        out[name] = {"oos_r2": float(1 - ((yte - pr) ** 2).sum() / (yte ** 2).sum()),
                     "hit_test": float((np.sign(yte) == d)[yte != 0].mean()), "net_test": float(net.mean()),
                     "t_test": S4.nw_t(net), "long_share": float((d > 0).mean())}
    out["ridge"]["alpha"] = a
    out["always_long"] = {"net_test": float((yte - cost).mean()), "t_test": S4.nw_t(yte - cost),
                          "hit_test": float((yte > 0).mean())}
    out["days_test"], out["n_signs"] = int(te.sum()), len(f)
    return out


def summarise(res):
    cells = hits = 0
    best = []
    for t, r in res.items():
        for fam in ("A", "B", "B_20h"):
            for c, v in r.get(f"alone_{fam}", {}).items():
                cells += 1
                ok = (v["t_test"] or 0) >= 2
                hits += ok
                best.append({"target": t, "family": fam, "sign": c, "kind": "alone", **v})
            for c, tv in r.get(f"tails_{fam}", {}).items():
                for side, v in tv.items():
                    if v["days_test"] >= 20:
                        cells += 1
                        ok = (v["t_test"] or 0) >= 2
                        hits += ok
                        best.append({"target": t, "family": fam, "sign": f"{c} {side} 10%", "kind": "tail", **v})
        for c, v in (r.get("calendar") or {}).items():
            if v:
                cells += 1
                hits += (v["t_test"] or 0) >= 2
                best.append({"target": t, "family": "calendar", "sign": c, "kind": "flag", **v})
    best = sorted(best, key=lambda v: -(v.get("t_test") or -9))
    return {"cells": cells, "t_ge_2": hits, "share": hits / cells if cells else None,
            "chance_share": 0.023, "top30": best[:30]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    ap.add_argument("--minutes", type=float, default=150)
    a = ap.parse_args()
    w = bars_frame(a.cache)
    w = w[(w.index.year >= S1.YEARS[0]) & (w.index <= S1.LAST)]
    cal = pd.DatetimeIndex(w.index)
    px = btc(a.cache)
    b16 = btc_at(px, cal, 16)
    b09 = btc_at(px, cal, 9)
    extra = pd.DataFrame(index=cal)
    extra["btc_day"] = b16 / b16.shift() - 1
    extra["btc_night"] = b09.shift(-1) / b16 - 1
    extra = extra.join(breadth(a.w1)).join(news(a.w1, a.cache, cal)).join(earnings(a.w1, cal))
    cflags = calendar_flags(cal, pd.read_csv(MACRO_CSV)).join(extra[["c_big_earn"]])
    print(f"[sector5] {len(w)} days; extra columns {list(extra.columns)[:12]} ...", flush=True)
    A, B = features(w, extra)
    res = {}
    for t in TARGETS:
        cost = cost_of(t)
        oa, ob = own_features(w, extra, t)
        XA = pd.concat([A, oa], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
        XB = pd.concat([B, ob], axis=1)
        y_cc = w[f"r_{t}"].shift(-1)
        y_oc = w[f"oc_{t}"].shift(-1)
        y_20 = w[f"n2c_{t}"]
        r = {}
        r["alone_A"], r["tails_A"] = run_family(XA, y_cc, cost)
        r["alone_B"], r["tails_B"] = run_family(XB, y_oc, cost)
        ah = XB[[c for c in XB.columns if "after_hours" in c]]
        r["alone_B_20h"], r["tails_B_20h"] = run_family(ah, y_20, cost)
        cf = cflags.copy()
        cf["c_own_earn"] = extra[f"earn_{t}"] if t in BIG else False
        nxt = cf.shift(-1)  # flags about day t+1, known on day t
        cal_res = {}
        z = pd.concat([nxt, y_cc.rename("y")], axis=1).dropna(subset=["y"])
        tr, te = split_mask(z.index)
        for c in CALENDAR:
            fz = z[c].astype(float).fillna(0).astype(bool).to_numpy()
            yz = z["y"].to_numpy(float)
            cal_res[c] = flag_rule(fz[tr], yz[tr], fz[te], yz[te], cost)
        r["calendar"] = cal_res
        r["combined_A"] = combined(pd.concat([XA, nxt[CALENDAR].astype(float)], axis=1), y_cc, cost)
        r["combined_B"] = combined(XB, y_oc, cost)
        res[t] = r
        print(f"[sector5] {t}: A ridge {r['combined_A']['ridge']['hit_test']:.3f}, "
              f"B ridge {r['combined_B']['ridge']['hit_test']:.3f}", flush=True)
    out = {"targets": res, "summary": summarise(res),
           "note": ("2019-2023, Alpaca SIP 5-minute bars adjustment=all, W1 breadth and Benzinga headlines, Binance "
                    "BTCUSDT hourly, Fed/BLS release dates. Family A: signs at the 16:00 close of t vs the t+1 "
                    "close-to-close return. Family B: signs known by 09:25 on t+1 vs the t+1 open-to-close return. "
                    "B_20h: after-hours moves vs the 20:00-to-next-close return. Fit 2019-2021, scored 2022-2023. "
                    "Cost 0.02% round trip for ETFs, 0.05% for the 7 stocks. Descriptive only.")}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage5.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.replace(Path(a.out) / "SECTOR1_stage5.json")
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "top30"}, default=float), flush=True)
    for v in out["summary"]["top30"][:15]:
        print(v["target"], v["family"], v["sign"], round(v.get("t_test") or 0, 2), flush=True)


if __name__ == "__main__":
    main()
