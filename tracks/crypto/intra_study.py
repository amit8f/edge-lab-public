"""CRY-INTRA-BIG and CRY-INTRA-MEME (research/prereg/CRYINTRA.md): every 4 hours, for coins whose 28-day trend is on,
a model picks long entries held 8 or 24 hours. Train to 2023, choose on 2024, one test on Jan-Sep 2025.

    python -m tracks.crypto.intra_study --stage data|rows|fit|test --cache cache --out research/results/CRYINTRA
"""
import argparse
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.crypto import cdata as C
from tracks.crypto import trend_study as T

BASKETS = {"BIG": {"coins": T.BASKETS["CRYBIG"]["coins"], "cost": 0.005},
           "MEME": {"coins": T.BASKETS["CRYMEME"]["coins"], "cost": 0.01}}
PERP = {"SHIBUSDT": "1000SHIBUSDT", "PEPEUSDT": "1000PEPEUSDT", "BONKUSDT": "1000BONKUSDT",
        "FLOKIUSDT": "1000FLOKIUSDT"}
HS, CUTS, MODELS = (8, 24), (0.05, 0.10, 0.20), ("xgb", "cat", "lgbm", "logit")
TRAIN_END, CHOOSE_END, TEST_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-10-01")
MET_FIRST = "2020-09-01"
MIN_CHOOSE, MIN_TEST, DRAWS, SEED = 60, 100, 1000, 20261119
FEATS = ["r1", "r4", "r24", "r72", "tb1", "tb4", "tb24", "vratio", "rv24", "rv168", "dhigh", "lead4", "lead24",
         "slot", "dow", "fund", "fund3d", "oi4", "oi24", "tls1"]


def perp(sym):
    return PERP.get(sym, sym)


# ---------- pure helpers (tested) ----------

def parse_metrics(raw):
    """Zip bytes of one daily metrics file -> (t, oi, tls)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    return pd.DataFrame({"t": pd.to_datetime(df["create_time"]), "oi": df["sum_open_interest"].astype(float),
                         "tls": df["sum_taker_long_short_vol_ratio"].astype(float)})


def regular(h):
    """Hourly candles -> a gap-free hourly grid indexed by the candle's CLOSE time (missing hours are NaN)."""
    h = h.drop_duplicates("t").set_index("t").sort_index()
    h = h.reindex(pd.date_range(h.index[0], h.index[-1], freq="h"))
    h.index = h.index + pd.Timedelta(hours=1)
    return h


def _asof(times, series):
    """series (sorted DatetimeIndex) value at or before each time; NaN when none."""
    s = series.dropna()
    if s.empty:
        return np.full(len(times), np.nan)
    idx = s.index.searchsorted(times, side="right") - 1
    out = np.where(idx >= 0, s.to_numpy()[np.clip(idx, 0, None)], np.nan)
    return out.astype(float)


def trend_on(g):
    """At each close time t of the grid: the last finished daily close (00 UTC at or before t) above the one 28 days
    earlier. 0 when either is missing."""
    c = g["c"]
    daily = c[c.index.hour == 0]
    day = g.index.floor("D")
    now = daily.reindex(day).to_numpy()
    back = daily.reindex(day - pd.Timedelta(days=T.LOOKBACK)).to_numpy()
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(now) & np.isfinite(back), (now > back).astype(float), 0.0)


def coin_rows(h, lead, fund, met, cost):
    """Decision rows (every 4 hours at the candle close) for one coin. h, lead: hourly candles of the coin and of the
    lead coin (BTC, or ETH for BTC); fund: (t, rate); met: (t, oi, tls). Inputs use data up to the decision time only;
    labels look forward H hours."""
    g, L = regular(h), regular(lead)
    c = g["c"]
    f = pd.DataFrame(index=g.index)
    for k in (1, 4, 24, 72):
        f[f"r{k}"] = c / c.shift(k) - 1
    for k in (1, 4, 24):
        f[f"tb{k}"] = g["tbqv"].rolling(k).sum() / g["qv"].rolling(k).sum()
    f["vratio"] = g["qv"].rolling(24).sum() / (g["qv"].rolling(168).sum() / 7)
    hr = c / c.shift(1) - 1
    f["rv24"], f["rv168"] = hr.rolling(24).std(), hr.rolling(168).std()
    f["dhigh"] = c / g["h"].rolling(168).max() - 1
    lc = L["c"].reindex(g.index)
    f["lead4"], f["lead24"] = lc / lc.shift(4) - 1, lc / lc.shift(24) - 1
    f["slot"], f["dow"] = g.index.hour, g.index.dayofweek
    t = g.index
    if len(fund):
        fs = fund.set_index("t")["rate"].sort_index()
        f["fund"] = _asof(t, fs)
        f["fund3d"] = _asof(t, fs.rolling("72h").mean())
    else:
        f["fund"] = f["fund3d"] = np.nan
    if len(met):
        ms = met.set_index("t").sort_index()
        oi = ms["oi"]
        now = _asof(t, oi)
        f["oi4"] = now / _asof(t - pd.Timedelta(hours=4), oi) - 1
        f["oi24"] = now / _asof(t - pd.Timedelta(hours=24), oi) - 1
        f["tls1"] = _asof(t, ms["tls"].rolling("1h").mean())
    else:
        f["oi4"] = f["oi24"] = f["tls1"] = np.nan
    f["trend"] = trend_on(g)
    for H in HS:
        f[f"net{H}"] = c.shift(-H) / c - 1 - cost
    f["c"] = c
    keep = (t.hour % 4 == 0) & c.notna().to_numpy() & f["rv168"].notna().to_numpy()
    out = f.loc[keep].replace([np.inf, -np.inf], np.nan)     # zero volume or open interest gives x/0
    out.index.name = "t"
    return out.reset_index()


def skip_cuts(R, train_end=TRAIN_END):
    """95th percentiles of funding and 24-hour OI change per coin on training rows (pooled for coins with fewer than
    200 training values)."""
    tr = R[R["t"] < train_end]
    pooled = {k: float(tr[k].quantile(0.95)) if tr[k].notna().any() else np.inf for k in ("fund", "oi24")}
    cuts = {}
    for s, g in R.groupby("symbol"):
        gt = tr[tr["symbol"] == s]
        cuts[s] = {k: float(gt[k].quantile(0.95)) if gt[k].notna().sum() >= 200 else pooled[k] for k in ("fund", "oi24")}
    return cuts


def gated(R, cuts):
    """Rows where the trend is on and the skip filter doesn't fire (NaN inputs never skip)."""
    fc = R["symbol"].map(lambda s: cuts[s]["fund"]).to_numpy(float)
    oc = R["symbol"].map(lambda s: cuts[s]["oi24"]).to_numpy(float)
    skip = (R["fund"].to_numpy() > fc) | (R["oi24"].to_numpy() > oc)
    return (R["trend"].to_numpy() > 0) & ~skip


def take(R, score, cut, H):
    """Entries where score >= cut, at most one open trade per coin (a trade blocks the coin for H hours)."""
    pick = np.flatnonzero(score >= cut)
    sub = R.iloc[pick].sort_values(["symbol", "t"])
    keep, last = [], {}
    for i, s, t in zip(sub.index, sub["symbol"], sub["t"], strict=True):
        if s in last and t < last[s]:
            continue
        keep.append(i)
        last[s] = t + pd.Timedelta(hours=H)
    return R.loc[sorted(keep)]


def clustered_t(net, day):
    """Mean / standard error clustered by entry day."""
    net = np.asarray(net, float)
    e = pd.Series(net - net.mean()).groupby(np.asarray(day)).sum().to_numpy()
    se = np.sqrt((e ** 2).sum()) / len(net)
    return float(net.mean() / se) if se > 0 else float("nan")


def make(name):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=4, verbosity=0)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0,
                                  allow_writing_files=False, thread_count=4)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=4, verbose=-1)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def fit_score(name, Rtr, H, Rs):
    """Fit on Rtr for horizon H -> (training scores, scores of each frame in Rs)."""
    X = np.where(np.isfinite(Rtr[FEATS].to_numpy(np.float32)), Rtr[FEATS].to_numpy(np.float32), np.nan)
    y = (Rtr[f"net{H}"].to_numpy() > 0).astype(int)
    m = make(name)
    m.fit(X, y)
    fin = lambda A: np.where(np.isfinite(A), A, np.nan)
    sc = lambda D: m.predict_proba(fin(D[FEATS].to_numpy(np.float32)))[:, 1]
    return sc(Rtr), [sc(D) for D in Rs]


def train_rows(R, ok, end, H):
    """Gated rows with a label whose exit is before `end` (purges labels that reach into the next period)."""
    return R[ok & (R["t"] + pd.Timedelta(hours=H) < end) & R[f"net{H}"].notna()]


# ---------- stages ----------

def stage_data(cache):
    d = C.cdir(cache)
    for b in BASKETS.values():
        for s in b["coins"]:
            C.download("spot", s, "1h", C.SPOT_FIRST, d / "spot1h" / f"{s}.parquet")
            p = perp(s)
            f = d / "funding" / f"{p}.parquet"
            if not f.exists():
                with ThreadPoolExecutor(C.THREADS) as ex:
                    raws = list(ex.map(lambda m, p=p: C.fetch(
                        f"{C.BASE}/futures/um/monthly/fundingRate/{p}/{p}-fundingRate-{m}.zip"), C.months(C.FIRST, C.LAST)))
                parts = [C.parse_funding(r) for r in raws if r]
                fr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"t": pd.to_datetime([]), "rate": []})
                C.atomic(fr.sort_values("t").drop_duplicates("t").reset_index(drop=True), f)
                print(f"[intra] {p} funding: {len(fr)}", flush=True)
            f = d / "metrics" / f"{p}.parquet"
            if not f.exists():
                days = pd.date_range(MET_FIRST, "2025-09-30", freq="D").strftime("%Y-%m-%d")
                with ThreadPoolExecutor(16) as ex:
                    raws = list(ex.map(lambda day, p=p: C.fetch(
                        f"{C.BASE}/futures/um/daily/metrics/{p}/{p}-metrics-{day}.zip"), days))
                parts = []
                for r in raws:
                    if r:
                        try:
                            parts.append(parse_metrics(r))
                        except (KeyError, ValueError, zipfile.BadZipFile) as e:
                            print(f"[intra] {p} bad metrics file: {e}", flush=True)
                m = (pd.concat(parts, ignore_index=True).sort_values("t").drop_duplicates("t").reset_index(drop=True)
                     if parts else pd.DataFrame({"t": pd.to_datetime([]), "oi": [], "tls": []}))
                assert m.empty or m["t"].max() < TEST_END, "holdout metrics read"
                C.atomic(m, f)
                print(f"[intra] {p} metrics: {len(m)} rows, {len(parts)} days"
                      + ("" if parts else " (NONE: open-interest inputs missing for this coin)"), flush=True)


def stage_rows(cache):
    d = C.cdir(cache)
    read = lambda s: pd.read_parquet(d / "spot1h" / f"{s}.parquet")
    btc, eth = read("BTCUSDT"), read("ETHUSDT")
    for name, b in BASKETS.items():
        parts = []
        for s in b["coins"]:
            h = read(s)
            assert h["t"].max() < TEST_END, "holdout candles read"
            R = coin_rows(h, eth if s == "BTCUSDT" else btc, pd.read_parquet(d / "funding" / f"{perp(s)}.parquet"),
                          pd.read_parquet(d / "metrics" / f"{perp(s)}.parquet"), b["cost"])
            R["symbol"] = s
            parts.append(R)
        R = pd.concat(parts, ignore_index=True)
        if name == "MEME":
            R["breadth"] = R.groupby("t")["trend"].transform("mean")
        C.atomic(R, d / "intra" / f"{name}.parquet")
        print(f"[intra] {name}: {len(R)} decision rows, trend on {R['trend'].mean():.2f}", flush=True)


def feats(name):
    return FEATS + (["breadth"] if name == "MEME" else [])


def stage_fit(cache, out):
    global FEATS
    base, res = list(FEATS), {}
    for name, b in BASKETS.items():
        FEATS = feats(name)
        R = pd.read_parquet(C.cdir(cache) / "intra" / f"{name}.parquet")
        R = R[R["t"] < CHOOSE_END].reset_index(drop=True)
        cuts = skip_cuts(R)
        ok = gated(R, cuts)
        ch = R[ok & (R["t"] >= TRAIN_END)]
        table = []
        for H in HS:
            Rtr = train_rows(R, ok, TRAIN_END, H)
            chH = ch[(ch["t"] + pd.Timedelta(hours=H) < CHOOSE_END) & ch[f"net{H}"].notna()]
            for m in MODELS:
                st, (sc,) = fit_score(m, Rtr, H, [chH])
                for q in CUTS:
                    tk = take(chH, sc, float(np.quantile(st, 1 - q)), H)
                    n = len(tk)
                    table.append({"model": m, "H": H, "top": q, "trades": n,
                                  "mean_net": float(tk[f"net{H}"].mean()) if n else float("nan"),
                                  "win": float((tk[f"net{H}"] > 0).mean()) if n else float("nan")})
                print(f"[intra {name}] H{H} {m} done ({len(Rtr)} train rows)", flush=True)
        ok_rows = [r for r in table if r["trades"] >= MIN_CHOOSE]
        best = max(ok_rows, key=lambda r: r["mean_net"]) if ok_rows else None
        gate24 = {H: float(ch[f"net{H}"].mean()) for H in HS}
        res[name] = {"table": table, "chosen": best, "gate_alone_2024": gate24, "skip_cuts": cuts,
                     "rows": int(len(R)), "gated_2024": int(len(ch))}
        print(f"[intra {name}] chosen {best}", flush=True)
    FEATS = base
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CRYINTRA_choose.json").write_text(json.dumps(res, indent=1))


def stage_test(cache, out):
    global FEATS
    base = list(FEATS)
    out = Path(out)
    ch = json.loads((out / "CRYINTRA_choose.json").read_text())
    res = {}
    for name, b in BASKETS.items():
        best = ch[name]["chosen"]
        if best is None:
            res[name] = {"verdict": "FAILED: no version had 60 trades in 2024"}
            continue
        FEATS = feats(name)
        H, m, q = best["H"], best["model"], best["top"]
        R = pd.read_parquet(C.cdir(cache) / "intra" / f"{name}.parquet").reset_index(drop=True)
        assert R["t"].max() < TEST_END
        cuts = skip_cuts(R)              # training rows only (t < 2024), same cut-offs as the choice stage
        ok = gated(R, cuts)
        Rtr = train_rows(R, ok, CHOOSE_END, H)
        te = R[ok & (R["t"] >= CHOOSE_END) & (R["t"] + pd.Timedelta(hours=H) < TEST_END) & R[f"net{H}"].notna()]
        st, (sc,) = fit_score(m, Rtr, H, [te])
        tk = take(te, sc, float(np.quantile(st, 1 - q)), H)
        net = tk[f"net{H}"].to_numpy()
        n = len(net)
        rng = np.random.default_rng(SEED)
        pool = te[f"net{H}"].to_numpy()
        draws = np.array([rng.choice(pool, size=n, replace=False).mean() for _ in range(DRAWS)]) if 0 < n <= len(pool) else np.array([])
        r = {"version": best, "trades": n, "mean_net": float(net.mean()) if n else float("nan"),
             "win": float((net > 0).mean()) if n else float("nan"),
             "t_day": clustered_t(net, tk["t"].dt.floor("D")) if n > 1 else float("nan"),
             "random_pct": float((draws < net.mean()).mean()) if len(draws) else float("nan"),
             "gate_alone": float(pool.mean()), "gate_points": int(len(pool)),
             "by_coin": {s: [int(len(g)), float(g[f'net{H}'].mean())] for s, g in tk.groupby("symbol")}}
        r["pass"] = bool(n >= MIN_TEST and r["mean_net"] > 0 and r["t_day"] >= 2.0 and r["random_pct"] >= 0.95
                         and r["mean_net"] > r["gate_alone"])
        r["verdict"] = "PASSED: start the forward book" if r["pass"] else "FAILED"
        res[name] = r
        print(f"[intra {name}] {r['verdict']} n={n} mean={r['mean_net']:.4f} t={r['t_day']:.2f} "
              f"rand={r['random_pct']:.2f} gate={r['gate_alone']:.4f}", flush=True)
    FEATS = base
    (out / "CRYINTRA_test.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "rows", "fit", "test"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="research/results/CRYINTRA")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.cache)
    elif a.stage == "rows":
        stage_rows(a.cache)
    elif a.stage == "fit":
        stage_fit(a.cache, a.out)
    else:
        stage_test(a.cache, a.out)


if __name__ == "__main__":
    main()
