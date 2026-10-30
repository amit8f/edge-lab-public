"""CH (research/prereg/CH.md): the walk-forward of the six chart-model arms and their test.

Stages (rows and image windows come from `tracks/main/ch_data.py` stage build):
- "check": 20 random triggers per track for the hand check (research/checks/CH.csv). Nothing else runs until
  at least 19 of 20 per track are marked right.
- "trees": XGB and LGBM arms, both tracks. For each test year Y (2019 .. 2025-09): fit on 2017 .. Y-2,
  predict Y-1 to fix the cut (80th percentile of the largest of the six predictions) and the size scale; refit
  on 2017 .. Y-1 (5-session purge before Y) and decide year Y. In-sample decisions are kept for rule 21.
- "img": the image arm, the same walk-forward, one fit per run step (resumable; CPU).
- "measure": rule 21, then the control, the daily test, rule 12 and 11, and the exploratory list.

Usage: python -m tracks.main.ch_study --stage check|trees|img|measure --cache cache/w1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main import ch_build as B
from tracks.main import ch_data as C
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

SEED = 20261030
TRACKS = ("main", "penny")
FIRST_TRAIN, FIRST_TEST, LAST_ENTRY = 2017, 2019, pd.Timestamp("2025-09-23")
CUT_Q, TOP_Q = 0.80, 0.99
SIZE_LO, SIZE_HI = 0.02, 0.10
PURGE_DAYS = 7                                       # 5 sessions
LAGS, BLOCK, N_BOOT = 5, 20, 10_000
MIN_TRADES, MIN_DAYS = 300, 100
CHECK = Path("research/checks/CH.csv")
CHECK_N, CHECK_PASS = 20, 19
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=200, reg_lambda=10.0, tree_method="hist")
LGB = dict(n_estimators=400, num_leaves=15, max_depth=4, learning_rate=0.05, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, min_child_samples=200, reg_lambda=10.0, deterministic=True, force_row_wise=True,
           verbose=-1)
META = ["symbol", "date", "cik", "track", "trig", "s", "entry_t", "entry", "atr"]


def outdir(cache):
    return C.chdir(cache) / "study"


def load_rows(cache, track=None):
    parts = [pd.read_parquet(f) for f in sorted((C.chdir(cache) / "rows").glob("*.parquet"))]
    r = pd.concat([p for p in parts if len(p)], ignore_index=True)
    r["month"] = r["date"].dt.strftime("%Y-%m")
    r["k"] = r.groupby("month").cumcount()           # row index inside its month's image file
    r = r[r["date"] <= LAST_ENTRY]
    return r if track is None else r[r["track"] == track].reset_index(drop=True)


def feature_names(rows):
    return [c for c in rows.columns if c not in META + B.LABELS + ["month", "k"]]


# ---------- hand check ----------

def stage_check(cache):
    if CHECK.exists():
        print(f"[ch check] {CHECK} exists; not redrawn")
        return
    r = load_rows(cache)
    out = []
    for t in TRACKS:
        g = r[r["track"] == t].reset_index(drop=True)
        k = np.sort(np.random.default_rng(SEED).choice(len(g), size=min(CHECK_N, len(g)), replace=False))
        out.append(g.iloc[k][["track", "symbol", "date", "trig", "s", "entry_t", "entry", "vwap_dist", "rvol", "gap",
                              "d_prev_c"]])
    c = pd.concat(out)
    c["date"] = c["date"].dt.strftime("%Y-%m-%d")
    c["start"] = c["date"]                           # tracks/hand_check.py pulls this day's 5-minute bars
    c["end"] = c["date"]
    c["timeframe"] = "5Min"
    c["ok"], c["note"] = "", ""
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(CHECK, index=False)
    print(f"[ch check] wrote {CHECK} ({len(c)} rows)")


def check_right():
    if not CHECK.exists():
        return None
    c = pd.read_csv(CHECK)
    ok = c["ok"].astype(str).str.lower()
    if not ok.isin(["y", "n"]).all():
        return None
    return {t: int((ok[c["track"] == t] == "y").sum()) for t in TRACKS}


# ---------- walk-forward ----------

def splits(rows, Y):
    y = rows["date"].dt.year
    start = pd.Timestamp(f"{Y}-01-01")
    fit_a = (y >= FIRST_TRAIN) & (y <= Y - 2)
    val = y == Y - 1
    fit_b = (y >= FIRST_TRAIN) & (rows["date"] < start - pd.Timedelta(days=PURGE_DAYS))
    test = y == Y
    return fit_a.to_numpy(), val.to_numpy(), fit_b.to_numpy(), test.to_numpy()


def decide(pred, cut, scale):
    """Largest of the six predictions: its label index, whether it trades, and its size (2% at the cut, 10% at
    the validation year's 99th percentile and above, linear in the validation percentile)."""
    best = pred.max(axis=1)
    pick = pred.argmax(axis=1)
    pct = np.interp(best, scale, np.linspace(0, 1, len(scale)))
    size = SIZE_LO + (SIZE_HI - SIZE_LO) * np.clip((pct - CUT_Q) / (TOP_Q - CUT_Q), 0, 1)
    return pick, best > cut, size


def huber_delta(y):
    return float(np.nanquantile(np.abs(y), 0.9))


def fit_trees(kind, X, Y, seed):
    """One regressor per label; rows with a missing label are left out of that label's fit."""
    models = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        d = huber_delta(Y[ok, j])
        if kind == "xgb":
            import xgboost as xgb
            m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=d, random_state=seed, **XGB)
        else:
            import lightgbm as lgb
            m = lgb.LGBMRegressor(objective="huber", alpha=d, random_state=seed, **LGB)
        m.fit(X[ok], Y[ok, j])
        models.append(m)
    return models


def predict(models, X):
    return np.column_stack([m.predict(X) for m in models])


def walk_forward(rows, fit, predict_fn, name, log=print):
    """Decisions for every test year (and the in-sample decisions of each year's final fit, for rule 21)."""
    X = rows[feature_names(rows)].to_numpy(np.float32)
    Yl = rows[B.LABELS].to_numpy(np.float64)
    out, ins = [], []
    for Y in range(FIRST_TEST, LAST_ENTRY.year + 1):
        a, v, b, t = splits(rows, Y)
        if not t.any() or not v.any():
            continue
        t0 = time.monotonic()
        m = fit(X[a], Yl[a], SEED + Y)
        pv = predict_fn(m, X[v]).max(axis=1)
        cut = float(np.quantile(pv, CUT_Q))
        scale = np.quantile(pv, np.linspace(0, 1, 101))
        m = fit(X[b], Yl[b], SEED + Y)
        for mask, sink in ((t, out), (b, ins)):
            p = predict_fn(m, X[mask])
            pick, trade, size = decide(p, cut, scale)
            d = rows.loc[mask, ["symbol", "date", "track", "month", "k", "trig"]].copy()
            d["label"] = np.array(B.LABELS)[pick]
            d["net"] = Yl[mask][np.arange(mask.sum()), pick]
            d["trade"], d["size"], d["pred"], d["year"] = trade, size, p.max(axis=1), Y
            sink.append(d)
        log(f"[ch {name}] {Y}: cut {cut:.4%}, {int(trade.sum())} trades of {int(t.sum())} "
            f"({time.monotonic() - t0:.0f}s)")
    return pd.concat(out, ignore_index=True), pd.concat(ins, ignore_index=True)


def stage_trees(cache):
    od = outdir(cache)
    od.mkdir(parents=True, exist_ok=True)
    for t in TRACKS:
        rows = load_rows(cache, t)
        for kind in ("xgb", "lgbm"):
            f = od / f"{kind}_{t}.parquet"
            if f.exists():
                continue
            dec, ins = walk_forward(rows, lambda X, Y, s, k=kind: fit_trees(k, X, Y, s), predict, f"{kind}-{t}")
            ins.to_parquet(od / f"{kind}_{t}_insample.parquet")
            dec.to_parquet(f)


# ---------- image arm ----------

H, W = 64, 96                                       # per panel; price in the top 48 rows, volume below


def render(intraday, daily):
    """(3, 64, 192) picture: left panel yesterday and today's 5-minute bars to s with VWAP and the 9 EMA, right
    panel the last 60 daily bars with the 20 and 50-day averages. Channel 0 bars, 1 lines, 2 volume."""
    img = np.zeros((3, H, 2 * W), np.float32)
    for side, (bars, lines) in enumerate(((intraday, "intra"), (daily, "daily"))):
        ok = np.isfinite(bars[:, 3])
        if ok.sum() < 2:
            continue
        b = bars[ok]
        n = len(b)
        cols = side * W + np.minimum((np.arange(n) * W) // max(n, 1), W - 1)
        c = b[:, 3]
        if lines == "intra":
            v = np.nan_to_num(b[:, 4])
            day2 = np.arange(n) >= max(0, n - int(np.isfinite(intraday[78:, 3]).sum()))
            vw = np.where(day2, np.cumsum(c * v * day2) / np.maximum(np.cumsum(v * day2), 1), np.nan)
            ls = [vw, B.ema(c, 9)]
        else:
            s = pd.Series(c)
            ls = [s.rolling(20).mean().to_numpy(), s.rolling(50).mean().to_numpy()]
        lo = np.nanmin(np.r_[b[:, 2], *[x[np.isfinite(x)] for x in ls]])
        hi = np.nanmax(np.r_[b[:, 1], *[x[np.isfinite(x)] for x in ls]])
        if not hi > lo:
            continue
        y = lambda p: np.clip(((hi - p) / (hi - lo) * 47).round(), 0, 47).astype(int)
        for i in range(n):
            img[0, y(b[i, 1]):y(b[i, 2]) + 1, cols[i]] = 1.0
            img[0, y(b[i, 0]), max(cols[i] - 1, side * W)] = 1.0
            img[0, y(b[i, 3]), min(cols[i] + 1, side * W + W - 1)] = 1.0
            for x in ls:
                if np.isfinite(x[i]):
                    img[1, y(x[i]), cols[i]] = 1.0
        vol = np.nan_to_num(b[:, 4])
        if vol.max() > 0:
            hv = (vol / vol.max() * 15).round().astype(int)
            for i in range(n):
                img[2, H - hv[i]:H, cols[i]] = 1.0
    return img


def images(cache, rows):
    """Rendered pictures for `rows` (month, k), month file by month file."""
    out = np.zeros((len(rows), 3, H, 2 * W), np.uint8)
    pos = {m: g.index.to_numpy() for m, g in rows.reset_index(drop=True).groupby("month")}
    ks = rows["k"].to_numpy()
    for m, idx in pos.items():
        z = np.load(C.chdir(cache) / "img" / f"{m}.npz")
        intr, dly = z["intraday"], z["daily"]
        for i in idx:
            out[i] = render(intr[ks[i]], dly[ks[i]]).astype(np.uint8)
    return out


def cnn():
    import torch.nn as nn
    blk = lambda i, o: [nn.Conv2d(i, o, (5, 3), padding=(2, 1)), nn.BatchNorm2d(o), nn.LeakyReLU(0.01),
                        nn.MaxPool2d((2, 1))]
    return nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), nn.Flatten(), nn.Dropout(0.5),
                         nn.Linear(128 * (H // 8) * 2 * W, 6))


def fit_cnn(imgs, Y, seed, max_rows=None):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Y)) if max_rows is None or len(Y) <= max_rows else np.sort(rng.choice(len(Y), max_rows, False))
    va = rng.random(len(idx)) < 0.1
    tr_i, va_i = idx[~va], idx[va]
    Yf = np.nan_to_num(Y.astype(np.float32))
    mask = np.isfinite(Y).astype(np.float32)
    d = torch.tensor([huber_delta(Y[:, j][np.isfinite(Y[:, j])]) for j in range(Y.shape[1])])
    net = cnn()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_f = torch.nn.HuberLoss(reduction="none", delta=1.0)
    batch = lambda ii: (torch.tensor(imgs[ii], dtype=torch.float32), torch.tensor(Yf[ii] / d.numpy()),
                        torch.tensor(mask[ii]))
    best, best_state, bad = np.inf, None, 0
    for ep in range(10):
        net.train()
        perm = rng.permutation(len(tr_i))
        for s in range(0, len(tr_i), 256):
            x, y, w = batch(np.sort(tr_i[perm[s:s + 256]]))
            opt.zero_grad()
            loss = (loss_f(net(x), y) * w).sum() / w.sum().clamp(min=1)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = []
            for s in range(0, len(va_i), 1024):
                x, y, w = batch(va_i[s:s + 1024])
                vl.append(float((loss_f(net(x), y) * w).sum() / w.sum().clamp(min=1)))
        v = float(np.mean(vl)) if vl else 0.0
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 1:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.scale = d
    return net


def predict_cnn(net, imgs):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), 1024):
            out.append((net(torch.tensor(imgs[s:s + 1024], dtype=torch.float32)) * net.scale).numpy())
    return np.concatenate(out)


def stage_img(cache, deadline, max_rows=None):
    """One (track, year) at a time; each finished year is saved, so a rerun continues."""
    od = outdir(cache) / "img"
    od.mkdir(parents=True, exist_ok=True)
    for t in TRACKS:
        if (outdir(cache) / f"img_{t}.parquet").exists():
            continue
        rows = load_rows(cache, t)
        pics = images(cache, rows)
        Yl = rows[B.LABELS].to_numpy(np.float64)
        for Y in range(FIRST_TEST, LAST_ENTRY.year + 1):
            f = od / f"{t}_{Y}.parquet"
            if f.exists():
                continue
            if time.monotonic() > deadline:
                return False
            a, v, b, te = splits(rows, Y)
            if not te.any() or not v.any():
                continue
            m = fit_cnn(pics[a], Yl[a], SEED + Y, max_rows)
            pv = predict_cnn(m, pics[v]).max(axis=1)
            cut, scale = float(np.quantile(pv, CUT_Q)), np.quantile(pv, np.linspace(0, 1, 101))
            m = fit_cnn(pics[b], Yl[b], SEED + Y, max_rows)
            parts = []
            for mask, kind in ((te, "test"), (b, "insample")):
                p = predict_cnn(m, pics[mask])
                pick, trade, size = decide(p, cut, scale)
                d = rows.loc[mask, ["symbol", "date", "track", "month", "k", "trig"]].copy()
                d["label"] = np.array(B.LABELS)[pick]
                d["net"] = Yl[mask][np.arange(mask.sum()), pick]
                d["trade"], d["size"], d["pred"], d["year"], d["kind"] = trade, size, p.max(axis=1), Y, kind
                parts.append(d)
            pd.concat(parts).to_parquet(f)
            print(f"[ch img-{t}] {Y}: cut {cut:.4%}")
        allp = pd.concat([pd.read_parquet(f) for f in sorted(od.glob(f"{t}_*.parquet"))])
        allp[allp["kind"] == "insample"].drop(columns="kind").to_parquet(outdir(cache) / f"img_{t}_insample.parquet")
        allp[allp["kind"] == "test"].drop(columns="kind").to_parquet(outdir(cache) / f"img_{t}.parquet")
    return True


# ---------- measure ----------

def with_control(dec, rows):
    """Each pick's control: another stock from the same day's triggered candidates of the same track, drawn
    with default_rng(SEED + date ordinal), traded with the pick's side and stop/target pair (its own label)."""
    by_day = {d: g for d, g in rows.groupby("date")}
    picks = dec[dec["trade"]].sort_values(["date", "symbol"]).reset_index(drop=True)
    ctrl = np.full(len(picks), np.nan)
    for d, g in picks.groupby("date"):
        pool = by_day[d].sort_values("symbol")
        rng = np.random.default_rng(SEED + d.toordinal())
        for i, r in g.iterrows():
            cand = pool[pool["symbol"] != r["symbol"]]
            if cand.empty:
                continue
            ctrl[i] = cand.iloc[int(rng.integers(len(cand)))][r["label"]]
    return picks.assign(ctrl=ctrl)


def daily(p, weight=False):
    p = p.dropna(subset=["net", "ctrl"])
    if weight:
        f = lambda g: pd.Series({"pick": np.average(g["net"], weights=g["size"]),
                                 "ctrl": np.average(g["ctrl"], weights=g["size"]), "n": len(g)})
        w = p.groupby("date").apply(f)
    else:
        w = p.groupby("date").agg(pick=("net", "mean"), ctrl=("ctrl", "mean"), n=("net", "size"))
    return w.assign(diff=w["pick"] - w["ctrl"])


def test(w, ff5, umd):
    m, se, t = nw_tstat(w["diff"], LAGS)
    lo, hi = block_bootstrap_ci(w["diff"], BLOCK, N_BOOT, SEED)
    pm, _, pt = nw_tstat(w["pick"], LAGS)
    mon = w["diff"].groupby(w.index.to_period("M")).sum()
    a = stats.factor_alpha(mon.to_numpy(float), mon.index, ff5, umd, 1) if len(mon) > 8 else None
    return {"trades": int(w["n"].sum()), "days": int(len(w)), "mean_minus_control": m, "se_hac": se, "t_hac": t,
            "ci95": [lo, hi], "picks_mean_net": pm, "picks_t": pt, "control_mean_net": float(w["ctrl"].mean()),
            "ff5_mom_alpha_monthly": a}


def stage_measure(cache, out):
    ok = check_right()
    res = {"idea": "CH", "prereg": "research/prereg/CH.md", "check_right": ok}
    out.mkdir(parents=True, exist_ok=True)
    if ok is None:
        raise SystemExit("[ch] fill in research/checks/CH.csv (ok = y/n) first")
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts()
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau, "arms": {}}
    od = outdir(cache)
    for t in TRACKS:
        if ok[t] < CHECK_PASS:
            res["arms"][t] = f"hand check {ok[t]}/{CHECK_N} < {CHECK_PASS}: nothing measured"
            continue
        rows = load_rows(cache, t)
        for kind, arm in (("xgb", "CHX"), ("lgbm", "CHL"), ("img", "CHI")):
            name = f"{arm}-{'M' if t == 'main' else 'P'}"
            f = od / f"{kind}_{t}.parquet"
            if not f.exists():
                res["arms"][name] = "not run"
                continue
            p = with_control(pd.read_parquet(f), rows)
            ins = with_control(pd.read_parquet(od / f"{kind}_{t}_insample.parquet"), rows)
            w, wi = daily(p), daily(ins)
            prim = test(w, ff5, umd)
            r21 = {"in_sample_mean_diff": float(wi["diff"].mean()), "out_of_sample_mean_diff": float(w["diff"].mean())}
            r21["overfit"] = bool(r21["out_of_sample_mean_diff"] <= 0 and r21["in_sample_mean_diff"] > 0) or bool(
                r21["out_of_sample_mean_diff"] > 0 and r21["in_sample_mean_diff"] > 2 * r21["out_of_sample_mean_diff"])
            a = prim["ff5_mom_alpha_monthly"]
            prim["passes_rule12"] = bool(prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0 and prim["picks_mean_net"] > 0
                                         and prim["trades"] >= MIN_TRADES and prim["days"] >= MIN_DAYS)
            prim["passes_rule11"] = bool(a is not None and a["alpha"] > 0 and a["t"] >= 2.0)
            prim["deflated_sharpe"] = stats.deflated_sharpe(w["diff"], both)
            prim["passes"] = bool(not r21["overfit"] and prim["passes_rule12"] and prim["passes_rule11"])
            prim["near_miss_rule19"] = bool(not prim["passes"] and not r21["overfit"] and prim["t_hac"] >= 1.5
                                            and prim["picks_mean_net"] > 0)
            ex = {"size_weighted": test(daily(p, weight=True), ff5, umd),
                  "long": test(daily(p[p["label"].str.startswith("long")]), ff5, umd),
                  "short": test(daily(p[p["label"].str.startswith("short")]), ff5, umd),
                  "pair_share": p["label"].value_counts(normalize=True).round(4).to_dict(),
                  "by_trigger": {k: float(daily(g)["diff"].mean()) for k, g in p.groupby("trig")},
                  "by_year": {str(y): float(g["diff"].mean()) for y, g in w.groupby(w.index.year)}}
            res["arms"][name] = {"rule21": r21, "primary": prim, "exploratory": ex}
            p.to_parquet(od / f"{kind}_{t}_trades.parquet")
        # benchmark: opening-range break, long above / short below, 1.5x/3x, on orb triggers
        orb = rows[rows["trig"] == "orb"].copy()
        orb["label"] = np.where(orb["d_or_h"] > 0, "long_1.5", "short_1.5")
        orb["net"] = orb[["long_1.5", "short_1.5"]].to_numpy()[np.arange(len(orb)), (orb["d_or_h"] <= 0).astype(int)]
        orb["trade"], orb["size"] = True, SIZE_LO
        res[f"orb_benchmark_{t}"] = test(daily(with_control(orb, rows)), ff5, umd)
    (out / "CH_result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["check", "trees", "img", "measure"], required=True)
    ap.add_argument("--cache", default="cache/w1")
    ap.add_argument("--out", default="research/results/CH")
    ap.add_argument("--budget-min", type=float, default=320)
    ap.add_argument("--img-max-rows", type=int, default=None)
    a = ap.parse_args()
    deadline = time.monotonic() + 60 * a.budget_min
    if a.stage == "check":
        stage_check(a.cache)
    elif a.stage == "trees":
        stage_trees(a.cache)
    elif a.stage == "img":
        stage_img(a.cache, deadline, a.img_max_rows)
    else:
        stage_measure(a.cache, Path(a.out))


if __name__ == "__main__":
    main()
