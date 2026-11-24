"""ANOM1: does an anomaly score (Isolation Forest + CUSUM on the stock's own last 60 sessions) improve BOOM1's
nightly top 20? (research/prereg/ANOM1.md)

    python -m tracks.main.anom1_study --stage candidates|run --cache cache/nx1 --out research/results/MODELS5

candidates: BOOM1's walk-forward (as boom2_study.stage_picks), keeping each night's top POOL by BOOM1 score with
their anomaly scores -> cache nx3/anom1/candidates.parquet. run: w chosen on 2019-2023, one look 2024-01..2025-09.
Summaries only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main import boom1_study as B
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

POOL = 100                      # BOOM1's top 100 per night are re-ranked (Amendment 1)
HIST = 60
TREES, IF_SEED = 200, 7
CUSUM_K, CUSUM_H, CUSUM_RUN = 0.5, 4.0, 5
W_GRID = (0.25, 0.5, 1.0)
CHOOSE = (2019, 2020, 2021, 2022, 2023)
TEST = (2024, 2025)
ANOM_COLS = ["lv", "rg", "gap", "r1", "ddv"]


def adir(cache):
    return X.nx3dir(cache) / "anom1"


# ---------- pure helpers (tested) ----------

def c_n(n):
    """Average path length of an unsuccessful search in a binary search tree of n points (Liu et al. 2008)."""
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2 * (np.log(n[big] - 1) + np.euler_gamma) - 2 * (n[big] - 1) / n[big]
    out[n == 2] = 1.0
    return out


def iforest_score(train, x, rng, trees=TREES):
    """Isolation Forest anomaly score of one point x against `train` (n x f), using all n points per tree (as
    sklearn with n <= 256) and depth limit ceil(log2 n). Only x's own path is grown, which gives the same path
    length distribution as growing whole trees. Higher = more unusual (0..1, 0.5 = ordinary)."""
    train, x = np.asarray(train, float), np.asarray(x, float)
    n, f = train.shape
    limit = int(np.ceil(np.log2(max(n, 2))))
    mask = np.ones((trees, n), bool)
    depth = np.zeros(trees)
    active = np.ones(trees, bool)
    for _ in range(limit):
        active &= mask.sum(1) > 1
        if not active.any():
            break
        feat = rng.integers(0, f, trees)
        vals = train[:, feat].T
        lo = np.where(mask, vals, np.inf).min(1)
        hi = np.where(mask, vals, -np.inf).max(1)
        active &= hi > lo
        split = lo + rng.random(trees) * (hi - lo)
        left = x[feat] < split
        keep = mask & np.where(left[:, None], vals < split[:, None], vals >= split[:, None])
        mask = np.where(active[:, None], keep, mask)
        depth += active
    h = depth + c_n(mask.sum(1))
    return float(2 ** (-h.mean() / c_n([n])[0]))


def cusum_alarm(hist, recent, k=CUSUM_K, h=CUSUM_H):
    """Two-sided CUSUM on `recent` values (oldest first, the last one is today) standardised by the mean and sd of
    `hist`; True when either sum is above h at today."""
    hist = np.asarray(hist, float)
    sd = hist.std()
    if not sd > 0:
        return False
    z = (np.asarray(recent, float) - hist.mean()) / sd
    hi = lo = 0.0
    for v in z:
        hi, lo = max(0.0, hi + v - k), max(0.0, lo - v - k)
    return bool(hi > h or lo > h)


def combined_top(g, w, top=B.TOP):
    """Top `top` of one night's pool by percentile rank of BOOM1's score + w x percentile rank of the anomaly score."""
    s = g["score"].rank(pct=True) + w * g["anom"].rank(pct=True)
    return g.loc[s.sort_values(ascending=False, kind="stable").index[:top]]


# ---------- stages ----------

def anomaly_inputs(cache, symbols):
    """Per (symbol, date): log relative volume, range / close, gap, return and the change in log dollar volume,
    from the chart cache (one series per segment)."""
    d = X.nx3dir(cache) / X.CHART
    parts = []
    for f in sorted(d.glob("c_*.parquet")):
        x = pd.read_parquet(f, columns=["symbol", "date", "seg", "vrel20", "rg_0", "gap", "r1", "log_dv"])
        parts.append(x[x["symbol"].isin(symbols)])
    x = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    x["lv"] = np.log(x["vrel20"].where(x["vrel20"] > 0))
    x["rg"] = x["rg_0"]
    x["ddv"] = x.groupby(["symbol", "seg"])["log_dv"].diff()
    return x[["symbol", "date", "seg", *ANOM_COLS]]


def score_candidates(cand, panel):
    """IF score and CUSUM alarm for each candidate from its own prior HIST sessions (same segment, complete rows)."""
    rng = np.random.default_rng(IF_SEED)
    by = {s: g.reset_index(drop=True) for s, g in panel.groupby("symbol")}
    ifs, alarm = np.full(len(cand), np.nan), np.zeros(len(cand), bool)
    for i, (s, d) in enumerate(zip(cand["symbol"], cand["d"])):
        g = by.get(s)
        if g is None:
            continue
        k = g["date"].searchsorted(d)
        if k >= len(g) or g["date"].iloc[k] != d or k < HIST:
            continue
        w = g.iloc[k - HIST:k + 1]
        if w["seg"].nunique() != 1:
            continue
        v = w[ANOM_COLS].to_numpy(float)
        if not np.isfinite(v).all():
            continue
        ifs[i] = iforest_score(v[:-1], v[-1], rng)
        alarm[i] = cusum_alarm(v[:-1, 0], v[-CUSUM_RUN:, 0])
    return ifs, alarm


def stage_candidates(cache):
    df = B.load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    out = []
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = df[df["d"] <= cut_d]
        te = df[(df["d"] >= start) & (df["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        if len(tr) > B.TRAIN_ROWS:
            tr = tr.iloc[np.sort(np.random.default_rng(B.SEED + yr).choice(len(tr), B.TRAIN_ROWS, replace=False))]
        m = B.classifier().fit(tr[B.INPUTS].to_numpy(np.float32), tr["boom"].to_numpy())
        te = te.assign(score=m.predict_proba(te[B.INPUTS].to_numpy(np.float32))[:, 1])
        for _, g in te.groupby("d", sort=True):
            if len(g) >= B.TOP * 5:
                out.append(g.nlargest(POOL, "score")[["symbol", "d", "nxt", "score", "boom"]].assign(n_night=len(g)))
        print(f"[anom1 candidates] {yr}: {sum(len(x) for x in out)} so far")
    cand = pd.concat(out, ignore_index=True)
    panel = anomaly_inputs(cache, set(cand["symbol"]))
    ifs, alarm = score_candidates(cand, panel)
    cand["if_score"], cand["cusum"] = ifs, alarm
    N.atomic_parquet(cand, adir(cache) / "candidates.parquet")
    print(f"[anom1 candidates] {len(cand)} rows, IF score on {np.isfinite(ifs).mean():.1%}, CUSUM alarms {alarm.mean():.1%}")


def nightly(c, pick):
    """Per night: share of the 20 picks (pick(g) -> rows) that moved 8%+, beside BOOM1's own top 20."""
    rows = []
    for d, g in c.groupby("d", sort=True):
        rows.append({"d": d, "x": float(pick(g)["boom"].mean()), "boom1": float(g.nlargest(B.TOP, "score")["boom"].mean())})
    return pd.DataFrame(rows)


def stage_run(cache, out):
    c = pd.read_parquet(adir(cache) / "candidates.parquet")
    c["anom"] = c["if_score"].fillna(0.5) + c["cusum"].astype(float)   # no history: an ordinary score
    c["year"] = c["d"].dt.year
    ch, te = c[c["year"].isin(CHOOSE)], c[c["year"].isin(TEST)]
    choose = {}
    for w in W_GRID:
        n = nightly(ch, lambda g, w=w: combined_top(g, w))
        choose[str(w)] = float((n["x"] - n["boom1"]).mean())
    w = float(max(W_GRID, key=lambda k: choose[str(k)]))
    n = nightly(te, lambda g: combined_top(g, w))
    gain = X.tstat((n["x"] - n["boom1"]).to_numpy(float))
    rng = np.random.default_rng(B.SEED)
    alone = nightly(te, lambda g: g.nlargest(B.TOP, "anom"))
    rand = nightly(te, lambda g: g.sample(B.TOP, random_state=int(rng.integers(1 << 30))))
    cus = nightly(te, lambda g: g.sort_values(["cusum", "score"], ascending=False).head(B.TOP))
    changed = [B.TOP - len(set(combined_top(g, w)["symbol"]) & set(g.nlargest(B.TOP, "score")["symbol"]))
               for _, g in te.groupby("d")]
    rep = {"rule": "ANOM1 passes if the nightly share of the top 20 moving 8%+, ANOM1 minus BOOM1 alone, has mean > 0 and "
                   "Newey-West t >= 2 (2024-01..2025-09)",
           "pool_per_night": POOL, "rows": int(len(c)), "if_scored_share": float(c["if_score"].notna().mean()),
           "cusum_alarm_share": float(c["cusum"].mean()), "choose_2019_2023_gain": choose, "w": w,
           "test": {"nights": int(len(n)), "boom1_hit": float(n["boom1"].mean()), "anom1_hit": float(n["x"].mean()),
                    "gain": gain, "picks_changed_per_night": float(np.mean(changed))},
           "reported": {"anomaly_alone_in_pool_hit": float(alone["x"].mean()), "random_20_of_pool_hit": float(rand["x"].mean()),
                        "cusum_first_hit": float(cus["x"].mean())}}
    rep["verdict"] = "passed" if gain["mean"] > 0 and gain["t_nw"] >= 2 else "failed"
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "ANOM1_result.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["candidates", "run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/MODELS5")
    a = ap.parse_args()
    cache = Path(a.cache)
    stage_candidates(cache) if a.stage == "candidates" else stage_run(cache, a.out)


if __name__ == "__main__":
    main()
