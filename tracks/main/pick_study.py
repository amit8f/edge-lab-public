"""PICK-MAIN and PICK-PENNY study (research/prereg/PICK.md, Amendment 1): news join and fit.

Stage join (every month 2018-12 .. 2025-09, resumable): each sampled decision row from pick_data gets
- the NX1 news inputs from single-symbol headlines with prior-session 16:00 ET <= t < the decision time,
- SPY's gap (today's open vs prior close), 5-session return and 20-day volatility, all from data before the
  decision (SPY's move since the open is the row's own spy_open from pick_data),
- the labels y_<w> = 1 if net_<w> >= 0.03 (NaN where the trade is undefined).
  Written atomically to <state>/joined/YYYY-MM.parquet.

Stage fit (train 2019-01..2022-12, validate 2023, refit 2019-01..2023-12). It reads only 2019-2023 months
(read_joined refuses any later month) and writes <out>/PICK_fit.json and <out>/models/.

Usage: python -m tracks.main.pick_study --stage join|fit --rows cache/w1 --nx1 cache/nx1 \
           --out research/results/PICK --state cache/pickfit
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tracks.main import pick_data as P
from tracks.news.nx1_study import FLAGS

HOLD_RET = 0.03                                    # the label: a trade that ends +3% or better
CAP = 1000.0                                       # minutes: no headline in the window
PRIOR_CLOSE_NS = 16 * 3600 * 10**9                 # 16:00 of the prior session
MIN_NS = 60 * 10**9
MID = {"main": 1.5, "penny": 2.0}                  # the middle stop width of each track
TIERS = {"main": 3, "penny": 1}
NORMAL_Q = (0.001, 0.0005, 0.0002)                 # top shares of normal-day training scores
MULTS = (1, 2, 4)                                  # bad-day strictness: the top share is divided by this
MIN_SEL = 100
SAMPLE_N = 20000
SEED = 0
TRAIN_YEARS, VAL_YEAR = (2019, 2022), 2023
FIT_YEARS = (2019, 2023)
FIRST, LAST = "2018-12", "2025-09"
LEARNERS = ["XGBoost", "CatBoost", "LightGBM", "ExtraTrees", "Logistic", "KRR"]
SAMPLED = {"ExtraTrees", "KRR"}
KRR_CHUNK, TREE_CHUNK = 2000, 200000

FLAG_COLS = [f"f_{k}" for k in FLAGS]
CHART = ["mins_open", "gap", "r_open", "r_prev", "r_5m", "r_15m", "r_30m", "hi_dist", "lo_dist", "range_open",
         "vol_pace", "swing30", "log_px", "log_dv", "spy_open"]
NEWS_ONLY = ["n_news", "fb_pos_mean", "fb_pos_max", "fb_neg_mean", "fb_neg_max", "lm_net", *FLAG_COLS, "mins_last"]
NEWS = NEWS_ONLY + ["log_px", "mins_open"]
COMBINED = CHART + NEWS_ONLY + ["market_score"]
VERSIONS = {"NEWS": NEWS, "CHART": CHART, "COMBINED": COMBINED}
MARKET = ["spy_gap", "spy_r5", "spy_vol20", "spy_open"]
FIT_NUM = sorted(set(CHART + NEWS_ONLY + MARKET))
NEWS_TONE = ["symbol", "t", "lm_neg", "lm_pos", "fb_pos", "fb_neg", *FLAG_COLS]


def naive(x):
    """Datetimes as tz-naive pandas datetime64[ns] Series."""
    s = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    return s.dt.tz_localize(None) if s.dt.tz is not None else s


def to_ns(x):
    return naive(x).to_numpy(dtype="datetime64[ns]").astype("int64")


def atomic(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


def atomic_json(obj, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(f)


# ---------- join (pure helpers, tested) ----------

def label(net):
    """y = 1 if the trade ends +3% or better after costs; NaN where the trade is undefined."""
    net = np.asarray(net, dtype=float)
    y = (net >= HOLD_RET).astype(float)
    y[np.isnan(net)] = np.nan
    return y


def spy_parts(spy):
    """SPY daily (date, ao, ac, c) -> per date: gap (the day's open vs the prior close, adjusted prices),
    5-session return and 20-day volatility of the log returns, both ending at the prior close."""
    s = spy.assign(date=naive(spy["date"]).to_numpy()).sort_values("date").set_index("date")
    ac = s["ac"].astype(float)
    lr = np.log(ac).diff()
    return pd.DataFrame({"spy_gap": s["ao"].astype(float) / ac.shift(1) - 1,
                         "spy_r5": ac.shift(1) / ac.shift(6) - 1,
                         "spy_vol20": lr.shift(1).rolling(20).std()})


def prior_open(d_ns, sess_ns):
    """The window start for each decision day: the prior session's 16:00 ET (sess_ns sorted, midnight ns)."""
    idx = np.clip(np.searchsorted(sess_ns, d_ns, side="left") - 1, 0, None)
    return sess_ns[idx] + PRIOR_CLOSE_NS


def news_features(rows, tone, lo):
    """NX1 inputs of the single-symbol headlines with lo <= t < the row's t (lo: window start per row, ns).
    No headline: counts, tone, flags 0 and minutes = CAP (the NX1 no-news values)."""
    out = np.zeros((len(rows), len(NEWS_ONLY)))
    c = {name: i for i, name in enumerate(NEWS_ONLY)}
    out[:, c["mins_last"]] = CAP
    t = to_ns(rows["t"])
    if len(rows) == 0 or len(tone) == 0:
        return pd.DataFrame(out, columns=NEWS_ONLY, index=rows.index)
    tone = tone.assign(tn=to_ns(tone["t"])).sort_values(["symbol", "tn"], kind="stable")
    by = {s: g for s, g in tone.groupby("symbol", sort=False)}
    for s, pos in rows.groupby("symbol").indices.items():
        g = by.get(s)
        if g is None:
            continue
        T = g["tn"].to_numpy()
        fbp, fbn = g["fb_pos"].to_numpy(float), g["fb_neg"].to_numpy(float)
        lm = g["lm_pos"].to_numpy(float) - g["lm_neg"].to_numpy(float)
        F = g[FLAG_COLS].to_numpy(float)
        i0 = np.searchsorted(T, lo[pos], side="left")
        i1 = np.searchsorted(T, t[pos], side="left")          # strictly before the decision time
        for j, p in enumerate(pos):
            a, b = i0[j], i1[j]
            if b <= a:
                continue
            out[p, c["n_news"]] = b - a
            out[p, c["fb_pos_mean"]], out[p, c["fb_pos_max"]] = fbp[a:b].mean(), fbp[a:b].max()
            out[p, c["fb_neg_mean"]], out[p, c["fb_neg_max"]] = fbn[a:b].mean(), fbn[a:b].max()
            out[p, c["lm_net"]] = lm[a:b].sum()
            out[p, [c[f] for f in FLAG_COLS]] = F[a:b].max(axis=0)
            out[p, c["mins_last"]] = min(CAP, (t[p] - T[b - 1]) / MIN_NS)
    return pd.DataFrame(out, columns=NEWS_ONLY, index=rows.index)


def join_rows(rows, tone, sess_ns, parts):
    """One month of decision rows -> the same rows with news inputs, SPY parts and labels."""
    out = rows.copy()
    lo = prior_open(to_ns(rows["d"]), sess_ns)
    news = news_features(rows, tone, lo)
    for c in NEWS_ONLY:
        out[c] = news[c].to_numpy()
    mk = parts.reindex(pd.DatetimeIndex(naive(rows["d"]))).to_numpy()
    for i, c in enumerate(["spy_gap", "spy_r5", "spy_vol20"]):
        out[c] = mk[:, i]
    for col in [c for c in rows.columns if c.startswith("net_")]:
        out[f"y_{col[4:]}"] = label(rows[col])
    return out


def read_tone(nx1, per):
    f = Path(nx1) / "nx1" / "tone" / f"tone_{per}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"NX1 tone for {per} is missing: {f}")
    return pd.read_parquet(f, columns=NEWS_TONE)


def stage_join(rows_cache, nx1, state):
    spy = pd.read_parquet(Path(nx1) / "nx1" / "spy.parquet")
    sess = np.unique(to_ns(spy["date"]))
    parts = spy_parts(spy)
    for m in pd.period_range(FIRST, LAST, freq="M"):
        f = Path(state) / "joined" / f"{m}.parquet"
        if f.exists():
            continue
        rf = P.pdir(rows_cache) / "rows" / f"{m}.parquet"
        if not rf.exists():
            raise FileNotFoundError(f"pick rows for {m} are missing: {rf}")
        rows = pd.read_parquet(rf)
        if len(rows) == 0:
            atomic(pd.DataFrame(), f)
            continue
        tone = pd.concat([read_tone(nx1, str(p)) for p in (m - 1, m)], ignore_index=True)
        atomic(join_rows(rows, tone, sess, parts), f)
        print(f"[pick join] {m}: {len(rows)} rows", flush=True)


# ---------- fit (pure helpers, tested) ----------

def fit_year_ok(month):
    """The fit stage reads 2019-2023 only; anything later is refused."""
    year = int(str(month)[:4])
    if not FIT_YEARS[0] <= year <= FIT_YEARS[1]:
        raise ValueError(f"the fit stage may not read {month} (2019-2023 only)")


def read_joined(state, month, cols):
    fit_year_ok(month)
    f = Path(state) / "joined" / f"{month}.parquet"
    names = pq.read_schema(f).names
    if not names:
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(f, columns=[c for c in cols if c in names])
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan                       # a label of a width no row of this month has
    return df


def load(state, track):
    widths = sorted({w for t in P.TRACKS for w in P.TRACKS[t]["widths"]})
    cols = ["track", "d", *FIT_NUM, *[f"y_{w}" for w in widths]]
    parts = []
    for y in range(FIT_YEARS[0], FIT_YEARS[1] + 1):
        for m in pd.period_range(f"{y}-01", f"{y}-12", freq="M"):
            df = read_joined(state, str(m), cols)
            if len(df):
                parts.append(df[df["track"] == track])
    D = pd.concat(parts, ignore_index=True)
    D["year"] = pd.to_datetime(D["d"]).dt.year.astype(np.int16)
    for c in FIT_NUM + [c for c in D.columns if c.startswith("y_")]:
        D[c] = D[c].astype(np.float32)
    return D


def market_params(d_tr):
    """Mean and std of the four parts: the three day values (one per day) and spy_open (per row)."""
    day = d_tr.drop_duplicates("d")
    p = {}
    for c in ["spy_gap", "spy_r5", "spy_vol20"]:
        p[c] = (float(day[c].mean()), float(day[c].std()))
    p["spy_open"] = (float(d_tr["spy_open"].mean()), float(d_tr["spy_open"].std()))
    return p


def market_score(D, p):
    """Mean of gap_z, spy_open_z, r5_z and -vol20_z (z-scored with the frozen parameters)."""
    z = {c: (D[c].to_numpy(float) - p[c][0]) / p[c][1] for c in MARKET}
    return ((z["spy_gap"] + z["spy_open"] + z["spy_r5"] - z["spy_vol20"]) / 4).astype(np.float32)


def tier_of(x, cuts):
    t = np.zeros(len(x), dtype=int)
    for c in cuts:
        t += (x > c)
    return t


def top_threshold(x, q):
    """The score above which the top share q of x lies."""
    return float(np.quantile(x, 1 - q))


def row_scores(S, tier, wpt):
    """Each row takes the score (or label) of the width model its tier chose."""
    out = np.full(len(tier), np.nan, dtype=np.float32)
    for t, w in wpt.items():
        m = tier == t
        out[m] = S[w][m]
    return out


def choose_width(S, Y, tier, normal, thr, widths, mid, tiers):
    """Per tier: among the top-scored normal-day rows of each width's model (score >= thr[w]), the width with
    the highest share of its own label = 1. Ties go to the width nearest the middle."""
    order = sorted(widths, key=lambda w: (abs(w - mid), w))
    wpt, table = {}, {}
    for t in tiers:
        table[str(t)], valid = {}, {}
        for w in widths:
            sel = normal & (tier == t) & (S[w] >= thr[w]) & ~np.isnan(Y[w])
            n = int(sel.sum())
            share = float(Y[w][sel].mean()) if n else float("nan")
            table[str(t)][str(w)] = {"n": n, "share": share}
            if n:
                valid[w] = share
        wpt[t] = mid if not valid else next(w for w in order if w in valid and valid[w] == max(valid.values()))
    return wpt, table


def choose_bar(s_va, y_va, bad_va, s_tr, bad_tr):
    """Pairs of (normal top share q, bad-day divisor m). Normal days take the top q of the training normal-day
    scores; bad days the top q/m of the same population (the bar is raised). Chosen on 2023: the best share of
    y=1 among selected labelled rows with at least MIN_SEL of them; if none, the most rows, flagged thin."""
    normal_tr = s_tr[~bad_tr]
    rows = []
    for q in NORMAL_Q:
        for m in MULTS:
            tn, tb = top_threshold(normal_tr, q), top_threshold(normal_tr, q / m)
            sel = np.where(bad_va, s_va >= tb, s_va >= tn) & ~np.isnan(y_va)
            n = int(sel.sum())
            rows.append({"normal_q": q, "bad_divisor": m, "thr_normal": tn, "thr_bad": tb, "n": n,
                         "share": float(y_va[sel].mean()) if n else float("nan")})
    ok = [r for r in rows if r["n"] >= MIN_SEL]
    if ok:
        best, flag = max(ok, key=lambda r: r["share"]), None
    else:
        best, flag = max(rows, key=lambda r: r["n"]), "thin"
    return best, rows, flag


def sample_idx(n):
    if n <= SAMPLE_N:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, SAMPLE_N, replace=False))


def make(name, n_features):
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=50, tree_method="hist", random_state=SEED, n_jobs=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=SEED, verbose=False,
                                  thread_count=-1, allow_writing_files=False)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                              verbose=-1, n_jobs=-1)
    if name == "ExtraTrees":
        return Pipeline([imp, ("et", ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20,
                                                          random_state=SEED, n_jobs=-1))])
    if name == "Logistic":
        return Pipeline([imp, ("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    if name == "KRR":
        return Pipeline([imp, ("sc", StandardScaler()),
                         ("krr", KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0 / n_features))])
    raise ValueError(name)


def fit_learner(name, X, y):
    if name in SAMPLED:
        i = sample_idx(len(y))
        X, y = X[i], y[i]
    m = make(name, X.shape[1])
    m.fit(X, y if name == "KRR" else y.astype(int))
    return m


def predict(name, m, X):
    chunk = KRR_CHUNK if name == "KRR" else TREE_CHUNK
    out = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        out.append(m.predict(xb) if name == "KRR" else m.predict_proba(xb)[:, 1])
    p = np.concatenate(out).astype(float) if out else np.empty(0)
    return np.clip(p, 1e-4, 1 - 1e-4) if name == "KRR" else p


EXT = {"XGBoost": ".json", "CatBoost": ".cbm", "LightGBM": ".txt"}


def save_model(name, m, base):
    """Native formats for the boosters, joblib for the rest. `base` is the file name without extension."""
    if name == "LightGBM":
        path = Path(f"{base}{EXT[name]}")
        m.booster_.save_model(str(path))
    elif name in EXT:
        path = Path(f"{base}{EXT[name]}")
        m.save_model(str(path))
    else:
        path = Path(f"{base}.joblib")
        joblib.dump(m, path)
    return str(path)


def fit_version(name, X, Y, year, bad, tier, n_tiers, widths, mid, out, tr):
    TR, VA = (year <= TRAIN_YEARS[1]), (year == VAL_YEAR)
    ym = Y[mid]
    k_tr, k_va = TR & ~np.isnan(ym), VA & ~np.isnan(ym)
    ll = {}
    for L in LEARNERS:                                  # learner: lowest 2023 log-loss on the middle width
        m = fit_learner(L, X[k_tr], ym[k_tr])
        ll[L] = float(log_loss(ym[k_va], predict(L, m, X[k_va]), labels=[0, 1]))
        del m
    chosen = min(LEARNERS, key=lambda L: ll[L])
    S = {}
    for w in widths:                                    # validation models: 2019-2022 only
        k = TR & ~np.isnan(Y[w])
        m = fit_learner(chosen, X[k], Y[w][k])
        S[w] = predict(chosen, m, X)
        del m
    thr = {w: top_threshold(S[w][TR & ~bad], NORMAL_Q[0]) for w in widths}
    wpt, width_table = choose_width({w: S[w][VA] for w in widths}, {w: Y[w][VA] for w in widths},
                                    tier[VA], ~bad[VA], thr, widths, mid, range(n_tiers))
    s_row = row_scores(S, tier, wpt)
    y_row = row_scores(Y, tier, wpt)
    best, bar_table, flag = choose_bar(s_row[VA], y_row[VA], bad[VA], s_row[TR], bad[TR])
    # refit the chosen learner per width on 2019-2023 and score every row with it
    SD, paths = {}, {}
    for w in widths:
        k = ~np.isnan(Y[w])
        m = fit_learner(chosen, X[k], Y[w][k])
        paths[str(w)] = save_model(chosen, m, str(out / "models" / f"{tr}_{name}_w{w}"))
        SD[w] = predict(chosen, m, X)
        del m
    sd_row = row_scores(SD, tier, wpt)
    normal_fit = sd_row[~bad]
    return {
        "learner_logloss_2023": ll, "chosen_learner": chosen,
        "widths_per_tier": {str(t): float(w) for t, w in wpt.items()},
        "width_validation_2023": width_table,
        "bar_validation_2023": {"pairs": bar_table, "chosen": best, "flag": flag,
                                "thresholds_2019_2022": {"normal": best["thr_normal"], "bad": best["thr_bad"]}},
        "bar_thresholds_refit_2019_2023": {"normal_q": best["normal_q"], "bad_divisor": best["bad_divisor"],
                                           "normal": top_threshold(normal_fit, best["normal_q"]),
                                           "bad": top_threshold(normal_fit, best["normal_q"] / best["bad_divisor"])},
        "validation_counts_2023": {"rows": int(VA.sum()), "bad_share": float(bad[VA].mean())},
        "models": paths,
    }


def stage_fit(state, out):
    out = Path(out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    result = {"inputs_sha256": hashlib.sha256(json.dumps(VERSIONS, sort_keys=True).encode()).hexdigest(),
              "versions": VERSIONS, "mid_width": MID, "train": "2019-01..2022-12", "validate": "2023",
              "refit": "2019-01..2023-12", "sample": {"n": SAMPLE_N, "seed": SEED, "learners": sorted(SAMPLED)},
              "tracks": {}}
    for tr in ("main", "penny"):
        D = load(state, tr)
        assert int(D["year"].max()) <= FIT_YEARS[1], "2024+ rows reached the fit stage"
        year = D["year"].to_numpy()
        mkt = market_params(D[year <= TRAIN_YEARS[1]])
        D["market_score"] = market_score(D, mkt)
        D = D[D["market_score"].notna()]                 # the first 20 sessions have no volatility
        year = D["year"].to_numpy()
        score = D["market_score"].to_numpy()
        cut = float(np.quantile(score[year <= TRAIN_YEARS[1]], 0.2))
        bad = score < cut
        sw = D["swing30"].to_numpy()
        n_tiers = TIERS[tr]
        cuts = list(np.quantile(sw[year <= TRAIN_YEARS[1]], [1 / 3, 2 / 3])) if n_tiers == 3 else []
        tier = tier_of(sw, cuts)
        widths = P.TRACKS[tr]["widths"]
        Y = {w: D[f"y_{w}"].to_numpy(np.float32) for w in widths}
        res = {"market": {"parts": mkt, "cut_20pct": cut, "bad_share_train": float(bad[year <= 2022].mean())},
               "tier_cuts_swing30": [float(c) for c in cuts],
               "rows": {"train": int((year <= 2022).sum()), "validate": int((year == 2023).sum())},
               "versions": {}}
        for name, feats in VERSIONS.items():
            X = D[feats].to_numpy(np.float32)
            res["versions"][name] = fit_version(name, X, Y, year, bad, tier, n_tiers, widths, MID[tr], out, tr)
            print(f"[pick fit] {tr} {name}: {res['versions'][name]['chosen_learner']}", flush=True)
            del X
        result["tracks"][tr] = res
        atomic_json(result, out / "PICK_fit.json")
    atomic_json(result, out / "PICK_fit.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["join", "fit"])
    ap.add_argument("--rows", default="cache/w1")
    ap.add_argument("--nx1", default="cache/nx1")
    ap.add_argument("--out", default="research/results/PICK")
    ap.add_argument("--state", default="cache/pickfit")
    a = ap.parse_args()
    if a.stage == "join":
        stage_join(a.rows, a.nx1, a.state)
    else:
        stage_fit(a.state, a.out)


if __name__ == "__main__":
    main()
