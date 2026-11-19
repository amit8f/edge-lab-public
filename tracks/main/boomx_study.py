"""BOOMX (research/prereg/BOOMX.md): open up the BOOM trees. Exploratory and descriptive, no rule.

Part A (boom): BOOM1's stock-days, label = an 8%+ move either way next session. Part B (direction): BOOM2's picks,
labels = first touch (+8% before -8%) and close above open. For both: single-input AUCs (overall, lowest/highest tenth,
per year), input groups, seven models trained 2019-2022 and scored on 2023, 2024 and 2025-01..09, CatBoost SHAP, a
readable depth-3 tree and the two strongest inputs as a 5 x 5 grid. Nothing after 2025-09-30 is read.
Writes research/results/BOOMX/BOOMX.json and BOOMX.md.

    python -m tracks.main.boomx_study --cache cache/nx1 --part boom|direction|all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main import boom1_study as B
from tracks.main import boom2_study as B2

SEED = 20261119
TRAIN = (2019, 2022)
TESTS = (2023, 2024, 2025)
LAST = pd.Timestamp("2025-09-30")
MAX_TRAIN = 600_000
SHAP_ROWS = 20_000
TOP10 = 10
GROUPS = {
    "price trend": ["r1", "r5", "r20", "r60", "r120", "mom12_1", "rsi14", "rsi2", "macd_h", "bb_pctb", "d_sma20",
                    "d_sma50", "d_sma200", "hi52", "lo52", "streak", "maxret20", "minret20", "clv", "oc",
                    *[f"rz_{i}" for i in range(10)]],
    "volatility and range": ["vol20", "vol60", "vol_ratio", "bb_width", "atr14", "rng_atr", *[f"rg_{i}" for i in range(10)]],
    "volume and size": ["vrel20", "vrel5_20", "obv20", "log_dv", "log_px", *[f"vr_{i}" for i in range(10)]],
    "gap and pre-market": ["gap", "pm_gap", "pm_dv"],
    "news": [*B.NEWS, "pre_offering"],
    "market": ["spy_r1", "spy_r5", "spy_d200", "spy_vol20"],
    "BOOM1 score": ["score"],
}


# ---------- pure helpers (tested) ----------

def auc(y, x):
    """Rank AUC of x for binary y (NaN rows dropped); NaN if one class is missing."""
    from sklearn.metrics import roc_auc_score
    y, x = np.asarray(y), np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], x[ok]))


def one_input(df, col, ycol):
    """AUC, label rate in the lowest and highest tenth, AUC per year."""
    x = df[col].to_numpy(float)
    y = df[ycol].to_numpy()
    out = {"auc": auc(y, x), "n": int(np.isfinite(x).sum())}
    ok = np.isfinite(x)
    if ok.sum() >= 100:
        lo, hi = np.nanquantile(x, [0.1, 0.9])
        out["rate_low10"] = float(y[ok & (x <= lo)].mean())
        out["rate_high10"] = float(y[ok & (x >= hi)].mean())
    out["by_year"] = {str(k): auc(g[ycol], g[col]) for k, g in df.groupby("year")}
    return out


def group_share(imp):
    """{group: share of total importance} for an {input: importance} dict."""
    tot = sum(imp.values()) or 1.0
    out = {g: sum(imp.get(c, 0.0) for c in cs) / tot for g, cs in GROUPS.items()}
    out["other"] = max(0.0, 1.0 - sum(out.values()))
    return {k: float(v) for k, v in out.items() if v > 0}


def grid(df, a, b, ycol, k=5):
    """k x k label-rate grid over the quantile bins of inputs a (rows) and b (columns)."""
    d = df[[a, b, ycol]].dropna()
    qa = pd.qcut(d[a].rank(method="first"), k, labels=False)
    qb = pd.qcut(d[b].rank(method="first"), k, labels=False)
    t = d.groupby([qa, qb])[ycol].agg(["mean", "count"])
    return {"rows": a, "cols": b, "rate": [[float(t.loc[(i, j), "mean"]) if (i, j) in t.index else None for j in range(k)]
                                           for i in range(k)],
            "n": [[int(t.loc[(i, j), "count"]) if (i, j) in t.index else 0 for j in range(k)] for i in range(k)]}


def confident(y, p, share=0.1):
    """Label rate among the top `share` of p and (1 - rate) among the bottom `share`."""
    y, p = np.asarray(y), np.asarray(p)
    hi, lo = np.quantile(p, [1 - share, share])
    return {"top_rate": float(y[p >= hi].mean()), "bottom_other_rate": float(1 - y[p <= lo].mean())}


# ---------- models ----------

def models(cols):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    return {
        "CatBoost": (CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=SEED,
                                        verbose=0, thread_count=-1, allow_writing_files=False), False),
        "LightGBM": (LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=50, random_state=SEED, n_jobs=-1, tree_method="hist"), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=100, max_features=0.3, n_jobs=-1,
                                            random_state=SEED), True),
        "Logistic": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), True),
        "Tree depth 3": (DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=SEED), True),
    }


def fit_all(tr, tests, cols, ycol, extra=None):
    """Fits every model on tr and returns {model: {year: {auc, extra...}}}, the CatBoost model, the depth-3 tree."""
    from sklearn.tree import export_text
    med = tr[cols].median()
    out, keep = {}, {}
    for name, (m, fill) in models(cols).items():
        X = (tr[cols].fillna(med) if fill else tr[cols]).to_numpy(np.float32)
        m.fit(X, tr[ycol].to_numpy())
        keep[name] = m
        res = {}
        for yr, te in tests.items():
            Xt = (te[cols].fillna(med) if fill else te[cols]).to_numpy(np.float32)
            p = m.predict_proba(Xt)[:, 1]
            res[yr] = {"auc": auc(te[ycol], p), **(extra(te, p) if extra else {})}
        out[name] = res
        print(f"[boomx] {ycol} {name}: " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in res.items()), flush=True)
    tree = export_text(keep["Tree depth 3"], feature_names=cols, show_weights=False, decimals=3)
    return out, keep["CatBoost"], tree


def shap_table(m, te, cols):
    from catboost import Pool
    s = te.sample(min(SHAP_ROWS, len(te)), random_state=SEED)
    sv = m.get_feature_importance(Pool(s[cols].to_numpy(np.float32)), type="ShapValues")[:, :-1]
    out = {}
    for i, c in enumerate(cols):
        x = s[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(sv[:, i])
        sign = float(np.corrcoef(x[ok], sv[ok, i])[0, 1]) if ok.sum() > 50 and np.std(x[ok]) > 0 and np.std(sv[ok, i]) > 0 else None
        out[c] = {"mean_abs": float(np.abs(sv[:, i]).mean()), "effect_sign": sign}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean_abs"]))


def study(df, cols, ycol, extra=None, label=""):
    df = df[(df["year"] >= TRAIN[0]) & (df["d"] <= LAST)].reset_index(drop=True)
    tr = df[df["year"] <= TRAIN[1]]
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(MAX_TRAIN, random_state=SEED)
    tests = {str(y): df[df["year"] == y] for y in TESTS}
    rep = {"label": label, "rows": len(df), "base_rate": float(df[ycol].mean()),
           "base_by_year": {str(k): float(g[ycol].mean()) for k, g in df.groupby("year")}}
    singles = {c: one_input(df, c, ycol) for c in cols}
    rep["single_inputs"] = dict(sorted(singles.items(), key=lambda kv: -abs((kv[1]["auc"] if np.isfinite(kv[1]["auc"]) else 0.5) - 0.5)))
    rep["models"], cat, rep["tree_depth3"] = fit_all(tr, tests, cols, ycol, extra)
    te = pd.concat(tests.values())
    rep["shap"] = shap_table(cat, te, cols)
    rep["shap_groups"] = group_share({c: v["mean_abs"] for c, v in rep["shap"].items()})
    top = list(rep["shap"])[:TOP10]
    m = models(top)["CatBoost"][0].fit(tr[top].to_numpy(np.float32), tr[ycol].to_numpy())
    rep["models"]["CatBoost top 10"] = {y: {"auc": auc(t[ycol], p := m.predict_proba(t[top].to_numpy(np.float32))[:, 1]),
                                            **(extra(t, p) if extra else {})} for y, t in tests.items()}
    rep["top10"] = top
    rep["grid"] = grid(df, top[0], top[1], ycol)
    return rep


def top20(te, p):
    """Share of each day's top 20 by p that boomed."""
    t = te.assign(p=p)
    return {"top20_boom": float(t.groupby("d", group_keys=False).apply(lambda g: g.nlargest(B.TOP, "p"))["boom"].mean())}


def conf(ycol):
    return lambda te, p: confident(te[ycol].to_numpy(), p)


# ---------- report ----------

def md(rep):
    L = ["# BOOMX results (exploratory, research/prereg/BOOMX.md)", ""]
    for part, r in rep.items():
        for x in r.values():
            L += [f"## {part}: {x['label']}", "", f"Rows {x['rows']:,}; base rate {x['base_rate']:.3f}.", "",
                  "| model | " + " | ".join(next(iter(x['models'].values())).keys()) + " |",
                  "|---|" + "---|" * len(next(iter(x['models'].values())))]
            for mname, yrs in x["models"].items():
                L.append(f"| {mname} | " + " | ".join(
                    f"{v['auc']:.3f}" + (f" (top20 {v['top20_boom']:.2f})" if "top20_boom" in v else "")
                    + (f" (conf {v['top_rate']:.2f}/{v['bottom_other_rate']:.2f})" if "top_rate" in v else "")
                    for v in yrs.values()) + " |")
            L += ["", "Strongest single inputs (AUC on its own; 0.5 = nothing; low/high tenth label rate):", ""]
            for c, s in list(x["single_inputs"].items())[:15]:
                L.append(f"- {c}: AUC {s['auc']:.3f}, lowest tenth {s.get('rate_low10', float('nan')):.3f}, "
                         f"highest tenth {s.get('rate_high10', float('nan')):.3f}")
            L += ["", "SHAP (CatBoost, 2023-2025/09), top 15, with the sign of the effect:", ""]
            for c, s in list(x["shap"].items())[:15]:
                sg = s["effect_sign"]
                L.append(f"- {c}: {s['mean_abs']:.4f}" + (f", higher input -> {'more' if sg > 0 else 'less'} ({sg:+.2f})" if sg is not None else ""))
            L += ["", "Input groups' share of SHAP: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(x["shap_groups"].items(), key=lambda kv: -kv[1])),
                  "", "Depth-3 tree (2019-2022):", "", "```", x["tree_depth3"], "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--part", default="all", choices=["boom", "direction", "all"])
    ap.add_argument("--out", default="research/results/BOOMX")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "BOOMX.json"
    rep = json.loads(f.read_text()) if f.exists() else {}
    if a.part in ("direction", "all"):
        from tracks.main.boom_short import d6_cols
        p = B2.load_all(a.cache)
        p = p[p["d"] <= LAST].copy()
        p["first_up"] = (p["lab"] == 1).astype(int)
        p["close_up"] = (p["close"] > p["open"]).astype(int)
        cols = d6_cols()
        rep["direction"] = {
            "first_touch": study(p[p["lab"].isin([1, -1])], cols, "first_up", conf("first_up"), "up-first vs down-first (8%)"),
            "close": study(p, cols, "close_up", conf("close_up"), "close above open"),
        }
        f.write_text(json.dumps(rep, indent=1, default=float))
    if a.part in ("boom", "all"):
        df = B.load(a.cache)
        df["year"] = df["nxt"].dt.year
        df["d"] = df["nxt"]
        rep["boom"] = {"boom": study(df, B.INPUTS, "boom", top20, "8%+ move next session")}
        f.write_text(json.dumps(rep, indent=1, default=float))
    (out / "BOOMX.md").write_text(md(rep))


if __name__ == "__main__":
    main()
