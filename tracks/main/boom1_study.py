"""BOOM1 step 1: narrow each day's market to the 20 stocks most likely to move 8%+ during the day (research/prereg/BOOM1.md).

Filter only, no trades. At the close of day d, score every $5+ liquid stock on its chance that the next session's
high or low reaches 8% away from its open, using chart and volume inputs at d's close and the news that came
out before the next session's open. Reads the nx3-study `chart_bars` files and NX3's trade files (news scores
only, never headline text). Writes research/results/BOOM1/BOOM1_walkforward.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import chart_features as C
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

SEED = 20261107
MIN_PX, MIN_DV = 5.0, 1e6
FIRST = pd.Timestamp("2016-01-01")
TRAIN_ROWS = 600_000
TOP = 20
BOOM, BIG = 0.08, 0.15
NEWS = ["pre_news", "pre_articles", "pre_fb_neg", "pre_fb_pos", "pre_lm_net", "pre_fb_max_neg", "pre_fb_max_pos"]
INPUTS = C.FEATURES + C.MARKET + NEWS


def pre_open_news(cache):
    """Per (symbol, session): real NX3 events decided before or at the open (news that came out after the
    previous close and before the session opened), with the strongest tone of that session's pre-open news."""
    parts = []
    cols = ["symbol", "d", "kind", "mins_open", "n_articles", "fb_neg", "fb_pos", "lm_net", "fb_max_neg", "fb_max_pos"]
    for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet")):
        x = pd.read_parquet(f, columns=cols)
        parts.append(x[(x["kind"] == "real") & (x["mins_open"] <= 0)])
    x = pd.concat(parts, ignore_index=True)
    g = x.groupby(["symbol", "d"])
    return pd.DataFrame({"pre_news": 1.0, "pre_articles": g["n_articles"].sum(), "pre_fb_neg": g["fb_neg"].max(),
                         "pre_fb_pos": g["fb_pos"].max(), "pre_lm_net": g["lm_net"].mean(),
                         "pre_fb_max_neg": g["fb_max_neg"].max(), "pre_fb_max_pos": g["fb_max_pos"].max()}).reset_index()


def load(cache):
    d = X.nx3dir(cache) / X.CHART
    spy = pd.read_parquet(d / "spy.parquet")[["date", *C.MARKET]]
    cols = ["symbol", "date", "c", "dv20", "ao1", "ac1", "nxt", "ac", *C.FEATURES]
    parts = []
    for f in sorted(d.glob("c_*.parquet")):
        x = next_range(pd.read_parquet(f, columns=cols))
        parts.append(x[(x["date"] >= FIRST) & (x["c"] >= MIN_PX) & (x["dv20"] >= MIN_DV) & x["vol20"].notna()
                       & x["nxt"].notna() & (x["nxt"] <= N.LAST_EXIT)])
    df = pd.concat(parts, ignore_index=True).merge(spy, on="date", how="inner")
    df["move"] = df["ac1"] / df["ao1"] - 1
    df["up_move"], df["down_move"] = df["ah1"] / df["ao1"] - 1, 1 - df["al1"] / df["ao1"]
    df = df[np.isfinite(df["move"]) & np.isfinite(df["up_move"]) & np.isfinite(df["down_move"])]
    news = pre_open_news(cache).rename(columns={"d": "nxt"})
    df = df.merge(news, on=["symbol", "nxt"], how="left")
    df[["pre_news", "pre_articles"]] = df[["pre_news", "pre_articles"]].fillna(0.0)
    df["boom"] = ((df["up_move"] >= BOOM) | (df["down_move"] >= BOOM)).astype(np.int8)
    return df.rename(columns={"date": "d"}).reset_index(drop=True)


def next_range(x):
    """The next session's all-adjusted high and low (ah1, al1), recovered from that row's range / close (rg_0) and
    close location (clv), only where the next row is the next session (nxt)."""
    x = x.sort_values(["symbol", "date"]).reset_index(drop=True)
    rng = x["rg_0"] * x["ac"]
    al = x["ac"] - x["clv"] * rng
    ah = al + rng
    same = (x["symbol"].shift(-1) == x["symbol"]) & (x["date"].shift(-1) == x["nxt"])
    x["ah1"] = np.where(same, ah.shift(-1), np.nan)
    x["al1"] = np.where(same, al.shift(-1), np.nan)
    return x


def classifier():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=SEED,
                              verbose=0, thread_count=-1, allow_writing_files=False)


def day_rates(te, score, top=TOP):
    """Per day: share of the top `top` by score that reached 8% (and 15%) from the open, up and down counts."""
    te = te.assign(s=score)
    rows = []
    for d, g in te.groupby("d", sort=True):
        if len(g) < top * 5:
            continue
        t = g.nlargest(top, "s")
        rows.append({"d": d, "hit": float(t["boom"].mean()),
                     "big": float(((t["up_move"] >= BIG) | (t["down_move"] >= BIG)).mean()),
                     "up": int((t["up_move"] >= BOOM).sum()), "down": int((t["down_move"] >= BOOM).sum()),
                     "close_hit": float((t["move"].abs() >= BOOM).mean()),
                     "base": float(g["boom"].mean()), "n": len(g)})
    return pd.DataFrame(rows, columns=["d", "hit", "big", "up", "down", "close_hit", "base", "n"])


def run(cache, out):
    df = load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    print(f"[boom1] {len(df)} stock-days, base boom rate {df['boom'].mean():.4f}, "
          f"pre-open news on {df['pre_news'].mean():.4f}")
    res = {"MODEL": [], "VOL": [], "VOL_NEWS": [], "MAX": [], "RANDOM": []}
    imp = {}
    for yr in N.TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
        tr = df[df["d"] <= cut_d]
        te = df[(df["d"] >= start) & (df["d"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        if len(tr) > TRAIN_ROWS:
            tr = tr.iloc[np.sort(np.random.default_rng(SEED + yr).choice(len(tr), TRAIN_ROWS, replace=False))]
        m = classifier().fit(tr[INPUTS].to_numpy(np.float32), tr["boom"].to_numpy())
        imp[yr] = dict(zip(INPUTS, map(float, m.get_feature_importance())))
        p = m.predict_proba(te[INPUTS].to_numpy(np.float32))[:, 1]
        vol = te["vol20"].to_numpy(float)
        res["MODEL"].append(day_rates(te, p).assign(year=yr))
        res["VOL"].append(day_rates(te, vol).assign(year=yr))
        res["VOL_NEWS"].append(day_rates(te, vol * (1 + te["pre_news"].to_numpy(float))).assign(year=yr))
        res["MAX"].append(day_rates(te, te["maxret20"].abs().to_numpy(float)).assign(year=yr))
        res["RANDOM"].append(day_rates(te, np.random.default_rng(SEED + yr).random(len(te))).assign(year=yr))
        print(f"[boom1] {yr}: model hit {res['MODEL'][-1]['hit'].mean():.3f}, vol rule {res['VOL'][-1]['hit'].mean():.3f}, "
              f"base {res['MODEL'][-1]['base'].mean():.3f}")
    cat = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    t = X.tstat
    rep = {"universe": {"min_price": MIN_PX, "min_dv20": MIN_DV, "stock_days": int(len(df)), "base_boom_rate": float(df["boom"].mean())},
           "top": TOP, "boom": BOOM, "arms": {}}
    for k, x in cat.items():
        rep["arms"][k] = {"hit_rate": t(x["hit"].to_numpy(float)), "big_rate": float(x["big"].mean()),
                          "close_hit_rate": float(x["close_hit"].mean()),
                          "lift_vs_base": float(x["hit"].mean() / x["base"].mean()),
                          "up_share_of_booms": float(x["up"].sum() / max((x["up"] + x["down"]).sum(), 1)),
                          "per_year_hit": {int(y): float(g["hit"].mean()) for y, g in x.groupby("year")}}
    m = cat["MODEL"].merge(cat["VOL"], on="d", suffixes=("", "_vol"))
    rep["model_minus_vol"] = t((m["hit"] - m["hit_vol"]).to_numpy(float))
    a, g = rep["arms"]["MODEL"], rep["model_minus_vol"]
    yrs_ok = all(v > rep["arms"]["VOL"]["per_year_hit"][y] for y, v in a["per_year_hit"].items())
    rep["model_beats_vol_every_year"] = bool(yrs_ok)
    rep["verdict"] = "passed" if (a["lift_vs_base"] >= 3 and g["mean"] > 0 and g["t_nw"] >= X.GATE_T) else "failed"
    rep["importance_top15"] = {k: float(v) for k, v in pd.DataFrame(imp).mean(axis=1).sort_values(ascending=False).head(15).items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BOOM1_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep
