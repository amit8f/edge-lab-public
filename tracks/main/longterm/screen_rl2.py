"""RL2 historical screen (research/prereg/RL2.md): RL1's panel and rank label with four event inputs (insider
buying, analyst actions, earnings surprise, news count) and a tamer XGBoost. Runs through study.yml on the
shared cache (RL1's bars, M5's Form 4 buys, M6's headlines, M7's company facts).

Usage: python -m tracks.main.longterm.screen_rl2 --cache <dir> [--out research/results/RL2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main.longterm import analyst, earnings, insider, news_history, sec_data
from tracks.main.longterm import screen_rl1 as R1

ROOT = R1.ROOT
SEED = R1.SEED
XGB = dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5,
           min_child_weight=200, reg_lambda=10.0, tree_method="hist", random_state=SEED)
EVENT_INPUTS = ["insider", "analyst", "sue", "news_n"]
RAW_INPUTS = R1.RAW_INPUTS + EVENT_INPUTS
INPUTS = [f"r_{k}" for k in RAW_INPUTS]
NEWS_FIRST, NEWS_LAST = "2016-10", "2025-09"
HEADLINES_FROM = pd.Timestamp("2017-01-01")
NEWS_DAYS = 30


# ---------- pure helpers (tested) ----------

def trailing_n(keys, days, window_days=NEWS_DAYS):
    """Per (symbol, date) in keys: headlines with day in (date - window, date]. days: symbol, day."""
    k = keys.reset_index(drop=True)
    out = np.zeros(len(k), dtype=int)
    groups = {s: np.sort(g["day"].to_numpy("datetime64[ns]")) for s, g in days.groupby("symbol")}
    dates = pd.to_datetime(k["date"]).astype("datetime64[ns]").to_numpy()
    for i, (s, t) in enumerate(zip(k["symbol"].to_numpy(), dates)):
        d = groups.get(s)
        if d is not None:
            out[i] = np.searchsorted(d, t, "right") - np.searchsorted(d, t - np.timedelta64(window_days, "D"), "right")
    return out


def event_inputs(uni, buys, an_events, sue, news_days):
    """The four event inputs on RL1's panel (raw values; ranked later with the rest)."""
    d = uni.reset_index(drop=True).copy()
    v, _ = insider.trailing_buys(d[["cik", "date"]], buys)
    d["insider"] = v / d["mcap"].to_numpy(float)
    up, down = analyst.trailing_counts(d[["symbol", "date"]], an_events)
    tot = up + down
    d["analyst"] = np.where(tot > 0, (up - down) / np.sqrt(np.maximum(tot, 1)), 0.0)
    d["news_n"] = trailing_n(d[["symbol", "date"]], news_days).astype(float)
    early = d["date"] < HEADLINES_FROM
    d.loc[early, ["analyst", "news_n"]] = np.nan
    t = sue.dropna(subset=["sue"])[["cik", "avail", "sue"]].sort_values("avail")
    m = pd.merge_asof(d[["cik", "date"]].reset_index().sort_values("date").astype({"cik": "int64"}),
                      t.rename(columns={"avail": "sue_avail"}).astype({"cik": "int64"}).assign(
                          sue_avail=lambda x: pd.to_datetime(x["sue_avail"]).astype("datetime64[ns]")),
                      left_on="date", right_on="sue_avail", by="cik", direction="backward").set_index("index")
    fresh = (d["date"] - m["sue_avail"].reindex(d.index)).dt.days < earnings.WINDOW_DAYS
    d["sue"] = np.where(fresh, m["sue"].reindex(d.index), np.nan)
    return d


def add_inputs(uni):
    g = uni.groupby("date")
    return R1.add_inputs(uni).assign(**{f"r_{k}": g[k].rank(pct=True) for k in EVENT_INPUTS})


def fit(X, y):
    from xgboost import XGBRegressor
    return XGBRegressor(**XGB).fit(X, y)


def walk_forward(uni, ends1, inputs=INPUTS, years=R1.TEST_YEARS):
    """RL1's walk-forward with one model on the rank label. Returns test predictions, the in-sample long-short
    mean and IC per year (rule 21) and the gain importance per year."""
    preds, ins, imp = [], [], {}
    lab_end = uni["date"].map(ends1)
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        tr = uni[(lab_end < start) & uni["ret1"].notna()]
        te = uni[(uni["date"] >= start) & (uni["date"] < pd.Timestamp(f"{yr + 1}-01-01"))]
        if tr.empty or te.empty:
            continue
        m = fit(tr[inputs].to_numpy(float), tr["y_rank"].to_numpy(float))
        preds.append(te[["symbol", "date", "ret1"]].assign(p=m.predict(te[inputs].to_numpy(float))))
        s = tr[["date", "ret1"]].assign(p=m.predict(tr[inputs].to_numpy(float)))
        ls = R1.long_short(s, "p")
        ins.append({"year": yr, "ls_mean": float(ls["ls"].mean()), "months": int(len(ls)), "ic": R1.ic(s, "p")})
        g = m.get_booster().get_score(importance_type="gain")
        imp[str(yr)] = {k: float(g.get(f"f{i}", 0.0)) for i, k in enumerate(inputs)}
    return pd.concat(preds, ignore_index=True), ins, imp


def verdict(pred, ins, base_pred, ff5=None, umd=None, trials=None):
    ls = R1.long_short(pred, "p")
    res = {"rl2": R1.summary(ls["ls"]), "control_random": R1.summary(R1.long_short(pred, "p", seed=SEED)["ls"]),
           "ic_oos": R1.ic(pred, "p")}
    w = [x["months"] for x in ins]
    i_ls = float(np.average([x["ls_mean"] for x in ins], weights=w))
    res["insample"] = {"ls_mean": i_ls, "ic": float(np.average([x["ic"] for x in ins], weights=w)), "per_year": ins}
    res["overfit_rule21"] = bool(i_ls > R1.OVERFIT * max(res["rl2"]["mean"], 0.0))
    bl = R1.long_short(base_pred, "p")
    both = ls.merge(bl, on="date", suffixes=("", "_base"))
    res["rl1_inputs_same_model"] = R1.summary(bl["ls"])
    res["minus_rl1_inputs"] = R1.summary(both["ls"] - both["ls_base"])
    c = R1.long_short(pred, "p", seed=SEED).merge(ls, on="date", suffixes=("_c", ""))
    res["minus_control"] = R1.summary(c["ls"] - c["ls_c"])
    res["near_miss_rule19"] = bool(res["minus_control"]["t_hac"] >= 1.5 and res["rl2"]["mean"] > 0
                                   and not res["overfit_rule21"])
    if ff5 is not None:
        alpha = stats.factor_alpha(ls["ls"], stats.month_after(ls["date"]), ff5, umd, R1.LAGS)
        res["rl2"]["factor_alpha"] = alpha
        res["rl2"]["deflated_sharpe"] = stats.deflated_sharpe(ls["ls"], trials)
        res["trading_passes"] = bool(stats.screen_pass(res["rl2"], alpha) and not res["overfit_rule21"])
    res["per_year"] = {str(y): float(g["ls"].mean()) for y, g in ls.groupby(ls["date"].dt.year)}
    return res, ls


# ---------- data (disk; RL1's prepare fetches what is missing) ----------

def news_days(cache):
    """Single-symbol headlines as (symbol, day), month by month (no text kept)."""
    out = []
    for per in pd.period_range(NEWS_FIRST, NEWS_LAST, freq="M"):
        m = news_history.load_range(str(per), str(per), cache / "news", columns=["id", "created_at", "symbols"])
        s = m["symbols"].fillna("")
        m = m[(s != "") & ~s.str.contains(",")]
        day = pd.to_datetime(m["created_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        out.append(pd.DataFrame({"symbol": m["symbols"].to_numpy(), "day": day.astype("datetime64[ns]").to_numpy()}))
    return pd.concat(out, ignore_index=True)


def analyst_events(cache):
    keep = []
    for per in pd.period_range(NEWS_FIRST, NEWS_LAST, freq="M"):
        m = news_history.load_range(str(per), str(per), cache / "news", columns=["created_at", "headline", "symbols"])
        m = m[m["headline"].map(analyst.classify) != 0]
        keep.append(m)
    return analyst.events(pd.concat(keep, ignore_index=True))


def run(cache, out, budget_min=290):
    cache, out = Path(cache), Path(out)
    deadline = time.monotonic() + budget_min * 60
    if not news_history.fetch_range(NEWS_FIRST, NEWS_LAST, cache / "news", deadline):
        print("NEWS DOWNLOAD INCOMPLETE: rerun to continue (cached months are kept)")
        return None
    p = R1.prepare(cache, deadline)
    if p is None:
        return None
    uni, ends1, funnel = p
    last = pd.Timestamp(R1.LAST_SIGNAL)
    buys = sec_data.dera_insider_buys(2015, pd.Period(last, "Q"), cache / "dera")
    buys = buys[buys["filed"] <= last]
    facts = sec_data.facts_for(cache / "companyfacts.zip", uni["cik"].unique(), cache / "facts")
    q = earnings.quarterly_ni(facts)
    sue = earnings.sue_table(q[q["avail"] <= last])
    uni = add_inputs(event_inputs(uni, buys, analyst_events(cache), sue, news_days(cache)))
    funnel.update({f"with_{k}": int(uni[k].notna().sum()) for k in EVENT_INPUTS} |
                  {"insider_nonzero": int((uni["insider"] > 0).sum()), "analyst_nonzero": int((uni["analyst"] != 0).sum())})
    print("FUNNEL", json.dumps(funnel), flush=True)
    pred, ins, imp = walk_forward(uni, ends1)
    base_pred, _, _ = walk_forward(uni, ends1, inputs=R1.INPUTS)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts(ROOT / "research" / "IDEAS.md")
    res, ls = verdict(pred, ins, base_pred, ff5, umd, both)
    res = {"idea": "RL2", "prereg": "research/prereg/RL2.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_sha256": sha5, "umd_sha256": shau, "funnel": funnel, "xgb_params": XGB, "importance": imp} | res
    out.mkdir(parents=True, exist_ok=True)
    (out / "RL2_screen.json").write_text(json.dumps(res, indent=1, default=str))
    ls.to_csv(out / "RL2_monthly.csv", index=False)
    print(json.dumps({k: res.get(k) for k in ("rl2", "insample", "overfit_rule21", "minus_rl1_inputs", "minus_control",
                                              "trading_passes", "near_miss_rule19")}, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/RL2")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
