"""LONG1-LONG5 (research/prereg/LONG.md): why NX3-HI's buys are weak, and four registered fixes for the long side.

    python -m tracks.news.long_study --stage run --cache cache/nx1 --out research/results/LONG

Reads what earlier studies left in the NX3 cache: models5 picks (NX3-ALL-HI walk-forward, 2019-2025-09), the
trades_v2 inputs, COST1's quotes, EXEC1's 10:00/10:30 prices and the daily chart closes. No new download.
LONG1 (descriptive) looks at 2019-2023 only. LONG2-5 choose on 2019-2023 and are tested once on 2024-01..2025-09.
Writes summaries only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.news import cost1_study as K
from tracks.news import exec1_study as E
from tracks.news import models5_study as M
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X

CAP = 3
CHOOSE, TEST = M.CHOOSE, M.TEST
CONFIRM_T = ("10:00", "10:30")
HOLDS = (3, 5)
MARKET_RULES = ("sma20", "sma50", "sma200", "calm")
CALM_FIT = ("2016-01-01", "2022-12-31")
SEED = 20261128


# ---------- pure helpers (tested) ----------

def cap_longs(p, cap=CAP):
    """NX3-HI's slots: the first `cap` longs of each day in time order (input order breaks ties, as X.picks)."""
    lo = p[p["side"] > 0]
    return lo.sort_values(["d", "t"], kind="stable").groupby("d", sort=False).head(cap)


def day_book(d, net, days):
    """Equal-weight daily return of a book over `days`: mean net of that day's trades, 0 on a day with none
    (cash). NaN nets count as skipped."""
    s = pd.Series(np.asarray(net, float), index=pd.DatetimeIndex(d)).dropna()
    return s.groupby(level=0).mean().reindex(pd.DatetimeIndex(days), fill_value=0.0)


def confirm(entry_t, entry, ret, hs_in, d, px, hs_t, hhmm):
    """LONG2: a long entering before T waits to T and buys there only if the price at T is above its first
    entry price; otherwise it is skipped (ret NaN). Others unchanged. Returns (ret, half-spread, skipped)."""
    r2, h2, _ = E.delayed(entry_t, entry, ret, hs_in, d, px, hs_t, hhmm)
    tt = np.array([E.at_time(x, hhmm) for x in pd.to_datetime(d)], dtype="datetime64[ns]")
    early = pd.to_datetime(entry_t).to_numpy("datetime64[ns]") < tt
    px, entry = np.asarray(px, float), np.asarray(entry, float)
    skip = early & ~(np.isfinite(px) & (px > entry))
    return np.where(skip, np.nan, r2), h2, skip


def hold_ret(ret, ac_next, ac_k):
    """LONG3: the close-to-close return carried from NX3's exit close to the k-th session's close."""
    return (1 + np.asarray(ret, float)) * np.asarray(ac_k, float) / np.asarray(ac_next, float) - 1


def market_ok(spy, rule, calm_cut):
    """LONG5: per date, whether the rule held at that date's close (used for the NEXT session's buys)."""
    ac = spy["ac"]
    if rule == "calm":
        return spy["rv20"] < calm_cut
    n = int(rule[3:])
    return ac > ac.rolling(n).mean()


def gate(own_day, diff_day, own_trade):
    """Pass: own net per trade > 0, own daily mean NW t >= 2, and the paired daily gain over plain NX3-HI
    longs has mean > 0 with NW t >= 1.5."""
    o, g = X.tstat(own_day), X.tstat(diff_day)
    ok = bool(own_trade > 0 and o["t_nw"] >= 2 and g["mean"] > 0 and g["t_nw"] >= 1.5)
    return {"own_day": o, "gain_vs_plain": g, "own_net_per_trade": float(own_trade), "verdict": "passed" if ok else "failed"}


def cell(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": len(x)}
    return {"n": len(x), "net": float(x.mean()), "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))),
            "win": float((x > 0).mean()), "thin": bool(len(x) < 30)}


def splits(t, net):
    """LONG1's cells for one set of trades (descriptive, per-trade t, not clustered)."""
    t = t.assign(_n=np.asarray(net, float))
    out = {}
    tod = t["entry_t"] - t["d"]
    out["news_time"] = {k: cell(t.loc[m, "_n"]) for k, m in {
        "before_or_at_open": tod <= pd.Timedelta(hours=9, minutes=31),
        "9:31-11:00": (tod > pd.Timedelta(hours=9, minutes=31)) & (tod <= pd.Timedelta(hours=11)),
        "after_11:00": tod > pd.Timedelta(hours=11)}.items()}
    t["jump"] = t["price"] / t["prev_c"] - 1
    for col, name in (("jump", "jump_before_entry"), ("conf", "confidence"), ("log_dv", "dollar_volume"),
                      ("log_px", "price"), ("r20", "past_20d_return"), ("vol20", "volatility")):
        q = pd.qcut(t[col].rank(method="first"), 3, labels=["low", "mid", "high"]) if t[col].notna().sum() > 10 else None
        out[name] = {} if q is None else {str(k): cell(t.loc[q == k, "_n"]) for k in ("low", "mid", "high")}
    out["spy_prev_day"] = {"up": cell(t.loc[t["spy_r1"] > 0, "_n"]), "down": cell(t.loc[t["spy_r1"] <= 0, "_n"])}
    out["news_type"] = {k: cell(t.loc[t[f"f_{k}"] > 0, "_n"]) for k in N.FLAGS}
    out["news_type"]["none_flagged"] = cell(t.loc[(t[[f"f_{k}" for k in N.FLAGS]] <= 0).all(axis=1), "_n"])
    out["year"] = {str(y): cell(g["_n"]) for y, g in t.groupby("year")}
    out["all"] = cell(t["_n"])
    return out


# ---------- data ----------

def load(cache):
    """models5 picks + entry time + COST1 quotes + EXEC1 prices at T + inputs + later closes, one row a trade."""
    p = E.trades(cache)
    p = p[pd.notna(p["entry_t"])].reset_index(drop=True)
    qf = sorted(K.cdir(cache).glob("q_*.parquet"))
    xf = sorted(E.xdir(cache).glob("x_*.parquet"))
    if not qf or not xf:
        raise SystemExit("COST1 quotes or EXEC1 prices missing; run nx3-study stages cost1 and exec1 first")
    p = p.merge(pd.concat([pd.read_parquet(f) for f in qf], ignore_index=True), on=["symbol", "d", "t"], how="left")
    p = p.merge(pd.concat([pd.read_parquet(f) for f in xf], ignore_index=True), on=["symbol", "d", "t"], how="left")
    cols = ["symbol", "d", "t", "prev_c", *X.INPUTS]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet"))],
                   ignore_index=True).drop_duplicates(["symbol", "d", "t"])
    p = p.drop(columns=[c for c in cols[3:] if c in p.columns]).merge(ev, on=["symbol", "d", "t"], how="left")
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"]).sort_values()
    i = caldates.searchsorted(p["d"])
    for k in HOLDS:
        p[f"exit_{k}"] = [caldates[j + k] if j + k < len(caldates) else pd.NaT for j in i]
    pan = M.chart_panel(cache, set(p["symbol"]))[["symbol", "date", "ac"]].drop_duplicates(["symbol", "date"])
    for c in ["next", *[f"exit_{k}" for k in HOLDS]]:
        p = p.merge(pan.rename(columns={"date": c, "ac": f"ac_{c}"}), on=["symbol", c], how="left")
    p["flat"] = N.cost_frac(p["price"], p["side"] < 0, p["days"])
    p["base"] = E.net_moc(p["side"], p["ret"], p["flat"], p["price"], p["hs_in"])
    return p


def spy_table(cache):
    spy = M.spy_states(cache)
    cut = float(spy[(spy["date"] >= CALM_FIT[0]) & (spy["date"] <= CALM_FIT[1])]["rv20"].quantile(0.8))
    dates = spy["date"].to_numpy()
    rules = {r: market_ok(spy, r, cut).to_numpy() for r in MARKET_RULES}
    return dates, rules, cut


def rule_on(days, dates, flag):
    """For each trading day, the rule's value at the previous SPY close (False when unknown)."""
    j = np.searchsorted(dates, np.asarray(days, dtype="datetime64[ns]")) - 1
    return np.where(j >= 0, flag[np.clip(j, 0, None)], False)


# ---------- the arms ----------

def plain(lo):
    return lo["base"].to_numpy(float)


def arm_confirm(lo, hhmm):
    r2, h2, skip = confirm(lo["entry_t"], lo["entry"], lo["ret"], lo["hs_in"], lo["d"], lo[f"px_{hhmm}"],
                           lo[f"hs_{hhmm}"], hhmm)
    n = E.net_moc(lo["side"], np.nan_to_num(r2), lo["flat"], lo["price"], h2)
    return np.where(np.isnan(r2), np.nan, n), skip


def arm_hold(lo, k):
    r = hold_ret(lo["ret"], lo["ac_next"], lo[f"ac_exit_{k}"])
    days = (lo[f"exit_{k}"] - lo["d"]).dt.days.to_numpy(float)
    flat = N.cost_frac(lo["price"], np.zeros(len(lo), bool), days)
    return E.net_moc(lo["side"], r, flat, lo["price"], lo["hs_in"])


def long_model_scores(p):
    """LONG4: for each year from 2020, an XGBoost regressor fitted on earlier years' NX3-ALL-HI longs (their
    walk-forward picks, so out-of-sample confidences) predicts each long's net; inputs NX3's 37 plus confidence.
    Training ends 10 sessions before the year starts (purge)."""
    import xgboost as xgb
    lo = p[p["side"] > 0].copy()
    feats = [*X.INPUTS, "conf"]
    lo["score"] = np.nan
    params = {**X.XGB, "random_state": SEED}
    for yr in sorted(lo["year"].unique()):
        if yr < 2020:
            continue
        start = pd.Timestamp(f"{yr}-01-01")
        tr = lo[(lo["next"] < start - pd.Timedelta(days=14)) & np.isfinite(lo["base"])]
        te = lo["year"] == yr
        if len(tr) < 500:
            continue
        m = xgb.XGBRegressor(**params).fit(tr[feats].to_numpy(float), tr["base"].to_numpy(float))
        lo.loc[te, "score"] = m.predict(lo.loc[te, feats].to_numpy(float))
    return lo["score"]


def evaluate(lo_c, net, days):
    """Own book and paired gain over plain capped longs on the same days."""
    own = day_book(lo_c["d"], net, days)
    base = day_book(lo_c["d"], plain(lo_c), days)
    return own.to_numpy(float), (own - base).to_numpy(float)


def stage_run(cache, out):
    p = load(cache)
    p = p[p["exit_5"].notna() & (p["exit_5"] <= N.LAST_EXIT)].reset_index(drop=True)   # one trade set for all arms
    lo_all = p[p["side"] > 0]
    lo = cap_longs(p)
    ch, te = lo["year"].isin(CHOOSE).to_numpy(), lo["year"].isin(TEST).to_numpy()
    days_ch = np.sort(p.loc[p["year"].isin(CHOOSE), "d"].unique())
    days_te = np.sort(p.loc[p["year"].isin(TEST), "d"].unique())
    rep = {"trades": {"all": len(p), "longs_uncapped": len(lo_all), "longs_capped": len(lo),
                      "shorts_uncapped": int((p["side"] < 0).sum())},
           "missing": {"later_close": float(lo["ac_exit_5"].isna().mean()), "entry_quote": float(lo["hs_in"].isna().mean()),
                       "price_10:00": float(lo["px_10:00"].isna().mean())}}

    # LONG1, descriptive, 2019-2023 only
    pc = p[p["year"].isin(CHOOSE)]
    rep["LONG1"] = {"capped_longs": splits(lo[ch], plain(lo[ch])),
                    "uncapped_longs": splits(pc[pc["side"] > 0], pc.loc[pc["side"] > 0, "base"]),
                    "uncapped_shorts": splits(pc[pc["side"] < 0], pc.loc[pc["side"] < 0, "base"]),
                    "hold_drift_capped_longs": {str(k): cell(arm_hold(lo[ch], k)) for k in HOLDS},
                    "gross_vs_cost_capped_longs": {"gross": float((lo.loc[ch, "ret"]).mean()),
                                                   "net": float(plain(lo[ch]).mean())}}

    # LONG2: confirmation entry
    arms = {hm: arm_confirm(lo, hm) for hm in CONFIRM_T}
    choose = {hm: float(np.mean(evaluate(lo[ch], arms[hm][0][ch], days_ch)[1])) for hm in CONFIRM_T}
    hm = max(CONFIRM_T, key=lambda k: choose[k])
    n2, sk = arms[hm]
    own, diff = evaluate(lo[te], n2[te], days_te)
    _, h2, _ = confirm(lo["entry_t"], lo["entry"], lo["ret"], lo["hs_in"], lo["d"], lo[f"px_{hm}"], lo[f"hs_{hm}"], hm)
    kc = K.keep_mask(h2, E.C_SKIP)[te]
    rep["LONG2"] = {"choose_2019_2023": choose, "T": hm, "skipped_share_test": float(sk[te].mean()),
                    "with_cost1_skip": {"kept_share": float(kc.mean()), "net_per_trade": float(np.nanmean(n2[te][kc]))},
                    **gate(own, diff, float(np.nanmean(n2[te])))}

    # LONG3: hold longer
    hold = {k: arm_hold(lo, k) for k in HOLDS}
    choose = {str(k): float(np.mean(evaluate(lo[ch], hold[k][ch], days_ch)[1])) for k in HOLDS}
    k = max(HOLDS, key=lambda x: choose[str(x)])
    own, diff = evaluate(lo[te], hold[k][te], days_te)
    rep["LONG3"] = {"choose_2019_2023": choose, "sessions": k, **gate(own, diff, float(np.nanmean(hold[k][te])))}

    # LONG4: long-only model as a filter (score > 0), then NX3-HI's 3 slots
    sc = long_model_scores(p)
    keep = p.index.isin(sc[sc > 0].index)
    lo4 = cap_longs(p[keep | (p["side"] < 0)])
    t4 = lo4[lo4["year"].isin(TEST)]
    base_te = lo[te]
    own = day_book(t4["d"], plain(t4), days_te)
    diff = own - day_book(base_te["d"], plain(base_te), days_te)
    c4 = lo4[lo4["year"].isin([y for y in CHOOSE if y >= 2020])]
    rep["LONG4"] = {"scored_share_2020_on": float(sc[p.loc[sc.index, "year"] >= 2020].notna().mean()),
                    "kept_share_test": float((sc[p.loc[sc.index, "year"].isin(TEST)] > 0).mean()),
                    "check_2020_2023": cell(plain(c4)),
                    "with_cost1_skip": cell(plain(t4)[K.keep_mask(t4["hs_in"], E.C_SKIP)]),
                    **gate(own.to_numpy(float), diff.to_numpy(float), float(plain(t4).mean()) if len(t4) else float("nan"))}

    # LONG5: market filter, rule chosen on 2019-2023
    dates, rules, calm = spy_table(cache)
    res = {}
    for r in MARKET_RULES:
        okd = rule_on(p["d"].to_numpy(), dates, rules[r])
        lo5 = cap_longs(p[okd | (p["side"] < 0).to_numpy()])
        res[r] = lo5
    choose = {}
    for r, lo5 in res.items():
        c = lo5[lo5["year"].isin(CHOOSE)]
        own = day_book(c["d"], plain(c), days_ch)
        choose[r] = float((own - day_book(lo[ch]["d"], plain(lo[ch]), days_ch)).mean())
    r = max(MARKET_RULES, key=lambda x: choose[x])
    t5 = res[r][res[r]["year"].isin(TEST)]
    own = day_book(t5["d"], plain(t5), days_te)
    diff = own - day_book(base_te["d"], plain(base_te), days_te)
    rep["LONG5"] = {"choose_2019_2023": choose, "rule": r, "calm_cut_rv20": calm,
                    "days_on_test": float(rule_on(days_te, dates, rules[r]).mean()),
                    **gate(own.to_numpy(float), diff.to_numpy(float), float(plain(t5).mean()) if len(t5) else float("nan"))}

    rep["plain_capped_longs_test"] = {"own_day": X.tstat(day_book(base_te["d"], plain(base_te), days_te).to_numpy(float)),
                                      "net_per_trade": float(plain(base_te).mean()), "trades": len(base_te)}
    write(out, "LONG.json", rep)


def write(out, name, rep):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v.get("verdict") if isinstance(v, dict) else v for k, v in rep.items()}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/LONG")
    a = ap.parse_args()
    stage_run(Path(a.cache), a.out)


if __name__ == "__main__":
    main()
