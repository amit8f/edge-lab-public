"""BOOM2: which way will a BOOM1 stock move? (research/prereg/BOOM2.md).

Stages (nx3-study workflow):
- boom2_picks: re-run BOOM1's walk-forward and save each day's 20 picks with their inputs (cache nx3/boom2).
- boom2_minutes: 1-minute SIP bars (raw) from 4:00 to the close for each pick's session; pre-market gap at 9:14,
  the first-touch label from the open (+-8%) and from 9:51 (+-4%), and the minute high/low for the data check.
- boom2: the arms D0-D6 and the primary tests. Writes research/results/BOOM2/BOOM2_walkforward.json.
Scores and counts only, never headline text.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import alpaca_headers
from tracks.main import boom1_study as B
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import nxe_study as E

SEED = 20261107
BAR, BAR951 = 0.08, 0.04
PM_LAST = pd.Timedelta(hours=9, minutes=14)
T951 = pd.Timedelta(hours=9, minutes=51)
COVER = 0.30
MIN_CELL = 30
MIN_D2 = 150
PRICE_CUTS, GAP_CUTS = (10.0, 30.0), (0.02, 0.05)
RULES = ("long", "short", "follow_gap", "fade_gap")


def bdir(cache):
    return X.nx3dir(cache) / "boom2"


# ---------- pure helpers (tested) ----------

def first_touch(h, l, c, entry, bar):
    """Minute highs/lows from the entry bar on: (label, exit price). label +1 up-first, -1 down-first, 0 none,
    2 ambiguous (both barriers inside one bar). None-label exits at the last close."""
    up, dn = entry * (1 + bar), entry * (1 - bar)
    for hi, lo in zip(h, l):
        a, b = hi >= up, lo <= dn
        if a and b:
            return 2, np.nan
        if a:
            return 1, up
        if b:
            return -1, dn
    return 0, (c[-1] if len(c) else np.nan)


def pick_row(b, prev_c, open_t, close_t, dv20):
    """One pick's session (1-minute bars t, o, h, l, c, v; naive ET) -> label inputs and outcomes."""
    day = open_t.normalize()
    pm = b[(b["t"] < open_t) & (b["t"] <= day + PM_LAST)]
    rs = b[(b["t"] >= open_t) & (b["t"] < close_t)]
    if rs.empty or rs["t"].iloc[0] > open_t + pd.Timedelta(minutes=5):
        return None
    o = float(rs["o"].iloc[0])
    lab, px = first_touch(rs["h"].to_numpy(float), rs["l"].to_numpy(float), rs["c"].to_numpy(float), o, BAR)
    f5 = rs[rs["t"] < open_t + pd.Timedelta(minutes=5)]
    r951 = rs[rs["t"] >= day + T951]
    pre = rs[rs["t"] < day + T951]
    rec = {"open": o, "close": float(rs["c"].iloc[-1]), "m_hi": float(rs["h"].max()), "m_lo": float(rs["l"].min()),
           "lab": lab, "exit": px,
           "touched_by951": bool(len(pre) and (pre["h"].max() >= o * (1 + BAR) or pre["l"].min() <= o * (1 - BAR))),
           "pm_last": float(pm["c"].iloc[-1]) if len(pm) else np.nan,
           "pm_dv": float((pm["c"] * pm["v"]).sum()) / dv20 if dv20 and dv20 > 0 else np.nan,
           "first5": float(f5["c"].iloc[-1]) / o - 1 if len(f5) else np.nan,
           "rv5": float((f5["c"] * f5["v"]).sum()) / (dv20 / 78) if dv20 and dv20 > 0 else np.nan}
    rec["pm_gap"] = (rec["pm_last"] / prev_c - 1) if np.isfinite(rec["pm_last"]) and prev_c > 0 else (o / prev_c - 1)
    if len(r951):
        e = float(r951["o"].iloc[0])
        l2, p2 = first_touch(r951["h"].to_numpy(float), r951["l"].to_numpy(float), r951["c"].to_numpy(float), e, BAR951)
        rec |= {"e951": e, "lab951": l2, "exit951": p2}
    else:
        rec |= {"e951": np.nan, "lab951": 0, "exit951": np.nan}
    return rec


def rule_side(df, rule):
    g = np.sign(df["pm_gap"].to_numpy(float))
    return {"long": np.ones(len(df)), "short": -np.ones(len(df)), "follow_gap": g, "fade_gap": -g}[rule]


def hits(side, lab):
    return np.where(side == 0, np.nan, (side == lab).astype(float))


def net(side, entry, exitp, price):
    """Net return of a one-day trade on `side` after NX3's cost model; NaN where there is no call."""
    s = np.asarray(side, float)
    r = np.asarray(exitp, float) / np.asarray(entry, float) - 1
    c = N.cost_frac(np.asarray(price, float), s < 0, np.ones(len(s)))
    return np.where(s == 0, np.nan, s * r - c)


def daily(df, col):
    return df.groupby("d")[col].mean().dropna()


def paired(df, a, b):
    m = df.assign(_x=df[a] - df[b]).groupby("d")["_x"].mean().dropna()
    return X.tstat(m.to_numpy(float))


def cell_ids(df):
    p = np.digitize(df["open"].to_numpy(float), PRICE_CUTS)
    g = np.digitize(np.abs(df["pm_gap"].to_numpy(float)), GAP_CUTS)
    s = (df["pm_gap"].to_numpy(float) >= 0).astype(int)
    return p * 6 + g * 2 + s


# ---------- stages ----------

def stage_picks(cache):
    df = B.load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    off = offering_flags(cache)
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
                out.append(g.nlargest(B.TOP, "score"))
        print(f"[boom2 picks] {yr}: {sum(len(x) for x in out)} picks so far")
    p = pd.concat(out, ignore_index=True).merge(off, on=["symbol", "nxt"], how="left")
    p["pre_offering"] = p["pre_offering"].fillna(0.0)
    N.atomic_parquet(p, bdir(cache) / "picks.parquet")
    return p


def offering_flags(cache):
    parts = []
    for f in sorted((X.nx3dir(cache) / X.TRADES).glob("t_*.parquet")):
        x = pd.read_parquet(f, columns=["symbol", "d", "kind", "mins_open", "f_offering"])
        parts.append(x[(x["kind"] == "real") & (x["mins_open"] <= 0)])
    x = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "d", "f_offering"])
    g = x.groupby(["symbol", "d"])["f_offering"].max().rename("pre_offering").reset_index()
    return g.rename(columns={"d": "nxt"})


def stage_minutes(cache, deadline):
    picks = pd.read_parquet(bdir(cache) / "picks.parquet", columns=["symbol", "d", "nxt", "c", "dv20"])
    ses = X.sessions(X.calendar(cache)).set_index("date")
    h = alpaca_headers()
    for m, gm in picks.groupby(picks["nxt"].dt.to_period("M"), sort=True):
        f = bdir(cache) / "minutes" / f"m_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        out = []
        for day, g in gm.groupby("nxt"):
            if day not in ses.index:
                continue
            o_t, c_t = ses.loc[day, "open_t"], ses.loc[day, "close_t"]
            bars = E.bars_1min(sorted(g["symbol"]), day + pd.Timedelta(hours=4), c_t, h)
            for r in g.itertuples(index=False):
                b = bars.get(r.symbol)
                rec = pick_row(b, float(r.c), o_t, c_t, float(r.dv20)) if b is not None and len(b) else None
                if rec:
                    out.append({"symbol": r.symbol, "nxt": day} | rec)
        N.atomic_parquet(pd.DataFrame(out), f)
        print(f"[boom2 minutes] {m}: {len(gm)} picks, {len(out)} with bars", flush=True)
    return True


def load_all(cache):
    p = pd.read_parquet(bdir(cache) / "picks.parquet")
    mm = pd.concat([pd.read_parquet(f) for f in sorted((bdir(cache) / "minutes").glob("m_*.parquet"))],
                   ignore_index=True)
    df = p.merge(mm, on=["symbol", "nxt"], how="inner")
    df["year"] = df["nxt"].dt.year
    df["d"] = df["nxt"]
    return df


def side_d6(tr, te, rng_seed=SEED):
    from catboost import CatBoostClassifier
    cols = [*B.INPUTS, "score", "pm_gap", "pm_dv", "pre_offering"]
    y = tr["lab"].to_numpy(int)
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=rng_seed,
                           verbose=0, thread_count=-1, allow_writing_files=False, loss_function="MultiClass")
    m.fit(tr[cols].to_numpy(np.float32), y)
    k = list(m.classes_)

    def sc(d):
        pr = m.predict_proba(d[cols].to_numpy(np.float32))
        up = pr[:, k.index(1)] if 1 in k else 0
        dn = pr[:, k.index(-1)] if -1 in k else 0
        return up - dn
    cut = np.quantile(np.abs(sc(tr)), 1 - COVER)
    s = sc(te)
    side = np.where(np.abs(s) >= cut, np.sign(s), 0.0)
    imp = dict(zip(cols, map(float, m.get_feature_importance())))
    return side, s, imp


def run(cache, out):
    df = load_all(cache)
    amb = int((df["lab"] == 2).sum())
    qa = {"picks_with_bars": int(len(df)), "ambiguous_dropped": amb,
          "daily_vs_minute_up_gap_gt_2pts": float((np.abs((df["ah1"] / df["ao1"] - 1) - (df["m_hi"] / df["open"] - 1)) > 0.02).mean()),
          "daily_vs_minute_down_gap_gt_2pts": float((np.abs((1 - df["al1"] / df["ao1"]) - (1 - df["m_lo"] / df["open"])) > 0.02).mean()),
          "boom_daily": float(df["boom"].mean()),
          "boom_minute": float(((df["m_hi"] / df["open"] - 1 >= BAR) | (1 - df["m_lo"] / df["open"] >= BAR)).mean())}
    df = df[df["lab"] != 2].reset_index(drop=True)
    price = df["open"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    df["ctrl"] = rng.choice([-1.0, 1.0], len(df))
    df["ctrl_net"] = net(df["ctrl"], df["open"], df["exit"], price)
    df["ctrl951"] = rng.choice([-1.0, 1.0], len(df))
    df["ctrl951_net"] = net(df["ctrl951"], df["e951"], df["exit951"], price)
    lab = df["lab"].to_numpy(int)
    for r in RULES:
        df[f"h_{r}"] = hits(rule_side(df, r), lab)
    rep = {"qa": qa, "d0": d0(df), "arms": {}}
    test = sorted(y for y in df["year"].unique() if y >= min(df["year"]) + 1)
    sides = {a: np.zeros(len(df)) for a in ("D1-NEWS", "D1-NONEWS", "D2", "D3", "D6")}
    best = np.empty(len(df), dtype=object)
    imp = {}
    for y in test:
        tr, ix = df[df["year"] < y], np.flatnonzero(df["year"] == y)
        te = df.iloc[ix]
        b = max(RULES, key=lambda r: np.nanmean(tr[f"h_{r}"]))
        best[ix] = b
        g = np.sign(te["pm_gap"].to_numpy(float))
        news = te["pre_news"].to_numpy(float) > 0
        sides["D1-NEWS"][ix] = np.where(news, g, 0)
        sides["D1-NONEWS"][ix] = np.where(~news, -g, 0)
        sides["D2"][ix] = np.where(te["pre_offering"].to_numpy(float) > 0, -1.0, 0)
        cells_tr, cells_te = cell_ids(tr), cell_ids(te)
        bside = rule_side(te, b)
        d3 = bside.copy()
        for c in np.unique(cells_te):
            t = tr[cells_tr == c]
            if len(t) >= MIN_CELL:
                d3[cells_te == c] = 1.0 if (t["lab"] == 1).sum() >= (t["lab"] == -1).sum() else -1.0
        sides["D3"][ix] = d3
        s6, _, imp[int(y)] = side_d6(tr, te)
        sides["D6"][ix] = s6
    df["best_rule"] = best
    tm = df["year"].isin(test).to_numpy()
    base_h = np.array([df[f"h_{b}"].iat[i] if b else np.nan for i, b in enumerate(best)], dtype=float)
    for a, s in sides.items():
        x = df.assign(side=s, h=hits(s, lab), net=net(s, df["open"], df["exit"], price),
                      base=np.where(s != 0, base_h, np.nan))[tm & (s != 0)]
        x["pmc"] = x["net"] - x["ctrl_net"]
        rep["arms"][a] = summarise(x, a)
    # D4 at 9:51: follow the first 5-minute bar, +-4% barriers from the 9:51 price
    s4 = np.sign(df["first5"].fillna(0).to_numpy(float))
    x = df.assign(side=s4, h=hits(s4, df["lab951"].to_numpy(int)), net=net(s4, df["e951"], df["exit951"], price))
    x = x[tm & (s4 != 0) & np.isfinite(x["net"])]
    x["pmc"] = x["net"] - x["ctrl951_net"]
    rep["arms"]["D4"] = summarise(x, "D4", base=False)
    rep["d6_importance_top15"] = {k: float(v) for k, v in pd.DataFrame(imp).mean(axis=1).sort_values(ascending=False).head(15).items()} if imp else {}
    rep["verdicts"] = {a: verdict(a, r, rep) for a, r in rep["arms"].items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BOOM2_walkforward.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep


def d0(df):
    lab = df["lab"]
    g = np.sign(df["pm_gap"])
    def shares(m):
        return {"n": int(m.sum()), "up_first": float((lab[m] == 1).mean()), "down_first": float((lab[m] == -1).mean()),
                "none": float((lab[m] == 0).mean())}
    allm = np.ones(len(df), bool)
    pb = np.digitize(df["open"], PRICE_CUTS)
    return {"all": shares(allm), "gap_up": shares(g > 0), "gap_down": shares(g < 0),
            "news": shares(df["pre_news"] > 0), "no_news": shares(df["pre_news"] <= 0),
            "price": {lbl: shares(pb == i) for i, lbl in enumerate(("$5-10", "$10-30", "$30+"))},
            "rule_hit": {r: float(np.nanmean(df[f"h_{r}"])) for r in RULES},
            "touched_8pct_by_951": float(df["touched_by951"].mean())}


def summarise(x, a, base=True):
    t = X.tstat
    r = {"calls": int(len(x)), "hit": float(x["h"].mean()) if len(x) else float("nan"),
         "net_per_trade": float(x["net"].mean()) if len(x) else float("nan"),
         "net_2x_cost_per_trade": float((x["net"] - N.cost_frac(x["open"].to_numpy(float), x["side"].to_numpy() < 0,
                                                                 np.ones(len(x)))).mean()) if len(x) else float("nan"),
         "pick_minus_control": t(daily(x, "pmc").to_numpy(float)) if len(x) else {},
         "long": {"n": int((x["side"] > 0).sum()), "hit": float(x.loc[x["side"] > 0, "h"].mean()),
                  "net": float(x.loc[x["side"] > 0, "net"].mean())},
         "short": {"n": int((x["side"] < 0).sum()), "hit": float(x.loc[x["side"] < 0, "h"].mean()),
                   "net": float(x.loc[x["side"] < 0, "net"].mean())},
         "per_year": {int(y): {"n": len(g), "hit": float(g["h"].mean()), "net": float(g["net"].mean())}
                      for y, g in x.groupby("year")}}
    if base and len(x):
        r["base_hit"] = float(x["base"].mean())
        r["hit_minus_best_rule"] = paired(x, "h", "base")
    return r


def verdict(a, r, rep):
    pmc = r.get("pick_minus_control") or {}
    econ = bool(pmc.get("mean", np.nan) > 0 and pmc.get("t_nw", np.nan) >= X.GATE_T)
    if a == "D4":
        return "passed" if econ and r["net_per_trade"] > 0 else "failed"
    if a == "D2":
        if r["calls"] < MIN_D2:
            return "too few, not judged"
        base = rep["d0"]["all"]["down_first"]
        return "passed" if econ and r["short"]["hit"] >= base + 0.08 else "failed"
    hm = r.get("hit_minus_best_rule") or {}
    need = 0.03 if a == "D3" else 0.04
    return "passed" if econ and hm.get("mean", np.nan) >= need and hm.get("t_nw", np.nan) >= X.GATE_T else "failed"


def diagnose(cache, out):
    """POST HOC, not pre-registered (BOOM2.md): why D6's shorts make money and its longs don't. Re-fits D6 with
    the same seed and walk-forward, then reports the picks' open-to-close drift, always-long / always-short net,
    and D6's score deciles. Writes research/results/BOOM2/BOOM2_diagnosis.json."""
    df = load_all(cache)
    df = df[df["lab"] != 2].reset_index(drop=True)
    price = df["open"].to_numpy(float)
    df["oc"] = df["close"] / df["open"] - 1
    test = sorted(y for y in df["year"].unique() if y >= min(df["year"]) + 1)
    s6, sc = np.zeros(len(df)), np.full(len(df), np.nan)
    for y in test:
        ix = np.flatnonzero(df["year"] == y)
        side, score, _ = side_d6(df[df["year"] < y], df.iloc[ix])
        s6[ix], sc[ix] = side, score
    tm = df["year"].isin(test).to_numpy()
    x = df[tm].assign(score=sc[tm], side=s6[tm])
    lab = x["lab"].to_numpy(int)
    nl = net(np.ones(len(x)), x["open"], x["exit"], x["open"])
    ns = net(-np.ones(len(x)), x["open"], x["exit"], x["open"])
    x = x.assign(net_long=nl, net_short=ns, dec=pd.qcut(x["score"], 10, labels=False, duplicates="drop"))
    touched = lab != 0
    rep = {"note": "POST HOC, not pre-registered; hypotheses only, decided by a forward test",
           "picks": int(len(x)),
           "open_to_close_mean": float(x["oc"].mean()), "open_to_close_median": float(x["oc"].median()),
           "open_to_close_by_year": {int(k): float(v) for k, v in x.groupby("year")["oc"].mean().items()},
           "none_exit_close_mean": float(x.loc[lab == 0, "oc"].mean()),
           "always_long_net": float(np.nanmean(nl)), "always_short_net": float(np.nanmean(ns)),
           "always_short_net_by_year": {int(k): float(v) for k, v in x.groupby("year")["net_short"].mean().items()},
           "cost_mean": float(np.nanmean(N.cost_frac(price[tm], np.zeros(tm.sum(), bool), np.ones(tm.sum())))),
           "d6_touched_accuracy": {
               "long": float((lab[(x["side"] > 0).to_numpy() & touched] == 1).mean()),
               "short": float((lab[(x["side"] < 0).to_numpy() & touched] == -1).mean())},
           "deciles": {int(k): {"n": len(g), "score": float(g["score"].mean()), "up_first": float((g["lab"] == 1).mean()),
                                "down_first": float((g["lab"] == -1).mean()), "oc": float(g["oc"].mean()),
                                "net_long": float(g["net_long"].mean()), "net_short": float(g["net_short"].mean())}
                       for k, g in x.groupby("dec")}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BOOM2_diagnosis.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep
