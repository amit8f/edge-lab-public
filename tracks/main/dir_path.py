"""DIR-SSTOP, DIR-BRKT and DIR-AGE (research/prereg/DIRECTION2.md) on BOOM2's picks and their stored minute paths.

Stage `dirpath` (nx3-study workflow): reads cache nx3/boom2 (picks, minutes, paths) and, for DIR-AGE, the Benzinga
months in NX1's cache (timestamps only). Writes research/results/DIR/DIRPATH.json (summary numbers only).

Within a bar the order of high and low is unknown: a bar that reaches both a trade's stop and its target is
ambiguous and the trade is dropped (arm and control alike). Exit checks for DIR-BRKT start on the bar after the fill.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracks.main import boom2_study as B2

FIRST, LAST = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-09-30")
IDEAS = {"DIR-BRKT": 6, "DIR-SSTOP": 7, "DIR-AGE": 9}
T935, T951, T1500 = (pd.Timedelta(hours=9, minutes=35), pd.Timedelta(hours=9, minutes=51),
                     pd.Timedelta(hours=15))
BOOT, BOOT_SEED = 2000, 0


# ---------- pure helpers (tested) ----------

def run_trade(o, h, l, c, i0, entry, side, stop, target, check_first=True):
    """Follows a trade from bar i0 (the entry bar). stop/target are prices. Returns (status, exit): status 'stop',
    'target', 'close' or 'ambiguous'. A bar that opens past the stop exits at its open (gap); bar i0 is checked only
    when check_first."""
    for i in range(i0 if check_first else i0 + 1, len(c)):
        if i > i0 and (o[i] - stop) * side <= 0:
            return "stop", float(o[i])
        hit_s = (l[i] <= stop) if side > 0 else (h[i] >= stop)
        hit_t = (h[i] >= target) if side > 0 else (l[i] <= target)
        if hit_s and hit_t:
            return "ambiguous", float("nan")
        if hit_s:
            return "stop", float(stop)
        if hit_t:
            return "target", float(target)
    return "close", float(c[-1]) if len(c) else float("nan")


def sstop_trade(o, h, l, c, side, stop_pct, tgt_pct=0.08):
    """DIR-SSTOP: entry at the first bar's open, stop stop_pct against the side, target tgt_pct with it."""
    e = float(o[0])
    st, x = run_trade(o, h, l, c, 0, e, side, e * (1 - side * stop_pct), e * (1 + side * tgt_pct))
    return st, e, x


def bracket(t, o, h, l, c, day, k=0.02):
    """DIR-BRKT trigger: P = close of the last bar before 9:35; first bar from 9:35 (before 15:00) whose high reaches
    P(1+k) -> long, whose low reaches P(1-k) -> short; both in one bar -> None. Fill = the stop, or the bar's open when
    it opened past it. Returns (bar index, side, fill, P) or None."""
    t = pd.DatetimeIndex(t)
    pre = np.flatnonzero(t < day + T935)
    if not len(pre):
        return None
    P = float(c[pre[-1]])
    up, dn = P * (1 + k), P * (1 - k)
    for i in np.flatnonzero((t >= day + T935) & (t < day + T1500)):
        a, b = h[i] >= up, l[i] <= dn
        if a and b:
            return None
        if a:
            return i, 1, float(max(up, o[i])), P
        if b:
            return i, -1, float(min(dn, o[i])), P
    return None


def bracket_trade(o, h, l, c, i, side, fill, P, ctrl_side):
    """Arm and control from the same fill: stop distance |fill - P| / fill, target distance |P(1 +- 8%) - fill| / fill,
    each taken against / with the given side. Returns ((status, exit) arm, (status, exit) control)."""
    ds = abs(fill - P) / fill
    dt = abs(P * (1 + 0.08 * side) - fill) / fill
    out = []
    for s in (side, ctrl_side):
        out.append(run_trade(o, h, l, c, i, fill, s, fill * (1 - s * ds), fill * (1 + s * dt), check_first=False))
    return out[0], out[1]


def age_call(news_t, day, move):
    """DIR-AGE: newest pre-open headline at 8:30-9:29 -> follow the move's sign; before 7:30 -> fade; else 0."""
    if not len(news_t) or not np.isfinite(move) or move == 0:
        return 0.0
    newest = pd.Timestamp(max(news_t))
    if newest >= day + pd.Timedelta(hours=8, minutes=30):
        return float(np.sign(move))
    if newest < day + pd.Timedelta(hours=7, minutes=30):
        return float(-np.sign(move))
    return 0.0


def random_sides(days, idea):
    days = pd.DatetimeIndex(days)
    out = np.empty(len(days))
    for d in np.unique(days):
        ix = np.flatnonzero(days == d)
        out[ix] = np.random.default_rng(int(pd.Timestamp(d).strftime("%Y%m%d")) + idea).choice([-1.0, 1.0], len(ix))
    return out


def boot_days(x, n=BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "days": len(x)}
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)),
            "days": len(x)}


def per_day(df, col):
    return boot_days(df.groupby("d")[col].mean().to_numpy())


def balanced_accuracy(side, lab):
    side, lab = np.asarray(side, float), np.asarray(lab, float)
    r = [float((side[lab == k] == k).mean()) for k in (1, -1) if (lab == k).any()]
    return float(np.mean(r)) if len(r) == 2 else float("nan")


def net(side, entry, exitp, price):
    return B2.net(side, entry, exitp, price)


def ok(b, gain=None):
    return b["lo"] is not None and b["lo"] > 0 and (gain is None or b["mean"] >= gain)


# ---------- the three tests ----------

def test_sstop(rows):
    """rows: per pick d, open (price tier), and the path arrays. -> DIR-SSTOP's report."""
    ctl = random_sides(rows["d"], IDEAS["DIR-SSTOP"])
    rec = []
    for r, cs in zip(rows.itertuples(index=False), ctl):
        o, h, l, c = r.po, r.ph, r.pl, r.pc
        x = {"d": r.d, "price": r.open}
        for name, side, sp in (("s4", -1, 0.04), ("s3", -1, 0.03), ("s6", -1, 0.06), ("plain", -1, 0.08), ("ctl", cs, 0.04)):
            st, e, ex = sstop_trade(o, h, l, c, side, sp)
            x[name] = np.nan if st == "ambiguous" else float(net([side], [e], [ex], [r.open])[0])
            x[f"{name}_st"] = st
            x[f"{name}_g"] = np.nan if st == "ambiguous" else side * (ex / e - 1)
        x["oc"] = c[-1] / o[0] - 1
        x["d6"] = getattr(r, "d6", np.nan)
        rec.append(x)
    t = pd.DataFrame(rec)
    t = t[np.isfinite(t["s4"]) & np.isfinite(t["plain"]) & np.isfinite(t["ctl"])]
    n = len(t)
    own = per_day(t, "s4")
    vs_plain = per_day(t.assign(_x=t["s4"] - t["plain"]), "_x")
    vs_ctl = per_day(t.assign(_x=t["s4"] - t["ctl"]), "_x")
    passed = n >= 1000 and ok(own) and own["mean"] > 0 and ok(vs_plain, 0.0025) and ok(vs_ctl)
    q = np.nanquantile(t["oc"], [0.01, 0.05, 0.5, 0.95, 0.99]) if n else [np.nan] * 5
    bear = t[t["d6"] <= np.nanquantile(t["d6"], 0.1)] if np.isfinite(t["d6"]).any() else t.iloc[0:0]
    return {"verdict": "too few" if n < 1000 else "passed" if passed else "failed", "trades": n,
            "mean_net": float(t["s4"].mean()), "mean_net_2x_cost": float((2 * t["s4"] - t["s4_g"]).mean()),
            "own_per_day": own, "minus_plain_per_day": vs_plain, "minus_control_per_day": vs_ctl,
            "plain_mean_net": float(t["plain"].mean()), "control_mean_net": float(t["ctl"].mean()),
            "stop3_mean_net": float(t["s3"].mean()), "stop6_mean_net": float(t["s6"].mean()),
            "exits_stop4": t["s4_st"].value_counts(normalize=True).round(4).to_dict(),
            "open_to_close_pct": dict(zip(["p1", "p5", "p50", "p95", "p99"], [float(x) for x in q])),
            "open_to_close_mean": float(t["oc"].mean()),
            "d6_bearish_tenth_stop4": {"n": len(bear), "mean_net": float(bear["s4"].mean()) if len(bear) else None},
            "per_year": {int(y): {"n": len(g), "mean_net": float(g["s4"].mean())}
                         for y, g in t.groupby(pd.DatetimeIndex(t["d"]).year)}}


def test_brkt(rows, k=0.02):
    ctl = random_sides(rows["d"], IDEAS["DIR-BRKT"])
    rec, trig = [], 0
    for r, cs in zip(rows.itertuples(index=False), ctl):
        b = bracket(r.pt, r.po, r.ph, r.pl, r.pc, r.d, k)
        if b is None:
            continue
        trig += 1
        i, side, fill, P = b
        (sa, xa), (sc, xc) = bracket_trade(r.po, r.ph, r.pl, r.pc, i, side, fill, P, cs)
        if "ambiguous" in (sa, sc):
            continue
        rec.append({"d": r.d, "side": side, "arm": float(net([side], [fill], [xa], [r.open])[0]),
                    "ctl": float(net([cs], [fill], [xc], [r.open])[0]), "g": side * (xa / fill - 1), "exit": sa})
    t = pd.DataFrame(rec, columns=["d", "side", "arm", "ctl", "g", "exit"])
    n = len(t)
    vs = per_day(t.assign(_x=t["arm"] - t["ctl"]), "_x") if n else boot_days([])
    passed = n >= 1000 and ok(vs, 0.0025) and t["arm"].mean() > 0
    return {"verdict": "too few" if n < 1000 else "passed" if passed else "failed", "k": k, "trades": n,
            "trigger_share": float(trig / len(rows)) if len(rows) else None,
            "mean_net": float(t["arm"].mean()) if n else None,
            "mean_net_2x_cost": float((2 * t["arm"] - t["g"]).mean()) if n else None,
            "control_mean_net": float(t["ctl"].mean()) if n else None, "minus_control_per_day": vs,
            "exits": t["exit"].value_counts(normalize=True).round(4).to_dict(),
            "per_side": {s: {"n": int((t["side"] == v).sum()), "mean_net": float(t.loc[t["side"] == v, "arm"].mean())
                             if (t["side"] == v).any() else None} for s, v in (("long", 1), ("short", -1))}}


def test_age(df):
    """df: picks with d, open, side (age call), mv (the 9:35 move), e951, exit951, lab951, f_earnings."""
    x = df[(df["side"] != 0) & (df["lab951"] != 2) & np.isfinite(df["exit951"]) & np.isfinite(df["e951"])].copy()
    if x.empty:
        return {"calls": 0, "verdict": "too few"}
    x["ctrl"] = random_sides(x["d"], IDEAS["DIR-AGE"])
    x["net"] = net(x["side"], x["e951"], x["exit951"], x["open"])
    x["ctrl_net"] = net(x["ctrl"], x["e951"], x["exit951"], x["open"])
    g = np.sign(x["mv"].to_numpy(float))
    ba = balanced_accuracy(x["side"], x["lab951"])
    best = np.nanmax([balanced_accuracy(g, x["lab951"]), balanced_accuracy(-g, x["lab951"])])
    vs = per_day(x.assign(_x=x["net"] - x["ctrl_net"]), "_x")
    n = len(x)
    passed = n >= 500 and ba >= best + 0.03 and ok(vs) and x["net"].mean() > 0
    fresh = x["side"] == g
    return {"verdict": "too few" if n < 500 else "passed" if passed else "failed", "calls": n,
            "balanced_accuracy": ba, "ba_best_plain": best, "ba_needed": best + 0.03,
            "mean_net": float(x["net"].mean()), "control_mean_net": float(x["ctrl_net"].mean()),
            "minus_control_per_day": vs,
            "fresh": {"n": int(fresh.sum()), "mean_net": float(x.loc[fresh, "net"].mean()) if fresh.any() else None},
            "stale": {"n": int((~fresh).sum()), "mean_net": float(x.loc[~fresh, "net"].mean()) if (~fresh).any() else None}}


# ---------- stage ----------

def load_rows(cache):
    from tracks.news import nx3_study as X
    df = B2.load_all(cache)
    df = df[(df["nxt"] >= FIRST) & (df["nxt"] <= LAST) & (df["open"] >= 5.0)].reset_index(drop=True)
    paths = pd.concat([pd.read_parquet(f) for f in sorted((X.nx3dir(cache) / "boom2" / "paths").glob("p_*.parquet"))],
                      ignore_index=True)
    g = {k: v.sort_values("t") for k, v in paths.groupby(["symbol", "nxt"])}
    arr = []
    for s, d in zip(df["symbol"], df["nxt"]):
        p = g.get((s, d))
        arr.append((p["t"].to_numpy(), *(p[c].to_numpy(float) for c in "ohlc")) if p is not None and len(p) else None)
    keep = [a is not None for a in arr]
    df = df[keep].reset_index(drop=True)
    a = [x for x in arr if x is not None]
    for j, col in enumerate(("pt", "po", "ph", "pl", "pc")):      # the minute path; "c" stays the previous close
        df[col] = [x[j] for x in a]
    return df


def wf_d6(rows, full):
    """D6's walk-forward score for each row (trained on picks of earlier years only), for the reported subset."""
    s = np.full(len(rows), np.nan)
    full = full[full["lab"] != 2]
    for y in sorted(rows["year"].unique()):
        tr = full[full["year"] < y]
        ix = np.flatnonzero(rows["year"].to_numpy() == y)
        if len(tr) < 500 or not len(ix):
            continue
        s[ix] = B2.side_d6(tr, rows.iloc[ix])[1]
    return s


def run(cache, out):
    from tracks.news import w1_data as W
    picks = load_rows(cache)
    rows = picks.assign(d=picks["nxt"], d6=wf_d6(picks, B2.load_all(cache)))
    rep = {"idea_count": "150 (165 with phase 1)", "picks_with_paths": len(rows),
           "DIR-SSTOP": test_sstop(rows), "DIR-BRKT": test_brkt(rows), "DIR-BRKT_k3": test_brkt(rows, 0.03)}
    # DIR-AGE: newest single-symbol headline from the previous close to 9:29
    news = []
    for per in pd.period_range(FIRST.to_period("M") - 1, LAST.to_period("M"), freq="M"):
        n = W.news_month(cache, per)
        n = n[n["symbols"].fillna("").ne("") & ~n["symbols"].fillna("").str.contains(",")]
        news.append(pd.DataFrame({"symbol": n["symbols"].to_numpy(), "t": W.et_time(n["created_at"]).to_numpy()}))
    news = pd.concat(news, ignore_index=True)
    from tracks.news import nx3_study as X
    ses = X.sessions(X.calendar(cache)).set_index("date")
    by = {k: np.sort(v["t"].to_numpy()) for k, v in news.groupby("symbol")}
    side, mv = [], []
    for r in rows.itertuples(index=False):
        tt = by.get(r.symbol, np.array([], dtype="datetime64[ns]"))
        lo = ses.loc[r.d, "prev_close_t"] if r.d in ses.index else r.d
        w = tt[(tt > np.datetime64(lo)) & (tt < np.datetime64(r.d + pd.Timedelta(hours=9, minutes=30)))]
        t = pd.DatetimeIndex(r.pt)
        j = np.flatnonzero(t <= r.d + T935)
        m = float(r.pc[j[-1]]) / float(r.c) - 1 if len(j) and float(r.c) > 0 else np.nan
        mv.append(m)
        side.append(age_call(w, r.d, m))
    rep["DIR-AGE"] = test_age(rows.assign(side=side, mv=mv))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "DIRPATH.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/DIR")
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
