"""PICK-MAIN and PICK-PENNY: the one test on 2024-01-02 .. 2025-09-30 (research/prereg/PICK.md, Amendment 2).

Stage rows (resumable, month by month; 2023-12 is a swing warm-up only): every decision row of every universe
stock gets pick_data's chart inputs (vectorised), NX1 news inputs and the frozen market score; each version scores
it with its frozen per-width models and its tier's width. Rows that clear any version's normal-day bar are kept,
with one random same-minute control per (kept row, version it clears). Trades are simulated only for those.
Written atomically to <state>/rows/YYYY-MM.parquet; the 5-minute bars are not kept.

Stage test: the picks, the $5,000 book and the three registered rules per track and version ->
<out>/PICK_test.json. Only COMBINED decides.

Usage: python -m tracks.main.pick_test --stage rows|test --w1 cache/w1 --nx1 cache/nx1 --fitstate cache/pickfit \
           --fit research/results/PICK --state cache/picktest --budget-min 320
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main import pick_data as P
from tracks.main import pick_study as S
from tracks.main.longterm.stats import nw_tstat
from tracks.news import w1_data as W

WARM, FIRST, LAST = "2023-12", "2024-01", "2025-09"
VERSIONS = list(S.VERSIONS)
BOOT, SEED, LAGS, MIN_TRADES = 2000, 0, 5, 30
SIZE_LO, SIZE_HI, TOP_Q = 0.10, 0.40, 0.99
BOOK = 5000.0
INPUT_COLS = ["mins_open", "gap", "r_open", "r_prev", "r_5m", "r_15m", "r_30m", "hi_dist", "lo_dist", "range_open",
              "vol_pace", "swing30", "log_px", "log_dv", "spy_open"]


# ---------- rows (pure helpers, tested) ----------

def inputs_all(b, ks, prev_c, dv20, swing, spy):
    """pick_data.inputs for every bar index in ks at once (same values, tested)."""
    c, o, h, lo, v = (b[x].to_numpy(float) for x in ("c", "o", "h", "l", "v"))
    ks = np.asarray(ks, dtype=int)
    px = c[ks - 1]
    o0 = o[0]
    cv = np.cumsum(c * v)[ks - 1]
    hmax = np.maximum.accumulate(h)[ks - 1]
    lmin = np.minimum.accumulate(lo)[ks - 1]
    mins = ks * 5.0

    def r(n):
        j = ks - 1 - n
        return np.where(ks > n, px / c[np.clip(j, 0, None)] - 1, np.nan)
    if spy is not None and len(spy):
        sc, so0 = spy["c"].to_numpy(float), float(spy["o"].iat[0])
        sp = np.where(len(spy) >= ks, sc[np.clip(ks - 1, 0, len(sc) - 1)] / so0 - 1, np.nan)
    else:
        sp = np.full(len(ks), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = cv / (dv20 * mins / 390) if dv20 > 0 else np.full(len(ks), np.nan)
    return pd.DataFrame({"mins_open": mins, "gap": o0 / prev_c - 1, "r_open": px / o0 - 1, "r_prev": px / prev_c - 1,
                         "r_5m": r(1), "r_15m": r(3), "r_30m": r(6), "hi_dist": px / hmax - 1, "lo_dist": px / lmin - 1,
                         "range_open": hmax / lmin - 1, "vol_pace": pace, "swing30": swing, "log_px": np.log(px),
                         "log_dv": float(np.log(dv20)), "spy_open": sp})


def day_inputs(day, uni, bars, swings):
    """All decision rows of one day (the same rows pick_data.day_rows would sample from), with chart inputs and the
    bar index k of each row (kept in `k`)."""
    spy = bars.get("SPY")
    parts = []
    for sym, (tr, prev_c, dv20) in uni.items():
        b = bars.get(sym)
        hist = swings.get(sym, [])
        sw = float(np.mean(hist)) if len(hist) >= 10 else np.nan
        if b is None or len(b) < 2 or not np.isfinite(sw) or b["t"].iat[0].strftime("%H:%M") != "09:30":
            continue
        hm = b["t"].dt.strftime("%H:%M")
        ks = np.flatnonzero((hm >= P.DEC_FIRST) & (hm <= P.DEC_LAST))
        if not len(ks):
            continue
        x = inputs_all(b, ks, prev_c, dv20, sw, spy)
        x.insert(0, "k", ks)
        x.insert(0, "t", b["t"].to_numpy()[ks])
        x.insert(0, "track", tr)
        x.insert(0, "d", day)
        x.insert(0, "symbol", sym)
        parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_model(path):
    p = str(path)
    if p.endswith(".json"):
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(p)
        return "XGBoost", m
    if p.endswith(".cbm"):
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(p)
        return "CatBoost", m
    if p.endswith(".txt"):
        import lightgbm
        return "LightGBM", lightgbm.Booster(model_file=p)
    m = joblib.load(p)
    return ("KRR" if hasattr(m, "named_steps") and "krr" in m.named_steps else "Logistic/ExtraTrees"), m


def predict(kind, m, X):
    if kind == "LightGBM":
        return m.predict(X)
    if kind == "KRR":
        return np.clip(m.predict(X), 1e-4, 1 - 1e-4)
    return m.predict_proba(X)[:, 1]


class Frozen:
    """The frozen fit of one track: market parameters and cut, tier cuts, and per version the per-width models,
    the width per tier and the refit bars."""

    def __init__(self, fit, track, root="."):
        T = fit["tracks"][track]
        self.track = track
        self.mkt = {k: tuple(v) for k, v in T["market"]["parts"].items()}
        self.cut = float(T["market"]["cut_20pct"])
        self.tier_cuts = list(T["tier_cuts_swing30"])
        self.v = {}
        for name in VERSIONS:
            V = T["versions"][name]
            self.v[name] = {"feats": fit["versions"][name],
                            "wpt": {int(t): float(w) for t, w in V["widths_per_tier"].items()},
                            "models": {float(w): load_model(Path(root) / p) for w, p in V["models"].items()},
                            "normal": float(V["bar_thresholds_refit_2019_2023"]["normal"]),
                            "bad": float(V["bar_thresholds_refit_2019_2023"]["bad"])}

    def tiers(self, sw):
        return S.tier_of(np.asarray(sw, float), self.tier_cuts)

    def score(self, name, D, tier):
        V = self.v[name]
        X = D[V["feats"]].to_numpy(np.float32)
        need = sorted(set(V["wpt"].values()))
        Sc = {w: predict(*V["models"][w], X) for w in need}
        return S.row_scores(Sc, tier, V["wpt"])


def control_pick(cands, day, t, sym, version):
    """One other symbol from `cands` (same track, same minute), seeded by (day, minute, symbol, version)."""
    pool = sorted(s for s in cands if s != sym)
    if not pool:
        return None
    h = hashlib.sha256(f"{pd.Timestamp(day).date()}|{pd.Timestamp(t):%H:%M}|{sym}|{version}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return pool[int(rng.integers(len(pool)))]


def keep_rows(D, frozen):
    """D: one day's scored rows of one track (score_<V>, tier). -> kept pick rows (role pick) and control rows
    (role ctrl, version, partner)."""
    clears = {V: D[f"score_{V}"].to_numpy() >= frozen.v[V]["normal"] for V in VERSIONS}
    anyc = np.zeros(len(D), bool)
    for m in clears.values():
        anyc |= m
    picks = D[anyc].assign(role="pick", version="", partner="")
    by_t = {t: set(g["symbol"]) for t, g in D.groupby("t")}
    idx = {(s, t): i for i, (s, t) in enumerate(zip(D["symbol"], D["t"]))}
    ctl = []
    for V, m in clears.items():
        for r in D[m].itertuples(index=False):
            c = control_pick(by_t[r.t], r.d, r.t, r.symbol, V)
            if c is not None:
                ctl.append(D.iloc[idx[(c, r.t)]].to_dict() | {"role": "ctrl", "version": V, "partner": r.symbol})
    return pd.concat([picks, pd.DataFrame(ctl)], ignore_index=True) if ctl else picks


def add_trades(K, bars, track):
    p = P.TRACKS[track]
    for w in p["widths"]:
        nets, kinds = [], []
        for s, k, sw in zip(K["symbol"], K["k"], K["swing30"]):
            n, kind = P.trade(bars[s], int(k), p["tp"], w * sw, p["cost"])
            nets.append(n)
            kinds.append(kind)
        K[f"net_{w}"], K[f"exit_{w}"] = np.asarray(nets, float), kinds
    return K


def stage_rows(w1, nx1, fit_dir, state, deadline, cal):
    fit = json.loads((Path(fit_dir) / "PICK_fit.json").read_text())
    frozen = {tr: Frozen(fit, tr) for tr in ("main", "penny")}
    spy = pd.read_parquet(Path(nx1) / "nx1" / "spy.parquet")
    sess = np.unique(S.to_ns(spy["date"]))
    parts = S.spy_parts(spy)
    daily = W.read_rows(w1, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", pd.Timestamp(f"{WARM}-01") - pd.Timedelta(days=10)), ("c", ">=", 1.0),
                         ("dv20", ">=", 1e6)])
    by_day = {d: g for d, g in daily.groupby("date")}
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    sf = Path(state) / "swings.json"
    swings = json.loads(sf.read_text()) if sf.exists() else {}
    h = alpaca_headers()
    for m in pd.period_range(WARM, LAST, freq="M"):
        f = Path(state) / "rows" / f"{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        tone = None if str(m) == WARM else pd.concat([S.read_tone(nx1, str(x)) for x in (m - 1, m)], ignore_index=True)
        out, n_all = [], 0
        for day in dates[(dates >= m.start_time) & (dates <= m.end_time)]:
            i = dates.get_loc(day)
            if i == 0:
                continue
            uni = P.universe(by_day.get(dates[i - 1], daily.iloc[0:0]))
            bars = P.bars_5min(sorted(uni) + ["SPY"], day, h)
            if tone is not None:
                D = day_inputs(day, uni, bars, swings)
                n_all += len(D)
                if len(D):
                    out.append(score_day(D, tone, sess, parts, frozen, bars))
            for sym, b in bars.items():
                s = P.swing30(b)
                if np.isfinite(s):
                    swings[sym] = (swings.get(sym, []) + [s])[-P.SWING_N:]
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        S.atomic(res, f)
        tmp = sf.with_suffix(".tmp")
        tmp.write_text(json.dumps(swings))
        tmp.replace(sf)
        print(f"[pick test rows] {m}: {n_all} decision rows, {len(res)} kept or control rows", flush=True)
    return True


def score_day(D, tone, sess, parts, frozen, bars):
    J = S.join_rows(D, tone, sess, parts)
    out = []
    for tr, g in J.groupby("track"):
        F = frozen[tr]
        g = g.reset_index(drop=True)
        g["market_score"] = S.market_score(g, F.mkt)
        g["bad"] = g["market_score"].to_numpy() < F.cut
        g["tier"] = F.tiers(g["swing30"])
        for V in VERSIONS:
            g[f"score_{V}"] = F.score(V, g, g["tier"].to_numpy())
        K = keep_rows(g, F)
        if len(K):
            out.append(add_trades(K, bars, tr))
    keep = ["symbol", "d", "t", "track", "tier", "bad", "market_score", "role", "version", "partner",
            *[f"score_{V}" for V in VERSIONS]]
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keep)
    return res[keep + [c for c in res.columns if c.startswith(("net_", "exit_"))]]


# ---------- test (pure helpers, tested) ----------

def size_of(score, bar, top):
    if not top > bar:
        return SIZE_LO
    return float(np.clip(SIZE_LO + (SIZE_HI - SIZE_LO) * (score - bar) / (top - bar), SIZE_LO, SIZE_HI))


def book_day(P_):
    """One day's picks (t, score, size wanted) in the registered order -> sizes taken (0 = skipped)."""
    order = P_.sort_values(["t", "score"], ascending=[True, False], kind="stable").index
    room, taken = 1.0, pd.Series(0.0, index=P_.index)
    for i in order:
        want = P_.at[i, "want"]
        take = min(want, room)
        if take < SIZE_LO - 1e-12:
            continue
        taken[i] = take
        room -= take
    return taken


def picks_of(R, frozen_v, version, track, top):
    """Test rows of one track -> one version's picks with their control's net, size and day book weight."""
    wpt = frozen_v["wpt"]
    pk = R[(R["role"] == "pick") & (R["track"] == track)].copy()
    sc = pk[f"score_{version}"].to_numpy()
    bar = np.where(pk["bad"].to_numpy(bool), frozen_v["bad"], frozen_v["normal"])
    pk = pk[sc >= bar].assign(score=sc[sc >= bar], bar=bar[sc >= bar])
    pk = pk.sort_values("t", kind="stable").drop_duplicates(["symbol", "d"], keep="first")
    w = pk["tier"].map(wpt).to_numpy(float)
    pk["net"] = [r[f"net_{x}"] for r, x in zip(pk.to_dict("records"), w)]
    pk["exit"] = [r[f"exit_{x}"] for r, x in zip(pk.to_dict("records"), w)]
    ct = R[(R["role"] == "ctrl") & (R["version"] == version) & (R["track"] == track)]
    cw = ct["tier"].map(wpt).to_numpy(float)
    cnet = {(p, d, t): r[f"net_{x}"] for p, d, t, r, x in zip(ct["partner"], ct["d"], ct["t"], ct.to_dict("records"), cw)}
    pk["ctrl_net"] = [cnet.get((s, d, t), np.nan) for s, d, t in zip(pk["symbol"], pk["d"], pk["t"])]
    pk = pk[np.isfinite(pk["net"].to_numpy(float))].copy()
    pk["want"] = [size_of(s, b, top) for s, b in zip(pk["score"], pk["bar"])]
    pk["size"] = pd.concat([book_day(g) for _, g in pk.groupby("d")]) if len(pk) else pd.Series(dtype=float)
    return pk


def day_boot(day, a, b, n=BOOT, seed=SEED):
    """95% interval of mean(a) - mean(b), resampling whole days (a, b are per-trade 0/1 with their day)."""
    days = np.unique(day)
    if len(days) < 2:
        return [float("nan"), float("nan")]
    ix = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        sel = np.concatenate([ix[d] for d in rng.choice(days, len(days))])
        out[k] = np.nanmean(a[sel]) - np.nanmean(b[sel])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def t_of(x):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 3:
        return {"mean": float("nan"), "t_nw": float("nan"), "days": int(np.isfinite(x).sum())}
    m, se, t = nw_tstat(x, LAGS)
    return {"mean": m, "se_nw": se, "t_nw": t, "days": int(np.isfinite(x).sum())}


def evaluate(pk, test_days):
    """The three registered rules on one version's picks (only rows the book took count as trades)."""
    tk = pk[pk["size"] > 0]
    n = len(tk)
    y = (tk["net"].to_numpy(float) >= S.HOLD_RET).astype(float)
    cn = tk["ctrl_net"].to_numpy(float)
    yc = np.where(np.isfinite(cn), (cn >= S.HOLD_RET).astype(float), np.nan)
    d = tk["d"].to_numpy()
    ci = day_boot(d, y, yc) if n else [float("nan")] * 2
    book = (tk["size"] * tk["net"]).groupby(tk["d"]).sum()
    diff = (tk["net"] - tk["ctrl_net"]).groupby(tk["d"]).mean()
    allday = book.reindex(pd.DatetimeIndex(test_days), fill_value=0.0)
    r2, r3 = t_of(book.to_numpy()), t_of(diff.to_numpy())
    rules = {"1_share_ci_above_0": bool(ci[0] > 0), "2_net_t_2": bool(r2["mean"] > 0 and r2["t_nw"] >= 2),
             "3_minus_random_t_2": bool(r3["mean"] > 0 and r3["t_nw"] >= 2)}
    verdict = "too few" if n < MIN_TRADES else ("pass" if all(rules.values()) else "fail")
    ex = tk["exit"].value_counts(normalize=True).to_dict() if n else {}
    yrs = {str(k): {"n": len(g), "share3": float((g["net"] >= S.HOLD_RET).mean()), "mean_net": float(g["net"].mean())}
           for k, g in tk.groupby(pd.to_datetime(tk["d"]).dt.year)}
    return {"trades": n, "picks_before_book": len(pk), "verdict": verdict, "rules": rules,
            "share3": float(np.nanmean(y)) if n else float("nan"), "share3_control": float(np.nanmean(yc)) if n else float("nan"),
            "share3_diff_ci95": ci, "mean_net": float(tk["net"].mean()) if n else float("nan"),
            "mean_net_control": float(np.nanmean(cn)) if n else float("nan"),
            "book_daily_days_with_picks": r2, "book_daily_all_days": t_of(allday.to_numpy()),
            "pick_minus_control_daily": r3, "share_of_days_with_a_pick": float(len(book) / max(1, len(test_days))),
            "exits": {str(k): float(v) for k, v in ex.items()}, "per_year": yrs,
            "bad_vs_other": {("bad" if k else "other"): {"n": len(g), "mean_net": float(g["net"].mean())}
                             for k, g in tk.groupby(tk["bad"].astype(bool))},
            "per_tier": {str(int(k)): {"n": len(g), "mean_net": float(g["net"].mean())} for k, g in tk.groupby("tier")},
            "book_mean_daily_pct": float(100 * book.mean()) if len(book) else float("nan")}


def top_scores(fitstate, fit, frozen):
    """Per track and version: the 99th percentile of 2019-2023 normal-day scores (refit models, sampled rows) at or
    above the normal bar; the sizing's top."""
    out = {}
    for tr, F in frozen.items():
        D = S.load(fitstate, tr)
        D["market_score"] = S.market_score(D, F.mkt)
        D = D[D["market_score"].notna()].reset_index(drop=True)
        normal = D["market_score"].to_numpy() >= F.cut
        tier = F.tiers(D["swing30"])
        out[tr] = {}
        for V in VERSIONS:
            s = F.score(V, D, tier)[normal]
            above = s[s >= F.v[V]["normal"]]
            out[tr][V] = float(np.quantile(above, TOP_Q)) if len(above) else F.v[V]["normal"]
    return out


def stage_test(fit_dir, fitstate, state, out, cal):
    fit = json.loads((Path(fit_dir) / "PICK_fit.json").read_text())
    frozen = {tr: Frozen(fit, tr) for tr in ("main", "penny")}
    files = [Path(state) / "rows" / f"{m}.parquet" for m in pd.period_range(FIRST, LAST, freq="M")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"test rows missing for {missing}; run stage rows until it prints done")
    R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    R["d"] = pd.to_datetime(R["d"])
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    test_days = dates[(dates >= pd.Timestamp(f"{FIRST}-01")) & (dates <= pd.Period(LAST, "M").end_time)]
    tops = top_scores(fitstate, fit, frozen)
    res = {"test": f"{FIRST}..{LAST}", "rows_kept": int((R["role"] == "pick").sum()), "tops": tops, "tracks": {}}
    for tr, F in frozen.items():
        res["tracks"][tr] = {}
        for V in VERSIONS:
            pk = picks_of(R, F.v[V], V, tr, tops[tr][V])
            res["tracks"][tr][V] = evaluate(pk, test_days)
            print(f"[pick test] {tr} {V}: {res['tracks'][tr][V]['verdict']} ({res['tracks'][tr][V]['trades']} trades)",
                  flush=True)
    res["primary"] = {("PICK-MAIN" if tr == "main" else "PICK-PENNY"): res["tracks"][tr]["COMBINED"]["verdict"]
                      for tr in frozen}
    S.atomic_json(res, Path(out) / "PICK_test.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rows", "test"])
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--nx1", default="cache/nx1")
    ap.add_argument("--fit", default="research/results/PICK")
    ap.add_argument("--fitstate", default="cache/pickfit")
    ap.add_argument("--state", default="cache/picktest")
    ap.add_argument("--budget-min", type=float, default=320)
    a = ap.parse_args()
    cal = alpaca.trading_calendar(pd.Timestamp(f"{WARM}-01") - pd.Timedelta(days=10), pd.Timestamp("2025-10-01"))
    if a.stage == "rows":
        done = stage_rows(a.w1, a.nx1, a.fit, a.state, time.monotonic() + 60 * a.budget_min, cal)
        print("[pick test rows] done" if done else "[pick test rows] time budget used; rerun to continue", flush=True)
    else:
        stage_test(a.fit, a.fitstate, a.state, a.fit, cal)


if __name__ == "__main__":
    main()
