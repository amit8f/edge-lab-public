"""PICK3: the same-day picker with a moving bar (research/prereg/PICK3.md).

Stage scan (resumable, month by month): 2023-09 warms the swing only; 2023-10 to 2023-12 are scored to warm the
moving bar (no trades); 2024-01 to 2025-09 are traded. Every main decision row is scored by PICK2's committed COMBINED
refit models (r = the mean raw prediction of the three learners). Session d's bars come from the previous 60
sessions' highest r: normal rows clear at the 30th percentile, bad-market rows at the 70th, and r > 0. A stock's
first clearing row is its only candidate; at most 2 buys a day, each with one random same-minute control.
Each month's trades go to <state>/trades/YYYY-MM.parquet, the daily maxima to <state>/maxima.json.

Stage check: the registered rules on 2024-01-02 .. 2025-09-30 -> <out>/PICK3_check.json.

Usage: python -m tracks.main.pick3_study --stage scan|check --w1 cache/w1 --nx1 cache/nx1 --state cache/pick3 \
           --out research/results/PICK3 --budget-min 320
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.main import pick2_study as P2
from tracks.main import pick_data as P
from tracks.main import pick_study as S
from tracks.main import pick_test as T
from tracks.main.longterm.stats import nw_tstat
from tracks.news import w1_data as W

TRACK, VERSION = "main", "COMBINED"
SWING_WARM, BAR_WARM, FIRST, LAST = "2023-09", "2023-10", "2024-01", "2025-09"
WINDOW = 60
Q_NORMAL, Q_BAD = 0.30, 0.70
MAX_BUYS = 2
USD, BOOK = 1000.0, 5000.0
MIN_TRADES, LAGS = 150, 5
FIT = Path("research/results/PICK2/PICK2_fit.json")


# ---------- pure helpers (tested) ----------

def bars_from(maxima, window=WINDOW):
    """(normal bar, bad bar) from the previous sessions' highest scores; None until `window` sessions exist."""
    m = np.asarray([x for x in maxima[-window:] if np.isfinite(x)], float)
    if len(maxima) < window or not len(m):
        return None
    return float(np.quantile(m, Q_NORMAL)), float(np.quantile(m, Q_BAD))


def picks_of_day(g, bar_normal, bar_bad, n=MAX_BUYS):
    """One day's scored rows (symbol, t, r, bad) -> the buys: first clearing row a stock, time order, higher r first."""
    thr = np.where(g["bad"].to_numpy(bool), bar_bad, bar_normal)
    r = g["r"].to_numpy(float)
    c = g[(r >= thr) & (r > 0)].sort_values(["t", "r"], ascending=[True, False], kind="stable")
    c = c.drop_duplicates("symbol", keep="first")
    return c.head(n)


def check(tk, days):
    """The registered rules on the traded rows (net, ctrl_net, d, bad, exit)."""
    n = len(tk)
    if not n:
        return {"trades": 0, "verdict": "rejected", "reason": "no trades"}
    net, cn = tk["net"].to_numpy(float), tk["ctrl_net"].to_numpy(float)
    book = (tk["net"] * USD / BOOK).groupby(tk["d"]).sum()
    m, _se, t = nw_tstat(book.to_numpy(), LAGS) if len(book) >= 3 else (float("nan"),) * 3
    diff = float(np.nanmean(net - cn))
    rules = {"trades_150": n >= MIN_TRADES, "mean_net_gt_0": float(net.mean()) > 0, "book_t_2": bool(t >= 2),
             "beats_control": diff > 0}
    yrs = {str(k): {"n": len(x), "mean_net": float(x["net"].mean())} for k, x in tk.groupby(pd.to_datetime(tk["d"]).dt.year)}
    return {"trades": n, "days": len(days), "trades_per_day": n / max(1, len(days)),
            "share_days_with_trade": len(book) / max(1, len(days)), "mean_net": float(net.mean()),
            "median_net": float(np.median(net)), "share_plus3": float((net >= S.HOLD_RET).mean()),
            "mean_ctrl": float(np.nanmean(cn)), "pick_minus_ctrl": diff, "book_daily_mean": float(m),
            "book_t_nw": float(t), "rules": rules,
            "verdict": "not rejected" if all(rules.values()) else "rejected",
            "exits": {str(k): float(v) for k, v in tk["exit"].value_counts(normalize=True).items()},
            "per_year": yrs,
            "bad_vs_other": {("bad" if k else "other"): {"n": len(x), "mean_net": float(x["net"].mean())}
                             for k, x in tk.groupby(tk["bad"].astype(bool))}}


# ---------- scoring ----------

class Model:
    def __init__(self, fit, root="."):
        f = fit["frozen_from_PICK"]
        self.mkt = {k: tuple(v) for k, v in f["market"].items()}
        self.cut, self.tier_cuts = float(f["cut_20pct"]), list(f["tier_cuts_swing30"])
        self.wpt = {int(k): float(w) for k, w in f["widths_per_tier"][VERSION].items()}
        self.models = [P2.load_model(Path(root) / ll["model"]) for ll in fit["versions"][VERSION]["learners"].values()]
        self.feats = S.VERSIONS[VERSION]

    def score(self, g):
        g["market_score"] = S.market_score(g, self.mkt)
        g["bad"] = g["market_score"].to_numpy() < self.cut
        g["tier"] = S.tier_of(g["swing30"].to_numpy(float), self.tier_cuts)
        X = g[self.feats].to_numpy(np.float32)
        g["r"] = np.mean([P2.predict(m, X) for m in self.models], axis=0)
        return g


def trade_rows(K, bars, wpt):
    """Each row's net and exit at its tier's stop width."""
    p = P.TRACKS[TRACK]
    out = [P.trade(bars[s], int(k), p["tp"], wpt[int(t)] * sw, p["cost"])
           for s, k, t, sw in zip(K["symbol"], K["k"], K["tier"], K["swing30"])]
    return [o[0] for o in out], [o[1] for o in out]


def day_trades(g, bars, model, bb, day):
    pk = picks_of_day(g, *bb)
    if not len(pk):
        return pk
    pk = pk.copy()
    pk["net"], pk["exit"] = trade_rows(pk, bars, model.wpt)
    by_t = {t: set(x["symbol"]) for t, x in g.groupby("t")}
    idx = {(s, t): i for i, (s, t) in enumerate(zip(g["symbol"], g["t"]))}
    ctl = []
    for r in pk.itertuples(index=False):
        c = T.control_pick(by_t[r.t], day, r.t, r.symbol, "PICK3")
        ctl.append(None if c is None else g.iloc[idx[(c, r.t)]])
    cn = [np.nan] * len(pk)
    for i, row in enumerate(ctl):
        if row is not None:
            cn[i] = trade_rows(pd.DataFrame([row]), bars, model.wpt)[0][0]
    pk["ctrl_net"] = cn
    pk["bar_normal"], pk["bar_bad"] = bb
    return pk[["symbol", "d", "t", "tier", "bad", "r", "net", "exit", "ctrl_net", "bar_normal", "bar_bad"]]


# ---------- stages ----------

def stage_scan(w1, nx1, state, deadline, cal):
    model = Model(json.loads(FIT.read_text()))
    spy = pd.read_parquet(Path(nx1) / "nx1" / "spy.parquet")
    sess = np.unique(S.to_ns(spy["date"]))
    parts = S.spy_parts(spy)
    daily = W.read_rows(w1, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", pd.Timestamp(f"{SWING_WARM}-01") - pd.Timedelta(days=10)),
                         ("date", "<=", pd.Period(LAST, "M").end_time), ("c", ">=", 5.0), ("dv20", ">=", 5e6)])
    by_day = {d: g for d, g in daily.groupby("date")}
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    st = Path(state)
    sf, mf = st / "swings.json", st / "maxima.json"
    swings = json.loads(sf.read_text()) if sf.exists() else {}
    maxima = json.loads(mf.read_text()) if mf.exists() else []
    h = alpaca_headers()
    for m in pd.period_range(SWING_WARM, LAST, freq="M"):
        f = st / "trades" / f"{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        scored = str(m) >= BAR_WARM
        traded = str(m) >= FIRST
        tone = pd.concat([S.read_tone(nx1, str(x)) for x in (m - 1, m)], ignore_index=True) if scored else None
        out, n_rows = [], 0
        for day in dates[(dates >= m.start_time) & (dates <= m.end_time)]:
            i = dates.get_loc(day)
            uni = {s: v for s, v in P.universe(by_day.get(dates[i - 1], daily.iloc[0:0])).items() if v[0] == TRACK}
            bars = P.bars_5min(sorted(uni) + ["SPY"], day, h)
            if scored:
                D = T.day_inputs(day, uni, bars, swings)
                if len(D):
                    g = model.score(S.join_rows(D, tone, sess, parts))
                    n_rows += len(g)
                    bb = bars_from([x for _, x in maxima])
                    if traded and bb is not None:
                        pk = day_trades(g, bars, model, bb, day)
                        if len(pk):
                            out.append(pk)
                    mx = float(np.nanmax(g["r"].to_numpy(float))) if np.isfinite(g["r"]).any() else float("nan")
                    maxima.append([str(day.date()), mx])
            for sym, b in bars.items():
                s = P.swing30(b)
                if np.isfinite(s):
                    swings[sym] = (swings.get(sym, []) + [s])[-P.SWING_N:]
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        S.atomic(res, f)
        for obj, path in ((swings, sf), (maxima, mf)):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj))
            tmp.replace(path)
        print(f"[pick3 scan] {m}: {n_rows} rows scored, {len(res)} trades", flush=True)
    return True


def stage_check(state, out, cal):
    files = [Path(state) / "trades" / f"{m}.parquet" for m in pd.period_range(FIRST, LAST, freq="M")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"trades missing for {missing}; run stage scan until it prints done")
    tk = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tk = tk[np.isfinite(tk["net"].to_numpy(float))] if len(tk) else tk
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    days = dates[(dates >= pd.Timestamp(f"{FIRST}-01")) & (dates <= pd.Period(LAST, "M").end_time)]
    res = {"period": f"{FIRST}..{LAST}", "rules": "research/prereg/PICK3.md"} | check(tk, days)
    mx = json.loads((Path(state) / "maxima.json").read_text())
    res["daily_max_r_quantiles"] = {str(q): float(np.nanquantile([x for _, x in mx], q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)}
    S.atomic_json(res, Path(out) / "PICK3_check.json")
    print(f"[pick3 check] {res['verdict']}: {res['trades']} trades, {res.get('trades_per_day', 0):.2f} a day", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["scan", "check"])
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--nx1", default="cache/nx1")
    ap.add_argument("--state", default="cache/pick3")
    ap.add_argument("--out", default="research/results/PICK3")
    ap.add_argument("--budget-min", type=float, default=320)
    a = ap.parse_args()
    cal = alpaca.trading_calendar(pd.Timestamp(f"{SWING_WARM}-01") - pd.Timedelta(days=10), pd.Timestamp("2025-10-01"))
    if a.stage == "check":
        stage_check(a.state, a.out, cal)
        return
    if stage_scan(a.w1, a.nx1, a.state, time.monotonic() + 60 * a.budget_min, cal):
        stage_check(a.state, a.out, cal)
        print("[pick3 scan] done", flush=True)
    else:
        print("[pick3 scan] time budget used; rerun to continue", flush=True)


if __name__ == "__main__":
    main()
