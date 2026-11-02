"""TOM1 (research/prereg/TOM1.md): hold the market over month ends and into market holidays.

One stage, "measure". Primary series: Ken French's daily market total return (Mkt-RF + RF), 2000 on. SPY and QQQ
(Alpaca SIP, adjustment=all, 2016 on) are the tradability check. Before any event return is printed, the run
checks that French's market and SPY agree day by day (correlation of daily returns >= 0.98) and stops if not.

Usage: python -m tracks.main.tom1_study --stage measure --out research/results/TOM1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm.stats import nw_tstat

SEED, N_PLACEBO, LAGS, COST = 20261102, 10_000, 3, 0.0002
START, HOLDOUT = pd.Timestamp("2000-01-03"), pd.Timestamp("2025-10-01")
ETF_START = pd.Timestamp("2016-01-04")
MIN_EVENTS, MIN_CORR = 300, 0.98
# weekdays the market closed without a scheduled holiday (attacks, storms, national days of mourning)
UNSCHEDULED = pd.to_datetime(["2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11", "2007-01-02",
                              "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])


def holidays(days):
    """Weekday full-day closures between the first and last trading day, minus unscheduled ones."""
    wk = pd.bdate_range(days[0], days[-1])
    return wk.difference(days).difference(UNSCHEDULED)


def events(days, hol):
    """[(kind, buy_idx, sell_idx)] on the trading-day index `days`: month-end buys at the close of the 2nd-to-last
    day of a month, sells at the close of the 3rd day of the next; pre-holiday buys two sessions before the
    holiday and sells at the close of the session before it. A pre-holiday window inside a month-end hold is
    dropped (counted once, as month end). Only windows inside `days`."""
    d = pd.Series(np.arange(len(days)), index=days)
    per = days.to_period("M")
    out, held = [], set()
    months = per.unique()
    for m, nxt in zip(months[:-1], months[1:]):
        cur, nx = d[per == m].to_numpy(), d[per == nxt].to_numpy()
        if len(cur) < 2 or len(nx) < 3:
            continue
        b, s = cur[-2], nx[2]
        out.append(("month_end", int(b), int(s)))
        held.update(range(b + 1, s + 1))
    for h in hol:
        s1 = d.index.searchsorted(h) - 1   # session just before the holiday
        if s1 < 1 or s1 in held:
            continue
        out.append(("pre_holiday", int(s1 - 1), int(s1)))
    return sorted(out, key=lambda e: e[1])


def window_net(logr, b, s):
    """Net return of buying at close b and selling at close s."""
    return float(np.expm1(logr[b + 1:s + 1].sum()) - COST)


def measure_series(r, ev, rng=None, n_placebo=0):
    """Per-event net, year-baseline excess and (optionally) the placebo p-value on one daily return series."""
    days = r.index
    logr = np.log1p(r.to_numpy())
    cum = np.concatenate([[0.0], np.cumsum(logr)])
    years = days.year.to_numpy()
    buy_days = {b for _, b, _ in ev}
    base, pool = {}, {}
    rows = []
    for kind, b, s in ev:
        L, y = s - b, years[b]
        if (y, L) not in base:
            starts = np.where((years == y) & (np.arange(len(days)) + L < len(days)))[0]
            nets = np.expm1(cum[starts + L + 1] - cum[starts + 1]) - COST
            base[(y, L)] = nets.mean()
            pool[(y, L)] = nets[~np.isin(starts, list(buy_days))]
        net = window_net(logr, b, s)
        rows.append({"kind": kind, "buy": days[b], "sell": days[s], "hold": L, "net": net,
                     "excess": net - base[(y, L)], "year": int(y)})
    df = pd.DataFrame(rows)
    p = None
    if rng is not None and n_placebo and len(df):
        keys = list(zip(df["year"], df["hold"]))
        draws = np.zeros((n_placebo, len(df)))
        for j, k in enumerate(keys):
            draws[:, j] = rng.choice(pool[k], n_placebo) - base[k]
        p = float((draws.mean(axis=1) >= df["excess"].mean()).mean())
    return df, p


def summarize(df):
    if not len(df):
        return {"n": 0}
    m, se, t = nw_tstat(df["excess"].to_numpy(), LAGS)
    return {"n": int(len(df)), "mean_excess": m, "nw_t": t, "own_mean_net": float(df["net"].mean()),
            "hit_rate": float((df["net"] > 0).mean())}


def etf_returns(sym):
    raw = fetch_daily_bars([sym], ETF_START.date().isoformat(), "2025-10-31", "all", alpaca_headers())
    c = to_frames(raw)[sym]["c"]
    return c.pct_change().dropna()


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    ffd, sha = stats.fetch_french(stats.FF5 + "_daily")
    mkt = (ffd["Mkt-RF"] + ffd["RF"]).rename("mkt")
    mkt.index = pd.DatetimeIndex(mkt.index)
    mkt = mkt[(mkt.index >= START - pd.Timedelta(days=10)) & (mkt.index < HOLDOUT)]
    spy, qqq = etf_returns("SPY"), etf_returns("QQQ")

    # data check before any event return
    both = pd.concat([mkt, spy.rename("spy")], axis=1, join="inner")
    corr = float(both.corr().iloc[0, 1])
    print(f"data check: French market vs SPY daily returns, {len(both)} days, corr {corr:.4f}")
    res = {"idea": "TOM1", "french_sha256": sha, "corr_mkt_spy": corr, "corr_days": int(len(both))}
    if corr < MIN_CORR:
        res["stopped"] = f"corr {corr:.4f} < {MIN_CORR}; check the data before reading returns"
        (out / "TOM1_result.json").write_text(json.dumps(res, indent=2))
        print(res["stopped"])
        return

    def keep(ev, days):
        return [e for e in ev if days[e[1]] >= START and days[e[2]] < HOLDOUT]

    days = mkt.index
    ev = keep(events(days, holidays(days)), days)
    print(f"funnel: {len(ev)} events ({sum(k == 'month_end' for k, _, _ in ev)} month end, "
          f"{sum(k == 'pre_holiday' for k, _, _ in ev)} pre-holiday), {days[ev[0][1]]:%Y-%m-%d} .. "
          f"{days[ev[-1][2]]:%Y-%m-%d}")
    df, p = measure_series(mkt, ev, np.random.default_rng(SEED), N_PLACEBO)
    prim = summarize(df)
    prim["placebo_p"] = p

    etf = {}
    for name, r in (("SPY", spy), ("QQQ", qqq)):
        r = r[r.index < HOLDOUT]
        d2 = r.index
        e2 = [e for e in events(d2, holidays(d2)) if d2[e[2]] < HOLDOUT and e[1] >= 1]
        x, _ = measure_series(r, e2)
        etf[name] = {"all": summarize(x), **{k: summarize(x[x["kind"] == k]) for k in ("month_end", "pre_holiday")}}
        x.to_csv(out / f"TOM1_events_{name}.csv", index=False)

    passed = (prim["n"] >= MIN_EVENTS and prim["mean_excess"] > 0 and prim["nw_t"] >= 3.0 and p < 0.01
              and prim["own_mean_net"] > 0 and etf["SPY"]["all"].get("own_mean_net", -1) > 0)
    near = (not passed and prim["mean_excess"] > 0 and prim["own_mean_net"] > 0 and prim["nw_t"] >= 1.5)
    p2, both_n = stats.idea_counts()
    res.update({
        "primary": prim, "verdict": "PASS" if passed else ("NEAR-MISS (rule 19)" if near else "FAIL"),
        "by_kind": {k: summarize(df[df["kind"] == k]) for k in ("month_end", "pre_holiday")},
        "by_year": {int(y): summarize(g) for y, g in df.groupby("year")},
        "periods": {"2000-2012": summarize(df[df["year"] <= 2012]), "2013-2025": summarize(df[df["year"] >= 2013])},
        "etf": etf, "cost_round_trip": COST, "idea_count": p2, "idea_count_with_phase1": both_n,
    })
    df.to_csv(out / "TOM1_events_market.csv", index=False)
    (out / "TOM1_result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("verdict", "primary", "by_kind", "etf")}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["measure"], default="measure")
    ap.add_argument("--cache", default=None)  # unused; the workflow passes one
    ap.add_argument("--out", default="research/results/TOM1")
    ap.add_argument("--budget-min", type=int, default=290)
    a = ap.parse_args()
    run(Path(a.out))


if __name__ == "__main__":
    main()
