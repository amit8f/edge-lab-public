"""T1 historical screen (research/prereg/T1.md): the turn-of-the-year rebound of December's biggest
penny losers, 2016-2024, against a same-year random control.

Bars (raw and adjusted, Alpaca SIP, delisted kept) are fetched in ~500-symbol batches and reduced at
once to one row per symbol and year (universe fields on the formation day, year-to-date return, entry
and exit), cached per batch.

Usage: python -m tracks.penny.t1_screen --cache <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm import bars_batch

YEARS = range(2016, 2025)
BARS_START, BARS_END = "2015-12-01", "2025-01-31"
COST, SEED, N_BOOT = 0.02, 20261017, 10_000
GAP_DAYS, MIN_YEAR_BARS, MIN_BARS = 15, 200, 60
FORM_BACK, EXIT_DAY = 5, 5          # formation: 5th-to-last December trading day; exit: 5th January one
COLS = ["year", "symbol", "group", "close", "dv20", "ytd", "gross", "exit_day", "exit_reason"]


def year_dates(tdays, y):
    """(P: last trading day of y-1, F: formation day, E: entry day, X: exit day) for year y."""
    t = pd.DatetimeIndex(tdays)
    dec = t[(t.year == y) & (t.month == 12)]
    jan = t[(t.year == y + 1) & (t.month == 1)]
    return t[t.year == y - 1][-1], dec[-FORM_BACK], dec[-1], jan[EXIT_DAY - 1]


def group_of(close, dv20):
    if 1.0 <= close < 5.0 and dv20 >= 5e5:
        return "penny"
    if close >= 5.0 and 5e5 <= dv20 < 5e6:
        return "small5"
    return None


def symbol_years(sym, raw, adj, tdays):
    """Rows (COLS) for one symbol. raw/adj: frames indexed by date with o, h, l, c, v (zero-volume
    bars already dropped)."""
    if raw is None or adj is None or not len(raw):
        return []
    seg = pd.Series((raw.index.to_series().diff().dt.days > GAP_DAYS).cumsum().to_numpy(), index=raw.index)
    out = []
    for y in YEARS:
        P, F, E, X = year_dates(tdays, y)
        if F not in raw.index or P not in raw.index or seg[P] != seg[F]:
            continue
        s = seg[F]
        inseg = raw[(seg == s).to_numpy()]
        upto = inseg.loc[:F]
        if len(upto[upto.index.year == y]) < MIN_YEAR_BARS or len(upto) < MIN_BARS:
            continue
        dv20 = float((upto["c"] * upto["v"]).tail(20).median())
        g = group_of(float(raw.at[F, "c"]), dv20)
        if g is None or F not in adj.index or P not in adj.index:
            continue
        ytd = adj.at[F, "c"] / adj.at[P, "c"] - 1
        gross, xday, why = np.nan, None, "no entry bar"
        if E in raw.index and E in adj.index and seg[E] == s:
            after = adj.loc[E:X]
            after = after[(seg.reindex(after.index) == s).to_numpy()]
            xday = after.index[-1]
            gross = after["c"].iloc[-1] / adj.at[E, "o"] - 1
            why = "exit day" if xday == X else "last close (stopped trading)"
        out.append({"year": y, "symbol": sym, "group": g, "close": float(raw.at[F, "c"]), "dv20": dv20,
                    "ytd": float(ytd), "gross": float(gross), "exit_day": None if xday is None else f"{xday:%Y-%m-%d}",
                    "exit_reason": why})
    return out


def pick(rows, group):
    """Per year: the losers (bottom decile by year-to-date return, ties by symbol) and the control (as
    many names drawn from the rest with default_rng(SEED + year)), both before dropping missing entries."""
    out = []
    for y, g in rows[rows["group"] == group].groupby("year", sort=True):
        g = g.sort_values(["ytd", "symbol"]).reset_index(drop=True)
        n = len(g) // 10
        if n < 1:
            continue
        rest = g.iloc[n:].sort_values("symbol").reset_index(drop=True)
        k = np.random.default_rng(SEED + int(y)).choice(len(rest), size=min(n, len(rest)), replace=False)
        out.append(g.iloc[:n].assign(role="loser"))
        out.append(rest.iloc[np.sort(k)].assign(role="control"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS + ["role"])


def yearly(sel):
    """Per year: losers' and controls' mean net return (entries with a bar only) and their difference."""
    s = sel.dropna(subset=["gross"]).assign(net=lambda d: d["gross"] - COST)
    t = s.pivot_table(index="year", columns="role", values="net", aggfunc="mean")
    t["n_losers"] = s[s["role"] == "loser"].groupby("year").size()
    t["diff"] = t["loser"] - t["control"]
    return t.dropna(subset=["diff"])


def primary(t):
    d = t["diff"].to_numpy()
    h = stats.ols_hac(d, np.empty((len(d), 0)), 1)
    rng = np.random.default_rng(SEED)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return {"years": int(len(d)), "mean": float(d.mean()), "t_hac": h["t"],
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "by_year": {int(y): round(float(v), 4) for y, v in t["diff"].items()}}


def alpha(sel, t, fac, tdays):
    """Loser-year net return minus its year's control mean, on the 6 factors summed over the year's
    holding window, clustered by year."""
    s = sel[(sel["role"] == "loser")].dropna(subset=["gross"])
    s = s[s["year"].isin(t.index)]
    ex = (s["gross"] - COST - s["year"].map(t["control"])).to_numpy()
    cal = pd.DatetimeIndex(tdays)
    wins = {y: cal[(cal >= year_dates(tdays, y)[2]) & (cal <= year_dates(tdays, y)[3])] for y in t.index}
    F = stats.window_factors(fac, [wins[y] for y in s["year"]])
    out = stats.ols_cluster(ex, F, s["year"].to_numpy())
    out["factors"] = list(fac.columns) + ["(summed over the holding window)"]
    return out


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    h = alpaca_headers()
    tdays = list(alpaca.trading_calendar(BARS_START, BARS_END)["date"])
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]))
    parts = []
    for b in range(0, len(syms), 500):
        f = cache / f"t1_rows_{b // 500:03d}.parquet"
        if not f.exists():
            batch = syms[b:b + 500]
            end = alpaca.sip_end(BARS_END)
            raw = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="raw", headers=h))
            adj = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="all", headers=h))
            rows = [r for s in batch for r in symbol_years(s, raw.get(s), adj.get(s), tdays)]
            tmp = f.with_suffix(".tmp")
            pd.DataFrame(rows, columns=COLS).to_parquet(tmp)
            tmp.replace(f)
        parts.append(pd.read_parquet(f))
        print(f"[t1] batch {b // 500 + 1}/{-(-len(syms) // 500)}")
    rows = pd.concat(parts, ignore_index=True)
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fac = ffd.join(umdd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]]
    p2, both = stats.idea_counts()
    res = {"idea": "T1", "prereg": "research/prereg/T1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_daily_sha256": sha5, "umd_daily_sha256": shau, "cost_round_trip": COST}
    for group in ("penny", "small5"):
        sel = pick(rows, group)
        t = yearly(sel)
        r = primary(t)
        r["losers_gross_mean"] = float(sel.loc[sel["role"] == "loser", "gross"].mean())
        r["losers_median_dv20"] = float(sel.loc[sel["role"] == "loser", "dv20"].median())
        r["n_losers_by_year"] = {int(y): int(n) for y, n in t["n_losers"].items()}
        if group == "penny":
            a = alpha(sel, t, fac, tdays)
            r["factor_alpha"] = a
            r["deflated_sharpe"] = stats.deflated_sharpe(t["diff"].to_numpy(), both)
            r["passes"] = bool(r["mean"] > 0 and r["t_hac"] >= stats.T_PASS and r["ci95"][0] > 0
                               and a["alpha"] > 0 and a["t"] >= stats.ALPHA_T_PASS)
            out.mkdir(parents=True, exist_ok=True)
            chk = sel[sel["role"] == "loser"].sample(n=min(20, (sel["role"] == "loser").sum()), random_state=SEED)
            chk.assign(checked_by="", ok="", note="").to_csv(out / "T1_rows_check.csv", index=False)
            res["PRIMARY"] = r
        else:
            res["descriptive_small5"] = r
    out.mkdir(parents=True, exist_ok=True)
    (out / "T1_result.json").write_text(json.dumps(res, indent=1, default=str))
    rows.to_parquet(out / "T1_rows.parquet")
    print(json.dumps(res, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
