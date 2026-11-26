"""SECTOR1 stage 6 (research/prereg/SECTOR1.md, Amendment 5): (A) do the market's first reactions to CPI, jobs and
FOMC releases carry on through the day and the next day; (B) do smaller stocks catch up with their industry's
leaders a day or five later. 2019-2023 only; directions from 2019-2021, scored on 2022-2023. Descriptive.

    python -m tracks.main.sector6_study --w1 cache/w1 --out research/results/SECTOR1 --cache cache/sector1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import industry
from core.env import alpaca_headers
from tracks.main import sector1_study as S1
from tracks.main import sector2_study as S2
from tracks.main import sector4_study as S4
from tracks.main import sector5_study as S5
from tracks.news import w1_data as W

ET = "America/New_York"
TARGETS = S5.TARGETS
EVENTS = {"CPI": (510, 540, 960), "JOBS": (510, 540, 960), "FOMC": (840, 870, 960)}  # start, end of reaction, close
N_LEAD, MIN_GROUP = 3, 6
COST = S1.COST


# ---------- pure helpers (tested) ----------

def clock_prices(b):
    """5-minute bars (t UTC, c) -> wide frame: ET date x minute of day, the price at that minute (the close of the bar
    that ends then)."""
    et = pd.to_datetime(b["t"], utc=True).dt.tz_convert(ET)
    x = pd.DataFrame({"date": et.dt.tz_localize(None).dt.normalize(),
                      "hm": et.dt.hour * 60 + et.dt.minute + 5, "c": b["c"].to_numpy(float)})
    return x.drop_duplicates(["date", "hm"], keep="last").pivot(index="date", columns="hm", values="c")


def at(px, hm, back=15):
    """Price at minute hm, or the last price up to `back` minutes before it."""
    cols = [m for m in range(hm - back, hm + 1, 5) if m in px.columns]
    return px[cols].ffill(axis=1).iloc[:, -1] if cols else pd.Series(np.nan, index=px.index)


def follow_stats(sig_tr, y_tr, sig_te, y_te, cost):
    """Trade the 2019-2021 direction of the relation between a reaction and a later move."""
    ok_tr, ok_te = np.isfinite(sig_tr) & np.isfinite(y_tr), np.isfinite(sig_te) & np.isfinite(y_te)
    s_tr, y_tr, s_te, y_te = sig_tr[ok_tr], y_tr[ok_tr], sig_te[ok_te], y_te[ok_te]
    if len(y_tr) < 8 or len(y_te) < 8:
        return None
    d = float(np.sign(np.mean(np.sign(s_tr) * y_tr))) or 1.0       # +1 = follow the reaction, -1 = fade it
    pos = d * np.sign(s_te)
    net = pos * y_te - cost
    return {"days_train": len(y_tr), "days_test": len(y_te), "dir": "follow" if d > 0 else "fade",
            "corr_train": float(np.corrcoef(s_tr, y_tr)[0, 1]), "corr_test": float(np.corrcoef(s_te, y_te)[0, 1]),
            "hit_train": float((np.sign(y_tr) == d * np.sign(s_tr)).mean()),
            "hit_test": float((np.sign(y_te) == pos).mean()), "net_test": float(net.mean()),
            "t_test": float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if net.std() > 0 else None}


def groups(sic, ff, date):
    """SIC 3-digit industry where it has at least MIN_GROUP stocks that day, else the FF12 sector."""
    key = pd.Series(np.where(pd.notna(sic), "s" + (sic // 10).astype("Int64").astype(str), "f" + ff.astype(str)))
    n = key.groupby([date.to_numpy(), key.to_numpy()]).transform("size").to_numpy()
    return np.where((n >= MIN_GROUP) & pd.notna(sic).to_numpy(), key.to_numpy(), "f" + ff.astype(str).to_numpy())


def leaders(d):
    """Adds lead (the group's top-N_LEAD previous-day dollar volume stocks' mean return, the stock left out) and
    is_lead."""
    d = d.copy()
    rk = d.groupby(["date", "grp"])["pdv"].rank(ascending=False, method="first")
    d["is_lead"] = rk <= N_LEAD
    r = d["ret"].where(d["is_lead"], 0.0)
    s = r.groupby([d["date"], d["grp"]]).transform("sum")
    c = d["is_lead"].groupby([d["date"], d["grp"]]).transform("sum")
    own = d["is_lead"].to_numpy()
    d["lead"] = ((s - np.where(own, d["ret"], 0.0)) / (c - own.astype(int))).where(c - own.astype(int) > 0)
    return d


def forward(d, k):
    """Return over the next k trading days of the same series (close t to close t+k)."""
    g = d.groupby(["symbol", "seg"], sort=False)["ac"]
    return g.shift(-k) / d["ac"] - 1


def spread_series(d, col, y, q=0.1):
    """Daily top-decile minus bottom-decile mean of y by col."""
    def one(x):
        x = x.dropna(subset=[col, y])
        if len(x) < 50:
            return np.nan
        lo, hi = x[col].quantile(q), x[col].quantile(1 - q)
        return x.loc[x[col] >= hi, y].mean() - x.loc[x[col] <= lo, y].mean()
    return d.groupby("date").apply(one).dropna()


# ---------- part A ----------

def part_a(cache):
    h = alpaca_headers()
    px = {s: clock_prices(pd.concat([b for y in S1.YEARS if len(b := S4.fetch(s, y, cache, h))], ignore_index=True))
          for s in TARGETS}   # one pivot: a day can span two yearly files (UTC vs ET)
    close = {s: at(p, 960) for s, p in px.items()}
    macro = pd.read_csv(S5.MACRO_CSV)
    macro = macro[macro["scheduled"].astype(str).str.lower() == "true"]
    out = {}
    for ev, (a, b, c) in EVENTS.items():
        days = pd.DatetimeIndex(pd.to_datetime(macro.loc[macro["event"] == ev, "date"]))
        react = {s: (at(p, b) / at(p, a) - 1).reindex(days) for s, p in px.items()}
        rest = {s: (at(p, c) / at(p, b) - 1).reindex(days) for s, p in px.items()}
        nxt = {s: (cl.shift(-1) / cl - 1).reindex(days) for s, cl in close.items()}
        tr, te = days.year <= 2021, days.year >= 2022
        big = react["TLT"].abs() >= react["TLT"][tr].abs().median()
        res = {}
        for t in TARGETS:
            cost = S5.cost_of(t)
            r = {}
            for sig in ("SPY", "TLT", "own"):
                s = react[t if sig == "own" else sig].to_numpy(float)
                for name, y in (("rest_of_day", rest[t]), ("next_day", nxt[t])):
                    yv = y.to_numpy(float)
                    r[f"{sig}_{name}"] = follow_stats(s[tr], yv[tr], s[te], yv[te], cost)
                    m = big.to_numpy()
                    r[f"{sig}_{name}_big"] = follow_stats(s[tr & m], yv[tr & m], s[te & m], yv[te & m], cost)
            res[t] = r
        out[ev] = {"days": len(days), "reaction_minutes": [a, b], "targets": res,
                   "mean_abs_reaction_SPY": float(react["SPY"].abs().mean()),
                   "mean_abs_reaction_TLT": float(react["TLT"].abs().mean())}
        print(f"[sector6] {ev}: {len(days)} days, SPY rest {res['SPY']['SPY_rest_of_day']}", flush=True)
    return out


# ---------- part B ----------

def stock_frame(w1, cache):
    ciks = set()
    for p in W.batch_files(w1):
        ciks |= {int(c) for c in pd.read_parquet(p, columns=["cik"])["cik"].dropna().unique()}
    sic = industry.sic_codes(ciks, cache)
    frames = []
    for y in S1.YEARS:
        f = [("date", ">=", pd.Timestamp(f"{y - 1}-12-01")), ("date", "<=", pd.Timestamp(f"{y + 1}-01-15"))]
        d = W.read_rows(w1, ["symbol", "date", "o", "h", "l", "c", "ao", "ac", "dv20", "seg", "cik"], f)
        d = d.drop_duplicates(["symbol", "date"])
        d = d[d["date"] <= S1.LAST]
        d = S1.prepare(d)
        d["f1"], d["f5"] = forward(d, 1), forward(d, 5)
        d = d[d["date"].dt.year == y]
        cik = d["cik"].astype("Int64")
        s = pd.array([sic.get(int(c)) if pd.notna(c) and sic.get(int(c)) else pd.NA for c in cik], dtype="Int64")
        d["sic"] = s
        d["ff"] = [industry.ff12(v) if pd.notna(v) else 12 for v in s]
        frames.append(d[["symbol", "date", "cik", "ret", "pdv", "sic", "ff", "f1", "f5"]])
        print(f"[sector6] stocks {y}: {len(d)} rows", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["grp"] = groups(d["sic"], d["ff"], d["date"])
    d["mkt"] = d.groupby("date")["ret"].transform("mean")
    return leaders(d)


def news_days(w1, cache, cal):
    S2.ensure_inputs(w1, cache, time.monotonic() + 3600)
    parts = []
    for per in pd.period_range("2018-12", "2023-12", freq="M"):
        n = W.news_month(w1, per)
        n = n[n["symbols"].fillna("") != ""]
        x = pd.DataFrame({"date": S2.to_session(W.et_time(n["created_at"]), cal), "symbol": n["symbols"].str.split(",")})
        parts.append(x.dropna(subset=["date"]).explode("symbol").drop_duplicates())
    return pd.concat(parts, ignore_index=True).drop_duplicates().assign(news=True)


def ev_stats(x, ycol):
    pnl = x["side"] * x[ycol]
    net = pnl - COST
    daily = net.groupby(x["date"]).mean()
    return {"trades": len(x), "days": int(x["date"].nunique()),
            "trades_per_day": float(len(x) / max(x["date"].nunique(), 1)),
            "hit": float((pnl > 0).mean()) if len(x) else None, "gross": float(pnl.mean()) if len(x) else None,
            "net": float(net.mean()) if len(x) else None, "t_day_nw": S4.nw_t(daily.to_numpy())}


def part_b(w1, cache):
    d = stock_frame(w1, cache)
    cal = pd.DatetimeIndex(sorted(d["date"].unique()))
    last_ok = cal[-6]
    d = d[d["date"] <= last_ok]
    nd = news_days(w1, cache, cal)
    d = d.merge(nd, on=["symbol", "date"], how="left")
    d["news"] = d["news"].eq(True)
    lead_news = d[d["is_lead"]].groupby(["date", "grp"])["news"].any().rename("lead_news")
    d = d.join(lead_news, on=["date", "grp"])
    f = d[~d["is_lead"]].dropna(subset=["lead", "ret"]).copy()
    f["gap"] = f["lead"] - f["ret"]
    f["period"] = np.where(f["date"].dt.year <= 2021, "2019-2021", "2022-2023")
    out = {"followers_rows": len(f), "sort": {}, "events": {}, "regression": {}}
    for per, x in f.groupby("period"):
        for y in ("f1", "f5"):
            s = spread_series(x, "gap", y)
            if y == "f5":       # overlapping 5-day windows: use every 5th day for the mean's t
                s5 = s.iloc[::5]
                t = S4.nw_t(s5.to_numpy())
            else:
                t = S4.nw_t(s.to_numpy())
            cost = 2 * COST
            out["sort"][f"{per}_{y}"] = {"days": len(s), "spread": float(s.mean()), "t": t,
                                          "net_after_costs": float(s.mean() - cost)}
    med = f["pdv"].median()
    big_lead = f["lead"].abs() >= 0.02
    lagging = (np.sign(f["ret"]) != np.sign(f["lead"])) | (f["ret"].abs() < 0.5 * f["lead"].abs())
    e = f[big_lead & lagging & ~f["news"]].copy()
    e["side"] = np.sign(e["lead"])
    for per, x in e.groupby("period"):
        for y in ("f1", "f5"):
            out["events"][f"{per}_{y}_all"] = ev_stats(x, y)
            out["events"][f"{per}_{y}_small"] = ev_stats(x[x["pdv"] < med], y)
            out["events"][f"{per}_{y}_large"] = ev_stats(x[x["pdv"] >= med], y)
            out["events"][f"{per}_{y}_leaders_had_news"] = ev_stats(x[x["lead_news"].eq(True)], y)
            out["events"][f"{per}_{y}_leaders_no_news"] = ev_stats(x[~x["lead_news"].eq(True)], y)
            for s_, nm in ((1, "up"), (-1, "down")):
                out["events"][f"{per}_{y}_{nm}"] = ev_stats(x[x["side"] == s_], y)
    z = f.dropna(subset=["f1"])
    tr, te = z[z["period"] == "2019-2021"], z[z["period"] == "2022-2023"]
    X = ["lead", "ret", "mkt"]
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[X].to_numpy(float)], tr["f1"].to_numpy(float), rcond=None)
    p = np.c_[np.ones(len(te)), te[X].to_numpy(float)] @ beta
    out["regression"] = {"coef": dict(zip(["const", *X], map(float, beta), strict=True)),
                         "corr_test": float(np.corrcoef(p, te["f1"])[0, 1]),
                         "rank_ic_test": float(pd.Series(p).corr(te["f1"].reset_index(drop=True), method="spearman"))}
    print(f"[sector6] part B: {json.dumps(out['sort'], default=float)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--out", default="research/results/SECTOR1")
    ap.add_argument("--cache", default="cache/sector1")
    ap.add_argument("--minutes", type=float, default=150)
    a = ap.parse_args()
    out = {"part_a": part_a(a.cache), "part_b": part_b(a.w1, a.cache),
           "note": ("2019-2023. Part A: first reaction (08:30-09:00 ET for CPI and jobs, 14:00-14:30 for scheduled FOMC) "
                    "of SPY, TLT or the target vs the rest of the day and the next day; direction from 2019-2021, "
                    "scored 2022-2023; costs 0.02% ETFs, 0.05% the seven. Part B: followers vs their SIC-3 (else FF12) "
                    "group's top-3 dollar-volume leaders; costs 0.20% round trip per side. Descriptive only.")}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "SECTOR1_stage6.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.replace(Path(a.out) / "SECTOR1_stage6.json")
    print(json.dumps(out["part_b"]["events"], default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
