"""V4 (research/prereg/V4.md): trading costs measured from SIP quotes for historical picks, a delisting
bound, and a sample of headlines for auditing the analyst classifier.

Quote lookups are cached per batch of picks (atomic parquet writes), so a run that hits its time
budget resumes where it stopped.

Usage: python -m validation.v4_raw --cache <dir> [--out <dir>] [--budget-min N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from tracks.main.longterm import analyst, bars_batch
from tracks.main.longterm.stats import nw_tstat

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
SEED = 20261015
# idea: (holding days, flat round-trip cost, Newey-West lags, picks costed (None = all))
IDEAS = {"M10": (21, 0.002, 6, None), "P3": (21, 0.02, 6, None),
         "M4": (126, 0.002, 5, 300), "M5": (126, 0.002, 5, 300), "M6": (126, 0.002, 5, 300), "M7": (126, 0.002, 5, 300)}
SHUMWAY = {"NYSE": -0.30, "AMEX": -0.30, "NASDAQ": -0.55}
MAX_REL_SPREAD = 0.5
BATCH = 250
NEWS_FIRST, NEWS_LAST = "2016-10", "2025-09"
ANALYST_WORDS = re.compile(r"\b(?:upgrade|downgrade|price target|rating|initiates|maintains|reiterates|analyst)",
                           re.IGNORECASE)


# ---------- costs (pure parts) ----------

def rel_spread(quotes):
    """Median (ask - bid) / mid over valid quotes (bid > 0, ask > bid, relative spread < 50%), or NaN."""
    if not quotes:
        return np.nan, 0
    q = pd.DataFrame(quotes)
    if not {"bp", "ap"} <= set(q.columns):
        return np.nan, 0
    bid, ask = q["bp"].astype(float), q["ap"].astype(float)
    mid = (bid + ask) / 2
    rs = (ask - bid) / mid
    ok = (bid > 0) & (ask > bid) & (rs < MAX_REL_SPREAD)
    return (float(rs[ok].median()) if ok.any() else np.nan), int(ok.sum())


def windows(cal, t, h):
    """(entry day, entry window, exit day, exit window) in UTC for a signal date t, or None past the
    calendar. Entry 09:31-09:40 New York time on the next trading day; exit the last 10 minutes before
    that day's close, h trading days after t."""
    days = cal["date"]
    k = int(days.searchsorted(pd.Timestamp(t), side="right"))
    if k + h - 1 >= len(cal):
        return None
    ny = lambda d, hm: pd.Timestamp(f"{d:%Y-%m-%d} {hm}", tz="America/New_York").tz_convert("UTC")
    e, x = cal.iloc[k], cal.iloc[k + h - 1]
    close = ny(x["date"], x["close"])
    return (e["date"], (ny(e["date"], "09:31"), ny(e["date"], "09:40")),
            x["date"], (close - pd.Timedelta(minutes=10), close))


def fetch_quotes(sym, start, end, get=alpaca.get):
    js = get(QUOTES_URL.format(sym=sym), {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                          "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                          "feed": "sip", "limit": 10000})
    return js.get("quotes") or []


def cost_picks(picks, cal, cache_dir, deadline, get=alpaca.get):
    """Per-pick entry and exit spreads (cached per batch); None when the time budget ran out."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    nb = -(-len(picks) // BATCH)
    for b in range(nb):
        f = cache / f"costs_{b:04d}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
            continue
        if time.monotonic() > deadline:
            print(f"[quotes] budget spent at batch {b + 1}/{nb}; rerun to continue")
            return None
        rows = []
        for _, p in picks.iloc[b * BATCH:(b + 1) * BATCH].iterrows():
            w = windows(cal, p["date"], p["h"])
            r = {"key": p["key"], "entry_spread": np.nan, "exit_spread": np.nan, "n_entry": 0, "n_exit": 0,
                 "exit_day_quotes": -1}
            if w is not None:
                r["entry_spread"], r["n_entry"] = rel_spread(fetch_quotes(p["symbol"], *w[1], get=get))
                r["exit_spread"], r["n_exit"] = rel_spread(fetch_quotes(p["symbol"], *w[3], get=get))
                if r["n_exit"] == 0:                  # quiet last 10 minutes, or no longer trading at all?
                    day_open = w[3][1].tz_convert("America/New_York").normalize() + pd.Timedelta(hours=9, minutes=30)
                    r["exit_day_quotes"] = len(fetch_quotes(p["symbol"], day_open.tz_convert("UTC"), w[3][1], get=get))
            rows.append(r)
        df = pd.DataFrame(rows)
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(f)
        parts.append(df)
        print(f"[quotes] batch {b + 1}/{nb}")
    return pd.concat(parts, ignore_index=True)


def load_picks(results=RESULTS):
    out = []
    rng = np.random.default_rng(SEED)
    for idea, (h, flat, lags, n) in IDEAS.items():
        p = Path(results) / idea / f"{idea}_screen_picks.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        ret = "ret21" if h == 21 else "ret6"
        df = df[df[ret].notna()].reset_index(drop=True).rename(columns={ret: "ret"})
        if n is not None and len(df) > n:
            df = df.iloc[np.sort(rng.choice(len(df), n, replace=False))].reset_index(drop=True)
        df = df.assign(idea=idea, h=h, flat=flat)
        df["key"] = df["idea"] + "|" + df["date"].dt.strftime("%Y-%m-%d") + "|" + df["symbol"]
        out.append(df[["key", "idea", "date", "symbol", "ret", "h", "flat"]])
    return pd.concat(out, ignore_index=True)


def summarize_costs(picks, costs, exchange, results=RESULTS):
    d = picks.merge(costs, on="key", how="left")
    d["measured"] = d["entry_spread"] / 2 + d["exit_spread"].fillna(d["entry_spread"]) / 2
    d["cost"] = d["measured"].fillna(d["flat"])
    d["no_exit_quotes"] = d["entry_spread"].notna() & (d["exit_day_quotes"] == 0)
    d["shumway"] = d["symbol"].map(exchange).map(SHUMWAY).fillna(-0.30)
    out = {}
    for idea, g in d.groupby("idea"):
        h, flat, lags, n = IDEAS[idea]
        m = g["measured"].dropna()
        r = {"picks": int(len(g)), "measured": int(len(m)), "flat": flat,
             "measured_mean": float(m.mean()) if len(m) else None, "measured_median": float(m.median()) if len(m) else None,
             "measured_p90": float(m.quantile(0.9)) if len(m) else None,
             "share_above_flat": float((m > flat).mean()) if len(m) else None,
             "no_exit_quotes": int(g["no_exit_quotes"].sum())}
        if n is None:                                   # every pick costed: recompute the primary series
            ser = pd.read_csv(Path(results) / idea / f"{idea}_primary_series.csv", parse_dates=["date"])
            um = ser.set_index("date")["uni_mean"]
            per_date = lambda x: x.assign(net=x["ret"] - x["cost"]).groupby("date")["net"].mean()
            per = per_date(g)
            mm, se, t = nw_tstat((per - um.reindex(per.index)).to_numpy(), lags)
            r["primary_measured_costs"] = {"mean": mm, "t_hac": t, "weeks": int(per.notna().sum())}
            adj = g.assign(ret=np.where(g["no_exit_quotes"], (1 + g["ret"]) * (1 + g["shumway"]) - 1, g["ret"]))
            per2 = per_date(adj)
            mm2, _, t2 = nw_tstat((per2 - um.reindex(per2.index)).to_numpy(), lags)
            r["primary_with_delisting_bound"] = {"mean": mm2, "t_hac": t2}
        out[idea] = r
    return out, d


# ---------- headline sample ----------

def headline_sample(news_dir, n=100, months=None):
    """100 headlines the classifier calls an action and 100 it calls no action that contain an analyst
    word; single-symbol headlines only, read month by month."""
    months = months or pd.period_range(NEWS_FIRST, NEWS_LAST, freq="M")
    act, pool = [], []
    for per in months:
        m = pd.read_parquet(Path(news_dir) / f"news_{per}.parquet", columns=["id", "created_at", "headline", "symbols"])
        s = m["symbols"].fillna("")
        m = m[(s != "") & ~s.str.contains(",")]
        lab = m["headline"].map(analyst.classify)
        act.append(m[lab != 0].assign(regex=lab[lab != 0]))
        kw = m[(lab == 0) & m["headline"].fillna("").str.contains(ANALYST_WORDS)]
        pool.append(kw.assign(regex=0))
    act, pool = pd.concat(act, ignore_index=True), pd.concat(pool, ignore_index=True)
    rng = np.random.default_rng(SEED)
    a = act.iloc[rng.choice(len(act), min(n, len(act)), replace=False)].assign(stratum="classified_action")
    p = pool.iloc[rng.choice(len(pool), min(n, len(pool)), replace=False)].assign(stratum="no_action_with_analyst_word")
    s = pd.concat([a, p], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["sid"] = np.arange(1, len(s) + 1)
    lab = {1: "up", -1: "down", 0: "none"}
    return (s[["sid", "symbols", "headline"]].rename(columns={"symbols": "symbol"}),
            s[["sid", "stratum", "regex"]].assign(regex=s["regex"].map(lab), news_id=s["id"], created_at=s["created_at"]),
            {"classified_actions": int(len(act)), "no_action_pool": int(len(pool))})


def run(cache, out, budget_min=290, get=alpaca.get):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_min * 60
    res = {"study": "V4", "prereg": "research/prereg/V4.md"}
    hl = out / "headlines_to_label.csv"
    if not hl.exists():
        to_label, regex, counts = headline_sample(cache / "news")
        to_label.to_csv(hl, index=False)
        regex.to_csv(out / "headlines_regex.csv", index=False)
        (out / "headline_pool_sizes.json").write_text(json.dumps(counts, indent=1))
        print("HEADLINES", counts)
    picks = load_picks()
    cal = alpaca.trading_calendar("2016-12-01", "2025-10-31")
    costs = cost_picks(picks, cal, cache / "v4_quotes", deadline, get=get)
    if costs is None:
        print("QUOTES INCOMPLETE: rerun this workflow to continue (cached batches are kept)")
        return None
    assets = bars_batch.assets_all()
    exchange = assets.sort_values("status").drop_duplicates("base_symbol").set_index("base_symbol")["exchange"]
    res["costs"], detail = summarize_costs(picks, costs, exchange)
    detail.to_csv(out / "V4_pick_costs.csv", index=False)
    (out / "V4_results.json").write_text(json.dumps(res, indent=1, default=str))
    print("COSTS", json.dumps(res["costs"], indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(RESULTS / "V4"))
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(a.cache, a.out, a.budget_min)


if __name__ == "__main__":
    main()
