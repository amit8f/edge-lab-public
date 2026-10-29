"""W2 (research/prereg/W2.md): W1a's short-seller group only, one tagged symbol, named in the headline. Stage
"check" builds the events from W1's cached headline months and writes the hand-check file (all events if
fewer than 50, else 50 at random); the return stages wait for that check (>= 80% right).

No headline text is written to the results folder; the check file carries the article URL and the matched
words only.

Usage: python -m tracks.news.w2_shorts --stage check --cache cache/w1 --out research/results/W2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.lanes import short_name
from core import edgar_live
from tracks.news import w1_data as D
from tracks.news import w1a_leaks as W

SEED, CHECK_N, CHECK_PASS = 20261024, 50, 0.80


def named(headline, symbol, name):
    """The headline names the company: its ticker as a whole word, $XYZ or (EXCHANGE: XYZ), or its short name
    (whole word, case-sensitive)."""
    h = headline or ""
    t = re.escape(symbol)
    if re.search(rf"(?<![\w$&.]){t}(?![\w&])", h) or re.search(rf"\${t}\b", h) or \
            re.search(rf"\((?:NASDAQ|NYSE|NYSE American|AMEX|Nasdaq|OTC)\s*:\s*{t}\)", h):
        return True
    n = short_name(name) if name else ""
    return bool(n) and re.search(r"(?<![\w&])" + re.escape(n) + r"(?![\w&])", h) is not None


def month_events(news, names):
    """W2's articles in one month of headlines (id, t, symbol, src_words, deal_words)."""
    c = W.classified(news)
    c = c[(c["group"] == "short_seller") & (c["n_symbols"] == 1)]
    if c.empty:
        return c
    h = news.set_index("id")["headline"]
    keep = [named(h.get(i), s, names.get(s)) for i, s in zip(c["id"], c["symbol"])]
    return c[keep]


def stage_check(cache, out):
    cal = D.calendar()
    _, names = edgar_live.ticker_map()
    parts = [month_events(D.news_month(cache, per), names) for per in pd.period_range("2016-01", D.NEWS_LAST, freq="M")]
    arts = pd.concat([p for p in parts if len(p)], ignore_index=True)
    ev = W.first_per_window(arts, W.DEDUP_DAYS, cal["date"])
    ev["prior_day"] = W.prior_sessions(ev["t"], cal["date"])
    rows = []
    for y in sorted(ev["prior_day"].dropna().dt.year.unique()):
        u = D.universe_year(cache, y).rename(columns={"date": "prior_day"})
        rows.append(ev[ev["prior_day"].dt.year == y].merge(u, on=["symbol", "prior_day"], how="inner"))
    ev = pd.concat(rows, ignore_index=True).sort_values(["t", "id"]).reset_index(drop=True)
    ev.to_parquet(D.w1dir(cache) / "w2_events.parquet")
    out.mkdir(parents=True, exist_ok=True)
    f = out / "W2_check.csv"
    if not f.exists():
        k = np.arange(len(ev)) if len(ev) <= CHECK_N else np.sort(
            np.random.default_rng(SEED).choice(len(ev), size=CHECK_N, replace=False))
        chk = ev.iloc[k][["id", "t", "symbol", "src_words", "deal_words"]]
        chk = chk.merge(W.load_details(cache)[["id", "url"]], on="id", how="left")
        chk.assign(ok="", note="").to_csv(f, index=False)
    cov = {"articles": int(len(arts)), "events_in_universe": int(len(ev)),
           "per_year": {str(y): int(n) for y, n in ev["t"].dt.year.value_counts().sort_index().items()},
           "by_source": {str(k): int(v) for k, v in ev["src_words"].value_counts().items()}}
    (out / "W2_coverage.json").write_text(json.dumps(cov, indent=1))
    print("W2", json.dumps(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/W2")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    stage_check(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
