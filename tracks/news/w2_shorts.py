"""W2 (research/prereg/W2.md): W1a's short-seller group only, one tagged symbol, named in the headline. Stage
"check" builds the events from W1's cached headline months and writes the hand-check file (all events if
fewer than 50, else 50 at random); the return stages wait for that check (>= 80% right).

No headline text is written to the results folder; the check file carries the article URL and the matched
words only.

Stage "urls" fills the check file's blank article links from the news API.

Stage "measure" (after the hand check passes, W1a's rule: at least 40 of 50 right) trades each event as W1a does
(entry 30 minutes after the article, exit at the close of trading day +5, signed stock minus SPY, C0's round trip
and borrow) next to a same-day random control; stage "report" writes the primary test.

Usage: python -m tracks.news.w2_shorts --stage check --cache cache/w1 --out research/results/W2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.lanes import short_name
from core import alpaca, edgar_live, stats
from tracks.news import w1_data as D
from tracks.news import w1a_leaks as W

SEED, CHECK_N, CHECK_PASS = 20261024, 50, 0.80
CHECK_FILE, EVENTS_FILE, MEASURE_DIR = "W2_check.csv", "w2_events.parquet", "w2_measure"


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


def fill_urls(chk, fetch):
    """Blank URLs in the check file from the news API, one call per symbol and day. Benzinga only serves the
    full slugged link now (/news/YY/MM/<id>/ returns an error), and W1a's details cache holds only W1a's ids.
    fetch(symbol, start, end) -> [article dicts with id and url]."""
    chk = chk.copy()
    chk["url"] = chk["url"].astype(object)
    for i, r in chk[chk["url"].isna() | (chk["url"].astype(str).str.strip() == "")].iterrows():
        t = pd.Timestamp(r["t"]).tz_localize(D.ET) if pd.Timestamp(r["t"]).tzinfo is None else pd.Timestamp(r["t"])
        got = {a.get("id"): a.get("url") for a in fetch(r["symbol"], t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1))}
        chk.at[i, "url"] = got.get(int(r["id"]))
    return chk


def stage_urls(out):
    f = out / "W2_check.csv"

    def fetch(sym, a, b):
        p = {"symbols": sym, "start": a.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": b.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "include_content": "false"}
        return alpaca.get(alpaca.NEWS_URL, p).get("news") or []
    chk = fill_urls(pd.read_csv(f), fetch)
    tmp = f.with_suffix(".csv.tmp")
    chk.to_csv(tmp, index=False)
    tmp.replace(f)
    print("W2 urls", int(chk["url"].notna().sum()), "of", len(chk))


def stage_measure(cache, out, budget_min):
    """W1a's trade and control code on W2's events; refuses until the hand check passed."""
    return W.stage_measure(cache, out, budget_min, CHECK_FILE, EVENTS_FILE, MEASURE_DIR)


def verdict(stats_, alpha, n_events):
    """W2's pre-registered reading: fewer than 100 events is inconclusive (forward log); otherwise the
    PROTOCOL rule 11-12 screen (mean > 0, HAC t >= 3, bootstrap CI above 0, FF5 + momentum alpha t >= 2)."""
    if n_events < W.MIN_EVENTS:
        return "inconclusive"
    if not alpha:
        return "failed"
    return "passed" if stats.screen_pass(stats_, alpha) else "failed"


def stage_report(cache, out):
    cal = D.calendar()
    rows = [json.loads(p.read_text()) for p in sorted((D.w1dir(cache) / MEASURE_DIR).glob("*.json"))]
    m = pd.DataFrame(rows)
    m["t"] = pd.to_datetime(m["t"])
    p2, both = stats.idea_counts()
    ok = m[m["status"] == "ok"]
    res = {"idea": "W2", "prereg": "research/prereg/W2.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "status_counts": m["status"].value_counts().to_dict(), "events_traded": int(len(ok))}
    s = W.group_stats(ok)
    alpha = None
    if s["dates"] > 2:
        per_day = ok.groupby("entry_day")["net"].mean().sort_index()
        alpha, sha5, shau = W.factor_alpha_events(per_day, cal)
        s["factor_alpha"] = alpha
        s["deflated_sharpe"] = stats.deflated_sharpe(per_day.to_numpy(), both)
        res["ff5_daily_sha256"], res["umd_daily_sha256"] = sha5, shau
    s["control"] = W.group_stats(ok.rename(columns={"net": "_n", "control_net": "net"}))
    res["primary"] = s
    res["verdict"] = verdict(s, alpha, len(ok))
    # exploratory, not tests
    res["by_price_group"] = {pg: W.group_stats(x) for pg, x in ok.groupby("price_group")}
    res["by_year"] = {str(y): W.group_stats(x) for y, x in ok.groupby(ok["t"].dt.year)}
    res["by_source"] = {str(k): W.group_stats(x) for k, x in ok.groupby("src_words") if len(x) >= 10}
    out.mkdir(parents=True, exist_ok=True)
    (out / "W2_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("status_counts", "events_traded", "verdict")}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "urls", "measure", "report"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/W2")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    if a.stage == "urls":
        stage_urls(Path(a.out))
    elif a.stage == "measure":
        if not stage_measure(Path(a.cache), Path(a.out), a.budget_min):
            print("[w2] measure unfinished or refused; rerun to continue")
    elif a.stage == "report":
        stage_report(Path(a.cache), Path(a.out))
    else:
        stage_check(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
