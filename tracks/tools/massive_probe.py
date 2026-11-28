"""Massive (free plan) /v2/reference/news: one weekday per month Sep 2025 .. now -> article count for the day
(first 1,000 pages at most per call, 5 calls a minute), publishers and how many carry tickers. Counts only, no
headline text (HI3.md: is there a free Benzinga source for Oct 2025 - Apr 2026?)."""
import collections
import os
import time

import pandas as pd
import requests

URL = "https://api.massive.com/v2/reference/news"


def get(params):
    for k in range(5):
        r = requests.get(URL, params=params | {"apiKey": os.environ["MASSIVE_API_KEY"]}, timeout=60)
        if r.status_code == 429:
            time.sleep(20 * (k + 1))
            continue
        if r.status_code != 200:
            print(f"[massive] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return None
        return r.json()
    return None


def day(d, pages=3):
    pub, n, tick, first, last = collections.Counter(), 0, 0, None, None
    p = {"published_utc.gte": f"{d:%Y-%m-%d}T00:00:00Z", "published_utc.lt": f"{d + pd.Timedelta(days=1):%Y-%m-%d}T00:00:00Z",
         "limit": 1000, "order": "asc", "sort": "published_utc"}
    url_next = None
    for _ in range(pages):
        j = get(p) if url_next is None else get({"cursor": url_next})
        time.sleep(13)
        if not j:
            break
        for a in j.get("results") or []:
            n += 1
            pub[(a.get("publisher") or {}).get("name")] += 1
            tick += bool(a.get("tickers"))
            first = first or a.get("published_utc")
            last = a.get("published_utc")
        nxt = j.get("next_url")
        if not nxt:
            break
        url_next = nxt.split("cursor=")[-1].split("&")[0]
    return n, pub, tick, first, last


if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for m in pd.period_range("2025-09", now.strftime("%Y-%m"), freq="M"):
        d = pd.Timestamp(f"{m}-15")
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        if d >= now:
            continue
        n, pub, tick, first, last = day(d)
        print(f"[massive] {d:%Y-%m-%d}: {n} articles (3 pages max), {tick} with tickers, {first}..{last}; "
              f"publishers {dict(pub.most_common(8))}", flush=True)
