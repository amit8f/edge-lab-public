"""Benzinga news API (trial key): one weekday per month Sep 2025 .. now -> articles that day (pages of 100, up to
20 pages), share with tickers, first/last created time. Counts only, no headline text (HI3.md)."""
import os
import time

import pandas as pd
import requests

URL = "https://api.benzinga.com/api/v2/news"


def page(d, p):
    prm = {"token": os.environ["BENZINGA_API_KEY"], "dateFrom": f"{d:%Y-%m-%d}", "dateTo": f"{d:%Y-%m-%d}",
           "pageSize": 100, "page": p, "displayOutput": "headline", "sort": "created:asc"}
    for k in range(4):
        r = requests.get(URL, params=prm, headers={"accept": "application/json"}, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (k + 1))
            continue
        if r.status_code != 200:
            print(f"[benzinga] {d:%Y-%m-%d} page {p}: HTTP {r.status_code} {r.text[:160]}", flush=True)
            return None
        return r.json()
    return None


if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for m in pd.period_range("2025-09", now.strftime("%Y-%m"), freq="M"):
        d = pd.Timestamp(f"{m}-15")
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        if d >= now:
            continue
        n, tick, first, last = 0, 0, None, None
        for p in range(20):
            j = page(d, p)
            if not j:
                break
            for a in j:
                n += 1
                tick += bool(a.get("stocks"))
                first = first or a.get("created")
                last = a.get("created")
            if len(j) < 100:
                break
        print(f"[benzinga] {d:%Y-%m-%d}: {n} articles{' (20-page cap)' if n >= 2000 else ''}, {tick} with tickers, "
              f"{first} .. {last}", flush=True)
