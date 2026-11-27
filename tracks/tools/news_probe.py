"""Prints how many Alpaca news articles come back for a few fixed windows (counts and keys only, no headlines)."""
import pandas as pd

from core import alpaca

WINDOWS = [("2023-03-01T13:00:00Z", "2023-03-01T15:00:00Z"), ("2025-03-03T13:00:00Z", "2025-03-03T15:00:00Z"),
           ("2025-09-02T13:00:00Z", "2025-09-02T15:00:00Z")]

if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC")
    f = "%Y-%m-%dT%H:%M:%SZ"
    wins = WINDOWS + [((now - pd.Timedelta(days=d)).strftime(f), (now - pd.Timedelta(days=d) + pd.Timedelta(hours=6)).strftime(f))
                      for d in (1, 3, 7, 14, 30, 60, 120, 240, 400)]
    # one weekday window per month, Sep 2025 .. now (HI3.md: where Alpaca's history starts again)
    for m in pd.period_range("2025-09", now.strftime("%Y-%m"), freq="M"):
        day = pd.Timestamp(f"{m}-15")
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        if day.tz_localize("UTC") < now:
            wins.append((f"{day:%Y-%m-%d}T13:00:00Z", f"{day:%Y-%m-%d}T19:00:00Z"))
    for s, e in wins:
        for sort in ("asc", "desc"):
            p = {"start": s, "sort": sort, "limit": 50, "include_content": "false"}
            if e:
                p["end"] = e
            j = alpaca.get(alpaca.NEWS_URL, p)
            if not (j.get("news") or []) and sort == "desc":
                j2 = alpaca.get(alpaca.NEWS_URL, p | {"symbols": "AAPL,TSLA,NVDA,AMZN,MSFT"})
                print(f"[probe]   with symbols: {len(j2.get('news') or [])}", flush=True)
            n = j.get("news") or []
            print(f"[probe] {s}..{e} sort={sort}: {len(n)} articles, keys {sorted(j)}, token {bool(j.get('next_page_token'))}",
                  flush=True)
