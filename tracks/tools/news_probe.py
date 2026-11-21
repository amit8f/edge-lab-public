"""Prints how many Alpaca news articles come back for a few fixed windows (counts and keys only, no headlines)."""
import pandas as pd

from core import alpaca

WINDOWS = [("2023-03-01T13:00:00Z", "2023-03-01T15:00:00Z"), ("2025-03-03T13:00:00Z", "2025-03-03T15:00:00Z"),
           ("2025-09-02T13:00:00Z", "2025-09-02T15:00:00Z")]

if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC")
    wins = WINDOWS + [((now - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"), None)]
    for s, e in wins:
        for sort in ("asc", "desc"):
            p = {"start": s, "sort": sort, "limit": 50, "include_content": "false"}
            if e:
                p["end"] = e
            j = alpaca.get(alpaca.NEWS_URL, p)
            n = j.get("news") or []
            print(f"[probe] {s}..{e} sort={sort}: {len(n)} articles, keys {sorted(j)}, token {bool(j.get('next_page_token'))}",
                  flush=True)
