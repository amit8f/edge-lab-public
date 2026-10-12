"""Historical Benzinga headlines from Alpaca, one parquet file per month (resumable, atomic writes)."""
import time
from pathlib import Path

import pandas as pd

from core import alpaca


def fetch_month(month, fetch=alpaca.news):
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return fetch_window(start, start + pd.offsets.MonthBegin(1), fetch)


def fetch_window(start, end, fetch=alpaca.news):
    """Every article created in [start, end), all symbols, as (id, created_at, headline, symbols)."""
    rows, token = [], None
    while True:
        page = fetch(start.strftime("%Y-%m-%dT%H:%M:%SZ"), end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     page_token=token)
        for a in page.get("news") or []:
            rows.append({"id": a.get("id"), "created_at": a.get("created_at"), "headline": a.get("headline"),
                         "symbols": ",".join(a.get("symbols") or [])})
        token = page.get("next_page_token")
        if not token:
            break
    return pd.DataFrame(rows, columns=["id", "created_at", "headline", "symbols"]).drop_duplicates("id")


def fetch_range(first_month, last_month, cache_dir, deadline=None, fetch=alpaca.news):
    """Fetch every missing month; stop early (returning False) once `deadline` (time.monotonic) passes."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for per in pd.period_range(first_month, last_month, freq="M"):
        f = cache / f"news_{per}.parquet"
        if f.exists():
            continue
        if deadline is not None and time.monotonic() > deadline:
            return False
        df = fetch_month(str(per), fetch)
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(f)
        print(f"[news] {per}: {len(df)} articles")
    return True


def load_range(first_month, last_month, cache_dir, columns=None):
    parts = [pd.read_parquet(Path(cache_dir) / f"news_{per}.parquet", columns=columns)
             for per in pd.period_range(first_month, last_month, freq="M")]
    return pd.concat(parts, ignore_index=True)
