# ORIGIN: rate limiter and GET from breakout-scanner pipeline/p02f_edgar_form4.py, daily index parser from
# shadow/sources.py, copied on 2026-10-11.
"""SEC EDGAR access under SEC's fair-access limit: request starts are spaced >= MIN_GAP seconds apart
(<= 8 per second, under the 10 per second cap), shared across threads."""
import re
import threading
import time

import pandas as pd
import requests

from core.env import sec_headers

MIN_GAP = 0.125
_lock = threading.Lock()
_next = [0.0]


def _wait_turn():
    with _lock:
        now = time.monotonic()
        start = max(now, _next[0])
        _next[0] = start + MIN_GAP
    time.sleep(max(0.0, start - now))


def get(url, headers=None, tries=8):
    """Response on 200, None on 404, raises after `tries` other failures."""
    headers = headers or sec_headers()
    last = "no response"
    for attempt in range(tries):
        _wait_turn()
        try:
            r = requests.get(url, headers=headers, timeout=60)
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        last = f"HTTP {r.status_code}"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def parse_form_index(text):
    """EDGAR daily form index text -> frame (form, company, cik, filed, file)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("----")) + 1
    rows = []
    for ln in lines[start:]:
        m = re.match(r"^(\S+(?: \S+)*?)\s{2,}(.*?)\s{2,}(\d+)\s+(\d{8})\s+(\S+)\s*$", ln)
        if m:
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=["form", "company", "cik", "filed", "file"])
    df["filed"] = pd.to_datetime(df["filed"], format="%Y%m%d")
    df["cik"] = df["cik"].astype(int)
    return df


def daily_form_index(d):
    """EDGAR daily form index for date d, or None when not published (yet, or a holiday)."""
    d = pd.Timestamp(d)
    q = (d.month - 1) // 3 + 1
    r = get(f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
    return None if r is None else parse_form_index(r.content.decode("latin-1"))
