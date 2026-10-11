# ORIGIN: request helpers adapted from breakout-scanner shadow/sources.py on 2026-10-11.
"""Alpaca REST helpers: retries, the 16-minute SIP delay of the free plan, calendar, assets and news.
Paper trading endpoint only."""
import time

import pandas as pd
import requests

from core.env import alpaca_headers
from core.instruments import classify

TRADE = "https://paper-api.alpaca.markets/v2"
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
# The free data plan refuses SIP bars from the last 15 minutes, so every bar request ends at now - 16 min.
SIP_DELAY = pd.Timedelta(minutes=16)


def utcnow():
    return pd.Timestamp.now(tz="UTC")


def sip_end(end, now=None):
    """RFC-3339 end for a bar request through day `end`: min(23:59:59 ET of that day, now - 16 min)."""
    now = now if now is not None else utcnow()
    day_end = pd.Timestamp(f"{pd.Timestamp(end):%Y-%m-%d} 23:59:59", tz="America/New_York").tz_convert("UTC")
    return min(day_end, now - SIP_DELAY).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url, params, headers=None, tries=8):
    """GET with backoff on network errors, 429 and 5xx; raises on other errors."""
    headers = headers or alpaca_headers()
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=120)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{url}: no answer after {tries} tries")


def trading_calendar(start, end):
    d = pd.DataFrame(get(f"{TRADE}/calendar", {"start": str(start), "end": str(end)}))
    d["date"] = pd.to_datetime(d["date"])
    d["early_close"] = d["close"] != "16:00"
    return d[["date", "open", "close", "early_close"]]


def active_assets():
    """Active us_equity assets with instrument classification and borrow flags."""
    a = pd.DataFrame(get(f"{TRADE}/assets", {"status": "active", "asset_class": "us_equity"}))
    return classify(a)


def news(start, end=None, page_token=None, limit=50):
    """One page of Benzinga news from Alpaca, ascending by time, all symbols, no article body."""
    p = {"start": start, "sort": "asc", "limit": limit, "include_content": "false"}
    if end:
        p["end"] = end
    if page_token:
        p["page_token"] = page_token
    return get(NEWS_URL, p)
