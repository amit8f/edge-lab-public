# ORIGIN: copied from breakout-scanner scanner/shared/bars.py at commit 111b71d on 2026-10-11.
# ORIGIN: copied (not imported) from C:\news-signal\research\shared\bars.py at commit 67922e1
# on 2026-09-28. Adapted for breakout-scanner; see git history of this file for changes.
"""SIP daily bar fetch with pagination and a REQUIRED, explicit `adjustment` argument.

The v1 Form 4 study's original fetch had no `adjustment` parameter at all, which meant Alpaca's
undocumented default (`raw`) was used, and real stock splits (NKLA's 1-for-30, AVGO's 10-for-1)
showed up as fake multi-hundred-percent single-day moves. This module makes that mistake
structurally harder to repeat: there is no default value for `adjustment`.
"""
import re
import time

import numpy as np
import pandas as pd
import requests

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def fetch_daily_bars(tickers, start, end, adjustment, headers, feed="sip", batch_size=50, pace_sec=0.3,
                     invalid_out=None):
    """Paginated multi-symbol daily bar fetch. `adjustment` has no default - callers must
    decide (`"all"` for total-return-style studies, `"split"` for parity with the live
    system's training pipeline, `"raw"` only if genuinely intended). Returns
    {symbol: [bar dict, ...]}, tolerating individual invalid symbols (dropped, not fatal to
    the whole batch)."""
    # ADAPTED (breakout-scanner): 429/5xx/network errors retry the same page with backoff
    # instead of restarting the batch, a batch restarted after an invalid-symbol removal
    # discards its partial pages (no duplicated rows), permanent failures raise instead of
    # looping forever, and removed symbols are reported via `invalid_out`.
    assert adjustment in ("raw", "split", "all"), f"adjustment must be explicit, got {adjustment!r}"
    assert feed == "sip", "this project always uses the SIP feed"
    all_bars = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), batch_size):
        batch = list(tickers[i:i + batch_size])
        while batch:
            batch_bars, page_token, restart = {}, None, False
            while True:
                params = {"symbols": ",".join(batch), "timeframe": "1Day", "start": start, "end": end,
                          "feed": feed, "adjustment": adjustment, "limit": 10000}
                if page_token:
                    params["page_token"] = page_token
                for attempt in range(8):
                    try:
                        r = requests.get(BARS_URL, headers=headers, params=params, timeout=90)
                    except requests.RequestException as e:
                        print(f"[bars] network error {e!r}, retry {attempt}")
                        time.sleep(5 * (attempt + 1))
                        continue
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(min(60, 5 * (attempt + 1)))
                        continue
                    break
                if r.status_code != 200:
                    m = re.search(r'invalid symbol: ([^"]+)', r.text)
                    if m and m.group(1) in batch:
                        batch.remove(m.group(1))
                        if invalid_out is not None:
                            invalid_out.append(m.group(1))
                        restart = True
                        break
                    raise RuntimeError(f"[bars] batch at {i} failed permanently: {r.status_code} {r.text[:200]}")
                payload = r.json()
                for sym, rows in (payload.get("bars") or {}).items():
                    batch_bars.setdefault(sym, []).extend(rows)
                page_token = payload.get("next_page_token")
                time.sleep(pace_sec)
                if not page_token:
                    break
            if not restart:
                for sym, rows in batch_bars.items():
                    all_bars.setdefault(sym, []).extend(rows)
                break
    return all_bars


def to_frames(raw_bars, drop_zero_volume=True):
    """Converts {symbol: [bar dicts]} into {symbol: DataFrame} indexed by UTC-naive date.
    Drops zero-volume bars by default: a halted/dormant listing's frozen last-known price is
    not a real tradable price (confirmed on DTC/OABI/PRME/CZFS in the v1 study)."""
    frames = {}
    for sym, rows in raw_bars.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["t"]).dt.tz_convert(None).dt.normalize()
        df = df.drop_duplicates("t").sort_values("t").set_index("t")
        if drop_zero_volume:
            df = df[df["v"] > 0]
        if len(df):
            frames[sym] = df
    return frames


def compute_adv(frames, start=None, end=None):
    """Average daily dollar volume (close * volume) per ticker over [start, end] (inclusive),
    from already-fetched daily bar frames (see to_frames)."""
    out = {}
    for sym, df in frames.items():
        d = df
        if start is not None:
            d = d[d.index >= pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        if len(d):
            out[sym] = float((d["c"] * d["v"]).mean())
    return out
