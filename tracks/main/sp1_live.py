"""SP1 forward log (research/prereg/SP1.md, near-miss; Amit 2026-10-31: log only, judged later). No orders.

Nightly, from EDGAR's daily form index:
1. Every Form 10-12B / 10-12B/A filed from 2025-10-01 on is kept in <state>/sp1/filings.jsonl (its own ledger: <state>/sp1/ledger.jsonl).
2. A company's first one, with no 10-12B in the 365 days before it (checked on its EDGAR submissions), is
   logged once as kind SP1_candidate.
3. Its first regular-way trading day comes from research/checks/SP1_live_dates.csv, filled by hand from the
   completion release (round 1's bug: the first bar can be a when-issued day). Once the entry session (the
   22nd, counting that day) has traded, the entry is logged once as kind SP1_entry: that session's raw SIP
   open, and the exit session 250 sessions later. The same-day random control is drawn when the book is
   judged, by SP1's seeded rule on point-in-time data, so it needs nothing logged now.

Usage: python -m tracks.main.sp1_live --state <dir>
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from core import alpaca, sec
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.ledger import Ledger

ET = "America/New_York"
FORMS = {"10-12B", "10-12B/A"}
FIRST_DAY = "2025-10-01"
STALE = 3                     # an index still missing after 3 business days was never published
WAIT, HOLD = 21, 250          # entry = the 22nd session counting the listing day; exit 250 sessions later
DATES = Path("research/checks/SP1_live_dates.csv")
SUBS = "https://data.sec.gov/submissions/CIK{:010d}.json"


def scan(root, yesterday, index=sec.daily_form_index):
    """Appends each new day's 10-12B rows to filings.jsonl; returns all rows so far."""
    f, cur = root / "filings.jsonl", root / "cursor.json"
    start = pd.Timestamp(json.loads(cur.read_text())["last_day"]) + pd.Timedelta(days=1) if cur.exists() \
        else pd.Timestamp(FIRST_DAY)
    for d in pd.bdate_range(start, yesterday):
        idx = index(d)
        if idx is None:
            if d >= yesterday - pd.offsets.BDay(STALE):
                break                                 # recent: not published yet, try again tomorrow
            idx = sec.parse_form_index("----\n")    # older: EDGAR was closed (e.g. 2025-12-24), no index
        rows = [{"form": r.form, "company": r.company, "cik": int(r.cik), "filed": f"{r.filed:%Y-%m-%d}"}
                for r in idx[idx["form"].isin(FORMS)].itertuples(index=False)]
        with open(f, "a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        tmp = cur.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_day": f"{d:%Y-%m-%d}"}))
        os.replace(tmp, cur)
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def earlier_10(cik, first, get=sec.get):
    """True when the company filed a 10-12B in the 365 days before `first` (EDGAR submissions)."""
    r = get(SUBS.format(int(cik)))
    if r is None:
        return None
    rec = r.json().get("filings", {}).get("recent", {})
    lo, hi = pd.Timestamp(first) - pd.Timedelta(days=365), pd.Timestamp(first)
    return any(fm in FORMS and lo <= pd.Timestamp(d) < hi for fm, d in zip(rec.get("form", []), rec.get("filingDate", [])))


def candidates(led, rows, get=sec.get):
    out = []
    if not rows:
        return out
    first = pd.DataFrame(rows).sort_values("filed").drop_duplicates("cik")
    for r in first.itertuples(index=False):
        key = f"{r.cik}"
        if led.has("SP1_candidate", key):
            continue
        prior = earlier_10(r.cik, r.filed, get)
        if prior is None:
            continue                                  # SEC didn't answer: try again tomorrow
        rec = {"cik": int(r.cik), "company": r.company, "first_10_12b": r.filed, "earlier_10_in_365d": prior,
               "candidate": not prior}
        out.append(led.write_json("SP1_candidate", key, f"candidates/{key}.json", rec))
    return out


def entries(led, dates, cal, today, opens):
    """Logs SP1_entry for each hand-dated company whose entry session has traded. `opens(symbol, day)` is the
    raw SIP open of that session (None if no bar)."""
    out = []
    days = pd.DatetimeIndex(cal["date"])
    for r in dates.itertuples(index=False):
        key = f"{int(r.cik)}"
        if led.has("SP1_entry", key) or pd.isna(r.regular_way_day):
            continue
        i = days.searchsorted(pd.Timestamp(r.regular_way_day))
        if i + WAIT >= len(days) or days[i + WAIT] > today:
            continue
        entry = days[i + WAIT]
        px = opens(r.symbol, entry)
        if px is None:
            continue
        rec = {"cik": int(r.cik), "symbol": r.symbol, "listing_day": f"{days[i]:%Y-%m-%d}",
               "entry_day": f"{entry:%Y-%m-%d}", "entry_open_raw": float(px),
               "exit_session": HOLD, "exit_day": f"{days[i + WAIT + HOLD]:%Y-%m-%d}" if i + WAIT + HOLD < len(days) else None,
               "source": getattr(r, "source", None)}
        out.append(led.write_json("SP1_entry", key, f"entries/{key}.json", rec))
    return out


def run(state, now=None, index=sec.daily_form_index, get=sec.get, opens=None, cal=None):
    root = Path(state) / "sp1"
    root.mkdir(parents=True, exist_ok=True)
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    yesterday = (now.tz_convert(ET) - pd.Timedelta(days=1)).normalize().tz_localize(None)
    led = Ledger(root)
    rows = scan(root, yesterday, index)
    cand = candidates(led, rows, get)
    ent = []
    if DATES.exists():
        dates = pd.read_csv(DATES)
        if len(dates):
            today = now.tz_convert(ET).normalize().tz_localize(None)
            cal = cal if cal is not None else alpaca.trading_calendar(FIRST_DAY, (today + pd.Timedelta(days=400)).date())
            if opens is None:
                h = alpaca_headers()

                def opens(sym, d):
                    b = to_frames(fetch_daily_bars([sym], d.date(), alpaca.sip_end(d), adjustment="raw", headers=h))
                    g = b.get(sym)
                    return float(g.loc[d, "o"]) if g is not None and d in g.index else None
            ent = entries(led, dates, cal, today, opens)
    print(f"sp1: {len(rows)} 10-12B filings logged, {len(cand)} new companies, {len(ent)} new entries")
    return rows, cand, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
