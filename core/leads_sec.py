"""SEC filings that often come before the headline, logged forward from EDGAR's daily form index.

For each complete day: every filing of the forms below, one JSON line each, to
<state>/leads/sec/<day>.jsonl. 8-Ks and Schedule 13Ds also get their acceptance time and 8-K items from
the filing header (one request each, under SEC's rate limit), so a later study can measure from the
first moment the filing was public. Each run fills the days from the cursor to yesterday (ET).

Usage: python -m core.leads_sec --state <dir> [--first-day YYYY-MM-DD]
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from core import sec

ET = "America/New_York"
FORMS = {"8-K", "SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A", "S-1", "S-1/A", "S-3", "424B1",
         "424B3", "424B4", "424B5", "NT 10-K", "NT 10-Q", "4"}
WITH_HEADER = {"8-K", "SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"}
HDR = re.compile(r"<(ACCEPTANCE-DATETIME|ITEMS)>([^<\r\n]+)")
FIRST_DAYS_BACK = 7


def parse_header(text):
    """(acceptance 'YYYY-MM-DD HH:MM:SS' ET, [items]) from an -index-headers.html page."""
    t = text.replace("&lt;", "<").replace("&gt;", ">")
    acc, items = None, []
    for k, v in HDR.findall(t):
        v = v.strip()
        if k == "ACCEPTANCE-DATETIME" and acc is None:
            acc = f"{v[:4]}-{v[4:6]}-{v[6:8]} {v[8:10]}:{v[10:12]}:{v[12:14]}"
        elif k == "ITEMS":
            items.append(v)
    return acc, items


def header(cik, file_path, get=sec.get):
    accn = re.search(r"(\d{10}-\d{2}-\d{6})", file_path).group(1)
    r = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index-headers.html")
    return (accn, *parse_header(r.text)) if r is not None else (accn, None, [])


def day_rows(day, index=sec.daily_form_index, get=sec.get):
    """Rows for one day, or None when the index is not published (a holiday, or not yet)."""
    idx = index(day)
    if idx is None:
        return None
    idx = idx[idx["form"].isin(FORMS)]
    rows = []
    for r in idx.itertuples(index=False):
        row = {"form": r.form, "company": r.company, "cik": int(r.cik), "filed": f"{r.filed:%Y-%m-%d}", "file": r.file}
        if r.form in WITH_HEADER:
            row["accession"], row["accepted_et"], row["items"] = header(r.cik, r.file, get)
        rows.append(row)
    return rows


def run(state_dir, now=None, first_day=None, index=sec.daily_form_index, get=sec.get):
    root = Path(state_dir) / "leads" / "sec"
    root.mkdir(parents=True, exist_ok=True)
    cursor = root / "cursor.json"
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    yesterday = (now.tz_convert(ET) - pd.Timedelta(days=1)).normalize().tz_localize(None)
    if cursor.exists():
        start = pd.Timestamp(json.loads(cursor.read_text())["last_day"]) + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(first_day) if first_day else yesterday - pd.Timedelta(days=FIRST_DAYS_BACK)
    fetched = f"{now:%Y-%m-%dT%H:%M:%SZ}"
    n = 0
    for d in pd.bdate_range(start, yesterday):
        rows = day_rows(d, index, get)
        if rows is None:
            # a weekday with no index: a holiday is final once a later day's index exists
            later = index(d + pd.offsets.BDay(1)) if d < yesterday else None
            if later is None:
                print(f"sec: {d:%Y-%m-%d} index not published yet; stopping")
                break
            rows = []
        path = root / f"{d:%Y-%m-%d}.jsonl"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps({"fetched_utc": fetched, **r}) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, path)
        cursor.write_text(json.dumps({"last_day": f"{d:%Y-%m-%d}", "updated_utc": fetched}) + "\n")
        print(f"sec: {d:%Y-%m-%d}: {len(rows)} filings")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--first-day")
    a = ap.parse_args()
    run(a.state, first_day=a.first_day)


if __name__ == "__main__":
    main()
