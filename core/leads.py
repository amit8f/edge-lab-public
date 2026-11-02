"""Leading-source loggers, forward only: Pentagon contract announcements and clinicaltrials.gov updates.

Both are public sources that often move before a company's own press release. They are logged raw so a
later pre-registered idea can test them; nothing here scores or trades.

- dod:    the daily "Contracts for <date>" page (posted ~5pm ET; contracts of $7.5M or more), saved as raw
          HTML to <state>/leads/dod/<date>.html. Days are taken from the RSS feed; a page that fails to
          download is retried on the next run. Opt-in (--dod): war.gov's CDN refuses cloud IPs (GitHub
          Actions got 403 on every page on 2026-10-15), so it only works from a home connection.
- trials: every industry-sponsored study whose last-update post date is d, one JSONL file per day
          (<state>/leads/trials/<d>.jsonl). Each run fills the days from the cursor to yesterday (ET).
          A study updated again before its day is fetched shows only its newer date, so a day fetched
          late can miss it; `fetched_utc` records when each day was read.

Usage: python -m core.leads --state <dir> [--first-day YYYY-MM-DD] [--dod]
"""
import argparse
import email.utils
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ET = "America/New_York"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
DOD_RSS = "https://www.war.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400&Site=945&max=20"
CTG_URL = "https://clinicaltrials.gov/api/v2/studies"
CTG_FIELDS = ("NCTId,BriefTitle,Condition,StudyType,Phase,LeadSponsorName,LeadSponsorClass,OverallStatus,"
              "WhyStopped,EnrollmentCount,EnrollmentType,StartDate,PrimaryCompletionDate,CompletionDate,"
              "LastUpdateSubmitDate,LastUpdatePostDate,ResultsFirstPostDate")
FIRST_DAYS_BACK = 7   # trials: the first run starts this many days before yesterday


def http_get(url, params=None):
    """GET with backoff on network errors, 429 and 5xx. Returns (status, text)."""
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
                             timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        return r.status_code, r.text
    return None, ""


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def parse_rss(xml):
    """[(et_date 'YYYY-MM-DD', link)] for the contract items of the RSS feed."""
    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
        link = re.search(r"<link>\s*(.*?)\s*</link>", item, re.S)
        pub = re.search(r"<pubDate>\s*(.*?)\s*</pubDate>", item, re.S)
        if not (link and pub) or "/Contract/" not in link.group(1):
            continue
        t = pd.Timestamp(email.utils.parsedate_to_datetime(pub.group(1))).tz_convert(ET)
        out.append((f"{t:%Y-%m-%d}", link.group(1)))
    return out


def run_dod(state_dir, get=http_get, now=None):
    """Save every contracts page in the feed that is not saved yet. Returns (saved, failed)."""
    root = Path(state_dir) / "leads" / "dod"
    status, xml = get(DOD_RSS)
    if status != 200:
        print(f"::warning::dod: RSS feed returned {status}")
        return 0, 1
    saved = failed = 0
    for day, link in parse_rss(xml):
        path = root / f"{day}.html"
        if path.exists():
            continue
        status, page = get(link)
        if status != 200 or "Access Denied" in page[:2000]:
            print(f"::warning::dod: {day} page returned {status}; retried next run")
            failed += 1
            continue
        _write_atomic(path, page)
        with open(root / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"day": day, "url": link, "fetched_utc": f"{_now(now):%Y-%m-%dT%H:%M:%SZ}"}) + "\n")
        saved += 1
    print(f"dod: {saved} pages saved, {failed} failed")
    return saved, failed


def trials_day(day, get=http_get):
    """All industry-sponsored studies with last-update post date == day (list of protocolSection dicts)."""
    q = f"AREA[LastUpdatePostDate]RANGE[{day},{day}] AND AREA[LeadSponsorClass]INDUSTRY"
    params = {"filter.advanced": q, "fields": CTG_FIELDS, "pageSize": 1000, "format": "json"}
    studies = []
    while True:
        status, text = get(CTG_URL, params)
        if status != 200:
            raise RuntimeError(f"clinicaltrials.gov returned {status} for {day}")
        page = json.loads(text)
        studies += [s["protocolSection"] for s in page.get("studies", [])]
        if not page.get("nextPageToken"):
            return studies
        params = {**params, "pageToken": page["nextPageToken"]}


def run_trials(state_dir, get=http_get, now=None, first_day=None):
    """Fill complete days from the cursor to yesterday (ET). Returns the number of days written."""
    root = Path(state_dir) / "leads" / "trials"
    cursor = root / "cursor.json"
    yesterday = (_now(now).tz_convert(ET) - pd.Timedelta(days=1)).normalize().tz_localize(None)
    if cursor.exists():
        start = pd.Timestamp(json.loads(cursor.read_text())["last_day"]) + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(first_day) if first_day else yesterday - pd.Timedelta(days=FIRST_DAYS_BACK)
    fetched = f"{_now(now):%Y-%m-%dT%H:%M:%SZ}"
    days = 0
    for d in pd.date_range(start, yesterday):
        day = f"{d:%Y-%m-%d}"
        rows = trials_day(day, get)
        lines = "".join(json.dumps({"fetched_utc": fetched, **s}, sort_keys=True) + "\n" for s in rows)
        _write_atomic(root / f"{day}.jsonl", lines)
        _write_atomic(cursor, json.dumps({"last_day": day, "updated_utc": fetched}) + "\n")
        print(f"trials: {day}: {len(rows)} industry studies")
        days += 1
    return days


def _now(now):
    return now if now is not None else pd.Timestamp.now(tz="UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--first-day")
    ap.add_argument("--dod", action="store_true")
    a = ap.parse_args()
    if a.dod:
        run_dod(a.state)
    run_trials(a.state, first_day=a.first_day)


if __name__ == "__main__":
    main()
