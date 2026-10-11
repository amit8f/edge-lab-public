"""News track, step 1: log every Benzinga headline from Alpaca's news API, forward only.

Each run fetches articles created since the last run's cursor (first run: the last 3 hours) and appends
them to one JSONL file per UTC day under <state>/news/raw/, de-duplicated by article id. It logs only;
no labels, scores or trades. Run by .github/workflows/news.yml every 30 minutes, with the state directory
checked out from the `state` branch.

Usage: python -m tracks.news.collect --state <dir>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from core import alpaca

KEEP = ["id", "created_at", "updated_at", "headline", "summary", "symbols", "author", "source", "url"]
FIRST_LOOKBACK = pd.Timedelta(hours=3)
OVERLAP = pd.Timedelta(minutes=10)   # re-read the cursor's last minutes; duplicates are dropped by id


def _day_file(root, created_at):
    return root / f"{created_at[:10]}.jsonl"


def _known_ids(path):
    if not path.exists():
        return set()
    return {json.loads(ln)["id"] for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def store(articles, root, fetched_utc):
    """Append new articles to their UTC-day files; returns the number written."""
    root.mkdir(parents=True, exist_ok=True)
    known, written = {}, 0
    for a in articles:
        path = _day_file(root, a["created_at"])
        if path not in known:
            known[path] = _known_ids(path)
        if a["id"] in known[path]:
            continue
        row = {k: a.get(k) for k in KEEP}
        row["fetched_utc"] = fetched_utc
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        known[path].add(a["id"])
        written += 1
    return written


def run(state_dir, fetch=alpaca.news, now=None):
    now = now if now is not None else alpaca.utcnow()
    root = Path(state_dir) / "news"
    cursor_path = root / "cursor.json"
    if cursor_path.exists():
        start = pd.Timestamp(json.loads(cursor_path.read_text())["last_created_at"]) - OVERLAP
    else:
        start = now - FIRST_LOOKBACK
    articles, token = [], None
    while True:
        page = fetch(start.strftime("%Y-%m-%dT%H:%M:%SZ"), page_token=token)
        articles.extend(page.get("news") or [])
        token = page.get("next_page_token")
        if not token:
            break
    fetched = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    n = store(articles, root / "raw", fetched)
    if articles:
        last = max(a["created_at"] for a in articles)
        prev = json.loads(cursor_path.read_text())["last_created_at"] if cursor_path.exists() else None
        if prev is None or last > prev:
            root.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(json.dumps({"last_created_at": last, "updated_utc": fetched}) + "\n")
    print(f"news: {len(articles)} fetched since {start:%Y-%m-%d %H:%M}Z, {n} new")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
