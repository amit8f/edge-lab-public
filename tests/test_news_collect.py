"""News collector on fake API pages (no network)."""
import json

import pandas as pd

from tracks.news.collect import run


def art(i, ts):
    return {"id": i, "created_at": ts, "updated_at": ts, "headline": f"h{i}", "summary": "", "symbols": ["AAA"],
            "author": "a", "source": "benzinga", "url": "u"}


def fake(pages):
    calls = []

    def fetch(start, page_token=None):
        calls.append((start, page_token))
        return pages[len(calls) - 1]
    return fetch, calls


def test_pages_dedupe_and_cursor(tmp_path):
    now = pd.Timestamp("2026-10-12T15:00:00Z")
    fetch, calls = fake([{"news": [art(1, "2026-10-12T13:00:00Z")], "next_page_token": "p2"},
                         {"news": [art(2, "2026-10-12T14:00:00Z")], "next_page_token": None}])
    assert run(tmp_path, fetch, now) == 2
    assert calls[0][0] == "2026-10-12T12:00:00Z" and calls[1][1] == "p2"
    cursor = json.loads((tmp_path / "news" / "cursor.json").read_text())
    assert cursor["last_created_at"] == "2026-10-12T14:00:00Z"

    # next run starts 10 minutes before the cursor; the repeated article is not written twice
    fetch, calls = fake([{"news": [art(2, "2026-10-12T14:00:00Z"), art(3, "2026-10-13T00:05:00Z")]}])
    assert run(tmp_path, fetch, now + pd.Timedelta(hours=10)) == 1
    assert calls[0][0] == "2026-10-12T13:50:00Z"
    day1 = (tmp_path / "news" / "raw" / "2026-10-12.jsonl").read_text().splitlines()
    day2 = (tmp_path / "news" / "raw" / "2026-10-13.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in day1] == [1, 2] and len(day2) == 1
