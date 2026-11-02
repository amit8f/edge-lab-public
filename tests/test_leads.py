"""Leading-source loggers on fake responses (no network)."""
import json

import pandas as pd

from core import leads

RSS = """<rss><channel><title>Contracts</title><link>https://www.war.gov/News/Contracts/</link>
<item><title>Contracts for Oct. 14, 2026</title>
<link>https://www.war.gov/News/Contracts/Contract/Article/2/contracts-for-oct-14-2026/</link>
<pubDate>Wed, 14 Oct 2026 21:04:17 GMT</pubDate></item>
<item><title>Contracts for Oct. 13, 2026</title>
<link>https://www.war.gov/News/Contracts/Contract/Article/1/contracts-for-oct-13-2026/</link>
<pubDate>Tue, 13 Oct 2026 21:00:04 GMT</pubDate></item></channel></rss>"""


def test_dod_saves_new_pages_and_retries_blocked(tmp_path):
    def get(url, params=None):
        if url == leads.DOD_RSS:
            return 200, RSS
        return (403, "Access Denied") if "/1/" in url else (200, "<html>contracts</html>")
    assert leads.run_dod(tmp_path, get) == (1, 1)
    assert (tmp_path / "leads/dod/2026-10-14.html").read_text() == "<html>contracts</html>"
    assert not (tmp_path / "leads/dod/2026-10-13.html").exists()
    assert leads.run_dod(tmp_path, get) == (0, 1)       # saved day skipped, blocked day retried


def test_trials_fills_days_to_yesterday_and_pages(tmp_path):
    calls = []

    def get(url, params):
        calls.append(params)
        day = params["filter.advanced"].split("RANGE[")[1][:10]
        if "pageToken" not in params:
            return 200, json.dumps({"studies": [{"protocolSection": {"id": day + "a"}}], "nextPageToken": "t"})
        return 200, json.dumps({"studies": [{"protocolSection": {"id": day + "b"}}]})
    now = pd.Timestamp("2026-10-15T03:30:00Z")           # 23:30 ET on Oct 14 -> yesterday is Oct 13
    assert leads.run_trials(tmp_path, get, now, first_day="2026-10-12") == 2
    rows = (tmp_path / "leads/trials/2026-10-13.jsonl").read_text().splitlines()
    assert [json.loads(r)["id"] for r in rows] == ["2026-10-13a", "2026-10-13b"]
    assert "INDUSTRY" in calls[0]["filter.advanced"]
    assert leads.run_trials(tmp_path, get, now) == 0      # cursor is at yesterday
    assert leads.run_trials(tmp_path, get, now + pd.Timedelta(days=1)) == 1
