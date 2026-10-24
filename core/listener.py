"""News listener (failure review 2026-10-24, decisions 8.3 and 9): logging only, no trades.

Every ~20 seconds from 4am to 8pm ET on weekdays it reads SEC's live filing feed and the press-release
wires, and every minute the outlet feeds of SYS2B lane 19, and writes one line per new item: when we first
saw it, the feed's own published time, the URL, the sha256 of the normalised title (never the title) and
the tickers found in the title and summary. After two weeks the speed report compares our first-seen time
with Benzinga's created_at (matched by title hash, or by ticker and time for filings) and with the price
move. Nothing here is a counted idea until a trade rule is registered on top of it.

Each line carries `window_s`, the seconds since that source's previous successful read, so the report
knows how early the item could have been seen; `null` marks the day's first read (a backlog of older
items, not discoveries). A `poll` line per cycle logs which sources answered.

The day's log (state/listen/live/<day>.jsonl) is append-only and hash-chained line by line (each line holds
the sha256 of the line before). When the session ends it is sealed into the ledger as
listen/days/<day>.jsonl (core/ledger.py).

Usage: python -m core.listener --state state [--minutes 5]   (exit 3 = the day's session is over)
       python -m core.listener --probe                         (read every source once, print counts)
"""
import argparse
import email.utils
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import edgar_live
from core.ledger import GENESIS, Ledger

ET_TZ = "America/New_York"
START_H, END_H = 4, 20        # 4am to 8pm ET, weekdays
POLL_S = 20                   # SEC feed and wires
OUTLET_EVERY = 3              # outlets every 3rd cycle (one minute)
EDGAR_PAGES = 3               # 100 filings a page; paged until a page holds one already seen
UA = {"User-Agent": "edge-lab research (RSS reader)"}

# Press-release wires. Which answer from GitHub is checked with `--probe` (probe-sources.yml); a dead one
# is logged as failed every cycle and never stops the others.
WIRES = [
    ("PR Newswire", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    ("GlobeNewswire", "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/"
                      "GlobeNewswire%20-%20News%20about%20Public%20Companies"),
    ("Business Wire", "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpRWQ=="),
    ("ACCESS Newswire", "https://www.accessnewswire.com/newsroom/api/rss"),
    ("Newsfile", "https://feeds.newsfilecorp.com/feed/all"),
]

EXCH = r"(?:NYSE American|NYSE Arca|NYSE MKT|NYSE|NASDAQ|Nasdaq(?: ?[A-Z]{2})?|Cboe(?: BZX)?|OTC ?[A-Za-z]*|OTCQ[XB])"
TICK = re.compile(rf"\b{EXCH}\s*:\s*([A-Z][A-Z0-9]{{0,4}}(?:\.[A-Z])?)\b")
CASHTAG = re.compile(r"(?<![\w$])\$([A-Z]{1,5})\b")


def norm_title(t):
    """Lowercase ASCII letters and digits, single spaces: the form both our log and Benzinga's are hashed in."""
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def title_sha(t):
    return hashlib.sha256(norm_title(t).encode()).hexdigest()


def tickers(*texts):
    """Exchange-tagged tickers ("(NASDAQ: ABCD)") and cashtags, in order of first appearance."""
    out = []
    for text in texts:
        for m in list(TICK.finditer(text or "")) + list(CASHTAG.finditer(text or "")):
            if m.group(1) not in out:
                out.append(m.group(1))
    return out


def _utc(pub):
    if not pub:
        return None
    for parse in (lambda x: pd.Timestamp(email.utils.parsedate_to_datetime(x)), pd.Timestamp):
        try:
            t = parse(pub.strip())
            return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()
        except (TypeError, ValueError):
            continue
    return None


def parse_feed(content):
    """[{title, link, published (UTC ISO or None), summary}] of an RSS or Atom document."""
    root = ET.fromstring(content)
    atom, dc, out = "{http://www.w3.org/2005/Atom}", "{http://purl.org/dc/elements/1.1/}", []
    for it in root.iter("item"):
        out.append((it.findtext("title"), it.findtext("link"), it.findtext("pubDate") or it.findtext(f"{dc}date"),
                    it.findtext("description")))
    for it in root.iter(f"{atom}entry"):
        ln = it.find(f"{atom}link")
        out.append((it.findtext(f"{atom}title"), ln.get("href") if ln is not None else None,
                    it.findtext(f"{atom}published") or it.findtext(f"{atom}updated"),
                    it.findtext(f"{atom}summary") or it.findtext(f"{atom}content")))
    return [{"title": " ".join(t.split()), "link": ln.strip(), "published": _utc(p),
             "summary": re.sub(r"<[^>]+>", " ", s or "")} for t, ln, p, s in out if t and ln]


def http_get(url):
    import requests
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    return r.content


def sec_get(url):
    from core import sec
    return sec.get(url, tries=2)


def now_utc():
    return datetime.now(timezone.utc)


def session(t):
    """'before', 'open' or 'over' for a UTC datetime against 4am-8pm ET on weekdays."""
    et = pd.Timestamp(t).tz_convert(ET_TZ)
    if et.weekday() >= 5 or et.hour >= END_H:
        return "over"
    return "before" if et.hour < START_H else "open"


class DayLog:
    """state/listen/live/<day>.jsonl: append-only, each line holding the sha256 of the line before."""

    def __init__(self, root, day):
        self.path = Path(root) / "listen" / "live" / f"{day}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev, self.keys, self.last_read = GENESIS, set(), {}
        if self.path.exists():
            for ln in self.path.read_text().splitlines():
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                self.prev = hashlib.sha256(ln.encode()).hexdigest()
                if rec.get("kind") == "item":
                    self.keys.add(rec["key"])
                elif rec.get("kind") == "poll":
                    for s in rec["ok"]:
                        self.last_read[s] = rec["t"]

    def add(self, rec):
        line = json.dumps(dict(rec, prev=self.prev), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.prev = hashlib.sha256(line.encode()).hexdigest()


def verify_day(path):
    """Line numbers (1-based) whose prev hash doesn't match the line before; empty = the chain holds."""
    bad, prev = [], GENESIS
    for i, ln in enumerate(Path(path).read_text().splitlines(), 1):
        if json.loads(ln).get("prev") != prev:
            bad.append(i)
        prev = hashlib.sha256(ln.encode()).hexdigest()
    return bad


def seal(root, day):
    """Register the finished day's log in the ledger (once). Returns the ledger entry or None."""
    p = Path(root) / "listen" / "live" / f"{day}.jsonl"
    led = Ledger(root)
    if not p.exists() or led.has("listen", day):
        return None
    return led.write_bytes("listen", day, f"listen/days/{day}.jsonl", p.read_bytes())


def seal_old(root, today):
    for p in sorted((Path(root) / "listen" / "live").glob("*.jsonl")):
        if p.stem < today:
            seal(root, p.stem)


def ticker_map(root, day, get=sec_get):
    """{cik: [tickers]} from SEC's current map, cached for the day in state/listen/tickers.json."""
    p = Path(root) / "listen" / "tickers.json"
    if p.exists():
        j = json.loads(p.read_text())
        if j.get("day") == day:
            return {int(k): v for k, v in j["by_cik"].items()}
    by_cik, _ = edgar_live.ticker_map(get=get)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"day": day, "by_cik": by_cik}))
    tmp.replace(p)
    return by_cik


def edgar_items(log, get, by_cik, t, window):
    """Item lines for filings not logged yet (one per accession and CIK)."""
    new = []
    for page in range(EDGAR_PAGES):
        r = get(edgar_live.CURRENT.format(form="", start=page * 100))
        rows = edgar_live.parse_atom(r.content.decode("latin-1")) if r is not None else []
        fresh = 0
        for x in rows:
            key = f"EDGAR|{x['acc']}|{x['cik']}"
            if key in log.keys:
                continue
            fresh += 1
            log.keys.add(key)
            new.append({"kind": "item", "source": "EDGAR", "key": key, "first_seen_utc": t, "window_s": window,
                        "published_utc": x["accepted"], "url": x["url"], "title_sha": None,
                        "tickers": by_cik.get(x["cik"], []), "cik": x["cik"], "role": x["role"],
                        "form": x["form"], "items": x["items"], "acc": x["acc"]})
        if fresh < len(rows) or len(rows) < 100:
            break
    return new


def feed_items(log, source, content, t, window):
    new = []
    for x in parse_feed(content):
        key = f"{source}|{x['link']}"
        if key in log.keys:
            continue
        log.keys.add(key)
        new.append({"kind": "item", "source": source, "key": key, "first_seen_utc": t, "window_s": window,
                    "published_utc": x["published"], "url": x["link"], "title_sha": title_sha(x["title"]),
                    "tickers": tickers(x["title"], x["summary"])})
    return new


def _window(log, source, t):
    last = log.last_read.get(source)
    return None if last is None else round((pd.Timestamp(t) - pd.Timestamp(last)).total_seconds(), 1)


def cycle(log, n, by_cik, get=http_get, get_sec=sec_get, clock=now_utc, outlets=None):
    """One read of every due source. Writes the new items, then one poll line. Returns the new-item count."""
    from adaptive.scouts import OUTLETS
    outlets = OUTLETS if outlets is None else outlets
    sources = [("EDGAR", None)] + WIRES + ([(o, u) for o, _, u in outlets] if n % OUTLET_EVERY == 0 else [])
    ok, failed, total = [], {}, 0
    for source, url in sources:
        t = clock().isoformat()
        try:
            if source == "EDGAR":
                new = edgar_items(log, get_sec, by_cik, t, _window(log, source, t))
            else:
                new = feed_items(log, source, get(url), t, _window(log, source, t))
        except Exception as e:  # noqa: BLE001 - one dead source never stops the others
            failed[source] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        for rec in new:
            log.add(rec)
        log.last_read[source] = t
        ok.append(source)
        total += len(new)
    log.add({"kind": "poll", "t": clock().isoformat(), "ok": ok, "failed": failed, "new": total})
    return total


def run(state, minutes, clock=now_utc, sleep=time.sleep, **kw):
    """Poll every POLL_S seconds for `minutes`, inside the session. 3 = the session is over (day sealed)."""
    stop = time.monotonic() + minutes * 60
    day = pd.Timestamp(clock()).tz_convert(ET_TZ).strftime("%Y-%m-%d")
    seal_old(state, day)
    if session(clock()) == "over":
        seal(state, day)
        return 3
    by_cik = kw.pop("by_cik", None)
    if by_cik is None:
        try:
            by_cik = ticker_map(state, day)
        except Exception as e:  # noqa: BLE001 - filings are still logged, with CIKs only
            print(f"ticker map failed: {type(e).__name__}", file=sys.stderr)
            by_cik = {}
    log, n = DayLog(state, day), 0
    while time.monotonic() < stop:
        s = session(clock())
        if s == "over":
            seal(state, day)
            return 3
        start = time.monotonic()
        if s == "open":
            cycle(log, n, by_cik, clock=clock, **kw)
            n += 1
        sleep(max(0.0, POLL_S - (time.monotonic() - start)))
    return 0


def probe():
    """Read every source once from where it runs (GitHub), printing status and counts, never titles."""
    from adaptive.scouts import OUTLETS
    for source, url in WIRES + [(o, u) for o, _, u in OUTLETS]:
        try:
            items = parse_feed(http_get(url))
            pub = sum(1 for x in items if x["published"])
            tk = sum(1 for x in items if tickers(x["title"], x["summary"]))
            print(f"{source}: OK {len(items)} items, {pub} dated, {tk} with tickers, newest "
                  f"{max((x['published'] for x in items if x['published']), default=None)}")
        except Exception as e:  # noqa: BLE001
            print(f"{source}: FAILED {type(e).__name__}: {str(e)[:150]}")
    try:
        r = sec_get(edgar_live.CURRENT.format(form="", start=0))
        rows = edgar_live.parse_atom(r.content.decode("latin-1"))
        print(f"EDGAR: OK {len(rows)} entries, newest {rows[0]['accepted'] if rows else None}")
    except Exception as e:  # noqa: BLE001
        print(f"EDGAR: FAILED {type(e).__name__}: {str(e)[:150]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args(argv)
    if a.probe:
        probe()
        return 0
    return run(a.state, a.minutes)


if __name__ == "__main__":
    sys.exit(main())
