import json
from datetime import datetime, timedelta, timezone

from core import listener as L
from core.ledger import Ledger

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Acme Corp (NASDAQ: ACME) Raises  Guidance</title><link>https://wire.example/a</link>
<pubDate>Fri, 23 Oct 2026 12:00:05 GMT</pubDate><description>&lt;p&gt;Acme (NYSE American: ACM.B) said&lt;/p&gt;</description></item>
<item><title>Beta $BETA wins contract</title><link>https://wire.example/b</link></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>8-K - Acme Corp (0000000123) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/123/000000012326000001-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-10-23 &lt;b&gt;AccNo:&lt;/b&gt; 0000000123-26-000001 &lt;b&gt;Size:&lt;/b&gt; 1 KB
&lt;br&gt;Item 2.02: Results</summary><updated>2026-10-23T08:00:01-04:00</updated></entry>
</feed>"""


class Resp:
    def __init__(self, text):
        self.content = text.encode("latin-1")


def clock_at(t0):
    state = {"t": t0}

    def clock():
        return state["t"]
    return clock, state


def test_norm_title_and_tickers():
    assert L.norm_title("Acme  Corp. (NASDAQ: ACME) – Raises Guidance!") == "acme corp nasdaq acme raises guidance"
    assert L.title_sha("ACME corp") == L.title_sha("acme, corp.")
    assert L.tickers("Acme (NASDAQ: ACME) and (NYSE American: ACM.B)", "see $BETA, not $5") == \
        ["ACME", "ACM.B", "BETA"]


def test_parse_feed_rss():
    items = L.parse_feed(RSS)
    assert [x["link"] for x in items] == ["https://wire.example/a", "https://wire.example/b"]
    assert items[0]["title"] == "Acme Corp (NASDAQ: ACME) Raises Guidance"
    assert items[0]["published"] == "2026-10-23T12:00:05+00:00"
    assert items[1]["published"] is None


def test_session_window():
    et = lambda h, d=23: datetime(2026, 10, d, h, 0, tzinfo=timezone.utc) + timedelta(hours=4)  # EDT = UTC-4
    assert L.session(et(3)) == "before"
    assert L.session(et(4)) == "open"
    assert L.session(et(19)) == "open"
    assert L.session(et(20)) == "over"
    assert L.session(et(10, d=24)) == "over"          # Saturday


def test_cycle_logs_new_items_once_without_titles(tmp_path):
    clock, st = clock_at(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc))
    log = L.DayLog(tmp_path, "2026-10-23")
    kw = dict(get=lambda url: RSS, get_sec=lambda url: Resp(ATOM), clock=clock, outlets=[])
    first = L.cycle(log, 0, {123: ["ACME"]}, **kw)
    st["t"] += timedelta(seconds=20)
    second = L.cycle(log, 1, {123: ["ACME"]}, **kw)
    assert first == 1 + 2 * len(L.WIRES) and second == 0
    lines = [json.loads(x) for x in log.path.read_text().splitlines()]
    items = [x for x in lines if x["kind"] == "item"]
    edgar = next(x for x in items if x["source"] == "EDGAR")
    assert edgar["tickers"] == ["ACME"] and edgar["form"] == "8-K" and edgar["items"] == ["2.02"]
    assert edgar["published_utc"] == "2026-10-23T12:00:01+00:00" and edgar["window_s"] is None
    assert "Guidance" not in log.path.read_text()        # no titles stored, only their hash
    wire = next(x for x in items if x["url"] == "https://wire.example/a")
    assert wire["title_sha"] == L.title_sha("Acme Corp (NASDAQ: ACME) Raises Guidance")
    assert wire["tickers"] == ["ACME", "ACM.B"]
    assert [x["new"] for x in lines if x["kind"] == "poll"] == [first, 0]
    assert L.verify_day(log.path) == []


def test_restart_keeps_keys_windows_and_chain(tmp_path):
    clock, st = clock_at(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc))
    kw = dict(get=lambda url: RSS, get_sec=lambda url: Resp(ATOM), clock=clock, outlets=[])
    L.cycle(L.DayLog(tmp_path, "2026-10-23"), 0, {}, **kw)
    st["t"] += timedelta(seconds=45)
    rss2 = RSS.replace(b"https://wire.example/b", b"https://wire.example/c")
    log = L.DayLog(tmp_path, "2026-10-23")              # a new job picks up the day
    assert L.cycle(log, 1, {}, **dict(kw, get=lambda url: rss2)) == len(L.WIRES)
    new = [json.loads(x) for x in log.path.read_text().splitlines()][-2]
    assert new["url"] == "https://wire.example/c" and new["window_s"] == 45.0
    assert L.verify_day(log.path) == []


def test_tampering_breaks_the_chain(tmp_path):
    clock, _ = clock_at(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc))
    log = L.DayLog(tmp_path, "2026-10-23")
    L.cycle(log, 0, {}, get=lambda url: RSS, get_sec=lambda url: Resp(ATOM), clock=clock, outlets=[])
    lines = log.path.read_text().splitlines()
    lines[1] = lines[1].replace("wire.example", "wire.exampel")
    log.path.write_text("\n".join(lines) + "\n")
    assert L.verify_day(log.path) == [3]


def test_failed_source_is_logged_and_others_go_on(tmp_path):
    clock, _ = clock_at(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc))
    log = L.DayLog(tmp_path, "2026-10-23")

    def get(url):
        if "prnewswire" in url:
            raise ConnectionError("down")
        return RSS
    L.cycle(log, 0, {}, get=get, get_sec=lambda url: Resp(ATOM), clock=clock, outlets=[])
    poll = json.loads(log.path.read_text().splitlines()[-1])
    assert list(poll["failed"]) == ["PR Newswire"] and "EDGAR" in poll["ok"] and len(poll["ok"]) == len(L.WIRES)


def test_outlets_every_third_cycle(tmp_path):
    clock, _ = clock_at(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc))
    log, urls = L.DayLog(tmp_path, "2026-10-23"), []

    def get(url):
        urls.append(url)
        return RSS
    outlets = [("Outlet", "small", "https://outlet.example/rss")]
    for n in range(4):
        L.cycle(log, n, {}, get=get, get_sec=lambda url: Resp(ATOM), clock=clock, outlets=outlets)
    assert urls.count("https://outlet.example/rss") == 2      # cycles 0 and 3


def test_run_seals_the_day_when_session_ends(tmp_path):
    t = {"t": datetime(2026, 10, 23, 23, 59, 30, tzinfo=timezone.utc)}   # 7:59:30pm EDT

    def clock():
        return t["t"]

    def sleep(s):
        t["t"] += timedelta(seconds=20)
    rc = L.run(tmp_path, 10, clock=clock, sleep=sleep, by_cik={}, get=lambda url: RSS,
               get_sec=lambda url: Resp(ATOM), outlets=[])
    assert rc == 3
    led = Ledger(tmp_path)
    e = led.entry("listen", "2026-10-23")
    assert e and (tmp_path / e["file"]).read_bytes() == (tmp_path / "listen/live/2026-10-23.jsonl").read_bytes()
    assert led.verify() == []
    assert L.run(tmp_path, 10, clock=clock, sleep=sleep, by_cik={}) == 3    # sealing twice is a no-op


def test_run_waits_before_four(tmp_path):
    t = {"t": datetime(2026, 10, 23, 7, 59, 30, tzinfo=timezone.utc)}    # 3:59:30am EDT
    calls = []

    def clock():
        return t["t"]

    def sleep(s):
        t["t"] += timedelta(seconds=20)
    rc = L.run(tmp_path, 0.01, clock=clock, sleep=sleep, by_cik={}, get=lambda url: calls.append(clock()) or RSS,
               get_sec=lambda url: Resp(ATOM), outlets=[])
    assert rc == 0 and calls and min(calls) >= datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)
