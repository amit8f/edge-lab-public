"""One-off check of outside data sources from GitHub's servers (the Claude sandbox can't reach them):
prints the HTTP status and the first lines of one sample of each, never any secret. Used to confirm
file formats before B0's code and the borrow logger depend on them.

Usage: python -m core.probe
"""
import ftplib
import io
import zipfile

import requests

UA = {"User-Agent": "edge-lab research probe"}


def show(name, fn):
    try:
        print(f"== {name}\n{fn()}\n")
    except Exception as e:  # noqa: BLE001 - a probe reports every failure and goes on
        print(f"== {name}\nFAILED: {type(e).__name__}: {str(e)[:300]}\n")


def http(url, n=600, headers=None, method="get", **kw):
    r = getattr(requests, method)(url, headers=headers or UA, timeout=60, **kw)
    return f"HTTP {r.status_code} {r.headers.get('content-type')}\n{r.text[:n]}"


def ibkr():
    """Each IBKR host in turn: a bare TCP connect to port 21 first (tells a blocked port from a slow
    data channel), then login, listing and the file itself."""
    import socket
    out = []
    for host in ("ftp2.interactivebrokers.com", "ftp3.interactivebrokers.com", "ftp.interactivebrokers.com"):
        try:
            socket.create_connection((host, 21), timeout=20).close()
            buf = io.BytesIO()
            with ftplib.FTP(host, timeout=60) as f:
                f.login("shortstock", "")
                names = f.nlst()
                f.retrbinary("RETR usa.txt", buf.write)
            lines = buf.getvalue().decode("latin-1").splitlines()
            out.append(f"{host}: OK files {names[:6]} lines {len(lines)}\n" + "\n".join(lines[:3]))
        except Exception as e:  # noqa: BLE001
            out.append(f"{host}: FAILED {type(e).__name__}: {str(e)[:200]}")
    return "\n".join(out)


def ftd():
    import os
    r = requests.get("https://www.sec.gov/files/data/fails-deliver-data/cnsfails202401a.zip",
                     headers={"User-Agent": os.environ.get("SEC_USER_AGENT", "edge-lab probe").strip()}, timeout=60)
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return f"{z.namelist()}\n" + z.read(z.namelist()[0]).decode("latin-1")[:400]


def ptr_pdf():
    """A House PTR PDF read with pypdf, as SYS2B lane 20 reads it."""
    from pypdf import PdfReader
    r = requests.get("https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20035648.pdf", headers=UA, timeout=60)
    text = "\n".join(pg.extract_text() or "" for pg in PdfReader(io.BytesIO(r.content)).pages)
    return f"HTTP {r.status_code}\n{text[:900]}"


def sys2b_lanes():
    """SYS2B lanes 15/16 (SIC lookups), 18, 19 and 20 through the scouts' own Api on live sources. Prints
    counts and symbols only, never headline text."""
    from collections import Counter
    import pandas as pd
    from adaptive import lanes as L
    from adaptive import scouts as S
    api, out = S.Api(), []
    by_cik, names = api.tickers()
    syms = set(names)
    idx = L.name_index(names, syms)
    out.append(f"tickers {len(names)}, name index {len(idx)}")
    hits = Counter()
    for outlet, _, url in S.OUTLETS:
        try:
            items = api.feed(url)
        except Exception as e:  # noqa: BLE001
            out.append(f"feed {outlet}: FAILED {type(e).__name__}")
            continue
        tagged = [s for it in items for s in L.name_hits(it["title"], idx, syms)]
        dated = sum(1 for it in items if it["published"])
        out.append(f"feed {outlet}: {len(items)} items, {dated} dated, tags {sorted(set(tagged))[:8]}")
        hits[outlet] = len(tagged)
    def part(name, fn):
        try:
            out.append(fn())
        except Exception as e:  # noqa: BLE001 - every part reports and the next one runs
            out.append(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")

    now = pd.Timestamp.now(tz="UTC")

    def buzz():
        ape, trend = api.apewisdom(), api.stocktwits_trending()
        cands = L.buzz_candidates(ape, trend, syms)
        got = [a for s in sorted(cands)[:10] if (a := L.buzz_alert(s, cands[s], api.stocktwits_stream(s), now))]
        return (f"apewisdom {len(ape)}, trending {trend[:10]}, candidates {sorted(cands)[:15]}, "
                f"buzz {[(a['symbol'], a['info']['users']) for a in got]}")

    def house():
        reps = api.house_reports(now.tz_convert("America/New_York").normalize().tz_localize(None)) or []
        lines = [f"house reports {len(reps)}"]
        for r in reps[:6]:
            tr = L.ptr_trades(api.ptr_text(r["pdf"]))
            lines.append(f"  {r['doc_id']} filed {r['filed']}: {[(x['symbol'], x['type']) for x in tr][:10]}")
        return "\n".join(lines)

    def sectors():
        cache = {"tickers": (by_cik, names)}
        sec = api.sectors(sorted(syms)[:300], cache, limit=300)
        return (f"sectors: {len(sec)} of 300 mapped, {Counter(sec.values()).most_common()}, "
                f"SIC known {sum(1 for v in cache['sic'].values() if v)}")

    part("buzz", buzz)
    part("house", house)
    part("sectors", sectors)
    return "\n".join(out)


def listener_sources():
    """Every source of the news listener (core/listener.py) once."""
    import contextlib
    from core import listener
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        listener.probe()
    return buf.getvalue()


def main():
    show("News listener sources", listener_sources)
    show("SYS2B lanes 15-20 on live sources", sys2b_lanes)
    show("House PTR PDF text (SYS2B lane 20)", ptr_pdf)
    show("StockTwits trending (SYS2B lane 18)", lambda: http("https://api.stocktwits.com/api/2/trending/symbols.json", 400))
    show("StockTwits symbol stream (SYS2B lane 18)", lambda: http("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json", 400))
    show("Reddit wallstreetbets new (SYS2B lane 18)", lambda: http("https://www.reddit.com/r/wallstreetbets/new.json?limit=5", 300))
    show("Reddit old domain (SYS2B lane 18)", lambda: http("https://old.reddit.com/r/stocks/new.json?limit=5", 300))
    show("Reddit RSS (SYS2B lane 18)", lambda: http("https://www.reddit.com/r/wallstreetbets/new/.rss", 300))
    show("ApeWisdom mentions (SYS2B lane 18)", lambda: http("https://apewisdom.io/api/v1.0/filter/all-stocks/page/1", 400))
    show("Betaville feed (SYS2B lane 19)", lambda: http("https://www.betaville.co.uk/feed/", 300))
    show("IBKR stock-loan file", ibkr)
    show("FINRA Reg SHO daily short volume", lambda: http("https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240102.txt", 400))
    show("FINRA consolidated short interest API (no key)", lambda: http(
        "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest", 600, method="post",
        headers={**UA, "Accept": "application/json"},
        json={"limit": 3, "compareFilters": [{"compareType": "equal", "fieldName": "symbolCode", "fieldValue": "AAPL"}]}))
    show("FINRA short interest files page", lambda: http("https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files", 300))
    show("SEC fails-to-deliver 2024-01 first half", ftd)


if __name__ == "__main__":
    main()
