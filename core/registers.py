"""Forward loggers for public registers that can show corporate events before the press does (W1
amendments, "new raw-source loggers"; no counted idea). Each source is written once per date into the
hash-chained ledger (core/ledger.py) on the `state-registers` branch:

- form144: EDGAR's daily form index rows for Form 144 (notice of a planned insider sale) and 144/A;
- sec_litigation: SEC litigation releases from the SEC's RSS feed (title, link, date, summary);
- ftc_early_termination: the FTC's early-termination notices page (acquiring and acquired party, date);
- house_ptr: the House Clerk's index of members' periodic transaction reports (stock trades, filed up to
  45 days late) filed in the 30 days to the date: name, state and district, filing date, document id and
  PDF link. Log only: published studies since 2013 mostly find no edge in following them. The Senate's
  equivalent site requires accepting terms in a browser session, so it isn't logged.

FINRA's weekly OTC volume needs no logger: its API serves the history on request (tracks/main/w1b_volume.py).

Usage: python -m core.registers --state <dir> [--date YYYY-MM-DD]
"""
import argparse
import html
import io
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from core import sec
from core.ledger import Ledger

LIT_FEEDS = ["https://www.sec.gov/enforcement-litigation/litigation-releases/rss",
             "https://www.sec.gov/rss/litigation/litreleases.xml"]
FTC_URL = "https://www.ftc.gov/legal-library/browse/early-termination-notices"
HOUSE_ZIP = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{y}FD.zip"
HOUSE_PDF = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{y}/{doc}.pdf"
PTR_DAYS = 30


def form144_rows(index):
    """Form 144 and 144/A rows of one daily form index frame (form, company, cik, filed, file)."""
    if index is None:
        return None
    f = index[index["form"].isin(["144", "144/A"])].copy()
    f["filed"] = f["filed"].dt.strftime("%Y-%m-%d")
    return f.to_dict("records")


def parse_rss(xml_bytes):
    """(title, link, published, summary) of each RSS item."""
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()  # noqa: E731
        out.append({"title": g("title"), "link": g("link"), "published": g("pubDate"),
                    "summary": re.sub(r"<[^>]+>", " ", html.unescape(g("description"))).strip()[:2000]})
    return out


def parse_ftc(page):
    """Rows of the FTC early-termination table: the text of each cell of each body row."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        if cells and any(cells):
            rows.append(cells)
    return rows


def parse_house(xml_bytes, day, days=PTR_DAYS):
    """Periodic transaction reports (FilingType P) in the House index filed in the `days` days to `day`."""
    root = ET.fromstring(xml_bytes)
    d1 = pd.Timestamp(day)
    out = []
    for m in root.iter("Member"):
        g = lambda tag: (m.findtext(tag) or "").strip()  # noqa: E731
        if g("FilingType") != "P":
            continue
        f = pd.to_datetime(g("FilingDate"), format="%m/%d/%Y", errors="coerce")
        if pd.isna(f) or not (d1 - pd.Timedelta(days=days) < f <= d1):
            continue
        name = " ".join(x for x in (g("Prefix"), g("First"), g("Last"), g("Suffix")) if x)
        out.append({"name": name, "state_district": g("StateDst"), "filed": f"{f:%Y-%m-%d}", "doc_id": g("DocID"),
                    "pdf": HOUSE_PDF.format(y=g("Year") or f.year, doc=g("DocID"))})
    return out


def fetch_house(day, get=sec.get):
    rows = []
    years = sorted({pd.Timestamp(day).year, (pd.Timestamp(day) - pd.Timedelta(days=PTR_DAYS)).year})
    for y in years:
        r = get(HOUSE_ZIP.format(y=y))
        if r is None:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                xml = z.read(next(n for n in z.namelist() if n.lower().endswith(".xml")))
        except (zipfile.BadZipFile, StopIteration):
            return None
        rows += parse_house(xml, day)
    return rows


def fetch_litigation(get=sec.get):
    for url in LIT_FEEDS:
        r = get(url)
        if r is not None:
            try:
                return url, parse_rss(r.content)
            except ET.ParseError:
                continue
    return None, None


def fetch_ftc(get=sec.get):
    r = get(FTC_URL)
    return None if r is None else parse_ftc(r.content.decode("utf-8", "replace"))


def run(state, day, get=sec.get, index_fn=sec.daily_form_index):
    """Log each source for `day` once; a source that can't be read is logged with its status, so a gap
    shows in the ledger instead of disappearing."""
    led = Ledger(state)
    d = f"{pd.Timestamp(day):%Y-%m-%d}"
    done = []
    for back in range(5, -1, -1):     # an index published late is picked up by the next runs
        dd = f"{pd.Timestamp(day) - pd.offsets.BDay(back):%Y-%m-%d}"
        if not led.has("form144", dd):
            rows = form144_rows(index_fn(dd))
            if rows is not None:
                led.write_json("form144", dd, f"form144/{dd}.json", {"date": dd, "rows": rows})
                done.append(f"form144 {dd} {len(rows)}")
    if not led.has("sec_litigation", d):
        url, items = fetch_litigation(get)
        led.write_json("sec_litigation", d, f"sec_litigation/{d}.json",
                       {"date": d, "source": url, "status": "ok" if items is not None else "unreadable",
                        "items": items or []})
        done.append(f"sec_litigation {len(items or [])}")
    if not led.has("ftc_early_termination", d):
        rows = fetch_ftc(get)
        led.write_json("ftc_early_termination", d, f"ftc_early_termination/{d}.json",
                       {"date": d, "source": FTC_URL, "status": "ok" if rows is not None else "unreadable",
                        "rows": rows or []})
        done.append(f"ftc {len(rows or [])}")
    if not led.has("house_ptr", d):
        rows = fetch_house(day, get)
        led.write_json("house_ptr", d, f"house_ptr/{d}.json",
                       {"date": d, "source": HOUSE_ZIP.format(y=pd.Timestamp(day).year),
                        "status": "ok" if rows is not None else "unreadable", "rows": rows or []})
        done.append(f"house_ptr {len(rows or [])}")
    print("[registers]", d, ", ".join(done) or "nothing new")


def last_business_day(now=None):
    """The last weekday before today in New York (the nightly job logs the day that just ended)."""
    t = (now or pd.Timestamp.now(tz="America/New_York")).tz_convert("America/New_York").normalize().tz_localize(None)
    return t - pd.offsets.BDay(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    run(a.state, pd.Timestamp(a.date) if a.date else last_business_day())


if __name__ == "__main__":
    main()
