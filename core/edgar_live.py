"""EDGAR's live filing feed for the SYS2B scouts (research/prereg/SYS2B.md): new filings by form type,
Form 4 details, the main text of a filing, full-text search, and the CIK-to-ticker map.

The "current filings" Atom feed lists filings as EDGAR accepts them (newest first, up to 100 a page).
Each entry carries the form, the company, its CIK and role (Filer, Issuer, Subject, Reporting, Filed by),
the accession number, the acceptance time and, for 8-Ks, the item numbers. Requests go through core.sec
(SEC's fair-access pacing and User-Agent). Checked from GitHub Actions on 2026-10-23.
"""
import html
import json
import re
import xml.etree.ElementTree as ET

import pandas as pd

from core import sec

CURRENT = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form}&company=&dateb=&owner=include"
           "&start={start}&count=100&output=atom")
FTS = "https://efts.sec.gov/LATEST/search-index"
TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
# feed queries (a prefix match on the form type); the exact forms kept are chosen by the lanes
FEED_FORMS = ("8-K", "4", "SCHEDULE 13D", "S-1", "S-3", "F-1", "F-3", "424B", "10-Q", "10-K", "NT 10", "DEF 14A",
              "SC TO-T")
MAX_PAGES = 4
NS = {"a": "http://www.w3.org/2005/Atom"}
TITLE = re.compile(r"^(?P<form>.+?) - (?P<company>.*) \((?P<cik>\d{10})\) \((?P<role>[^)]*)\)\s*$")


def parse_atom(text):
    """Entries of one feed page: form, company, cik, role, acc, filed, accepted (UTC), items, url."""
    root = ET.fromstring(text)
    out = []
    for e in root.findall("a:entry", NS):
        m = TITLE.match((e.findtext("a:title", "", NS) or "").strip())
        if not m:
            continue
        summ = html.unescape(e.findtext("a:summary", "", NS) or "")
        acc = re.search(r"AccNo:</b>\s*([\d-]+)", summ)
        filed = re.search(r"Filed:</b>\s*(\d{4}-\d{2}-\d{2})", summ)
        link = e.find("a:link", NS)
        out.append({"form": m["form"].strip(), "company": m["company"].strip(), "cik": int(m["cik"]),
                    "role": m["role"], "acc": acc.group(1) if acc else None,
                    "filed": filed.group(1) if filed else None,
                    "accepted": pd.Timestamp(e.findtext("a:updated", "", NS)).tz_convert("UTC").isoformat(),
                    "items": re.findall(r"Item (\d+\.\d+)", summ),
                    "url": link.get("href") if link is not None else None})
    return [x for x in out if x["acc"]]


def poll(seen, forms=FEED_FORMS, get=sec.get):
    """New feed entries (by accession, CIK and role) not in `seen`, oldest first. A form's feed is paged
    until a page holds an entry already seen. `seen` is updated in place."""
    new = []
    for form in forms:
        for page in range(MAX_PAGES):
            r = get(CURRENT.format(form=form.replace(" ", "%20"), start=page * 100))
            rows = parse_atom(r.content.decode("latin-1")) if r is not None else []
            fresh = [x for x in rows if f"{x['acc']}|{x['cik']}|{x['role']}" not in seen]
            new += fresh
            if len(fresh) < len(rows) or len(rows) < 100:
                break
    out, keys = [], set()
    for x in sorted(new, key=lambda x: x["accepted"]):
        k = f"{x['acc']}|{x['cik']}|{x['role']}"
        if k not in keys:
            keys.add(k)
            out.append(x)
    seen.update(keys)
    return out


def ticker_map(get=sec.get):
    """({cik: [tickers]}, {ticker: company name}) from SEC's company_tickers_exchange.json."""
    j = get(TICKERS).json()
    i_cik, i_t, i_n = (j["fields"].index(k) for k in ("cik", "ticker", "name"))
    by_cik, names = {}, {}
    for row in j["data"]:
        if row[i_t]:
            t = str(row[i_t]).upper().replace("-", ".")
            by_cik.setdefault(int(row[i_cik]), []).append(t)
            names.setdefault(t, str(row[i_n]))
    return by_cik, names


def folder(entry):
    """The filing's folder URL from its index-page link."""
    return entry["url"].rsplit("/", 1)[0] if entry.get("url") else None


def files(entry, get=sec.get):
    r = get(folder(entry) + "/index.json")
    return [] if r is None else [i["name"] for i in r.json()["directory"]["item"]]


def _txt(node, path):
    v = node.findtext(path)
    return v.strip() if v else ""


def parse_form4(text):
    """Form 4 XML -> issuer, owners and non-derivative transactions."""
    root = ET.fromstring(text)
    owners = []
    for o in root.findall("reportingOwner"):
        rel = o.find("reportingOwnerRelationship")
        owners.append({"name": _txt(o, "reportingOwnerId/rptOwnerName"),
                       "director": _txt(rel, "isDirector") in ("1", "true"),
                       "officer": _txt(rel, "isOfficer") in ("1", "true"),
                       "ten_pct": _txt(rel, "isTenPercentOwner") in ("1", "true"),
                       "title": _txt(rel, "officerTitle")} if rel is not None else
                      {"name": _txt(o, "reportingOwnerId/rptOwnerName"), "director": False, "officer": False,
                       "ten_pct": False, "title": ""})
    tx = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        def num(p):
            try:
                return float(_txt(t, p))
            except ValueError:
                return None
        tx.append({"code": _txt(t, "transactionCoding/transactionCode"),
                   "date": _txt(t, "transactionDate/value"),
                   "shares": num("transactionAmounts/transactionShares/value"),
                   "price": num("transactionAmounts/transactionPricePerShare/value"),
                   "ad": _txt(t, "transactionAmounts/transactionAcquiredDisposedCode/value")})
    return {"issuer_cik": int(_txt(root, "issuer/issuerCik") or 0),
            "symbol": _txt(root, "issuer/issuerTradingSymbol").upper(),
            "plan_10b5_1": _txt(root, "aff10b5One") in ("1", "true"),
            "owners": owners, "transactions": tx}


def form4(entry, get=sec.get):
    """Parsed Form 4 of a feed entry, or None when it has no XML."""
    xml = [f for f in files(entry, get) if f.endswith(".xml") and "index" not in f]
    if not xml:
        return None
    r = get(f"{folder(entry)}/{xml[0]}")
    return None if r is None else parse_form4(r.content)


def strip_html(text):
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return " ".join(html.unescape(t).split())


def doc_text(entry, limit=2500, get=sec.get):
    """The start of a filing's most telling document: a press-release exhibit (EX-99) when there is one,
    else the largest .htm/.txt document. Plain text, cut to `limit` characters; '' when none."""
    names = [f for f in files(entry, get) if re.search(r"\.(htm|html|txt)$", f, re.I)
             and "index" not in f.lower() and not re.match(r"\d{10}-\d{2}-\d{6}\.txt$", f)]
    if not names:
        return ""
    ex = [f for f in names if re.search(r"ex[-_]?99", f, re.I)]
    name = ex[0] if ex else names[0]
    r = get(f"{folder(entry)}/{name}")
    return "" if r is None else strip_html(r.content.decode("utf-8", "replace"))[:limit]


def fulltext(query, day, forms, get=sec.get):
    """Full-text search hits filed on `day`: [{"ciks", "acc", "form", "items", "filed"}] (first 100)."""
    url = (f"{FTS}?q={query}&dateRange=custom&startdt={day:%Y-%m-%d}&enddt={day:%Y-%m-%d}"
           f"&forms={','.join(forms)}")
    r = get(url)
    if r is None:
        return []
    out = []
    for h in json.loads(r.content)["hits"]["hits"]:
        s = h["_source"]
        out.append({"ciks": [int(c) for c in s.get("ciks", [])], "acc": s.get("adsh"), "form": s.get("form"),
                    "items": s.get("items") or [], "filed": s.get("file_date")})
    return out
