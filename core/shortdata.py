"""Short-side public data from FINRA and the SEC (B0, research/prereg/B0.md). Formats as seen from
GitHub's servers on 2026-10-15 (core/probe.py):

- FINRA Reg SHO daily short volume, https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt:
  'Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market' rows, then a row-count line. Volume
  reported to FINRA's facilities only (off-exchange), so the ratio, not the level, is the feature.
- FINRA consolidated short interest, POST
  https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest, no key: JSON rows with
  symbolCode, settlementDate ('YYYY-MM-DD'), currentShortPositionQuantity, marketClassCode, ...
- SEC fails-to-deliver, https://www.sec.gov/files/data/fails-deliver-data/cnsfails{YYYYMM}{a|b}.zip: one
  pipe-separated member 'SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE' (price may be
  '.'); 'a' covers settlement days 1-15, 'b' the rest of the month.
"""
import io
import time
import zipfile

import pandas as pd
import requests

from core import sec

REGSHO = "https://cdn.finra.org/equity/regsho/daily/{mkt}shvol{d:%Y%m%d}.txt"
FACILITIES = ("FNSQ", "FNQC", "FNYX", "FNRA")   # Nasdaq TRFs, NYSE TRF, ADF: the pieces of CNMS
SI_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SI_FIELDS = ["symbolCode", "settlementDate", "currentShortPositionQuantity", "marketClassCode"]
FTD_URL = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{m:%Y%m}{half}.zip"
UA = {"User-Agent": "edge-lab research"}


def http_get(url, tries=5):
    """Response, or None when the file doesn't exist (FINRA's CDN answers 403 for a missing key)."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except requests.RequestException as e:
            last = type(e).__name__
        else:
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                return None
            last = f"HTTP {r.status_code}"
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def parse_regsho(text):
    """Frame (symbol, short, total) from one daily file; the header and the count line are skipped."""
    rows = []
    for ln in text.splitlines():
        p = ln.split("|")
        if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8:
            rows.append((p[1].strip(), p[2], p[4]))
    d = pd.DataFrame(rows, columns=["symbol", "short", "total"])
    for c in ("short", "total"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna().groupby("symbol", as_index=False)[["short", "total"]].sum()


def regsho_day(d, get=http_get):
    """(frame, source) for trading day d: the consolidated CNMS file, else the sum of the facility files
    that exist, else (None, None)."""
    r = get(REGSHO.format(mkt="CNMS", d=pd.Timestamp(d)))
    if r is not None:
        return parse_regsho(r.text), "CNMS"
    parts, found = [], []
    for f in FACILITIES:
        r = get(REGSHO.format(mkt=f, d=pd.Timestamp(d)))
        if r is not None:
            parts.append(parse_regsho(r.text))
            found.append(f)
    if not parts:
        return None, None
    d = pd.concat(parts).groupby("symbol", as_index=False)[["short", "total"]].sum()
    return d, "+".join(found)


def parse_si(rows):
    d = pd.DataFrame(rows, columns=SI_FIELDS)
    out = pd.DataFrame({"symbol": d["symbolCode"].astype(str).str.strip(),
                        "settlement": pd.to_datetime(d["settlementDate"], errors="coerce"),
                        "short_qty": pd.to_numeric(d["currentShortPositionQuantity"], errors="coerce"),
                        "market": d["marketClassCode"]})
    return out.dropna(subset=["settlement", "short_qty"]).drop_duplicates(["symbol", "settlement"])


def short_interest(start, end, post=requests.post, page=5000, tries=5):
    """Every consolidated short-interest row with a settlement date in [start, end], paged."""
    rows, off = [], 0
    while True:
        body = {"limit": page, "offset": off, "fields": SI_FIELDS,
                "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": f"{pd.Timestamp(start):%Y-%m-%d}",
                                      "endDate": f"{pd.Timestamp(end):%Y-%m-%d}"}]}
        for attempt in range(tries):
            r = post(SI_URL, json=body, headers={**UA, "Accept": "application/json"}, timeout=120)
            if r.status_code == 200:
                break
            if r.status_code == 204:          # FINRA answers 204 when nothing matches
                return parse_si(rows)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"FINRA short interest {start}..{end} offset {off}: HTTP {r.status_code}")
        got = r.json() or []
        rows += got
        if len(got) < page:
            return parse_si(rows)
        off += page


def parse_ftd(zip_bytes):
    """Frame (date, symbol, fails) from one half-month zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for ln in text.splitlines():
        p = ln.split("|")
        if len(p) >= 4 and p[0].strip().isdigit():
            rows.append((p[0].strip(), p[2].strip(), p[3].strip()))
    d = pd.DataFrame(rows, columns=["date", "symbol", "fails"])
    d["date"] = pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce")
    d["fails"] = pd.to_numeric(d["fails"], errors="coerce")
    return d.dropna().query("symbol != ''").reset_index(drop=True)


def ftd_file(month, half, get=sec.get):
    """(frame, Last-Modified header or None) for one half-month file, or (None, None) if not posted."""
    r = get(FTD_URL.format(m=pd.Timestamp(month), half=half))
    if r is None:
        return None, None
    return parse_ftd(r.content), r.headers.get("Last-Modified")


def ftd_period_end(month, half):
    m = pd.Timestamp(month).replace(day=1)
    return m.replace(day=15) if half == "a" else m + pd.offsets.MonthEnd(0)


def ftd_mean_fails(d):
    """Per symbol: total fails over the file's settlement days divided by the number of those days (a day
    the symbol isn't listed counts as zero)."""
    n = d["date"].nunique()
    return (d.groupby("symbol")["fails"].sum() / max(n, 1)).rename("mean_fails")
