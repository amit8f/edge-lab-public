"""Independent bar pull for hand checks (PROTOCOL rule 14). Reads research/checks/<name>.csv with columns
symbol, start, end and writes research/checks/<name>_bars.json: each symbol's daily SIP bars over
[start, end] with both adjustment=raw and adjustment=all. A file with columns cik, accession instead writes
research/checks/<name>_edgar.json: each filing's acceptance time, form and filer names from its EDGAR index
page (not the header file the studies read). Plain HTTP on purpose, so a check never reuses a study's code."""
import csv
import json
import os
import re
import sys
import time

import requests

URL = "https://data.alpaca.markets/v2/stocks/{s}/bars"
EDGAR = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{acc}-index.htm"


def bars(sym, start, end, adjustment, headers):
    out, token = [], None
    while True:
        p = {"timeframe": "1Day", "start": start, "end": end, "feed": "sip", "adjustment": adjustment,
             "limit": 10000, **({"page_token": token} if token else {})}
        for attempt in range(4):
            r = requests.get(URL.format(s=sym), params=p, headers=headers, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        j = r.json()
        out += [{k: b[k] for k in ("t", "o", "h", "l", "c", "v")} for b in j.get("bars") or []]
        token = j.get("next_page_token")
        if not token:
            return out


def edgar_index(cik, acc, headers):
    """Acceptance time, form and filer names from a filing's EDGAR index page."""
    for attempt in range(5):
        r = requests.get(EDGAR.format(cik=int(cik), nodash=acc.replace("-", ""), acc=acc), headers=headers, timeout=60)
        if r.status_code not in (429, 503):
            break
        time.sleep(2 ** attempt)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    html = r.text
    acc_t = re.search(r'infoHead">Accepted</div>\s*<div class="info">([^<]+)<', html)
    form = re.search(r'id="formName">\s*<strong>([^<]+)<', html)
    names = re.findall(r'class="companyName">([^<(]+)', html)
    time.sleep(0.15)  # SEC's limit is 10 requests a second
    return {"accepted": acc_t and acc_t.group(1).strip(), "form": form and form.group(1).strip(),
            "names": [n.strip() for n in names]}


def main_edgar(name, rows):
    headers = {"User-Agent": os.environ["SEC_USER_AGENT"].strip()}
    res = [{**r, **edgar_index(r["cik"], r["accession"], headers)} for r in rows]
    tmp = f"research/checks/{name}_edgar.json.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, f"research/checks/{name}_edgar.json")
    print(f"{len(res)} filings, {sum('error' in x for x in res)} errors")


def main(name):
    with open(f"research/checks/{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "accession" in rows[0]:
        return main_edgar(name, rows)
    headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
               "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    res = []
    for r in rows:
        res.append({**r, "raw": bars(r["symbol"], r["start"], r["end"], "raw", headers),
                    "all": bars(r["symbol"], r["start"], r["end"], "all", headers)})
    tmp = f"research/checks/{name}_bars.json.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, f"research/checks/{name}_bars.json")
    print(f"{len(res)} rows, {sum(len(x['raw']) for x in res)} raw bars")


if __name__ == "__main__":
    main(sys.argv[1])
