"""Independent bar pull for hand checks (PROTOCOL rule 14). Reads research/checks/<name>.csv with columns
symbol, start, end and writes research/checks/<name>_bars.json: each symbol's daily SIP bars over
[start, end] with both adjustment=raw and adjustment=all. Plain HTTP on purpose, so a check never reuses the
study's own bar code."""
import csv
import json
import os
import sys
import time

import requests

URL = "https://data.alpaca.markets/v2/stocks/{s}/bars"


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


def main(name):
    headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
               "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    with open(f"research/checks/{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
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
