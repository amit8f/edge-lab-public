"""V0: which raw data sources the GitHub runner can reach (research/prereg/V1.md, "Sources"). Prints and
saves HTTP status codes only, never a credential or a response body.

Usage: python -m validation.v0_sources --cache <dir> [--budget-min N]
"""
import argparse
import json
from pathlib import Path

import requests

from core.env import alpaca_headers, sec_headers

OUT = Path(__file__).resolve().parents[1] / "research" / "results" / "V0"
SOURCES = {
    "ken_french_ff5_daily": ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                             "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", None),
    "stooq_daily": ("https://stooq.com/q/d/l/?s=aapl.us&i=d", None),
    "nasdaq_historical": ("https://api.nasdaq.com/api/quote/AAPL/historical?assetclass=stocks&fromdate=2024-01-02"
                          "&todate=2024-01-10&limit=10", None),
    "edgar_full_index": ("https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/company.idx", "sec"),
    "edgar_daily_index": ("https://www.sec.gov/Archives/edgar/daily-index/2024/QTR1/", "sec"),
    "finra_short_volume": ("https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240102.txt", None),
    "finra_short_interest_api": ("https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
                                 "?limit=1", None),
    "opentimestamps_calendar": ("https://a.pool.opentimestamps.org", None),
    "alpaca_quotes": ("https://data.alpaca.markets/v2/stocks/AAPL/quotes?start=2024-01-02T15:00:00Z"
                      "&end=2024-01-02T15:00:05Z&feed=sip&limit=5", "alpaca"),
}


def probe(url, kind):
    headers = {"User-Agent": "edge-lab research"}
    if kind == "sec":
        headers = sec_headers()
    elif kind == "alpaca":
        headers = {**headers, **alpaca_headers()}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        return {"status": r.status_code, "bytes": len(r.content)}
    except requests.RequestException as e:
        return {"status": None, "error": type(e).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--budget-min", type=float, default=0)
    ap.parse_args()
    res = {name: probe(url, kind) for name, (url, kind) in SOURCES.items()}
    for k, v in res.items():
        print(k, v)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
