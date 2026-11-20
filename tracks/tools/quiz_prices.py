"""Print daily SIP bars (adjustment=all) for a few symbols, for checking hand-read news calls. Prices only.

    python -m tracks.tools.quiz_prices WIX,ICL,ZG,AVAV,CVS,SPY 2026-11-03 2026-11-19
"""
import sys

from core import alpaca

BARS = "https://data.alpaca.markets/v2/stocks/bars"


def main(symbols, start, end):
    j = alpaca.get(BARS, {"symbols": symbols, "timeframe": "1Day", "start": start, "end": alpaca.sip_end(end),
                          "feed": "sip", "adjustment": "all", "limit": 10000})
    for sym, bars in sorted(j.get("bars", {}).items()):
        prev = None
        for b in bars:
            gap = f"{b['o'] / prev - 1:+.2%}" if prev else ""
            day = f"{b['c'] / prev - 1:+.2%}" if prev else ""
            print(f"{sym} {b['t'][:10]} o {b['o']:.2f} c {b['c']:.2f} gap {gap} close-to-close {day}")
            prev = b["c"]


if __name__ == "__main__":
    main(*sys.argv[1:4])
