"""How BOOM's live picks moved on their day: high and low from the open (SIP daily bars, adjustment=all).

    python -m tracks.tools.boom_check research/results/boom_live_picks.json
"""
import json
import sys

import pandas as pd

from core import alpaca

BARS = "https://data.alpaca.markets/v2/stocks/bars"


def main(path):
    picks = json.load(open(path))
    days = sorted({p[0] for p in picks})
    rows = []
    for d in days:
        syms = sorted({p[1] for p in picks if p[0] == d})
        j = alpaca.get(BARS, {"symbols": ",".join(syms), "timeframe": "1Day", "start": d, "end": alpaca.sip_end(d),
                              "feed": "sip", "adjustment": "all", "limit": 10000})
        for s, bs in (j.get("bars") or {}).items():
            b = bs[0]
            rows.append({"d": d, "symbol": s, "up": b["h"] / b["o"] - 1, "down": b["l"] / b["o"] - 1,
                         "oc": b["c"] / b["o"] - 1})
    t = pd.DataFrame(rows)
    print(f"picks with a bar: {len(t)} of {len(picks)}")
    print(f"high >= +8% from open: {(t['up'] >= 0.08).mean():.1%}")
    print(f"low <= -8% from open: {(t['down'] <= -0.08).mean():.1%}")
    print(f"either: {((t['up'] >= 0.08) | (t['down'] <= -0.08)).mean():.1%}  both: {((t['up'] >= 0.08) & (t['down'] <= -0.08)).mean():.1%}")
    print(f"open-to-close median {t['oc'].median():+.2%}, mean {t['oc'].mean():+.2%}, up share {(t['oc'] > 0).mean():.1%}")
    for d, g in t.groupby("d"):
        print(f"{d}: n {len(g)}, +8% {(g['up'] >= 0.08).sum()}, -8% {(g['down'] <= -0.08).sum()}, oc median {g['oc'].median():+.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
