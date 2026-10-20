"""Plan v3 position sizing (section 3 and additions 2, 4, 5).

- Risk sizing: a position risks `risk_frac` (1%) of the book down to its stop.
- Caps: at most 10% of the book in one stock; shorts at most 5% at entry; at 1x, total long + short
  notional never exceeds the book's equity.
- Liquidity: at most 1% of 5-minute dollar volume (penny) or 0.5% of daily dollar volume (main).
- Trim: a position whose value grows past 10% of the book is cut back to 10%.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    risk_frac: float = 0.01
    max_name: float = 0.10
    max_short: float = 0.05
    max_gross: float = 1.0
    stop_cap: float = 0.15          # main 0.15, penny 0.25 (Q43)
    liq_frac: float = 0.005         # main: share of daily dollar volume; penny: share of 5-minute volume


MAIN = Limits()
PENNY = Limits(stop_cap=0.25, liq_frac=0.01)


def size(equity, side, stop_dist, price, liq_dollars, gross_now, lim):
    """Shares to trade (0 when nothing fits) and the reason for the binding limit.

    stop_dist: fraction from entry to stop (> 0). liq_dollars: daily dollar volume (main) or average
    5-minute dollar volume (penny). gross_now: notional already open, long + short.
    """
    if stop_dist <= 0 or price <= 0 or equity <= 0:
        return 0, "bad input"
    if stop_dist > lim.stop_cap:
        return 0, "stop wider than cap"
    caps = {
        "risk": equity * lim.risk_frac / stop_dist,
        "name cap": equity * (lim.max_short if side == "short" else lim.max_name),
        "liquidity": liq_dollars * lim.liq_frac,
        "gross 1x": max(0.0, equity * lim.max_gross - gross_now),
    }
    why = min(caps, key=caps.get)
    shares = int(caps[why] // price)
    return shares, why


def trim(equity, position_value, lim):
    """Dollars to cut from a position worth `position_value` (absolute) so it is back at the name cap."""
    cap = equity * lim.max_name
    return max(0.0, abs(position_value) - cap)
