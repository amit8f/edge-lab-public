# SHORT-FAST: take the quick move after bad news, exit within hours

Registered 2026-11-06, before any SHORT-FAST number is computed. Source: Amit, 2026-11-06 ("pick those things up fast
and act", "be creative"). NX3, NX3-BARS, SHORT-FIX, SHORT-FIX2 and SHORT-ML were seen first (shorts held to the next
close earn about zero on their own), so this history can only reject (PROTOCOL rule 4); a forward book decides. One
idea. Phase-2 count 110 -> 111 (phase 1 + phase 2 = 126).

## Question

Do NX3's shorts make money after costs if we exit within hours of the entry, before the news is fully priced and the
move fades, instead of holding to the next close?

## Rules

- Events and picks: NX3's own (NX3.md: walk-forward models, firing cut, bar 1.0, first 3 longs and 3 shorts a day in
  time order, entry one minute after the news, costs from config/risk.yaml, borrow for 1 day).
- Exit: FAST_PRIMARY = 60 minutes after the entry (the first bar at or after it). Raw 1-minute SIP bars.
- Control: as many of the day's random real events as picks, same sides, same exit.
- Exploratory exits (not rules): 15, 30, 120 minutes and the same session's close; the long side at each exit.

## Primary test (history)

Shorts, 60-minute exit, all of: own net per day after costs > 0 with NW (5 lags) t >= 1.5; pick-minus-control
mean > 0 with NW t >= 1.5; at least 300 short trades. The other exits and the long side are post hoc.

## Forward book

If it passes: ledger shadow book `nx3fast` next to NX3, no orders, $5,000, the same short picks exited after 60
minutes; rule: NW t >= 1.5 on its own net at 30 closed trades; an Alpaca paper account only per PROTOCOL rule 20.

## Prediction (written first)

Probability the primary passes: about 0.3. Fast exits cut the borrow and overnight-gap noise, but costs are a
bigger share of a smaller move.
