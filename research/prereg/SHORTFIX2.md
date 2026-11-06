# SHORT-FIX2: NX3 shorts only in a weak market

Registered 2026-11-06, before any SHORT-FIX2 number is computed. Source: Amit, 2026-11-06 ("keep trying adjusting";
before that, "yes" to fixing the shorts). SHORT-FIX (idea 108) failed its primary test (t 0.88); its post hoc arm
"weak market only" showed +1.14% a day over random shorts (t 2.57, positive in 6 of 7 years). That history was seen,
so this test can only reject (PROTOCOL rule 4); a forward book decides. One idea. Phase-2 count 108 -> 109
(phase 1 + phase 2 = 124).

## Question

Random shorts lose money, so beating them is not enough. Do NX3's shorts, taken only when SPY's 5-day return before
the event is below 0, make money after costs by themselves?

## Rules

NX3's shorts (NX3.md: events, walk-forward models, firing cut, entry minute and price, next-close exit, costs), 3 slots
a day in time order, confidence bar 1.75, only on events where spy_r5 < 0. No liquidity filter (it added nothing).
The same filter applies to the control pool.

## Primary test (history)

All of: pick-minus-control mean > 0 and NW (5 lags) t >= 1.5; mean absolute net per day of the picks (the shorts'
own result after costs, not versus control) > 0 with NW t >= 1.5; at least 300 short trades.

## Reported, no rule

The same at bar 1.0 (NX3's own); per-year results versus control and absolute.

## Forward book

If it passes: ledger shadow book `nx3sf` next to NX3, no orders: longs at bar 1.75 (NX3-BARS table) plus these
shorts, $5,000, 3+3 slots. Rule: book minus control, NW t >= 1.5 at 30 closed trades; an Alpaca paper account only
per PROTOCOL rule 20.

## Prediction (written first)

Probability the primary passes: about 0.3. The absolute-profit condition is the hard one: in the BARS table shorts
at 1.75 earn about +0.06% a trade over all years.
