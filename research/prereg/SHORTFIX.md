# SHORT-FIX: NX3 shorts only in a weak market, only liquid stocks, bar 1.75

Registered 2026-11-06, before any SHORT-FIX number is computed. Source: Amit, 2026-11-06 ("what can we do to help
with the short", then "yes do so, test it and if it is good merge it with the bar we found at 1.75"). NX3-BARS's
fired-trade table was seen first (shorts lost in most years 2019-2024; the long bar 1.75 was chosen on 2019-2024),
so this history can only reject (PROTOCOL rule 4); a forward book decides. One idea. Phase-2 count 107 -> 108
(phase 1 + phase 2 = 123).

## Question

NX3's shorts lose in most years. Do they earn money after costs if (a) the confidence bar is 1.75, (b) a short is
taken only when SPY's 5-day return before the event is below 0, and (c) only when the stock is liquid?

## Rules

Everything is NX3's (NX3.md: events, walk-forward models, firing cut, entry minute and price, next-close exit, costs)
for the SHORT side, 3 slots a day in time order, plus:
- bar: confidence |e|/cut >= 1.75 (chosen from the NX3-BARS table; no other value is tried for the primary);
- weak market: spy_r5 < 0 (SPY close-to-close over the 5 sessions before the event day);
- liquid: 20-day dollar volume >= $20M and price >= $5. History has no borrow data, so liquidity stands in for
  easy to borrow; live trading uses Alpaca's easy-to-borrow flag as well.
- The filters also apply to the same-day random control pool (control shorts come from the filtered events).

## Primary test (history)

Daily pick-minus-control for the filtered short book, pooled over the test days 2019..2025-09: mean > 0 and
Newey-West (5 lags) t >= 1.5 (the near-miss bar), AND at least 300 short trades (else "too thin").

## Reported, no rule

Exploratory arms: bar 1.75 alone, weak market alone, liquid alone; per-year results; trades and days counted.

## Forward book

If the primary passes: ledger shadow book `nx3sf` beside NX3, no orders: longs at bar 1.75 (the NX3-BARS choice)
plus these shorts, $5,000, 3+3 slots, from the next trading day. Rule: book minus control, NW t >= 1.5 at 30 closed
trades; an Alpaca account only per PROTOCOL rule 20. If it fails, the shorts stay off the live books.

## Prediction (written first)

The weak-market filter removes most of the losing years, and liquidity removes squeezes. Probability the primary
passes: about 0.35; probability the filtered shorts have at least 300 trades: about 0.6.
