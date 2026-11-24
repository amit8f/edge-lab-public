# EXEC1: enter NX3 trades later in the morning, and exit at the closing auction

Registered 2026-11-24, before any EXEC1 number is computed. Source: Amit, 2026-11-24 ("ok test them"), after COST1
showed the spread is what turns NX3-ALL-HI's history negative. One idea (EXEC-10). Phase-2 count 189 -> 190 (phase 1
+ phase 2 = 205). The closing-auction part is a cost measurement like C0 and is not counted.

## Mechanism (rule 10)

Spreads are widest right after the open: COST1 measured a median round trip of 0.58% for NX3-ALL-HI entries in the
first 30 minutes against 0.20% later. A trader who waits pays the market maker less. The price of waiting is the
news drift that happens before the later entry, which goes to whoever traded first. EXEC-10 asks which is bigger.

The exit side has no such trade-off. Every NX3 label already exits at the next session's official close, which is
the closing auction price. A market-on-close order fills at exactly that price, with no bid/ask to cross. So the
exit spread COST1 charged (the quote at 15:59) is an execution choice, not a property of the strategy.

An opening-auction entry is not tested: for news that arrives before the open, the model's input r_open uses the
opening price, which is only known after the opening auction, so a market-on-open order can't use the model's call.

## Data

NX3-ALL-HI's trades 2019-01 to 2025-09 as COST1 (`models5_study` picks; entry price and time from
`cache/nx1/nx3/trades_v2`; COST1's quotes in `cache/nx1/nx3/cost1`). New: for each trade whose entry minute is
before 10:30, the stock's 1-minute SIP bars (adjustment raw, the basis of the entry price) from 9:30 to 10:31, and
the NBBO in force at 9:45, 10:00 and 10:30 read as COST1 (`cost1_study.quote_at`).

## Costs (both arms)

Entry: the quoted half-spread at the entry minute (COST1's, or the one at T for a delayed trade); a trade with no
valid quote keeps the flat cost, as COST1. Exit: 0 spread (market-on-close at the official close). Commission and
borrow as `adaptive/costs.py`.

## Rule (EXEC-10)

A trade whose entry minute is before T instead enters at the open of the first 1-minute bar at or after T (within
10 minutes, else the trade keeps its original entry). Same side, size and exit. Trades entering at or after T are
unchanged. T in {9:45, 10:00, 10:30}, chosen on 2019-2023 by net per trade.

## Primary test (one look, 2024-01 to 2025-09)

Per day, sized NX3-ALL-HI return with the delay minus without it (both with the costs above): mean > 0 with
Newey-West t >= 2, and the delayed book's own net per trade > 0.

Pass: the live NX3-ALL-HI shadow copies log what a T entry would have paid, and a forward EXEC-10 copy ($5,000, no
orders) is added to the dashboard's Live tab.

## Reported, no rule

The closing-auction saving: mean and median exit half-spread COST1 measured at 15:59, and NX3-ALL-HI's net per trade
with and without it. Delayed trades: how many, the return they gave up before T, and the spread they saved. Per side;
pre-open news vs news during the session; EXEC-10 combined with COST1's skip (c = 0.5%) at the new entry time.

## Prediction (written first)

The closing auction saves about 0.1% a trade. Delaying saves about 0.3% of spread on the delayed trades but gives up
more than that in drift, because NX3's edge is fastest in the first minutes. P(pass) about 0.25.
