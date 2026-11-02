# NX3-ALL: NX3 without the 3+3 cap, sized by confidence (news track)

Registered 2026-10-31, before any NX3-ALL number is computed. Source: Amit, 2026-10-31: "why are we capping
it to 6 ... lets not cap it" (he chose an uncapped copy next to NX3 over changing NX3), and "if it high its
should be toward the 10% of the money meaning 500$ but if its less confidence more toward 5%". NX3's capped
history (gate passed 2026-10-24) and its first live days were seen first, so this history can only reject
(PROTOCOL rule 4); the forward book decides. One idea. Phase-2 count 88 -> 89 (phase 1 + phase 2 = 104).
NX3 itself is unchanged.

## Question

NX3 takes only the first 3 longs and 3 shorts the model fires each day. Do all the fired events, sized by how
confident the model is, still beat the same-day random control after costs?

## Rules

Everything is NX3's (NX3.md: events, inputs, walk-forward models refitted each test year 2019..2025-09 with
the same seeds, firing cut, entry minute and price, next-close exit, costs), except:
- Picks: every fired event of the day (e > 0 long, e < 0 short), no cap.
- Size: confidence c = |e| / cut (c >= 1 for a fired event). Each trade gets 5% of the $5,000 book at c = 1,
  rising in a straight line to 10% ($500) at c >= 2. If a day's sizes add up to more than the book, all of
  that day's sizes are scaled down by the same factor.
- Control: as many of that day's real events at random (seeded as NX3), same sides; control trade i takes
  pick i's size.

## Primary test (history)

Daily book return of the picks minus the control (sum over trades of size x net return), pooled over test
days, mean > 0 and Newey-West t (5 lags) >= 1.5: the near-miss bar Amit set for live variants (grill
2026-10-30). Pass: a $5,000 ledger shadow book on NX3's live job; an Alpaca paper account only after rule 20
(60 days and 30 closed trades, NW t >= 2).

## Reported, no rule

The same with equal sizes; picks minus the shuffled-news control; picks per day and the median share of the
book invested; per year (`NX3ALL_walkforward.json`).

## Prediction (written first)

Equal-size uncapped picks will be weaker than NX3's capped ones per trade (later events of the day get a worse
entry), and confidence sizing will win back part of it. Probability it passes: about 0.5.

## Clarification 2026-11-02: paper account EL-NX3ALL (decided before any NX3-ALL forward result exists)

Amit set up a paper account for the book (secrets `ALPACA_NX3ALL_*`, replacing EL-M3's account; M3's month-end
orders on Nov 30 need another account) and asked for the book to trade there. Nothing about the test changes: it
stays the ledger book minus its control. The account is for real fills, costs and borrow, as EL-NX3 is for NX3.
Orders (`tracks/news/nx3all_orders.py`): each fired event is sent as a market day order at its decision, in whole
shares at the IEX price, sized by the confidence rule on the $5,000 book (5% at confidence 1 up to 10% at 2 or more).
The ledger book scales a day's sizes down together after the close, which an order cannot wait for, so orders stop
once the day's sent sizes would pass 100% of the book (an order and ledger size can therefore differ on busy days).
Shorts only where the stock was easy to borrow. Exit: a market-on-close order on the next session for every
position held before that session's own entries. Nothing is sent without the secrets.
