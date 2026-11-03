# NX3-ALL-HI: NX3-ALL with a higher confidence bar

Registered 2026-11-03, before any NX3-ALL-HI number is computed. Source: Amit, 2026-11-03: "yes do also that"
(the high-bar version of the uncapped book, after NX3-HI). NX3-ALL's history and its first live day were seen
first, so this history can only reject (PROTOCOL rule 4); the forward book decides. One idea. Phase-2 count
101 -> 102 (phase 1 + phase 2 = 117).

## Question

NX3-ALL takes every fired event, sized 5% (confidence |e|/cut = 1) up to 10% (>= 2) of a $5,000 book. If only
events with |e|/cut >= 1.5 are taken (so sizes run 7.5% to 10%), does the confidence-sized book still beat the
same-day random control after costs?

## Rules

Everything is NX3-ALL's (NX3ALL.md: no 3+3 cap, sizes by confidence, a day adding up to more than the book is
scaled down together, control trade i takes pick i's side and size, next-close exit, costs), except fired events
with c < 1.5 are not taken. A short only where easy to borrow in live trading; history ignores borrow.

## Primary test (history)

`nx3-study stage=highbar_all`: daily confidence-sized pick book minus control, pooled over test days: mean > 0 and
Newey-West (5 lags) t >= 1.5 (the near-miss bar, grill 2026-10-30), at bar 1.5. Bars 1.25 and 2.0, the
comparison with NX3-ALL on shared days (bar 1.0) and per-year results are exploratory.

## Forward book

Ledger shadow book `nx3allhi` next to NX3 in NX3's live job, no orders, from 2026-11-04, same sizes as the
history. Rule: book minus control, NW t >= 1.5 at 30 closed trades per NX3ALL.md; an Alpaca account only per
PROTOCOL rule 20.

## Prediction (written first)

Picks per day fall by about half; the sized mean per day moves little, and the t may drop because there are fewer
days. Probability the primary test passes: about 0.5; probability it clearly beats NX3-ALL: about 0.15.
