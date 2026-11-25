# NX3-TIE: when the 3+3 slots are contested at one minute, take the most confident events, not the alphabetical ones

Registered 2026-11-24, before any NX3-TIE number is computed. Source: Amit, 2026-11-24 ("do all of this", the
tie-break fix from the status list), after the 2026-11-20 check of the NXM books (`nxm-books-check`): 147 of 168
capped picks started with "A". One idea. Phase-2 count 190 -> 191 (phase 1 + phase 2 = 206).

## Mechanism (rule 10)

Every event that arrives before the open is dated at the open, so on busy mornings many events tie on one minute. NX3
fills its 3 long and 3 short slots in time order and, within a tie, in symbol order: the slots go to whichever tickers
come first in the alphabet. The model's own confidence (`|e| / cut`) is known at the same moment and NX3-HI shows it
carries information (bar 1.5: +0.72% a day against +0.51% for NX3 vs random). The only gain is picking better events
among those already on offer; nobody trades against us. Chosen slots only change on days when more than 3 events of one
side tie at one minute.

## Rule

NX3 exactly (same models, same walk-forward, same costs, same cap of 3 longs and 3 shorts a day, same random control
and seeds), except events are taken in order of time, and within the same minute by `|e|` from high to low instead of
by symbol. No parameter is chosen. The -ALL books (no cap) are not affected.

## Primary test (history walk-forward, all test years as NX3's own gate)

1. NX3-TIE picks minus the same-day random control, daily, Newey-West t >= 2 (NX3's gate), AND
2. NX3-TIE minus plain NX3 on shared days: mean > 0 with NW t >= 1.5.

Pass: a $5,000-style ledger book `NX3-TIE` ($500 a trade, NX3's rules, no orders) beside NX3 from the next trading day,
on the dashboard's Live tab. Forward test as every shadow book: 60 trading days and 30 trades, picks minus control NW
t >= 2.

## Reported, no rule

Share of days on which the tie-break changes at least one slot; share of picks dated at the open; net per trade with
the flat cost for NX3 and NX3-TIE; per year; longs and shorts separately; the share of picks whose symbol starts with
"A" under each rule (the 2026-11-20 check found 87% for NX3).

## Prediction (written first)

The tie-break changes slots on about a third of days, picks higher-confidence events, and adds about +0.1% a day over
NX3 with a paired t near 1.5. P(pass) about 0.45.

## Result (2026-11-24, run 45476008061): FAILED

`research/results/NX3/NX3TIE_walkforward.json`. Walk-forward 2019-2025, 1,677 pick days, 8,208 picks each. NX3-TIE
picks minus control +0.60% a day, NW t 4.69 (NX3 itself +0.59%, t 4.99), so the first condition passes. NX3-TIE minus
NX3 on shared days +0.015% a day, NW t 0.20 (needs 1.5): failed. The tie-break changed at least one slot on 1,327 of
1,677 days (79%, the prediction said a third); symbols starting with "A" fell from 33% of picks to 11%. Flat-cost net
per trade +0.047% -> +0.086% (longs +0.13% -> +0.19%, shorts -0.065% -> -0.048%). By year the two arms are within noise
(TIE better in 2019, 2023, 2025; worse in 2020, 2024).

So the alphabetical order costs nothing measurable in history. One loose end: the 2026-11-20 check of 15 live days
found 87% of capped picks starting with "A", against 33% in history; the cause is not found and nothing here explains
it. No shadow book. NX3 and the running books are unchanged.
