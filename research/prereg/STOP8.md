# STOP8: NX3's hold with a hard 8% stop

Registered 2026-11-03, before any STOP8 number is computed. Source: Amit, 2026-11-03: trades of -9% to -12%
in NX3 and NX3-ALL ("check why some picks are doing really terrible"), answered "yes" to a registered history
test of a stop. NXE already tested ATR-scaled stops (DAY, EXA1, EXQ, NXT); a fixed percent stop of that size was
not tested, and NX3's first live days were seen first, so this history can only reject (PROTOCOL rule 4). One
idea (STOP5 and STOP12 are exploratory arms of it). Phase-2 count 102 -> 103 (phase 1 + phase 2 = 118).

## Question

Does holding NX3's picks to the next close but leaving early if the stock moves 8% against the trade keep the
pick-minus-control edge while cutting the worst trades?

## Rules

NX3's picks and controls and costs unchanged (NXE.md frame). Entry as in NX3. The stop is the entry price moved
8% against the trade, watched on the daily high/low of day d after the entry and of session d + 1; a session
opening past the stop fills at its open; a session reaching the stop fills at the stop; else the close of d + 1
(NXE's `frame_exit`, no target). The control trades with the same rule.

## Primary test (history)

NXE's bar for a quick check: daily pick-minus-control under the stop rule, pooled over test days, mean > 0 and
Newey-West t >= 1.5 (lags 2). Pass: a $5,000 ledger shadow book beside NX3 (no orders); an Alpaca account only
under PROTOCOL rule 20.

## Reported, no rule

- STOP8 minus H1 (NX3 itself) on the same picks, paired daily mean, NW t.
- Worst 1% of trades, worst month, share of trades stopped, long and short separately, per year.
- Exploratory arms: STOP5 and STOP12 (5% and 12%).
- A daily-bar stop assumes the fill at the stop price; real fills in fast moves are worse. Reported as a caveat.

## Prediction (written first)

The worst 1% of trades improves (from about -28% to about -14%), the mean per trade changes by less than 0.05%,
and the paired difference with H1 is not significant (the NXE exit tests found no rule that clearly beat the hold).
Probability the primary test passes: about 0.5; probability STOP8 beats H1 with paired t >= 1.5: about 0.15.
