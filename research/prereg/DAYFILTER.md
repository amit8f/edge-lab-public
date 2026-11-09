# DAYFILTER: a higher bar against the market's colour

Registered 2026-11-09, before any DAYFILTER number is computed. Source: Amit, 2026-11-09: "raise the bar for short
if it's a green day and raise the bar for buy if it's a red day". One idea. Phase-2 count 135 -> 136 (phase 1 +
phase 2 = 151). NX3-HI and NX3-BARS already raise the bar by side; this raises it by the market's colour, a
different cut of the same confidence score. NX3's live days were seen first, so history can only reject
(PROTOCOL rule 4).

## Question

Does asking for more confidence on trades that go against the market (shorts on a green day, longs on a red day)
improve NX3-ALL's sized pick-minus-control, or its own net?

## Rules

NX3-ALL unchanged (prereg NX3ALL.md): same models, seeds, fired events, confidence sizing 5%..10%, costs. The
market's colour is known at the entry: green when SPY at the event's entry time is above the prior close
(`spy_open` > 0); where that input is missing, the prior session's SPY return (`spy_r1` > 0). Red otherwise.
Registered arm: a short on a green day or a long on a red day needs confidence |e|/cut >= 1.5 (the NX3-HI
value); the other trades keep bar 1.0. Exploratory arm: 2.0. The control draws the same number of random trades
on the same day with the same sides, as in NX3-ALL.

## Primary test (history)

Sized pick-minus-control per day under the bar-1.5 rule: mean > 0 and Newey-West t >= 1.5 (NX3-ALL's near-miss
bar), AND the paired daily gain over plain NX3-ALL (bar 1.0, same days) mean > 0 with NW t >= 1.5. Pass: a $5,000
ledger shadow book beside NX3-ALL (no orders); an Alpaca account only under PROTOCOL rule 20.

## Reported, no rule

- Own net per day (sized, after costs) for each arm, per year, longs and shorts separately.
- Share of trades removed, and their own net (are the removed trades the bad ones?).
- Exploratory: bar 2.0.

## Prediction (written first)

The filter removes few trades (about 15%), the removed ones are not clearly worse than the kept ones, and the
paired gain over NX3-ALL is not significant (NX3-HI and NX3-BARS found that a higher bar by side did not beat
plain NX3). Probability the primary test passes: about 0.15.
