# SURV1-LONG: hold NX3-ALL-HI's longs with SURV1's survival model, forward only

Registered 2026-11-24, after SURV1's result and chosen from it, so history cannot test it (PROTOCOL rule 4): the
test is forward paper data only. Source: Amit, 2026-11-24 ("if it failed ... search why ... if it's fixable we can
try again"). One idea. Phase-2 count 185 -> 186 (phase 1 + phase 2 = 201).

## Mechanism (rule 10)

SURV1 lost overall but split by side: holding past the next close helped longs (+0.39% a trade on 2024-Sep 2025)
and hurt shorts (-0.13%). Good news keeps drifting for days (LESSONS 5: the move goes on past the next close); bad
news on shorts meets squeezes and borrow (LESSONS 6). Post hoc, so it may be luck.

## Rules

NX3-ALL-HI's live picks and sizes exactly as its ledger logs them (prereg NX3ALLHI.md). Shorts exit at the next
close (unchanged). Longs follow SURV1's rule with its frozen model: the LightGBM hazard model trained on 2019-2022
trades (prereg SURV1.md, Amendment 1) and p* = 0.3 (chosen on 2023); at each close k = 1..4 after entry the trade
keeps holding while the model's chance of reaching +3% by close 5 is >= 0.3, exits at the first close at or above +3%,
and at close 5 at the latest. Closes from Alpaca SIP daily bars, `adjustment=all`. Costs: `N.cost_frac` for the days
held. Ledger only ($5,000, NX3-ALL-HI's sizes, no orders), from the first settle after the model is exported.

## Primary test (forward)

After 100 closed long trades: SURV1-LONG's sized return minus NX3-ALL-HI's on the same long trades, per day: mean > 0
with NW t >= 2. Fail: mean <= 0 at 100 longs, or 200 longs without t >= 2.

## Reported

Share held past day 1, mean days held, the longs' own net, and NX3-ALL-HI's shorts beside it.

## Prediction (written first)

Most of the history gain was one or two strong years. P(pass) about 0.15.
