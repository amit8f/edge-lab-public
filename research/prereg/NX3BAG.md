# NX3-BAG: 25 bagged NX3 models, for steadier picks (news track)

Registered 2026-11-01, before any NX3-BAG number is computed. Source: the monthly review of 2026-11-01
(`/mnt/project-files/reviews/MONTHLY_2026-11-01.md`, idea A); Amit: "do both and we will see". NX3's
history and its rebuilt-cache rerun were seen first (a change of under 0.1% in the training events moved the
2020 fire cut from 0.36 to 0.27 and the result from +0.51% to +0.43% a day), so this history can only reject
(PROTOCOL rule 4); the forward book decides. One idea. Phase-2 count 91 -> 92 (phase 1 + phase 2 = 107).
NX3 itself is unchanged.

## Question

Does the average of 25 NX3 models, each trained on a bootstrap resample of the training days, beat the
same-day random control at least as NX3's near-miss bar asks, with fire cuts that move less when the event
file changes slightly?

## Rules

Everything is NX3's (NX3.md: events, 37 inputs, NX2's baseline-plus-news design, walk-forward refitted each
test year 2019..2025-09 on the rebuilt cache, purge, 3 longs + 3 shorts a day in time order, entry, next-close
exit, costs, controls), except the model:
- For each test year, draw 25 bootstrap samples of the training trading days (with replacement, as many days
  as the training set has; seed 20261101 + 100 x year + k, k = 0..24). Each sample takes every real and
  placebo event of its drawn days, repeated as often as the day was drawn.
- Member k is NX3's `Model` (baseline B on placebo events, news model N on real z - B, same XGBoost settings)
  fitted on sample k, with XGBoost's `random_state` = 20261024 + k.
- Bagged effect e = mean over the 25 members of N_k(x) - N_k(x0).
- Fire cut: the 90th percentile of |e| over all real training events (not a resample), as NX3.

## Primary test (history)

Picks minus same-day random control, per day, pooled over test days: mean > 0 and Newey-West t (5 lags)
>= 1.5 (the near-miss bar for live variants, grill 2026-10-30).

## Second gate (stability)

Five reruns, each on one bootstrap draw of the whole event file by trading day (seed 20261102 + r,
r = 0..4). In each rerun both NX3 (single model, as registered) and NX3-BAG are refitted for every test year.
For each test year, the coefficient of variation (standard deviation / mean) of the fire cut across the 5
reruns; the gate passes when NX3-BAG's mean CV over the test years is below NX3's.

Both gates must pass.

## Reported, no rule

Picks minus the shuffled-news control; bagged minus NX3 on the same days (NW t); the share of NX3-BAG's picks
that are NX3's; mean pairwise overlap (Jaccard) of the test picks across the 5 reruns, for both; per year
numbers (`NX3BAG_walkforward.json`).

## If it passes

A $5,000 ledger shadow book beside NX3 live, version 1 trained on all events to 2025-09-23, judged by PROTOCOL
rule 20 (60 trading days and 30 closed trades, NW t >= 2). An Alpaca paper account only after that.

## Prediction (written first)

Bagging should make the cuts steadier (about 0.8) and leave the mean about where NX3 is; it won't beat NX3 on
the same days (about 0.85 that the head-to-head t is under 1.5). Probability both gates pass: about 0.6.
