# NXM: NX3 with other model types (news track)

Registered 2026-10-28, before any NXM model is fit or scored. Source: Amit, 2026-10-28: "try other models like
the XGB to see how they react". NX3's XGBoost results on the same data were read first (its gate passed on
2026-10-24), so this history can only reject an arm (PROTOCOL rule 4); forward books decide. Five arms, each
counted: phase-2 count at registration 63 (main 28, penny 10, news 14, cross-track 10, crypto 1). Phase 1 +
phase 2 = 78.

## Question

Does NX3's result depend on XGBoost? If the same design with another learner also passes NX3's gate, the edge
sits in the data and the design. If only XGBoost passes, it is fragile.

## What stays NX3's

Everything but the learner: NX3's cached trades (`trades_v2`, the files its gate read), its 37 inputs and 13
baseline inputs, label, walk-forward (test years 2019 to 2025-09, 5-session purge), the baseline B on placebo
events and news model N on real z - B, the effect e = N(x) - N(x0), the fire cut at the training years' 90th
percentile of |e|, picks (first 3 fired longs and shorts in time order), the same-day random control, the
shuffled-news test, seeds (20261024 + year) and costs. Nothing from the 2025-10-01 holdout is read.

## Arms (settings fixed now, no tuning; `tracks/news/nxm_study.py`)

- `LGBM`: LightGBM 4.7.0 with XGBoost's settings where they map: 400 trees, depth 4 (15 leaves), learning rate
  0.05, row and column sampling 0.8, at least 50 events a leaf, Huber loss with delta 1.0, deterministic.
- `LIN`: Huber linear regression (scikit-learn 1.9.1, epsilon 1.35, alpha 1e-4) on standardized inputs, missing
  values at the training median plus a missing flag. Its effect is a weighted sum of the news inputs, so it
  can't read the news together with the price reaction.
- `RF`: random forest, 200 trees, a third of the inputs per split, at least 200 events a leaf, each tree on a
  30% sample, scikit-learn's own missing-value handling.
- `MLP`: neural net with hidden layers of 64 and 32, Adam, batch 1,024, L2 1e-3, at most 60 passes with early
  stopping (5 passes without gain on a random 10% of the training events), LIN's input preparation.
- `ENS`: the mean of the five models' effects (XGBoost, LGBM, LIN, RF, MLP), each divided by its own fire cut,
  fired at the training years' 90th percentile of that mean.

The XGBoost arm is NX3 rerun with its own code as a check: its pooled numbers must equal
`results/NX3/NX3_walkforward.json`. It is not a new idea.

## Primary test (per arm)

1. Rule 21 first. In-sample score: each year's model scored on its own training events with the same picks and
   control; the mean daily pick-minus-control over the seven fits (weighted by days), beside the pooled
   out-of-sample mean. An arm whose in-sample mean is more than twice its out-of-sample mean is reported as
   failed (overfit), whatever its test numbers.
2. NX3's gate: pooled 2019 to 2025-09 daily pick-minus-control and pick-minus-shuffled, both mean > 0 and
   Newey-West t (5 lags) >= 2.

An arm passes when it clears both.

## Head-to-head (rule, but it changes nothing by itself)

Per arm: daily mean net of its picks minus XGBoost's, on days both pick; NW t (5 lags); Holm across the five
arms at 5% two-sided. "Beats XGBoost" needs mean > 0 and Holm p < 0.05. NX3 stays frozen either way.

## Reported, no rule (how they react)

Per-year numbers; test fire rates on real and placebo events; picks' vol20 percentile and share before the open;
Spearman correlation of each arm's e/cut with XGBoost's; the share of each arm's picks XGBoost also picks (same
day and side); mean e/cut and long and short fire shares by decile of FinBERT net tone (fb_pos - fb_neg) and
for each of the 12 news-type flags. XGBoost's own in-sample score is shown too; NX3 was registered before rule
21, so it doesn't change NX3's status.

## What happens next (fixed now)

Each passing arm gets version 1 trained on all of 2016-10 .. 2025-09-23 (as NX3's was) and a $5,000 ledger
shadow book beside NX3's live book: the same live events and inputs (no extra API calls), its own fire cut,
picks and same-day control, and NX3's forward test (book minus control, NW t >= 2 at 100 closed trades or
2027-12-31), each reported with the number of NXM books tried. A failing arm gets no book. No arm replaces NX3.
Accounts and money follow PROTOCOL rules 9 and 18.

## Prediction (written first)

LGBM passes and picks mostly what XGBoost picks (overlap above 60%). LIN passes the control test but fails the
shuffled test (about 0.6), because NX3's edge needs the news read with the first price reaction. RF passes,
weaker than XGBoost. MLP about even odds. ENS has the highest mean but doesn't beat XGBoost at Holm p < 0.05.
About 0.15 that any arm beats XGBoost head-to-head, and about 0.5 that rule 21 flags the tree arms (boosted
trees on noisy returns usually score several times better in-sample).
