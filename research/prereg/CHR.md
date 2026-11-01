# CHR-P: CH's penny chart model with tamer trees (penny track)

Registered 2026-11-01, before any CHR-P model is fit or scored. Source: Amit, 2026-11-01, tapping "Penny only"
on the card "Register tamer chart-model arms for CH?" (CHR-M was not approved). One arm, one idea: phase-2 count
93 -> 94 (phase 1 + phase 2 = 109).

The fix (stronger regularisation) was chosen after seeing CH's rule 21 numbers (`results/CH/CH_result.json`:
CHX-P and CHL-P scored +5% in sample against about +0.6% out of sample), so by PROTOCOL rules 4 and 23 this
history can only reject the arm; the forward book decides. This is the NXR precedent (`prereg/NXR.md`).

## Question

Do LightGBM trees held back from memorising (depth 2, 500-row leaves, heavy sampling, an L2 penalty) keep CH's
penny edge over same-day random flagged stocks with an in-sample score no more than twice the out-of-sample one?

## What stays CH's

Everything but the learner (CH.md): the penny rows, the 68 inputs (none dropped), the six labels and their
costs, the walk-forward (test years 2019 to 2025-09, fit 2017 to Y-2, cut from year Y-1 at the 80th percentile,
refit with a 5-session purge), the decision rule and size scale, the same-day random control (same seed), the
tests and the 2025-10-01 holdout (untouched). Code: `tracks/main/ch_study.py`, stage `treesr`.

## Arm (settings fixed now, no tuning)

LightGBM, Huber loss with alpha at the labels' 90th percentile of |label|, one model per label: 300 trees,
depth 2 (4 leaves), learning rate 0.03, row and column sampling 0.5, at least 500 rows a leaf, L2 10,
deterministic, seed 20261030 + year.

## Primary test

As CH.md: rule 21 first (in-sample more than twice out-of-sample fails the arm); then PROTOCOL rule 12 (HAC
t >= 3.0, bootstrap CI above 0, picks' own mean net > 0, at least 300 trades and 100 days) and rule 11 (FF5 +
momentum alpha t >= 2). Deflated Sharpe reported with 94 trials. Penny shorts cost the track's 20% a year borrow
in history; history cannot tell which stocks were shortable.

## Reported, no rule

Long and short separately, per year, pair share, trigger type, size-weighted, and the arm beside CHX-P and
CHL-P (same days).

## If it passes

A $1,000 ledger shadow book, shorts only where Alpaca can borrow, judged by PROTOCOL rule 20 (60 trading days
and 30 closed trades, beats the control at NW t >= 2, own mean > 0). Real money is rule 9, unchanged. If it
fails rule 21 or the gate, it closes, with a DIAGNOSES.md entry (rule 24).

## Prediction (written first)

Rule 21 passes (ratio about 1.5). Mean over the control falls to about +0.3% a trade, t about 3. About 0.3 that
it clears rule 12 and 11; the main risk is penny-short borrow, which P1 showed is often missing.
