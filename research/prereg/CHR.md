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

## Live book (written 2026-11-01/02, after Amit chose the shadow book and before any forward data)

Amit, 2026-11-02: tapped "Shadow book" on the card "Give CHR-P a $1,000 penny shadow book, or close it?". CHR-P missed
rule 12 by t (2.91 against 3.0) and meets the rule 19 near-miss test on paper (hand check passed, t >= 1.5, picks'
own mean +0.01% > 0), so it goes forward under the Oct 24 near-miss rule. History scored it first
(`results/CH/CH_result.json`); this section fixes the forward details.

- **Frozen version 1** (`models/chr_p/`, stage `freeze`): the cut and size scale from a model fitted on 2017 .. 2024
  and scored on 2025-01-01 .. 2025-09-23; the final six models fitted on every penny row to 2025-09-23. Settings as
  above. sha256 of each file in `manifest.json`. Never refitted (no self-tuning for this book).
- **Scanner**: every 5 minutes from 9:50 to 15:30 ET on regular days, on bars that end 16 minutes before now, the
  CH.md triggers, penny universe (previous raw close $1 to $5, 20-day median raw dollar volume >= $1M), one decision
  per stock per day at its first trigger. Inputs are CH.md's 68, built by `ch_build.features` as in history.
- **Trade**: a $1,000 ledger book (log only, no orders). The model's largest prediction above the cut picks side and
  stop/target pair. Size 2% of the book at the cut, rising to 10% at the 99th percentile of the scale. Entry at the
  live quote when the decision is made; exit at the first stop or target touched, or at the close of the 5th session.
  Total open exposure at most 100% of the book (a trade that would pass it is skipped and logged); no leverage;
  shorts only where Alpaca marks the stock shortable and easy to borrow (others are logged as "not shortable").
  Costs are the `adaptive/costs.py` penny model.
- **Control**: for each pick, a random other stock from that day's triggered stocks, drawn with
  `default_rng(20261030 + date ordinal)`, same side, pair and size rule, logged beside it.
- **Judged by** PROTOCOL rule 20: 60 trading days and 30 closed trades, book minus control at NW t >= 2, own mean > 0.
  Days that fail the history download filter (CH.md) are flagged and reported separately.
- **Disclosed**: the live entry price is a delayed or IEX-based quote, history entered at the next 5-minute open, so the
  forward book can differ from history for that reason alone.
