# PICK2: the same-day long picker, trained on what a trade earns

Registered 2026-11-14, before any PICK2 number is computed. Amit chose "Core + bad-day" on the decision card
the same day. Source: Amit, 2026-11-14: "what can we
learn from the 3% picker? can we try and salvage it, adjust some stuff, use different formulas". Two ideas:
PICK2-MAIN and PICK2-BAD. Phase-2 count 158 -> 160 (phase 1 + phase 2 = 175). PICK-MAIN and PICK-PENNY stay FAILED.

## What PICK's test showed (seen before this was written)

PICK-MAIN, COMBINED, 2024-01 to 2025-09: 2,011 trades; 33% ended +3% or better (random 5%), but 55% hit the stop:
net -0.44% a trade. 2,004 of the 2,011 came from the jumpiest stop tier. It traded on 99% of days. On bad-market
rows it earned +1.0% a trade (159 trades); on other rows -0.57%. Because these were seen, **2024-01 to 2025-09
cannot test PICK2**: it is reported and labelled "seen". Everything is chosen on 2019-2023; a forward book decides.

## PICK2-MAIN

Everything is PICK-MAIN's (PICK.md with its amendments: universe, 5-minute data, decision times 9:35-14:00, entry at
the next bar's open, +8% target, the tier stops frozen in `PICK_fit.json`, the 15:55 exit, costs, the three input
sets, the frozen market score and bad-row cut), except:
- Target: the trade's net return after the stop, target and costs, divided by the stock's 30-minute swing (so a
  jumpy stock's +3% counts for less than a calm stock's), clipped to +-10. A number, losses included, in place of
  PICK's yes/no on +3%.
- Learners: XGBoost, CatBoost and LightGBM regressors with a Huber loss; the one with the highest mean net per
  trade of the 2023 book below wins.
- Picks: a row is a pick when its predicted net return (prediction x swing) is at or above the bar. At most 2 buys
  a day, the first two to clear it, one per stock. Bar candidates: the top 0.10%, 0.05%, 0.02% and 0.01% of
  2019-2022 training predictions; the one with the best 2023 mean net per trade with at least 100 trades in 2023.
  On bad rows the bar is the same (PICK2-BAD covers bad days).
- Size: PICK's rule (10% to 40% of a $5,000 ledger book by confidence).
- Periods: train 2019-01 to 2022-12, choose on 2023, refit on 2019-2023.
- Gate to start the book: the 2023 book's mean net per trade > 0. If not: "failed at validation", no book.
- Reported, labelled "seen", no rule: the refit models on 2024-01 to 2025-09 with PICK's random control.

## PICK2-BAD

PICK-MAIN's frozen COMBINED model, widths and bad-day bar exactly as committed (`research/results/PICK`), buying
only on bad rows (market score below the frozen cut). No refit, nothing chosen. Sizes and exits are PICK's. It has
no history test: its model was fitted on 2019-2023 and its 2024-2025 result is the post-hoc hint above.

## Forward test (each idea)

A $5,000 ledger book (no orders) from the first session after the live scan is built, with PICK's random control
(one random universe stock bought at the same minute, same rules). Pass: PROTOCOL rule 20: 60 trading days and 30
closed trades, mean net per trade > 0 with Newey-West t >= 2 on daily book returns, and net minus the random buys
> 0 with t >= 2. Judged then, or at 2027-12-31.

## Prediction (written first)

PICK2-MAIN's 2023 book is positive (chance 0.5) and it passes forward (0.15). PICK2-BAD passes forward: 0.15; bad
days are about a fifth of days, so 30 trades may take longer than 60 days.

## Amendment 1 (2026-11-14, before any PICK2 number exists): how the fit and the 2023 choice are computed

Written with the code (`tracks/main/pick2_study.py`) and before it has run on real data. Nothing above changes;
these are the details the text left open.

1. Training rows are PICK's (the fixed 3% sample of 2019-2023 main decision rows). Each input set (NEWS, CHART,
   COMBINED) keeps the tier stops PICK froze for it, so a row's target is its net return at its own tier's stop,
   divided by swing30. One model per input set and learner; as in PICK, only COMBINED decides, the other two are
   reported.
2. Huber loss with delta 1 (one swing): XGBoost's pseudo-Huber with slope 1, CatBoost `Huber:delta=1`, LightGBM
   `huber` with alpha 1. Trees, depth and learning rates are the ones PICK's classifiers used.
3. Bars: for each model, the top 0.10%, 0.05%, 0.02% and 0.01% of its own predicted net returns on its 2019-2022
   training rows. Bad and normal rows share one population and one bar.
4. The 2023 book is built from **every** 2023 main decision row, not the 3% sample, because "the first two to
   clear the bar" can only be found by looking at every row. The 2019-2022 models score a fresh full scan of 2023
   (5-minute bars, the same code path as PICK's test). A stock's first clearing row is its only candidate that
   day; at the same minute the higher prediction is bought first; a row with no valid trade is skipped.
5. Learner and bar are chosen together: among the 12 (learner, bar) books of an input set with at least 100 trades
   in 2023, the one with the highest mean net per trade. If none has 100 trades, or COMBINED's chosen book has
   mean net <= 0: "failed at validation", no PICK2-MAIN book.
6. Refit bars and the sizing top come from the 2019-2023 refit model's predictions on the sampled rows (the same
   share as the chosen bar; top = the 99th percentile of predictions at or above the bar, PICK's rule).
7. Models (validation and refit) are committed under `research/results/PICK2`, so an evicted cache cannot lose
   them.
