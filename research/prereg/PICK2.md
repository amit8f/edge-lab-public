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

## Result: PICK2-MAIN (2026-11-15, run 43652376291, `research/results/PICK2/PICK2_choose.json`)

**Failed at validation; no PICK2-MAIN book.** COMBINED, the input set that decides, had no (learner, bar) book with
100 trades in 2023: its best had 2. CHART had none either. NEWS's chosen book (LightGBM, top 0.10%) had 104 trades
at -0.09% a trade. Idea count unchanged: 167 (182 with phase 1).

POST-HOC / NOT PRE-REGISTERED: the bars were set from each model's predictions on its own training rows (item 3 of
Amendment 1), and CHART's and COMBINED's bars (+0.6% to +4.1% predicted net) were almost never reached in 2023.
[Correction 2026-11-15: the first reading here, that in-sample predictions spread wider than predictions on new
rows, was wrong. PICK2-OOF's out-of-fold bars came out higher, not lower (COMBINED XGBoost top 0.10%: +4.9% vs
+1.3%). Checked 2026-11-15 (data check below): the top 2019-2022 predictions come from 2020 and 2022, and 2023's
predictions sit well below either bar.]

## Amendment 2 (2026-11-15, before PICK2-BAD's first session): how the forward book runs

Written with the code (`tracks/main/pick2_bad_live.py`, workflow `pick2-bad.yml`) and before it has logged a session.

1. The book is logged after each close (21:45 UTC, with two catch-up runs), not by a loop during the session. Each
   finished session from 2026-11-16 is replayed through PICK's history code (`pick_test.day_inputs`, `score_day`,
   `picks_of`): every decision row 09:35-14:00 uses only 5-minute SIP bars that finished before it and headlines
   created before it, so no row sees anything a live scan could not have seen at that minute. The fill is the
   open of the bar starting at the decision time, as in history.
2. Universe: active Alpaca us_equity assets that `core/instruments.py` keeps (common-stock-like), raw close >= $5
   and 20-day median dollar volume >= $5M at the previous close (W1's `daily_rows`, raw SIP daily bars). History
   also required a match in SEC's DERA ticker map; the live book has no such map and drops that filter.
3. News: Alpaca's Benzinga articles from the previous session's 16:00 ET to 14:05 ET, made into NX1's headline rows
   (LM counts, FinBERT at NX1's pinned commit `research/results/NX1/nx1_v1_meta.json`, flags); no text is kept.
4. The usual 30-minute swing (mean of the last 20 sessions' swing30, at least 10) is warmed on the 20 sessions
   before 2026-11-16 on the first run, then carried forward in the state branch `state-pick2bad`.
5. Picks: only rows whose market score is below PICK's frozen cut, with a COMBINED score at or above PICK's frozen
   bad-day bar; sizes by PICK's rule over the bad picks alone (top = `PICK_test.json`'s main COMBINED top), each
   with PICK's random same-minute control. Each session is one ledger entry (kind `day`) with its trades,
   control results, book return and counts; nothing is ever sent to a broker.

## PICK2-OOF (registered 2026-11-15, before any number for it exists; count 168, 183 with phase 1)

Amit chose "Retry once" (decision card, 2026-11-15) after PICK2-MAIN failed at validation. Question: with buy bars
set where each model's scores on **unseen** rows actually fall, does PICK2-MAIN's 2023 book net more than zero?

Changes from PICK2-MAIN, and nothing else:
1. Bars come from out-of-fold predictions. For the validation models (trained 2019-2022): each year 2019..2022 is
   predicted by the same learner (same settings) fitted on the other three years; the bars are the same top shares
   (0.10%, 0.05%, 0.02%, 0.01%) of those pooled out-of-fold predicted net returns. For the refit (2019-2023): the
   same with five yearly folds; the sizing top is the 99th percentile of the out-of-fold predictions at or above
   the chosen bar.
2. The 2023 scan scores every 2023 main decision row with the **committed** validation models
   (`research/results/PICK2/models_val`, unchanged) against those bars, through the same code. The choice and gate
   are Amendment 1, items 4-5: among 12 (learner, bar) books with >= 100 trades in 2023, the best mean net; COMBINED
   decides; if none qualifies or its mean net <= 0, "failed at validation" and the idea ends.
3. If it passes: a $5,000 ledger book with PICK's random control from the first session after it is built, the
   refit model and refit out-of-fold bar, logged like PICK2-BAD; pass by PROTOCOL rule 20.

Honest limits, written first: 2023 is now read a second time with these same models (PICK2-MAIN's books at higher
bars are known), so a 2023 pass earns only a forward paper book, never a history claim. NEWS and CHART are
reported, not decisive.

Prediction (written first): COMBINED's 2023 book reaches 100 trades (0.8); its mean net is > 0 (0.3); it then
passes forward (0.1).

## Result: PICK2-OOF (2026-11-15, runs 43692605422 and 43695990216, `research/results/PICK2/PICK2_choose_oof.json`)

**Failed at validation; the idea ends.** With out-of-fold bars, COMBINED cleared no bar on any 2023 day (0 trades
for every learner and bar); CHART's best had 12 trades. NEWS's chosen book (LightGBM, top 0.10%) had 103 trades at
-0.09% a trade. Count 168 (183 with phase 1). No book; PICK2-BAD's forward book is unaffected.

## Data check after the two failures (2026-11-15, POST-HOC, run 43715521956, `research/results/PICK2/PICK2_diag.json`)

Amit asked whether 0 trades in 2023 meant a bug. Two checks, no rule:
1. Parity: on 2023-03-01, 06-01 and 09-01 the scan path rebuilt 2,644-2,779 of the training rows of the same stock
   and minute. Every input matched except swing30 (up to 0.4 points apart, because the check warms the swing on
   20 sessions instead of the run's full history), and predictions differ by at most 0.2 points. No bug.
2. Predictions by year on the training rows themselves (COMBINED XGBoost, predicted net): the top 0.1% is
   +0.03% in 2019, +2.9% in 2020, +0.2% in 2021, +1.1% in 2022 and +0.09% in 2023; the 2023 maximum is +0.6%,
   under the +1.3% bar. 3.4% of 2020's rows clear that bar and none of 2023's. The model predicts big net gains only
   in wild markets (its prediction is in units of the stock's swing, which was large in 2020 and 2022), so a bar
   set on 2019-2022 as a whole is set by 2020. CHART behaves the same way.
