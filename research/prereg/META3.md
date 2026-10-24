# META3: a simple keep-or-skip score for R1's picks, tested forward (main track)

Registered 2026-10-24, before any number for this score exists. Amit's request after META2 failed
(2026-10-24: find why a good idea fails, fix it and keep going). Phase-2 count at registration: 45 (main 23,
penny 7, news 7, cross-track 7, crypto 1). Phase 1 + phase 2 = 60.

## Where it comes from (and why the test is forward only)

META2's post-hoc diagnosis (`research/results/META2/META2_diagnosis.json`) found that its LightGBM model
memorised noise: in-sample AUC 0.82-0.90 against 0.48-0.52 out of sample. A pick's 5-day return spreads about
8.5% around its week's average, and no input explains more than a sliver of that. Three inputs had the same
sign against the excess return (pick minus its week's mean) in almost every year from 2017 to 2025:

- `ret5` (smaller drop is better): positive in 8 of 9 years, mean rank correlation +0.040.
- `vol20` (calmer stock is better): negative in 9 of 9, mean -0.029.
- `n_8k` (fewer 8-Ks in the 5 sessions to t is better): negative in 9 of 9, mean -0.024.

Each has a prior outside this data: Chan (2003) and R1's own split (drops without news reverse), the
low-volatility effect (Ang, Hodrick, Xing and Zhang 2006), and less extreme losers being more likely forced
selling than information. But these three were chosen by looking at all of 2017-2025, so any history test on
those years is biased upward. **The only test is forward.** The history number is reported as a hint.

- **Prediction written now:** about 60% this fails forward. The correlations are 0.02-0.04, so the edge,
  if real, is small next to the noise, and 30 weeks is a short test.

## Rule (no fitted parameters)

Each Friday, for R1's picks that week (`tracks/main/r1_live.py`):

1. `ret5`, as R1 logs it.
2. `vol20`: standard deviation (ddof 1) of the last 20 daily log returns of all-adjusted closes to t, within
   the current series (needs 21 bars).
3. `n_8k`: 8-K and 8-K/A filings by the stock's CIK with filing date from the 4th session before t through t
   (EDGAR daily form indexes; SEC's current ticker-to-CIK map).

Score = mean of three within-week percentile ranks (average ranks for ties): `ret5` ascending, `vol20`
descending, `n_8k` descending, so a higher score is a better pick. A missing input counts as the middle rank
(0.5). **Keep** a pick when its score is at or above the median score of that week's picks.

The live job logs the inputs, score and keep flag with each week's R1 record (best effort; when Friday's
EDGAR index isn't published by the run, `n_8k` is logged as missing). At judgement, all three inputs are
recomputed from the same sources with data up to each t only, and those recomputed flags are what counts;
the live log is the check that nothing was changed.

## Forward test (PROTOCOL rule 20)

First week: the R1 record for Friday 2026-10-30. Judged once, at the first date with >= 60 trading days
since that week and >= 30 kept picks closed: per week, mean `fwd5` of kept picks minus mean `fwd5` of all
picks (R1's own entry and exit: next open to 5th close). **Pass:** mean > 0 with Newey-West t (4 lags) >= 2.0,
and the kept picks' own mean after the 0.2% round trip > 0. Same-week random control: the same number of
picks drawn with numpy default_rng(20261024 + week ordinal), reported. A pass makes R1's book hold only kept
picks from the next Friday; a fail ends META3. No change to the rule before judgement.

## Reported alongside (not tests)

The history hint (2017-01 .. 2025-09, every week of R1's picks, same rule and test statistic), per year;
each of the three inputs alone (keep top half by that input); the top-5-by-score variant.
