# META1: a take-or-skip model for R1's picks (main track)

Registered 2026-10-24, before any of this idea's features were pulled or any model was fit. Amit's pick
(2026-10-24: "only picking the stock with the best confidence to win and not every stock that passed some
filter"; Plan v3 section 13, META). Phase-2 count at registration: 43 (main 21, penny 7, news 7,
cross-track 7, crypto 1). Phase 1 + phase 2 = 58.

## Idea

R1 (`prereg/R1.md`) buys up to 20 of last week's biggest losers that are still above their 200-day average.
Its screen missed the t >= 3 bar (picks minus control +0.29% a week, t 1.89) and it now runs as a FWD1
shadow book. META1 does not choose stocks. It looks at each R1 pick and says take or skip, so the book
holds only the picks most likely to beat the week's average. This is meta-labeling (Lopez de Prado,
*Advances in Financial Machine Learning*, ch. 3): a second model learns when the first one is right.

- **Mechanism:** R1 is paid for buying from forced, hurried sellers (Nagel 2012). Some drops are that;
  others are news that keeps pricing in (Chan 2003: drops with news keep falling, drops without news
  reverse). R1's own result already hints at it: picks with no 8-K in the 5 days earned +0.33% a week,
  picks with one +0.03%. A model that sees how violent, how liquid and how news-driven each drop was may
  tell the two apart.
- **Prediction written now:** about 65% this fails. The label is noisy (a 5-day return), and 8,790 picks
  is a small set to learn from.

## Data

- Primary picks: R1's picks as committed in `research/results/R1/R1_picks.csv` (role "pick", 8,790 picks,
  week-ends 2017-01-06 .. 2025-09-19), with R1's `fwd5` (adjusted exit close over adjusted entry open,
  minus 1, before costs).
- Inputs come from Alpaca SIP daily bars (`adjustment=all` for returns, `raw` for price and dollar volume),
  delisted stocks kept, fetched in ~500-symbol batches, plus SPY and EDGAR's filing index. Every input uses
  data up to and including the signal day t only (filings by filing date <= t). A >15-day gap ends a series.
- Holdout: nothing from 2025-10-01 on is used (R1's picks already stop before it).

## Label

For each pick in week w: 1 when its `fwd5` is above the mean `fwd5` of all of R1's picks that week, else 0.
So the model learns which picks beat their own week, not which weeks were good for the market.

## Inputs (18, fixed now)

From the stock's own series at t: 5-day return; its rank among that week's picks; 1-, 20-, 60- and
250-day returns (the 250-day one skips the last 20 days); 20-day volatility of daily returns; 5-day over
20-day volatility; the 5-day drop in units of 20-day volatility; distance from the 52-week high close;
distance above the 200-day average; log of 20-day median raw dollar volume; log of market cap (R1's
`mcap`); day-t volume over its 20-day median; number of 8-Ks filed in the 5 trading days to t. Market:
SPY's 5- and 20-day returns; the number of R1 picks that week.

## Model and walk-forward

LightGBM binary classifier with fixed settings, no tuning: 200 trees, learning rate 0.03, 15 leaves,
at least 100 rows per leaf, 80% feature and row sampling, seed 20261024. Expanding window: for each test
year Y from 2019 to 2025 (2025 = Jan .. Sep), train on all weeks before Y minus the last 2 weeks
(embargo, as the 5-day labels overlap), test on year Y's weeks.

**Rule:** take a pick when its predicted probability is at or above the median prediction on that
model's own training rows (so about half are taken; the cut is fixed by training data only).

## Primary test

Per test week (2019-01 .. 2025-09): mean `fwd5` of the taken picks minus mean `fwd5` of all picks that week
(weeks where nothing is taken count as 0). Over all test weeks: the mean, a Newey-West t (4 lags) and a
95% block-bootstrap CI (blocks of 4 weeks, 10,000 draws, seed 20261024). Costs cancel (same 0.2% round
trip on both sides), so the taken picks' own mean after 0.2% costs is reported next to it.

**Pass to a forward shadow book** (PROTOCOL rule 19, the near-miss lane, since META1 is a new rule under
Plan v3): mean > 0 with t >= 1.5, and the taken picks' own mean after costs > 0. Out-of-sample AUC is
reported. A failure ends META1; any retry is a new, counted idea.

Same-week random control: each week, the same number of picks drawn at random from that week's picks
(numpy default_rng(20261024 + week index)); its mean minus all picks is reported and should be near 0.

## Forward (if it passes)

A final model trained on all weeks to 2025-09 is frozen and versioned. Each Friday the R1 book
(`tracks/main/r1_live.py`) logs META1's probability and take/skip for every pick. Judged once (PROTOCOL
rule 20), at the first date with >= 60 trading days and >= 30 taken picks closed: weekly taken-minus-all
after costs > 0 with Newey-West t >= 2.0, and the taken picks' own mean after costs > 0. A pass makes the
R1 book hold only taken picks from the next Friday on. No retraining before that judgement; a retrained
model is a new version and its own registration.

## Reported alongside (not tests)

Results per test year; the top-5-by-probability variant (as FWD1's most-confident books); input
importance (gain) per model; the taken share; the same numbers split by market cap ($300M-$2B vs larger).

## Data QA (equity-data-qa)

Funnel: picks, picks with full inputs, drops with reasons. Min, max and percentiles of every input; the 20
largest |fwd5| picks checked against raw bars; any input that can't be computed point-in-time is dropped
and logged before the walk-forward runs.
