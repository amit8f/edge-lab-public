# NX2: NEWS-XGB version 2, the news-only effect (news track)

Registered 2026-10-23, before NX2's models are trained and before any NX2 return is computed. NX1's
walk-forward was read before this registration (that is why NX2 exists), so NX2 is a new idea, counted.
Phase-2 count at registration: 37 (news 5). Phase 1 + phase 2 = 52. NX1 keeps running as registered.

## Why a version 2

NX1's walk-forward (`research/results/NX1/NX1_walkforward.json`) fired on no-news placebo stocks as often
as on real news in every test year, and its top input by gain was 20-day volatility. Its label (top and
bottom 20% of raw 3-day market-adjusted returns) rewards picking jumpy stocks, news or not. M10 and P3 had
the same flaw (V5: "a volatility screen"). NX2 removes it by design rather than hoping the model avoids it.

## Mechanism (PROTOCOL rule 10)

As NX1: traders under-react to company news for a few days, most of all at small stocks and for news that
is hard to read (Tetlock 2007; Tetlock, Saar-Tsechansky & Macskassy 2008; Ke, Kelly & Xiu 2019). The bet
is on the news, so a stock with no news must score zero.

## Question

Does the part of an XGBoost prediction that comes from the news inputs (the prediction with the news minus
the same model's prediction for the same stock with no news) pick 3-day long and short trades that beat a
same-night random pick from the same news, after costs, and beat the same model fed the wrong night's news?

## Events, universe, entry and exit

NX1's, unchanged: nightly Benzinga batches, single-symbol headlines, the same 32 inputs, entry at the next
open, exit at the third session's close, the same universe and costs. NX2 reads NX1's cached event files
(the same real and placebo events); it adds no new data.

## Label

z = (NX1's 3-day market-adjusted return) / (vol20 x sqrt(3)), where vol20 is the 20-day standard deviation
of daily log returns known at d (NX1's input). Events with vol20 missing or 0 are dropped. z is clipped to
[-6, 6]. A jumpy stock's ordinary move is now an ordinary z.

## Models (fixed settings, no tuning; seed 20261023)

XGBoost regressors, depth 4, 400 trees, learning rate 0.05, subsample 0.8, column sample 0.8, min child
weight 50, objective `reg:pseudohubererror` (slope 1).
1. Baseline B (price only): trained on the placebo (no-news) events of the training years; inputs the 6
   stock and 2 market inputs; target z. It learns what a stock in that state does without news.
2. News model N: trained on the real events of the training years; inputs all 32 NX1 inputs; target
   z - B(x).
3. News effect e(x) = N(x) - N(x0), where x0 is x with the 23 news inputs at NX1's no-news values (novelty,
   stock and market inputs kept). A placebo event has x = x0, so its effect is exactly 0.

## Picks

Fired: |e| at or above the 90th percentile of |e| over the training years' real events (fixed with each
fit). Each night: the top 3 fired events by e go long, the bottom 3 fired by e go short (fewer when fewer
fire). Control: as NX1 (same number of that night's real events at random, seeded, same sides).

## Historical gate (walk-forward, all before the 2025-10-01 holdout)

Test years 2019 through 2025 (January to September), each trained on all earlier years with the last 5
sessions before the test year dropped (NX1's scheme). Pooled over all test nights, both must hold:
1. Picks minus control, after costs, per night: mean > 0 and Newey-West t (5 lags) >= 2.
2. Shuffled-news test: within each test night, the 23 news inputs are permuted among that night's real
   events (seeded) and picks are made again with the same models. Real picks minus shuffled picks, after
   costs, per night: mean > 0 and Newey-West t >= 2. This shows the content of the news matters, not just
   which stocks had news.
Also reported, no rule: per year the same two numbers; the median vol20 percentile of picks within their
night; the Spearman correlation of |e| with vol20; importance by input group for N; NX1's placebo numbers
(the fire share is 0 for placebo events by construction).

If both hold, a $5,000 ledger shadow book starts on the next nightly run (NX1's book rules and nightly
timing, version 1 trained on 2016-10 .. 2025-09-24). If either fails, NX2 is reported as failed and gets
no book.

## Primary test (forward, read at 100 closed trades, and at 2027-12-31)

The shadow book minus its same-night random control, after costs, daily, Newey-West t (5 lags). Pass:
t >= 2 with a positive mean, and the shuffled-news test on the same forward nights with a positive mean.
Then the book moves to its own paper account under SYS1's promotion rule. Fail: dropped.

## Prediction (written first)

Scaling by volatility removes most of NX1's apparent signal. The literature's effect is a few basis points
a day, smaller than costs at liquid stocks. I expect gate 1 to fail (pooled t between 0 and 2) with
probability about 0.6; if it passes, the edge sits in small stocks and earnings or guidance news.
