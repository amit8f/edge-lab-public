# CHART1: a chart-and-volume-only daily stock picker, five model families

Registered 2026-11-06, before any CHART1 number is computed. Source: Amit, 2026-11-06 ("do one that is not the 37
inputs that is mainly graph and bar related stuff ... on different models that fit it best"). Five ideas, one per
model arm (the reversal rule is the baseline, not counted). Phase-2 count 116 -> 121 (phase 1 + phase 2 = 136).
Daily reversal and momentum results from M-series screens and the literature were seen first, so this can only
reject; a forward book decides.

## Question

Can a model that sees only a stock's daily bars and volume (no news, no filings) pick next-day winners and losers
well enough to beat random picks after costs, and which model family suits chart data best?

## Universe and data

The `chart_bars` stage's files (see NX3TA.md): every symbol that ever had a NX3 news event 2016 .. 2025-09 (this is
almost every US stock that traded with real volume; delisted ones included) and SPY. A stock-day counts if the raw
close >= $5, the 20-day median raw dollar volume >= $5M, 20-day volatility exists and the next market session's bar
exists in the same series. Days 2016-01-01 .. 2025-09-29 (the trade ends by 2025-09-30; nothing from the
2025-10-01 holdout).

## Inputs (66), all known at the close of day d

The 32 classic inputs of `core/chart_features.py` (the 28 in NX3TA.md plus r1, r5, r20, vol20, log dollar volume,
log price), the last 10 days' bar shape (daily return / vol20, high-low range / close, volume / 20-day average) and
SPY's r1, r5, distance from its 200-day average and 20-day volatility.

## Trade and label

At the close of d: score every stock. Buy the 3 highest at the next session's open, short the 3 lowest at the same
open, close both at that session's close. Return from all-adjusted open and close; costs = NX3's (`cost_frac`, one
day, spread both sides at the raw open price, borrow on shorts). Label for training: (stock return - SPY return over
the same open-to-close) / vol20, clipped at +-6.

## Arms

- CAT: CatBoost, 400 iterations, depth 6, learning rate 0.05, L2 10, Huber loss.
- LGBM: LightGBM, 400 trees, 31 leaves, learning rate 0.03, 500-row leaves, Huber loss.
- ET: Extra Trees, 200 trees, max_features 0.33, 300-row leaves, 30% bootstrap, median-filled.
- RIDGE: ridge regression (alpha 100) on standardised, median-filled inputs (a linear model).
- SEQ: a small neural net (64-32 units) on only the 10-day bar shape plus SPY's 4 inputs: does the raw recent
  candle pattern carry anything without hand-made indicators?
- REV (baseline rule, not a model): buy the biggest 5-day losers and short the biggest winners (r5 / vol20).

## Walk-forward

Test years 2019 .. 2025-09. Each year trains on stock-days up to 5 sessions before the year starts, 400,000 rows
sampled at random (seed fixed). Control: the same number of random stocks from that day's universe, same sides.

## Primary test (history)

Per model arm: passes if pick-minus-control mean > 0 with NW (5 lags) t >= 2.0 AND the picks' own mean net return
per day (after costs) > 0. Five arms are tested, so one pass at t just above 2 is weak evidence; the report shows all
five and the forward book decides.

Reported, no rule: long side and short side net separately, each arm minus REV, per-year numbers, CatBoost's top 15
inputs. Exploratory: top 10 + bottom 10 instead of 3 + 3; long side only.

## Forward book

An arm that passes gets a $5,000 ledger shadow book (no orders; no Alpaca account is free), judged by PROTOCOL
rule 20.

## Prediction (written first)

Most likely result: picks beat random (extreme scores catch short-term reversal) but the 3 + 3 picks are the most
volatile stocks, where spreads eat the edge, so net per day <= 0. Probability any arm passes: about 0.25. Best family
expected: CatBoost or LightGBM; SEQ the weakest.
