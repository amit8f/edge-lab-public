# BOOM1 (step 1 of the boom picker): narrow the market to the 20 stocks most likely to move 8%+ during the day

Registered 2026-11-06, before any BOOM1 number is computed. Source: Amit, 2026-11-06 ("a formula ... narrowing it
down to a much smaller amount of stocks ... the best stocks we think are going to boom that day, long or short ...
step by step, testing, learning, fixing, adjusting"). One idea. Phase-2 count 126 -> 127 (phase 1 + phase 2 = 142).

This step only tests the filter (which stocks will move a lot). It makes no trades and claims no profit. Step 2
(BOOM2, registered later and only after this result) picks long or short on the short list and is judged after
costs. A big-move filter matters because an 8% move dwarfs the ~0.55% round-trip cost that sank CHART1.

## Universe and data

The `chart_bars` stage's daily files (NX3TA.md): stock-days with raw close >= $5, 20-day median dollar volume
>= $1M, 20-day volatility known and a next market session in the same series, 2016-01-01 .. 2025-09-29 (moves end
by 2025-09-30; the 2025-10-01 holdout is untouched).

## Label

Boom = during the next session the stock trades at least 8% above or 8% below its open (all-adjusted high and low,
recovered from the daily bar), so a trade can take the move and get out the same day (Amit, 2026-11-06: 8%, and
"it booms and I want to be out"). Reported, no rule: the share that closed 8%+ from the open, and 15%+ moves.

## Inputs, all known before the next session opens

The 66 CHART1 inputs at day d's close (classic chart and volume indicators, 10-day bar shape, SPY's mood) plus the
next session's pre-open news from NX3's events (decision at or before the open): a news flag, article count,
strongest negative and positive FinBERT tone, mean word tone, most negative and most positive headline tone.
Scores and counts only, never headline text.

## Model and walk-forward

CatBoost classifier (500 iterations, depth 6, rate 0.05, L2 10). Test years 2019 .. 2025-09; each year trains on
up to 600,000 random stock-days ending 5 sessions before the year starts.

Each day: the 20 highest-scored stocks. Hit rate = share of the 20 that boomed.

Not run until Amit's boom research is done (Amit, 2026-11-06); any change from that research is made here before
the first run.

Baselines: VOL (the 20 most volatile stocks by vol20, the obvious simple rule), VOL_NEWS (vol20 doubled when
there is pre-open news), MAX (the 20 with the largest daily move in the last 20 days; recent big movers repeat, Bali,
Cakici and Whitelaw 2011) and RANDOM (20 at random), added 2026-11-07 from the boom research report
(/mnt/project-files/reports/Predicting big intraday stock moves.md) before any BOOM1 number exists.

## Primary test

Passes if the model's daily hit rate is at least 3 times the day's base rate (all stocks) AND the model minus VOL,
paired by day, has mean > 0 with NW (5 lags) t >= 2.0. Reported, no rule: the 15%+ rate, the share of booms that went
up, per-year hit rates, whether the model beats VOL every year, CatBoost's top 15 inputs.

## Prediction (written first)

Base rate about 3-5%. VOL's top 20 about 25-35%. The model beats VOL (pre-open news and gaps add a lot): about 0.7.
