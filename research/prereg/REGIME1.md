# REGIME1: a hidden-Markov market regime switch for NX3-ALL-HI

Registered 2026-11-24, before any number of this idea is computed. Source: Amit, 2026-11-24 ("for now do all 5", picking from his model-families list: GARCH, HMM regimes, survival models, Isolation Forest / CUSUM, quantile regression / NGBoost). One idea. Phase-2 count 180 -> 181 (phase 1 + phase 2 = 196).

## Mechanism (rule 10)

Risk control, not alpha. NX3-ALL lost about 4% on 2026-11-06 when a crash day loaded it with shorts (252 vs 11)
and the market bounced. If stormy markets can be told apart from calm ones as they happen, cutting size in storms
trims the worst days. Published (Hamilton 1989; Ang & Bekaert 2002; Nystrup et al. 2017); regimes are crowded
and noisy, so expect little.

## Rules

Base events: NX3-ALL-HI (prereg NX3ALLHI.md) as logged in `cache/nx1/nx3/trades_v2` (same models, seeds, confidence >= 1.5, uncapped, costs `N.cost_frac`), 2019-01 to 2025-09. Daily bars from Alpaca SIP (`adjustment=all` for returns, `raw` for prices). Choose years 2019-2023; one look at 2024-01 to 2025-09; nothing from 2025-10-01 on (holdout). Every input uses data known before the entry.

A 2-state Gaussian HMM (`hmmlearn`, fixed seed 7) on SPY daily log return and SPY 20-day realised volatility,
fitted once on 2016-01 to 2022-12. Each day uses only the forward-filtered probability of the high-volatility
state p, from data up to the prior close. Rule: when p > 0.5, NX3-ALL-HI's day gross is cut to 50% and no side
may exceed 60% of that day's gross (the largest trades of the heavier side by confidence are kept).

Simple baseline (must also be beaten): the same cut when SPY's 20-day realised vol is above its 2016-2022 80th percentile.

## Primary test (one look, 2024-01 to 2025-09)

Paired daily difference (REGIME1 minus plain NX3-ALL-HI, net, per dollar of plain gross): passes if the worst-1%-day
loss is smaller AND the mean difference is >= 0 with Newey-West t >= 1.5 AND the mean difference vs the simple baseline is
>= 0. Pass: the rule is added to a $5,000 ledger shadow book beside NX3-ALL-HI.

## Reported, no rule

Share of days in the storm state; state flips per year; per-year results; longs and shorts.

## Prediction (written first)

Storm days are few (about 15%), the cut trims the worst days but also some of the best rebounds. P(pass) about 0.2.

## Amendment 1 (2026-11-24, code details, before any result)

`tracks/news/models5_study.py`. Inputs: SPY daily log return and its 20-day standard deviation (annualised) from the
all-adjusted close. The storm state is the HMM state with the higher mean volatility. Day d uses the state filtered
through the close before d (a forward filter written out by hand, so no later day can leak in). When a storm day
trades only one side, the 60% side cap can't apply; that day is only cut to 50% gross. Worst day, gains and own net
are measured per dollar of plain NX3-ALL-HI's gross for that day.

## Result (2026-11-24, run 45382657914): FAILED

`research/results/MODELS5/REGIME1_result.json`. One look at 2024-01..2025-09 (421 days). The HMM called 33% of test
days stormy (the plain vol switch 8%). It did shrink the worst 1% day from -12.1% to -8.5% of plain gross, but it gave
up more on the good days: paired gain -0.06% a day (NW t -0.79), and -0.06% a day vs the plain vol switch. Own net per
dollar of gross: plain +0.11% a day (t 0.50), REGIME1 +0.05% (t 0.27). As predicted, the cut trims bad days and
rebounds alike; no shadow book.
