# M10 report: short-term boom model, main track

Pre-registration: `research/prereg/M10.md` (commit aca22c6, before any result). Screen run
38680039626 on 2026-10-15. Phase-2 count 7 (main 6, penny 1); phase 1 + phase 2 = 22.

## Verdict: FAILED

| | Mean 21-day excess after 0.2% | Newey-West t (6 lags) | 95% block-bootstrap CI | Weeks |
|---|---|---|---|---|
| **Primary: XGBoost top 20** | **+0.55%** | **0.51** | -1.63% .. +2.60% | 348 |
| Random 20 (control) | -0.25% | -1.65 | -0.54% .. +0.06% | 348 |

The bar was mean > 0, t >= 3 and CI lower bound > 0. The median week was -1.17%: a few huge winners
(MARA, RIOT, AMC, NVAX in their spikes) carry the mean.

## What the model actually learned

It finds stocks that are about to move a lot, not stocks that are about to rise. Out-of-sample, 13.0%
of its picks landed in the month's top 2% (6.5 times chance), 13.2% rose 30% or more in 21 trading
days, and 9.8% fell 30% or more. Median pick return -1.9%, middle half -17% to +15%. Feature gain is
dominated by 63-day volatility, then 21-day volatility, distance from the 52-week low and price level.
That is a volatility screen, and high volatility on its own has no reliable reward.

## Funnel and QA

Panel: 4,796 common stocks named in Form 3/4/5 filings, 452 weekly dates (2017-01-06 .. 2025-08-29),
1,047,907 rows with a $1+ close and $1M median dollar volume, 60,371 rows dropped for no point-in-time
CIK; 1,384,080 single-symbol headlines, 288,194 analyst actions. Main universe 1,492 to 1,949 stocks a
week (767,879 rows). Label rate 2.03%. Walk-forward models for 2019 .. 2025, each trained only on
windows closed before its year started.

Largest 21-day returns are real events checked against raw prices (GME from $20.15 on 2020-12-24, AMC
May 2021, CCXI, MDGL). Universe mean +1.19% per 21 days against SPY +1.49%; the control sits at the
cost (-0.25% vs -0.2%). Truncated windows 0.2%.

## Exploratory (pre-listed; none is a finding)

| Variant | Mean | t | Share of picks in top 2% |
|---|---|---|---|
| LightGBM | +0.87% | 0.87 | 13.5% |
| Logistic | +1.22% | 1.07 | 13.2% |
| Blend of the three | +1.28% | 1.14 | 14.0% |
| XGBoost without news/analyst | +0.65% | 0.61 | 13.1% |
| XGBoost regression on return rank | +0.24% | 0.54 | 4.4% |
| Top 10 | +1.58% | 1.29 | |
| Top 50 | +0.50% | 0.54 | |
| Versus SPY | +0.24% | 0.20 | |
| Bottom 20 as a short (0.2% + 3%/yr borrow) | -0.22% | -0.46 | |
| Same picks, 5-day horizon | -0.18% | -0.62 | |

Per year: 2019 +1.8%, 2020 +6.7%, 2021 -3.6%, 2022 -1.5%, 2023 -0.4%, 2024 +0.0%, 2025 +0.8%. News
and analyst features add almost nothing (with: +0.55%, without: +0.65%).

## What happens next

M10 is not paper traded. As pre-registered, the refit model's current picks go on the dashboard in a
"High risk, experimental" section that states this verdict and the volatility numbers above.
