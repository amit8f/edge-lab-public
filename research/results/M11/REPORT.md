# M11 report: calculated-risk boom picks, main track

Pre-registration: `research/prereg/M11.md` (before any result). Screen run 38760815234 on 2026-10-15
(the first run, 38733624895, stopped on a cache without headlines and produced no result). Count at
registration: phase 2 13 (main 9), total 28. Running count at this report: phase 2 16 (main 10),
phase 1 + phase 2 31.

## Verdict: FAILED

| | Mean 21-day excess after 0.2% | Newey-West t (6 lags) | 95% block-bootstrap CI | Hit rate | Weeks |
|---|---|---|---|---|---|
| **Primary: M10 top 100, best 20 on quality/value/buybacks** | **+0.51%** | **0.76** | -0.85% .. +1.81% | 52% | 348 |
| M10 top 20, same weeks | +0.55% | 0.51 | | 43% | 348 |
| Random 20 from the same top 100 (control) | +0.56% | 0.76 | | 47% | 348 |
| Random 20 from the universe (control) | -0.05% | -0.40 | | 48% | 348 |

The bar was mean > 0, t >= 3 and CI lower bound > 0. The filter picked no better than a random draw
from the same 100 boom candidates, so the quality, value and buyback scores added nothing to the mean.

## What the filter did do

It cut the tails on both sides. Picks that fell 30% or more in 21 days dropped from 9.8% (M10) to
5.0%, and picks that rose 30% or more dropped from 13.2% to 9.0%. The median week moved from -1.9%
to -0.2%. That is a lower-risk version of the same volatility bet, not an edge. Each filter alone
(quality only, value only) and the wider pools (top 50, top 200) were all at t 0.1 to 0.4
(exploratory). By year: 2019 -1.0%, 2020 +3.1%, 2021 -0.7%, 2022 0.0%, 2023 +1.1%, 2024 -0.6%,
2025 +2.3%; the mean leans on 2020 and 2025.

## QA

348 weeks (2019-2025 walk-forward), 601,988 universe rows, SEC fundamentals for 98% of rows (96% inside
the top-100 pool), never fewer than 20 picks. Largest pick returns checked against known events: PCG
2019-01-18 +154% (rebound after its bankruptcy filing), TDS 2023-07-14 +152% (sale review), SMCI
2024-01 +132%, UPST 2023-05 +135%. Largest losses SM and PUMP in February 2020 (oil and COVID crash).
None looks like a data error.

## What follows

M11 ends here. It goes into V2's re-judging (factor-adjusted, luck-adjusted) with the other finished
screens and stays on the dashboard as a failed high-risk idea.
