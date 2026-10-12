# M6 screen result: fails, but the closest so far

Run 2026-10-12 on GitHub Actions (run 38273572603), code at commit f379176, pre-registration
`research/prereg/M6.md` committed earlier at d9077c0. Counts at reporting: phase 2 = 3 (main 3; M4, M5,
M6 all tested, none passed); phase 1 + phase 2 = 18.

## Primary test (as pre-registered)

Top 20 by net analyst actions over 90 days minus the equal-weight universe, 6 months, after the 0.2%
round trip:

| | Mean per 6 months | HAC t (5 lags) | Block-bootstrap 95% CI | Dates |
|---|---|---|---|---|
| M6 top 20 | +3.01% | 1.61 | -0.57% to +6.83% | 99 |
| Random 20 (control) | -1.22% | -2.21 | -2.26% to -0.11% | 99 |

**Verdict: the screen fails** (t below 2 and the CI includes zero). Per the stop rule, M6 ends and the
dashboard does not change.

Unlike M4 and M5, the result does not rest on a few outliers. The median date's excess is +3.2%, the
picks beat the universe on 72% of dates (the control: 38%), and the score sorts returns in order:
median 6-month return rises from 2.0% (below the 50th percentile) to 3.7%, 4.0%, 4.4%, 4.8%, 5.2% and
5.6% (top 1%). Per year: 2017 +5.8%, 2018 +3.1%, 2019 +9.9%, 2020 -4.0%, 2021 -1.1%, 2022 +2.1%,
2023 +7.7%, 2024 +2.6%, 2025 (Q1) -5.7%.

## Control check

The control's t of -2.21 includes the 0.2% cost, so its expected value is -0.2%, not 0. Against that,
it is -1.0% with t about -1.85. Random 20-stock baskets also miss the universe's rare huge winners
(GME 2020 and similar), which pushes their typical excess below the mean. M4's and M5's controls
(-0.52%, -0.31%) used other universes and were near zero. I read this as noise from that skew, not a
selection bug, but it is the largest control so far and should be watched.

## Funnel and classification QA

Benzinga articles 2016-10 .. 2025-09: 1,725,517. Single-symbol articles that the rules classify:
289,045; after removing repeats within 3 days: 288,194 (154,727 upgrades or target raises, 133,467
downgrades or cuts). 68% of universe symbol-dates had at least one action in the prior 90 days.

The 50 printed sample headlines are all classified correctly by the pre-registered rules (for example
"Credit Suisse Downgrades Five Below After 63% Rally" is -1, and "Morgan Stanley Maintains Underweight
on Roku, Raises Price Target to $50" is +1 because the rules count target raises regardless of the
rating). No bug fix and no rerun.

## Exploratory (pre-listed; none is a finding)

| Variant | Mean excess per 6 months | HAC t |
|---|---|---|
| Rating changes only | +2.2% | 1.89 |
| 30-day window | +2.7% | 1.72 |
| 12-month horizon | +7.0% | 2.00 |
| Top 50 | +2.5% | 1.67 |
| Vs SPY instead of the universe | +1.3% | 0.91 |

## What this means

M6 fails its own bar and stays off the dashboard. It is the first idea in this phase with a consistent,
ordered effect rather than a mean carried by outliers, and every variant points the same way. Testing
it again on the same 2017-2025 data would be fitting to noise. A new pre-registered forward test
(paper picks logged monthly from now, judged only on data after today) is the honest next step if the
owner wants it.
