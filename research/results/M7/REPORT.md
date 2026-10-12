# M7 screen result: fails (no edge)

Run 2026-10-12 on GitHub Actions (run 38293955035), code at commit bc09cb6, pre-registration
`research/prereg/M7.md` committed earlier at ed4dd85. Counts at reporting: phase 2 = 4 (main 4; M4-M7
all tested, none passed); phase 1 + phase 2 = 19.

## Primary test (as pre-registered)

| | Mean per 6 months | HAC t (5 lags) | Block-bootstrap 95% CI | Dates |
|---|---|---|---|---|
| M7 top 20 by SUE | -0.66% | -0.58 | -3.23% to +1.29% | 99 |
| Random 20 (control) | +0.32% | +0.52 | -0.90% to +1.52% | 99 |

**Verdict: the screen fails.** The picks beat the universe on 43% of dates (median date -1.3%). Per
the stop rule, M7 ends and the dashboard does not change.

## Funnel

Universe as M4 (167,371 eligible symbol-dates). Companies with quarterly net income: 2,840; quarters:
131,701; with a defined SUE: 108,579. 94.5% of universe symbol-dates had a SUE filed in the prior 120
days. Universe per date 1,353 .. 1,821. SUE values filed after their signal date: 0 (checked in code).

## Data QA

- SUE percentiles in the universe: 1% -6.4, 5% -2.5, median 0.1, 95% 2.8, 99% 5.8. Picks: median SUE
  6.5; 10% of picks above 20, max 902.
- The largest |SUE| values are one-off accounting events, not data errors: Q4 2017 tax-reform gains,
  Q1 2020 impairments (one company: -$10.2B net income in a quarter), and early-history quarters with
  tiny prior variation. Net income (as pre-registered) includes these items, so the top 20 is partly a
  list of one-time gains, which don't repeat. This is a design weakness of the registered signal, not a
  bug, and is not fixed by a rerun.
- Random control near zero.

## Exploratory (pre-listed; none is a finding)

| Variant | Mean excess per 6 months | HAC t |
|---|---|---|
| 12-month horizon | +2.0% | 0.78 |
| Top 50 | +0.7% | 0.71 |
| Vs SPY | -2.4% | -2.16 |

SUE deciles are flat (decile 1 -0.8%, decile 10 +0.6%, nothing in between ordered). Per year: 2017
-1.4%, 2018 -0.3%, 2019 -1.7%, 2020 -4.5%, 2021 -5.2%, 2022 +1.6%, 2023 +3.6%, 2024 +1.8%, 2025 +2.9%.
The filing-window reaction variant was not computed: the pipeline does not cache daily bars around
filing dates. It is listed as exploratory only and no result depends on it.

## What this means

Raw net-income surprises from SEC filings did not pick winners over 2017-2025. The published version
of this effect uses analyst consensus or EPS before one-off items, which this free data doesn't give
cleanly. Of the four ideas tested, analyst actions (M6) remains the only one with a consistent sign.
