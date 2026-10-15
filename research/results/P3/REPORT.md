# P3 report: short-term boom model, penny track

Pre-registration: `research/prereg/P3.md` (commit aca22c6, before any result). Screen run
38680039626 on 2026-10-15. Phase-2 count 7 (main 6, penny 1); phase 1 + phase 2 = 22.

## Verdict: FAILED

| | Mean 21-day excess after 2.0% | Newey-West t (6 lags) | 95% block-bootstrap CI | Weeks |
|---|---|---|---|---|
| **Primary: XGBoost top 20** | **-2.19%** | **-3.99** | -3.24% .. -1.07% | 348 |
| Random 20 (control) | -1.93% | -6.08 | -2.52% .. -1.29% | 348 |

Before costs the picks returned +1.49% per 21 days against +1.69% for the penny universe, so the model
did slightly worse than random even before the 2% round trip. Median week -2.87%; without each week's
single best pick the mean is -6.2%. Only 3.6% of picks landed in the top 2% (the label rate here is
2.4%), so the penny model barely beats chance at finding booms at all.

## Funnel and QA

Same panel as M10 (shared data, separate rows and models). Penny universe ($1 to under $5, $1M median
dollar volume, one row per company): 50 to 274 stocks a week, 57,767 rows, label rate 2.42%. The control
sits close to its cost (-1.93% vs -2.0%).

QA finding: WOLF on 2025-08-29 shows +2,264% in 21 days. That is Wolfspeed's bankruptcy reorganisation
(old shares exchanged), which the adjusted bars treat as one continuous series; it is not a tradable
gain. WOLF was not a pick that week, so it only lowers that one week's excess (by roughly 0.06% on the
348-week mean). It is left in, as the pre-registration fixed the data rules; it does not change the
verdict.

## Exploratory (pre-listed; none is a finding)

Every variant lost after costs: LightGBM -2.77% (t -4.82), logistic -2.61%, blend -2.66%, without news
-2.27%, regression -1.41%, top 10 -2.97%, top 50 -2.38%. The bottom 20 as a short also lost (-3.03%
after 2% and 20%/yr borrow, easy-to-borrow by today's list, which flatters it). The same picks over 5
days: -2.13% (t -9.5), because the 2% cost is paid on a much shorter hold. Every year was negative
(2019 -2.4% .. 2025 -4.3%).

## What happens next

P3 is not paper traded. As pre-registered, the refit model's picks are shown on the dashboard with
this verdict. The penny costs (2% a round trip) are the main obstacle: a monthly penny strategy needs
more than 2% a month of real edge before it breaks even.
