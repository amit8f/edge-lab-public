# M13 result: the rising 150-day average rule (Micha.Stocks) FAILED

Pre-registration: [prereg/M13.md](../../prereg/M13.md). Data: `M13_screen.json`, `M13_monthly.csv` (study run 38884794719).
Running idea count at this result: phase 2 = 20, phase 1 + phase 2 = 35 (33 at registration).

## Primary verdict (as pre-registered)

Each month-end, hold every eligible stock above a rising 150-day average (about 840 stocks on average)
and compare it with the whole eligible universe over the next month, after 0.2% x turnover.

| | Value |
|---|---|
| Mean, allowed minus universe, after costs | **+0.08% a month** (gross +0.12%) |
| HAC t (3 lags) | **0.66** (bar: 3.0) |
| Bootstrap 95% CI | -0.13% .. +0.31% |
| Alpha on FF5 + momentum | +0.11% a month, t 1.28 (bar: 2.0) |
| Deflated Sharpe (33 trials) | 0.06 |
| Months beating the universe | 61% |
| Worst month | -5.3% (2018-12) |
| Replication (Ken French momentum and market) | passes (corr 0.96 and 0.99) |

**FAILED.** The prediction written before the run (+0.0% to +0.3% a month, t under 3, no alpha) was right.
The rule mostly picks stocks the momentum factor already holds (momentum beta 0.20).

Funnel: 197,515 eligible stock-months on 115 dates; 108 dates have the 171 bars the rule needs (Alpaca
SIP starts in 2016, so the first months can't be scored). The pre-registration said 116 dates; one date
dropped out of M4's panel and the first seven lack history. This is a data limit, not a choice.

Control: a same-size random set lost -0.09% a month after costs (t -3.4). That is its cost: a random set
turns over 53% a month against the rule's 21%, at 0.2% per round trip. Gross it was +0.01%.

## Data QA

MA150 recomputed from raw bars for 5 random stock-dates: all match. Largest held monthly returns are real
events (GME Dec 2020 +1,611%, CNST, DDD, NVAX). With ~840 stocks held, one stock moves the mean by about
0.1% of its return.

## POST-HOC / NOT PRE-REGISTERED (exploratory list, never findings)

| Variant | Mean a month | t |
|---|---|---|
| 200-day average | +0.09% | 0.71 |
| Above 150-day, not requiring "rising" | 0.00% | 0.01 |
| 20-day average | -0.34% | -2.65 |
| Allowed minus not allowed | +0.27% | 1.37 |
| 3-month hold | +0.35% | 1.20 |
| Top 25 by distance above the average | +1.27% | 1.22 |
| Fama-MacBeth with momentum: rule coefficient | +0.17% | 0.93 |

Share of big losers (down 20%+ in the month): 2.8% of allowed stocks against 3.6% of the universe, so the
rule does avoid some crashes, as predicted, without improving the mean. The 20-day version (Micha's method 2)
loses money after costs. Per year the effect flips sign (best 2021 +0.43%, worst 2022 -0.20%).

M13 ends here (stop rule). Nothing goes to the dashboard or paper trading.
