# SD1: does accounting alpha survive year-old data on our free SEC data? (main track)

Registered 2026-10-24, before any SD1 result exists. Source: the weekly research scan
(`/mnt/project-files/research-scan/2026-10-24.md`, pick 3); Amit: "add all". Phase-2 count at
registration: 53 (main 27). Phase 1 + phase 2 = 68.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** As the published accounting anomalies: investors who underreact to profitability,
   issuance and investment until later news corrects the price.
2. **Why us.** Weak: these are big-fund factors. SD1 is mainly a check of our data path: if alpha
   survives a 12-month lag, our strict filing-date rule (slow but point-in-time) costs little.
3. **Published.** Fan and Li, "Stale Data, Persistent Alpha" (SSRN 7500938, posted 2026-10-19): across
   94 accounting anomalies, median monthly CAPM alpha 36 bps at a 4-month lag and 28 bps with year-old
   data.

## Data

M4's month-end panel and SEC companyfacts path (`prereg/M4.md`: original 10-K values counted from their
filing date). Signals: M4's four accounting families (quality: gross profitability and ROA; value; net
issuance; asset growth), each a within-date percentile rank, composite = their mean. Two versions per
month-end t:
- **Fresh:** the latest 10-K filed by t (M4 as is).
- **Stale:** the latest 10-K filed by t minus 12 months.
Dates 2017-01 .. the last month-end whose 1-month label closes before 2025-10-01.

## Primary test

Monthly long-short quintile of each composite (equal weight, 0.2% round trip per side), FF5+momentum alpha
with Newey-West t (3 lags). **Replication holds** if Fresh's alpha > 0 with t >= 2.0 and Stale's alpha is
at least half of Fresh's. If Fresh's alpha has t < 2.0, the result is "nothing to replicate on our data"
(reported as a fail). SD1 trades nothing; a trading use would be a new registration under rule 12.

Reported alongside: CAPM alpha (the paper's measure), a same-month random control, per-year results.

Prediction: ~65% "nothing to replicate" (M4's composite had no edge as a top-20 book).

## Clarification (2026-10-29, before any SD1 result; code `tracks/main/longterm/screen_sd1.py`)

- Panel: RL1's month-end rows (M4's price features and filters, next-month return) with M4's companyfacts
  path, so Fresh is M4's composite on the same rows. M4's filters are applied once, at t, on Fresh's market
  cap; Stale is scored on the same stocks.
- Stale takes every accounting input as of t minus 365 days (10-K values, share counts for issuance). E/P
  and B/M put the year-old earnings and book equity over t's market cap, so only the accounting is stale,
  not the price.
- A stock-month needs at least 3 of the 4 families; the composite is the mean of those it has.
- CAPM alpha is the same regression on Mkt-RF only, Newey-West 3 lags. The random control draws two random
  quintiles per month (seed 20261024 plus the date).
