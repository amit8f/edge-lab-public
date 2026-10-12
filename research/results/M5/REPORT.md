# M5 screen result: fails (no measured edge)

Run 2026-10-12 on GitHub Actions (run 38272994386), code at commit bd25a8c, pre-registration
`research/prereg/M5.md` committed earlier at bcbc568. Counts at reporting: phase 2 = 3 registered
(main 3; M4 and M5 tested, M6 pending); phase 1 + phase 2 = 18.

## Primary test (as pre-registered)

Top 20 by insider buying + buyback score minus the equal-weight eligible universe, 6 months, after the
0.2% round trip:

| | Mean per 6 months | HAC t (5 lags) | Block-bootstrap 95% CI | Dates |
|---|---|---|---|---|
| M5 top 20 | +2.54% | 0.72 | -2.87% to +10.83% | 99 |
| Random 20 (control) | -0.31% | -0.52 | -1.45% to +0.96% | 99 |

**Verdict: the screen fails** (needs t >= 2 and a CI above zero). Per the stop rule, M5 ends here and
the dashboard keeps M4's "no measured edge" label.

The mean flatters the score. The median date's excess is -0.7%, the picks beat the universe on 46% of
dates, and five dates from Jul-Nov 2020 carry the whole mean (GME's +1,600% in Aug/Sep 2020 picks
alone). Without Aug-Oct 2020 the mean excess is -0.3%. Per year: 2017 -2.2%, 2018 -0.1%, 2019 -7.7%,
2020 +27.0%, 2021 +5.8%, 2022 +0.3%, 2023 -1.0%, 2024 -1.6%, 2025 (Q1) +1.0%.

## Funnel

Same universe as M4 (symbol-dates passing $5 / $300M / $2M: 167,304), then net issuance defined:
160,603. Universe per date 1,445 .. 1,811 (median 1,622). DERA purchase rows (code P, acquired,
officer or director, original Form 4): 232,367 across 7,249 issuers. 26.8% of universe symbol-dates had
any insider buy in the prior 182 days; every pick had one.

## Data QA

- Largest raw purchase rows are filer typos (prices of $150,000 to $6,000,000 per share on
  small-company Form 4s). None reached the universe: the largest insider-buy / market-cap ratios in the
  universe are KALA 2020 ($120M on a $324M cap, 37%) and KOD 2022 (22%), both real director-affiliated
  fund purchases in offerings. Because the score uses ranks, a typo could only move a stock up the
  order, not dominate a mean.
- **Data bug found (net issuance).** 48 of 1,980 pick-rows have share-count log growth below -0.4, and
  some are impossible: MGPI, WTW and WWD at about -6.9 (= 1/1000) and NNBR at -13.4 (about 1/1,000,000).
  These are XBRL scale errors (cover-page shares reported in thousands or millions in one filing), and
  CRC 2017 (-2.2) looks like a reverse split the split correction missed. The same bug affects M4's net
  issuance family, so M4's exploratory "net issuance alone +4.1%" is contaminated too. It cannot rescue
  M5: the affected rows had ordinary 6-month returns (-25% .. +170%), and the result's sign depends on
  GME, whose issuance (-0.44) is real (2019-2020 buybacks). The fix (drop share records >= 100x away from
  the median of the company's filings) goes into the code for the live dashboard and future ideas;
  M5 is not re-scored, because a fail that a fix could only move by a fraction of its CI stays a fail.
- Random control near zero, so no sign of a selection bug.

## Exploratory (pre-listed; none is a finding)

| Variant | Mean excess per 6 months | HAC t |
|---|---|---|
| Insider buying alone | +1.7% | 0.91 |
| Number of buyers instead of dollar value | +2.7% | 0.98 |
| 12-month horizon | +10.1% | 1.28 |
| Top 50 | +1.1% | 0.50 |
| Vs SPY instead of the universe | +0.8% | 0.16 |

Calibration by score percentile (median 6-month return, share beating SPY): below 50th 1.8% / 41%,
50-80th 3.9% / 44%, 80-90th 4.1% / 45%, 90-95th 3.4% / 43%, 95-98th 4.0% / 45%, 98-99th 3.8% / 45%,
top 1% 2.9% / 44%. Above the median the score does not sort returns.

## What this means for the dashboard

Nothing changes. Insider buying, alone or with buybacks, showed no edge large enough to measure on
2017-2025 US stocks over $300M, so the dashboard won't call a stock a buy because insiders bought it.
