# M4 screen result: fails (no measured edge)

Run 2026-10-12 on GitHub Actions (run 38262504058), code at commit b21f6ed, pre-registration
`research/prereg/M4.md` committed earlier at e7927c7. Counts: phase 2 = 1 (main 1); phase 1 + phase 2 = 16.

## Primary test (as pre-registered)

Top 20 by composite minus the equal-weight eligible universe, 6 months, after the 0.2% round trip:

| | Mean per 6 months | HAC t (5 lags) | Block-bootstrap 95% CI | Dates |
|---|---|---|---|---|
| M4 top 20 | +1.20% | 0.94 | -1.35% to +3.80% | 99 |
| Random 20 (control) | -0.52% | -1.00 | -1.43% to +0.65% | 99 |

**Verdict: the screen fails** (needs t >= 2 and a CI above zero). Per the stop rule, M4 is not paper
traded; the dashboard may show the ranking only labelled "no measured edge".

## Deviation from the plan (data, not method)

The plan named 112 signal dates from 2015-12-31. Alpaca's SIP history starts in 2016, so no stock had
the 253-bar lookback before 2017-01-31: the test ran on the 99 dates 2017-01-31 .. 2025-03-31. Nothing
else changed. One pick in 99 x 20 had no next-day bar and was dropped, not replaced.

## Funnel

Alpaca us_equity assets 33,569 (active + inactive) -> common stock 6,278 -> named in a Form 3/4/5 by
2025-03 4,641 -> with SIP bars 4,601. Symbol-dates with a full lookback 305,902; unmapped to a CIK on
the date 21,728; with market cap 256,445; passing $5 / $300M / $2M dollar volume 167,304; with >= 4
factor families 163,628. Universe per date: 1,455 .. 1,861 (median 1,636). Family coverage: momentum
and volatility 100%, quality 99.7%, value 99.97%, net issuance 96.5%, asset growth 99.7%.

## Data QA

- 6-month return percentiles: 1% -61%, median +2.9%, 99% +106%, max +1,797% (GME, Sep 2020).
- The 20 largest 6-month returns are all real moves (GME 2020, IMMU's Gilead buyout, PLUG, NVAX,
  VKTX, IONQ 2024 ...); the 10 largest losses are real collapses (QTTB 2024, CORZ 2022 bankruptcy,
  KOD 2022, VAL 2020). No split artefacts.
- Equal-weight universe +5.4% per 6 months vs SPY +7.4%: consistent with equal weight lagging cap
  weight over 2017-2025.
- Series cut short (delisting or gap): 1.3% of returns.
- Random control near zero, so no sign of a selection bug.

## Exploratory (pre-listed; none is a finding)

| Variant | Mean excess per 6 months | HAC t |
|---|---|---|
| 12-month horizon | +2.5% | 0.98 |
| Top 50 | -0.0% | -0.03 |
| Vs SPY instead of the universe | -0.5% | -0.38 |
| Momentum alone | +1.7% | 0.50 |
| Low volatility alone | -3.1% | -1.25 |
| Quality alone | +2.1% | 1.38 |
| Value alone | +2.2% | 0.72 |
| Net issuance alone | +4.1% | 1.10 |
| Asset growth alone | +4.8% | 1.31 |

Per year (mean excess): 2017 -1.5%, 2018 +2.5%, 2019 -1.3%, 2020 -7.9%, 2021 +8.3%, 2022 +3.9%,
2023 +3.8%, 2024 +3.1%, 2025 (Q1 only) -3.4%. Composite deciles are flat (decile 1 -1.5%, deciles
3-10 between -0.0% and +0.4%): the score barely sorts 6-month returns.

Calibration used on the dashboard: top-20 picks returned a median +5.6% over 6 months (middle half
-5.1% to +16.8%) and beat SPY 47% of the time; SPY's median 6-month return over the same windows was
+8.4%.

## What it means

An equal mix of six well-known factors did not beat the average eligible stock with any reliability
over 2017-2025, and it lost to SPY. Net issuance and asset growth are the strongest single families
here, but they are exploratory and would need their own pre-registration and fresh data.
