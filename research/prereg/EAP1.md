# EAP1: the earnings announcement premium (main track)

Registered 2026-10-30, before any result of this idea was computed. Source: the books review's shelf
(2026-10-23); Amit picked it on 2026-10-30 ("NX2 log + 2 ideas"). One idea: phase-2 count at
registration 67 (main 30). Phase 1 + phase 2 = 82.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** Investors who buy stocks that are about to be in the news. Attention before an
   earnings release pulls in buyers, and the price rises in the days around the release whatever the
   news turns out to be (Frazzini and Lamont 2007; Barber, De George, Lehavy and Trueman 2013). Some of it
   is pay for carrying the release's risk (Savor and Wilson 2016), and that part doesn't go away.
2. **Why us.** Weak. Funds can trade this, and the weekly turnover costs everyone the same. The edge we
   might have is that we hold for one week only, when they hold for months.
3. **Published.** Yes, since 2007. Later work finds it smaller but still there. Expect a few basis points
   a day, so costs (0.2% a round trip) may well eat it.

## Question

After costs, do $5+ stocks expected to report earnings next week beat a same-week random $5+ stock that
is not expected to report, over that week?

## Universe and dates

- W1's cache (`tracks/news/w1_data.py`: Alpaca SIP daily bars, raw for price and dollar volume, all for
  returns; >15-day gaps end a series; zero-volume bars dropped; delisted stocks kept; CIK from SEC's
  point-in-time ticker map). It's the same pipeline PD1 and W1 use.
- Main universe on formation day t: raw close >= $5 and 20-day median raw dollar volume >= $5M, one row
  per company (highest dollar volume). W1's cache has no market cap, so the dollar-volume floor stands in
  for the main track's $300M cap.
- Earnings events: 8-Ks listing item 2.02 in SEC's submissions data, with their acceptance time (PD1's
  code, `tracks/penny/pd1_study.earnings_rows`), day 0 being the first session whose close comes after
  acceptance. One event per company per 30 calendar days.
- Formation days: the last session of each week, from 2017-01-06 on. No exit may reach 2025-10-01.

## Signal and trade

- **Expected to report next week.** On formation day t, a company is expected to report when it had an
  earnings event with day 0 in (t - 364, t - 357] calendar days. That is the same 7 days, exactly 52 weeks
  earlier. It uses only year-old dates, so there's no look-ahead.
- **Picks.** Every expected company in the main universe on t.
- **Trade.** Long at the next session's open, held 5 sessions (the entry day counts as the first), out
  at the close. A series that ends early exits at its last close. Costs are main's 0.2% round trip
  (`config/risk.yaml`).
- **Control.** The same number of stocks from t's main universe with no earnings event with day 0 in
  (t - 371, t - 350], drawn with `numpy default_rng(20261030 + t ordinal)`. Same entry and exit.

## Primary test

Weekly cohorts: picks' mean net return minus controls' mean net return. Mean, Newey-West t (4 lags) and a
95% block bootstrap (4-week blocks, 10,000 draws, seed 20261030).

Passes only when PROTOCOL rule 12 holds (HAC t >= 3.0, bootstrap CI above 0, picks' own mean net > 0,
at least 2,000 picks in all) and rule 11 holds (monthly sums of the weekly difference: FF5 + momentum
alpha > 0 with t >= 2). The deflated Sharpe is reported with 83 trials. A failure that meets rule 19 gets
a near-miss forward book. Otherwise the idea closes.

**Hand check before any return is read.** Draw 20 random picks. For each, check the year-old 8-K on
EDGAR (it lists item 2.02, it was accepted at the logged time, and it belongs to the company trading
under that symbol on t). At least 19 must be right.

## Exploratory (labelled exploratory now)

- A 10-session hold.
- Picks that really did report in the hold week, against those that didn't. This uses the actual
  date, so it's only a diagnostic of the forecast.
- Dollar-volume terciles, and results by year.
- The pre-registered same-week diff over the days -2..+1 around the actual day 0 (diagnostic).

## Prediction (written first)

Picks beat controls by about +0.1% to +0.3% a week before costs. After 0.2% costs the picks' own mean
is near zero, so the screen probably fails rule 12 on its own mean even if the difference is positive.
