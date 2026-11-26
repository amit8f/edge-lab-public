# LIT1: setups that "light up" when a few conditions are all on at once (main track)

Registered 2026-11-26, before any LIT1 number exists. Source: Amit, 2026-11-26: "check on some parameters that
when they align together we can predict if the stock worth buying / selling at like 70% accurate ... multiple types
of that ... 0-5 a day on average ... some criterias that applies on certain stuff so we can have high probability of
succeeding", then his answers Q1-Q6: Q1 a stop, and "if we are sure it will go up every a certain time to check it
and update the take profit"; Q2 "it can include everything it just need those certain stuff to be on to lit it up";
then "I dont know if once a day maybe once every 2-3 hours"; Q3-Q6 yes (find 2019-21, choose 2022-23, test once 2024-Sep 2025; pass at 60% wins with 70% reported; long and
short, main first; one idea for all families). One idea: phase-2 count 191 -> 192 (phase 1 + phase 2 = 207).

Mechanism (rule 10). 1. Other side: traders who react to one signal at a time (a gap, a headline, a volume spike)
when the combination says more; or nobody, if the combinations are just noise mined from history. 2. A fund can run
the same search on far more data; any combination that survives is likely small, short-lived or in names too small
for size. 3. Rule mining is published and known to overfit (data-snooping: White 2000, Sullivan, Timmermann and
White 1999); the search below uses three separate periods for that reason. Prior: low (about 10%). PICK-MAIN, a
high-confidence picker, failed on 2024-Sep 2025 (33% hit +3%, -0.44%/trade).

## Universe and data (point in time)

- PICK main universe: previous raw close >= $5 and 20-day dollar volume >= $5M (W1 daily rows, SIP, `adjustment=all`
  for returns, raw for price and dollar volume); delisted stocks kept; a >15-day gap ends a series.
- Feature day t uses data known at the 16:00 close of t: daily bars through t; Benzinga headlines of session t
  (16:00 t-1 to 16:00 t) and the last 5 sessions; SEC filings with filing date <= t (W1 filings).

## Conditions (each one is "on" or "off" for a stock on a day)

Continuous signs, each cut into four conditions at its 2019-2021 pooled 10th, 25th, 75th and 90th percentiles
("<= p10", "<= p25", ">= p75", ">= p90"):
- chart: return today, 5, 20, 60 days; close vs 20-day high and low; close vs 250-day high; 20-day volatility;
  today's range / close; volume vs its 20-day median; today's gap and open-to-close; where the close sits in the
  day's range; up or down days in a row; log price; log 20-day dollar volume;
- news: headline count today and over 5 sessions; mean headline tone (stage 5's word list); trading days since the
  last headline (capped at 60);
- market and sector: the universe's equal-weight return today and 5 days; its 20-day volatility; share of stocks up
  today; the stock's FF12 sector return today and 5 days (stock left out); stock minus sector today.
Flags: headline of each SECTOR1 type today (earnings, guidance, analyst, deal, fda, offering, contract); earnings
window (8-K item 2.02 filed t-1 or t); other 8-K; Form 4 filed; offering filing (S-1/S-3/424B); Monday; Friday;
turn of month; day before a market holiday; monthly options expiry.
Fed, CPI and jobs days are not used (our release-date file stops at 2023).

A setup is 1 to 3 conditions that are all on, with a side (long or short). It "lights up" on the first day all its
conditions are on after a day when they were not.

## Exits

- Search and choice (fixed, fast): entry at the next open; stop and take-profit each one sigma from the entry
  (sigma = the stock's 20-day standard deviation of daily returns, from day t); daily high and low decide, the stop
  first when both are touched on the same day, a gap through the stop fills at the open; out at the close of the
  10th trading day.
- Test (Amit's Q1 and his follow-up "maybe once every 2-3 hours", the trading rule): entry at the next open; stop
  one sigma away, live all the time on 5-minute SIP bars (`adjustment=raw` for fills, a gap through it fills at that
  bar's open); a take-profit check level one sigma away, looked at three times a day, at 11:30, 14:00 and 15:55 ET:
  if the price is beyond it and the setup is still on (its conditions as of the last close; at 15:55 the day's
  conditions with the 15:55 price as the close), the check level moves to one sigma beyond that price and the stop
  to one sigma behind it (never back); if the price is beyond it and the setup is off, exit at that price. Out at
  15:55 of the 10th trading day. Shorts mirror. 5-minute bars are fetched only for the stocks and days in a trade.
  The once-a-day version (check at the close only, on daily bars) is reported beside it, not deciding.
- One open trade per stock at a time. A win is a trade with net > 0. Costs: config/risk.yaml main round trip
  (0.20%); shorts ignore borrow in history (as NX3), reported apart.

## Find on 2019-2021, choose on 2022-2023

1. Score every single condition, every pair, and every triple made by adding one condition to the best 300 pairs
   (ranked by the lower 95% bound of their win rate), each long and short, on 2019-2021 light-ups. Keep setups with
   at least 150 trades, win rate >= 60% and mean net > 0.
2. On 2022-2023 take the 500 best kept setups (by that lower bound) and keep those with at least 60 trades, win
   rate >= 60% and mean net > 0. From these pick up to 10 families greedily by their 2022-2023 lower bound,
   skipping any whose trades overlap an already picked one by more than half (Jaccard of stock-days).
3. No family left: LIT1 fails without opening the test period.

## Primary test (one look, 2024-01-01 to 2025-09-30)

All picked families together, the three-checks-a-day exit, one trade per stock at a time (the family with the higher 2022-2023
bound takes the stock). Pass, all of:
1. at least 60% of trades win (net > 0); 70% is Amit's goal and is reported;
2. mean net per trade > 0 with a day-clustered HAC t >= 3.0 (PROTOCOL rule 12) on the daily mean trade net;
3. at least 100 trades;
4. beats random: on each trade's day, a random universe stock with the same side and the same exit (seed 192,
   200 draws); LIT1's mean net above the 95th percentile;
5. FF5 + momentum monthly alpha > 0 with t >= 2 (rule 11).
Reported, not deciding: trades per day (goal 0-5), each family alone, the fixed exit, longs and shorts, each year,
the deflated Sharpe with 207 trials.

Pass: a $5,000 ledger shadow book (no orders), each family its own line; penny ($1-5, $1M/day) gets its own
amendment before any penny number. Fail: written up, no book. Holdout from 2025-10-01 untouched.

## Amendment 1 (2026-11-26, same day, before any LIT1 number exists): more free data

Source: Amit, 2026-11-26 ("check if you need more data from elsewhere"). Added conditions, all point in time:
- **Insider trades** from the SEC's quarterly Insider Transactions Data Sets (Forms 3/4/5, 2018Q4-2025Q3, free):
  per stock, open-market purchases (code P) and sales (code S) by officers and directors, counted by filing date
  (a Form 4 counts on day t when filed on or before t; Forms accepted until 22:00 keep their date, as CLAUDE.md
  says, so day t uses filings dated <= t only when the nightly job runs after 22:00, which the forward book does).
  Flags: any purchase filed in the last 5 sessions; purchases above $100k in the last 20 sessions; any sale in the
  last 5 sessions; net purchase dollars over 20 sessions >= p90 / <= p10 (cut on 2019-2021). This replaces "Form 4
  filed" (which only says a form exists).
- **Release days**: Fed statement day, the day before it, CPI day, jobs-report day, about day t+1 (as SECTOR1 stage
  5), from research/macro_dates_2019_2023.csv plus a 2024-2025 file from the same Fed and BLS archives, committed
  before the test is run. This replaces the sentence that left them out.
- **Cross-asset state**, daily from Alpaca SIP bars (`adjustment=all`): today's and 5-day returns of VIXY, USO, TLT
  and UUP, each cut at p10/p25/p75/p90 like the other continuous signs.
Not available free, so not used: analyst earnings-estimate history (FMP's free key returns only recent quarters),
option prices, short interest (FINRA's terms forbid machine-learning use).
