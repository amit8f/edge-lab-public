# PD1: post-earnings drift in penny stocks (penny track)

Registered 2026-10-28, before any result of this idea was computed. Source: Amit asked for a sweep of new
ideas on 2026-10-28 and said "add all of those"; this was pick 3. Phase-2 count at registration: 56 (penny 10).
Phase 1 + phase 2 = 71.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** Holders of thin, ignored $1-5 stocks who under-react to an earnings release: few
   analysts, slow retail readers, and sellers who anchor on the old price. The price keeps drifting in the
   direction of the first reaction for weeks (Bernard and Thomas 1989).
2. **Why us.** The drift that survives is in microcaps, where a fund can't put meaningful money: positions
   of a few hundred dollars in stocks trading about $1M a day are too small for them and too big a share of
   volume for them to enter without moving the price.
3. **Published.** Yes, for decades. Martineau (2022) finds it gone in non-microcaps since 2006.
   Subrahmanyam (2025, UCLA Anderson Review, 2026-01-21) finds it in all US stocks from 2001 to 2024 at
   t 2.18 only because of microcaps (t 1.43 without them), measured from the price reaction rather than the
   accounting surprise. Expect it to be weak; our earlier earnings-style penny tests were not of this form.

## Question

After costs, do liquid $1-5 stocks whose price jumped most on an earnings release keep beating a same-day
random penny stock over the next 60 trading days?

## Universe and dates

- Penny universe on each event date: raw close from $1 to under $5, median 20-day dollar volume >= $500k,
  at least 60 bars, NYSE/Nasdaq/NYSE American, one row per company (highest dollar volume); delisted
  stocks kept. Alpaca SIP daily bars, `adjustment=raw` for price, dollar volume and fills,
  `adjustment=all` for returns. A >15-calendar-day gap ends a series; zero-volume bars dropped.
- Events: 8-Ks listing item 2.02 (results of operations) in SEC's submissions data, dated by acceptance
  time (core/sec.py). Day 0 is the first session whose close comes after acceptance (an 8-K accepted after
  4pm ET reacts the next session). One event per company per 30 calendar days (the first).
- Event dates 2017-01-01 on. No exit may reach 2025-10-01 (holdout).

## Signal and trade

- Reaction: the stock's return from the close before day 0 to the close of day +1, minus IWM's over the
  same window (`adjustment=all`).
- A reaction is "top" when it is at or above the 80th percentile of all reactions in the 365 calendar days
  before day 0 (only earlier events, so no look-ahead; at least 100 earlier events, else no signal).
- Long every "top" event at the open of day +2, hold 60 sessions, exit at the close. No stop.
- Costs: `config/risk.yaml` penny round trip (2%).
- Control: for each pick, one penny stock from the same day's universe with no item 2.02 8-K in the 30
  calendar days around day 0, drawn with `numpy default_rng(20261028 + day-0 ordinal)`, same entry and exit.

## Primary test

Monthly cohorts by entry month: mean over cohorts of (picks' mean net return minus controls' mean net
return), Newey-West t with 3 lags (the 60-day holds overlap about 3 months). Passes only under PROTOCOL
rule 12 (HAC t >= 3.0, bootstrap 95% CI resampling months above 0, picks' own mean net > 0, at least 300
events) and rule 11 (monthly alpha on FF5 + momentum > 0 at t >= 2). Deflated Sharpe with 71 trials.
A failure that meets PROTOCOL rule 19 gets a near-miss forward book; otherwise it closes.

Hand check before any return is read: 20 random events, each 8-K's acceptance time and item 2.02 checked
on EDGAR against the event table, and the day-0 session against the acceptance time; at least 19 right.

## Exploratory (labelled exploratory now)

The short side (bottom 20%, easy-to-borrow only where Alpaca's list allows, borrow 20%/year), holds of 20
and 40 sessions, the reaction measured over day 0 only, and the result by year and by price band
($1-2, $2-5).

Prediction: about 25% pass (the published t is barely 2 and our cost of 2% a round trip is heavy).

## Clarification (2026-10-29, before any PD1 event or return is built; code `tracks/penny/pd1_study.py`)

- Data: W1's cache (`tracks/news/w1_data.py`): its daily bars, CIK map and universe flag (raw close >= $1,
  median 20-day dollar volume >= $500k), with raw close under $5 added. W1's stock list is SEC's insider
  data sets' tickers on Alpaca's exchange list, so the exchange rule is Alpaca's.
- Events: Form 8-K only (an 8-K/A is a correction, not the release) listing item 2.02, from each penny
  company's SEC submissions record. Its acceptanceDateTime carries a "Z" but is EDGAR's ET clock; the hand
  check compares it with each filing's EDGAR index page.
- The penny universe is read on the last session before day 0 (known before the reaction starts).
- One event per company per 30 calendar days, counted from the acceptance time of the last kept event.
- Events from 2015-10 are built so the 365-day reaction percentile has history; signals start 2017-01-01.
