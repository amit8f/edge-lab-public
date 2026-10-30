# INS1: buy after opportunistic insider purchases (main track)

Registered 2026-10-30, before any INS1 number is computed. Source: Amit's grill 2026-10-30 round 2 Q5 (a slow
multi-week book from free SEC Form 4 data), after the research report
`/mnt/project-files/reports/Live strategies and adaptive exits.md`. One idea, counted (count in NXE.md).

## Mechanism (PROTOCOL rule 10)

Cohen, Malloy & Pomorski (2012): insiders who trade at irregular times ("opportunistic") know something; their
buys earned about 0.8% a month (1986-2007), while routine insiders' trades earned about zero. The seller of
the stock is a market that hasn't read the filing. Evidence ends in 2007, so this may have decayed.

## Events

- Data: SEC's quarterly Insider Transactions Data Sets (Form 4, non-derivative table), 2013 Q1 to 2025 Q3.
- Routine insider: traded (any code P or S) in the same calendar month in each of the 3 years before; an
  insider with trades in all 3 prior years but not routine is opportunistic; others are unclassified (skipped).
- Event: an open-market purchase (code P) by an opportunistic insider worth at least $25,000, by an officer or
  director, at a US common stock that on the filing date is $5+ with 20-day dollar volume at least $1M
  (raw SIP bars). Several insiders at one company within 5 sessions are one event (the first filing).
- Entry: the open of the session after the filing date (filings count from their filing date; the nightly job
  runs after 10pm ET). Exit: the close 20 sessions later. Label dates 2017-01 to 2025-08 so exits end before the
  2025-10-01 holdout.

## Test

Control: 5 random $5+ stocks (dv20 >= $1M) the same day, same holding. Monthly cohorts, NW t (11 lags), costs
from config/risk.yaml on $500 positions. Pass: rule 12 (t >= 3, CI > 0, own mean > 0, at least 200 events) and
rule 11 (FF5 + momentum alpha t >= 2); near-miss (rule 19): t >= 1.5 and own mean > 0, which gets a forward
shadow book. Hand check first: 20 random events, each checked against the Form 4 on EDGAR (insider, code P,
value, filing date); at least 19 right before any return is computed.

## Prediction (written first)

The effect shrank after publication (McLean & Pontiff) and the $5+ filter drops the small caps where it was
strongest: about +0.3% a month over control, t near 1. Probability of passing rule 12: about 0.1; near-miss:
about 0.3.
