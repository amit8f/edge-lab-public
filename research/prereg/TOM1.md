# TOM1: hold SPY over month ends and into market holidays (main track)

Registered 2026-11-02, before any TOM1 data was pulled or computed. Source: Amit's "yes to all" (2026-11-02 09:34
UTC) to the shortlist in `/mnt/project-files/reports/No news edges for big stocks.md` (item 2). Count 97 -> 100
with NR1 and DOD1 (115 with phase 1).

## Mechanism (PROTOCOL rule 10)

Pay-day and pension inflows arrive around month ends, and traders thin out before holidays; both have pushed index
prices up on those few days (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008). **Counterparty:** nobody
in particular; this is a timing premium from flows. **Decay:** widely known since the 1980s; prediction written
now: about 60% that it fails the gate.

## Amended 2026-11-02, before any data was pulled

Alpaca's SIP bars start in 2016, so SPY alone gives about 200 events, under the 300 needed. The primary series is
now the US market's daily total return from Ken French's library (Mkt-RF + RF, close to close, dividends included),
2000-01-03 to the last exit before 2025-10-01 (about 540 events). SPY from Alpaca (2016-01-04 on) is the
tradability check: SPY's own mean net over its events must also be > 0. Everything else below is unchanged.

## Events and trade (see the amendment for the period)

- Month end: buy SPY at the close of the second-to-last trading day of the month, sell at the close of the 3rd
  trading day of the next month (4 sessions held).
- Pre-holiday: buy SPY at the close two sessions before each NYSE full-day holiday, sell at the close of the
  session just before the holiday (1 session held). An overlap with a month-end window is counted once, as a
  month-end event.
- Closing-auction fills: cost 0.02% round trip (sale fees and auction slippage; SPY's spread is about 0.01%).
  SPY uses adjusted closes (`adjustment=all`, so dividends count).

## Primary test

Control: 10,000 placebo sets, each drawing for every event a random window of the same length starting on a
non-event day of the same calendar year (default_rng(20261102)). Event excess = event net return minus the mean net
return of all same-length windows in that year. Over events: mean excess, Newey-West t (3 lags), and the placebo
p-value (share of placebo sets whose mean is at least the events' mean). **Pass:** mean > 0, t >= 3.0, placebo
p < 0.01, events' own mean net > 0 (market and SPY), at least 300 events. Rule 11 (factor alpha) does not apply to the market
itself. Fewer than 300 events (expected about 540) or t between 1.5 and 3 with own mean > 0 is a rule 19 near-miss.
Pass or near-miss -> a DAY1 sleeve after rule 20 forward numbers, using the bot's MOC orders.

## Reported alongside

Month-end and pre-holiday separately; SPY and QQQ the same way (2016 on); per year; 2000-2012 vs 2013-2025.
