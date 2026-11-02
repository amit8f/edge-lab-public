# DOD1: buy listed companies after a big Pentagon contract that the news missed (main track)

Registered 2026-11-02, before any DOD1 data was pulled or computed. Source: Amit's "yes to all" (2026-11-02 09:34
UTC) to the shortlist in `/mnt/project-files/reports/No news edges for big stocks.md` (item 3). Phase-2 count with
NR1, TOM1 and DOD1 (same commit): 97 -> 100 (phase 1 + phase 2 = 115).

## Mechanism (PROTOCOL rule 10)

The Pentagon posts every contract of $7.5M or more around 5pm ET on the day it is signed. For a big prime
(Lockheed, RTX) one contract is noise; for a smaller listed company a contract worth 1% or more of a year's sales is
real news, and many of them get no headline, so the price adjusts slowly (limited attention: DellaVigna & Pollet
2009; the research report's DoD-award notes find the effect, where reported, sits in small recipients).
**Counterparty:** holders who don't read the daily contracts page.
**Decay:** the page is public and free; anyone with a scraper sees it. Prediction written now: about 75% this fails.

## Amended 2026-11-02, before any DOD1 data was pulled

Found while building it (probe run 41423470813): USAspending's search can't filter on the potential value, so
new awards are pulled when their first obligation is at least $1M, and each matched award's value is the larger
of that obligation and the base-and-all-options value on its USAspending record. An order whose first obligation
is under $1M is missed (disclosed). Companies are those with a raw close >= $5 and a 20-day median dollar volume
of at least $5M on some day (NR1's company-facts set, which holds the revenue). The daily contract pages refuse
cloud servers, so the hand check reads each award's USAspending record, the company's SEC revenue and Benzinga
instead of the page archive; the posting-date rule below is dropped (it needs the pages). Everything else is
unchanged.

## Events (2017-01-03 to the last exit before 2025-10-01)

- Source for history: USAspending.gov prime award transactions, awarding agency Department of Defense, new awards
  and new task/delivery orders only (modifications excluded), action date d. Value = base and all options value
  (closest to the "potential value" the daily page states).
- Recipient: the award's parent recipient matched to a US-listed common stock by exact cleaned name (suffixes like
  Inc/Corp/LLC removed) against SEC company names, then CIK to symbol as M4 does. Unmatched recipients are dropped
  and counted.
- Size: value >= 1% of the company's latest annual revenue in SEC XBRL facts with filing date <= d. Several awards
  to one company on one day are summed into one event.
- Price: raw close on d >= $5. Delisted kept.
- No news: no Benzinga article tagged with the stock from the open of d to the open of d+1.
- An event within 10 sessions of the same company's previous event is skipped (no overlapping trades).
- Timing proxy (history only, disclosed): the award is treated as public at 5pm ET on its action date. USAspending
  action dates can differ from the announcement date; the hand check below measures how often.

## Trade, control, test

- Buy at the open of d+1 (opening auction), sell at the close of the 10th session counted from entry (closing
  auction). Adjusted prices. Cost 0.05% round trip; also reported at the 0.41% average spread.
- Control: for each event, 5 random other $5+ stocks from the same day's M4 universe (default_rng(20261102 + date
  ordinal)), traded the same way.
- Per event: net return minus the mean net of its 5 controls. Events on the same entry day are averaged into one
  day. Over days: mean, Newey-West t (10 lags), block-bootstrap 95% CI (blocks of 10 days, 10,000 draws).
- **Pass** (rules 11-12): mean > 0, t >= 3.0, CI lower bound > 0, events' own mean net > 0, at least 300 trades and
  100 days, FF5 + momentum alpha t >= 2. Deflated Sharpe with 115 trials. Nothing is fitted, so rule 21 doesn't
  apply. A near-miss (including fewer than 300 events with t >= 1.5 and own mean > 0) goes to rule 19.
- Pass or near-miss -> a DAY1 sleeve fed by the forward logger after rule 20 forward numbers; the forward rule
  uses the value stated on the daily page and the page's posting time.

## Reported alongside (not tests)

Size buckets 1-3%, 3-10%, 10%+ of revenue; events with news vs without; 1-, 5- and 20-session holds; market cap
under and over $2B; per year; the 5% best and worst events.

## Data check (rule 14)

Before any return is read: 20 random events checked by hand against the daily contracts page archive (company
named, value within 10%, posting date vs action date) and Benzinga's site; 18 of 20 needed on company and value.
The posting-date gap is reported; if more than 4 of 20 were posted two or more business days after the action
date, entry moves to the open after action date + 2 business days, decided now, before any return.

## Forward logging

From 2026-11-02 the daily page is saved raw by `core/leads.py --dod`. war.gov refuses cloud IPs, so it runs from
Amit's PC (or any home connection) once a day after 6pm ET.
