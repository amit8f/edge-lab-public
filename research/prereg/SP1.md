# SP1: spin-offs after the forced selling (main track)

Registered 2026-10-30, before any result of this idea was computed. Source: the books review's shelf
(2026-10-23; Greenblatt, "You Can Be a Stock Market Genius"). Amit picked it on 2026-10-30 ("NX2 log + 2
ideas"). One idea: phase-2 count at registration 68 (main 31). Phase 1 + phase 2 = 83.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** The parent's holders get shares they never chose. Index funds that don't hold the
   new company's index, and managers who find it too small or outside their mandate, sell in the first
   weeks whatever the price. The new company's managers are often paid in its stock, so they want it to
   do well (Cusatis, Miles and Woolridge 1993).
2. **Why us.** Partly. Small spin-offs are too small for a large fund to buy a meaningful position, and
   a $5k account can buy them once the forced selling is over.
3. **Published.** Yes, since 1993 and in a popular book since 1997. Later studies find the excess smaller
   and concentrated in small spin-offs and in years with takeovers. Expect it to be weak.

## Question

After costs, do new spin-off stocks bought one month after they start trading beat same-day random $5+
stocks over the following year?

## Universe and dates

- W1's cache (`tracks/news/w1_data.py`), as in EAP1: Alpaca SIP daily bars, raw for price and dollar
  volume, all for returns; >15-day gaps end a series; zero-volume bars dropped; delisted stocks kept; CIK
  from SEC's point-in-time ticker map. Filings come from W1's EDGAR submissions table (forms and filing
  dates, 2015-10-01 .. 2025-09-30).
- **Events.** A company whose first Form 10-12B (or 10-12B/A) is filed on f0 >= 2015-10-01. Its listing
  day L is its first daily bar in W1's cache on or after f0 and within 365 days of it, and the company may
  have no bar in the 365 days before f0, so it's a new listing. One event per company.
- **Entry day.** The 22nd session counting L as the first, entered at the open. The stock must be in the
  main universe on the session before entry (raw close >= $5, 20-day median raw dollar volume >= $2M).
- Entries from 2016-01-01. No exit may reach 2025-10-01.

## Trade

- Long at the entry day's open, held 250 sessions, out at the close. A series that ends early (a
  takeover, a delisting) exits at its last close. No stop. Costs are main's 0.2% round trip.
- **Control.** Five stocks from the main universe on the session before entry, from companies with no
  Form 10-12B in the 3 years before, drawn with `numpy default_rng(20261031 + entry ordinal)`. Same entry
  and exit. The control return is the mean over those with an entry bar.

## Primary test

Each event's net return minus its control's net, grouped into monthly cohorts by entry month. Mean,
Newey-West t (11 lags, since the one-year holds overlap) and a 95% block bootstrap (12-month blocks,
10,000 draws, seed 20261031).

Passes only when PROTOCOL rule 12 holds (HAC t >= 3.0, bootstrap CI above 0, picks' own mean net > 0, at
least 60 events) and rule 11 holds (FF5 + momentum alpha of the cohort difference on factors summed over
each cohort's 12 months > 0 with t >= 2). The deflated Sharpe is reported with 83 trials. A failure that
meets rule 19 gets a near-miss forward book. Otherwise the idea closes. With 60-200 events, the result
will be thin, and that is reported plainly.

**Hand check before any return is read.** Draw 20 random events and check each one on EDGAR and in a
published news item: the company was spun off by a parent distributing its shares, the symbol and CIK
match, and L is its first day of regular trading (when-issued trading not counted). At least 19 must be
right. A miss in the matcher may be fixed and checked again on history (rule 23), and it's logged.

## Exploratory (labelled exploratory now)

- A 125-session hold.
- Entry at L + 1 (no wait).
- Below versus above the median dollar volume at entry, and results by year.

## Prediction (written first)

A small positive difference of +2% to +5% over the year, concentrated in the smaller half. HAC t below 2
with this few events, so it probably fails.
