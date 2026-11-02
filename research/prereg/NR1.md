# NR1: buy big stocks that fell hard against their sector with no news, at the auctions (main track)

Registered 2026-11-02, before any NR1 data was pulled or computed. Source: Amit's "yes to all" (2026-11-02 09:34
UTC) to the shortlist in `/mnt/project-files/reports/No news edges for big stocks.md` (item 1). Phase-2 count
with NR1, TOM1 and DOD1 (same commit): 97 -> 100 (phase 1 + phase 2 = 115).

## Mechanism (PROTOCOL rule 10)

- **Counterparty:** sellers who need out with no new information (fund outflows, stop-outs, rebalancing). A drop
  with no news tends to bounce; a drop on news tends to keep going (Chan 2003). Low-turnover drops reverse more
  (Medhat & Schmeling 2022, large caps about 0.84% a month). Measuring the drop against the sector removes
  sector-wide moves, which carry information.
- **What's new against R1** (`prereg/R1.md`, t 1.89): R1 used neither the no-news nor the low-turnover filter,
  measured against the whole market, and paid a 0.41% round trip. R1's own side table showed losers with no 8-K
  earned +0.33% a week vs +0.03% with one (an exploratory hint, disclosed). NR1 enters and exits in the opening
  and closing auctions, which have no bid-ask spread.
- **Decay:** plain industry reversal faded after 2002 (Do & Faff). Prediction written now: about 70% this fails.

## Universe, signal, trade

- Day t: every trading day from 2017-01-03 to the last day whose exit is before 2025-10-01 (holdout untouched).
- Universe on t: the 500 largest stocks by market cap in M4's universe on t (common stock, delisted kept, raw
  close >= $5, CIK-mapped), as a point-in-time stand-in for the S&P 500.
- Sector: the SPDR sector ETF from the stock's SEC SIC code (`ch_data.sector_etf`; unknown codes use SPY).
- Residual: the stock's 5-day return (adjusted close t over t-5) minus its sector ETF's over the same days.
- No news: no Benzinga article tagged with the stock from the open of t-4 to the close of t, and no 8-K item 2.02
  (earnings) filed in that window.
- Low turnover: the stock's mean volume over t-4..t below its own median daily volume over t-64..t-5.
- Picks: the 5 lowest residuals among stocks passing both filters, if below -3% (ties by symbol). Fewer is fine.
- Trade: buy at the open of t+1 (opening auction), sell at the close of the 5th session counted from entry
  (closing auction). Adjusted prices. Equal weight.
- Cost: 0.05% round trip (auction fills pay no spread; this covers sale fees and auction slippage). Also reported
  at the 0.41% average spread.

## Control and primary test

Control: for each pick, a random other stock from the same day's universe (default_rng(20261102 + date
ordinal)), traded the same way. Per day: mean net of picks minus mean net of control; days with no pick skipped
and counted. Over days: mean, Newey-West t (5 lags), block-bootstrap 95% CI (blocks of 5 days, 10,000 draws).
**Pass** (rules 11-12): mean > 0, t >= 3.0, CI lower bound > 0, picks' own mean net > 0, at least 300 trades and
100 days, and FF5 + momentum alpha t >= 2. Rule 21 doesn't apply (nothing is fitted). Deflated Sharpe with 115
trials. A near-miss goes to rule 19; a pass becomes a DAY1 sleeve after rule 20 forward numbers.

## Reported alongside (not tests)

All residual losers (no filters) vs control; each filter alone; the R1-style market-relative version; per year;
the 5% best and worst trades; the quick-kill check from the research (filtered group minus all-losers, kill below
+0.1% a week).

## Data check (rule 14)

20 random picks checked by hand against an independent SIP pull and Benzinga's site (no article in the window,
the residual and the turnover filter) before any return is read; 18 of 20 needed.
