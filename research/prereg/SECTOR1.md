# SECTOR1: a descriptive study of how stocks move, sector by sector (stage 1: daily data)

Registered 2026-11-19, before any number for it is computed. Source: Amit's 2026-11-16 request to understand "why each
stock goes up or down, by sector" before building more models; the six scoping answers were accepted on 2026-11-19.
**Descriptive only: no trading rule, no primary test, no p-values.** It is logged in IDEAS.md but is not counted as an
idea (count stays 170, 185 with phase 1). Any rule that follows from it is registered as a new idea on independent data.

## Question
For each sector, how large and how predictable-looking are daily moves, how much of them is the market or the sector,
how much happens overnight, and what do cost and liquidity look like?

## Data and groups
- W1's cached daily rows (Alpaca SIP, raw for universe flags, all-adjusted for returns, delisted kept, >15-day gaps end a
  series, zero-volume bars dropped). Dates 2019-01-01 to 2023-12-31 only; nothing from 2024 on is read.
- Universe: PICK main's: raw close >= $5 and 20-day median dollar volume >= $5M on the previous day.
- Sector: Fama-French 12 industries from the company's EDGAR SIC code (`core/industry.py`; the current code, not
  point-in-time: stated limit). Stocks with no CIK or SIC fall in "Other".

## Measures, per sector and per year (2019..2023) and pooled
1. Stock-days, distinct stocks, median price and median 20-day dollar volume.
2. Close-to-close return (adjusted): mean absolute value, median absolute value, 95th percentile of absolute value,
   share of stock-days with |return| >= 5% and >= 8%, and the up share among those.
3. Overnight gap (adjusted open over previous adjusted close) vs intraday (adjusted close over adjusted open): mean
   absolute value of each and the overnight share of their sum.
4. Co-movement: R-squared of the stock's daily return on the same day's equal-weight universe return, and on its
   sector's equal-weight return (the stock excluded from its own sector mean); median over stocks.
5. Own-news days: the share of |return| >= 8% days on which the universe return was within +-1% and the sector's
   equal-weight return (stock excluded) within +-2%. ("Stock-specific big days"; a proxy, not a news count.)
6. Cost view: median (high-low)/close, and cost from `config/risk.yaml` against the sector's median absolute intraday move.

## Not in stage 1 (a dated amendment will register them before their results exist)
Earnings days, news counts and types, rates and oil, and time of day. They need EDGAR 8-K dates, the Benzinga cache and
5-minute bars; stage 2 will use the same sectors and years.

## Exploratory/limits (stated in advance)
Sector means hide dispersion; SIC codes are current, not historical; the equal-weight market is the universe, not SPY;
overnight gaps include splits only to the extent the adjusted series does. Cells with fewer than 30 stocks are flagged.

## Prediction (written first)
Health and Business equipment show the largest absolute moves and the highest share of stock-specific big days; Utilities
and Money the smallest, with the highest market R-squared; overnight is more than half of the absolute move in most sectors.
