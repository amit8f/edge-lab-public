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

## Stage 1 result (run 44481499966, recorded 2026-11-19)

Pooled 2019-2023, 3,440 stocks. "Other" (743 stocks) holds companies whose SIC matches no FF12 range or is missing.

| sector | stocks | mean abs move | 8%+ days | overnight share | market R2 | sector R2 | stock-specific 8% days |
|---|---|---|---|---|---|---|---|
| Health | 497 | 2.5% | 4.1% | 30% | 0.17 | 0.22 | 33% |
| Energy | 104 | 2.5% | 3.8% | 37% | 0.22 | 0.58 | 10% |
| Business equipment | 496 | 2.2% | 3.0% | 35% | 0.29 | 0.34 | 28% |
| Shops | 236 | 2.0% | 2.7% | 34% | 0.32 | 0.32 | 25% |
| Other | 743 | 2.0% | 2.6% | 35% | 0.31 | 0.30 | 28% |
| Consumer durables | 69 | 2.0% | 2.5% | 37% | 0.39 | 0.40 | 24% |
| Telecom | 48 | 1.9% | 2.2% | 32% | 0.27 | 0.28 | 28% |
| Chemicals | 71 | 1.8% | 2.0% | 36% | 0.38 | 0.42 | 22% |
| Manufacturing | 231 | 1.9% | 2.0% | 36% | 0.40 | 0.43 | 25% |
| Consumer nondurables | 100 | 1.8% | 1.9% | 33% | 0.29 | 0.30 | 29% |
| Money | 551 | 1.7% | 1.8% | 36% | 0.42 | 0.50 | 16% |
| Utilities | 94 | 1.4% | 1.1% | 33% | 0.27 | 0.50 | 16% |

By year, the 8%+ day rate swings 2x to 6x within every sector: 2020 and 2022 high, 2019 and 2023 low (Health 2.1%,
6.4%, 4.0%, 5.1%, 2.6%). Up share of 8% days is 50-54% everywhere. The cost (0.2% round trip) is 13-23% of a median
intraday move.

Prediction check: Health largest (right) but Energy, not Business equipment, second; Utilities smallest (right);
Money has the highest market R2 (right); overnight is a third of the move, not more than half (wrong).

## Amendment 1 (2026-11-19, before any stage-2 number exists): stage 2, what drives the moves

Still descriptive only, not counted. Same universe, sectors and years (2019-2023), same stage-1 stock-days.

Drivers, each marked per stock-day t:
- **Earnings window:** the company filed an 8-K with item 2.02 on day t or on the trading day before (a filing
  after the close moves the next day). From W1's EDGAR filings (filing date only, no time).
- **News:** at least one Benzinga headline naming the symbol from 16:00 ET on the previous trading day to 16:00 ET
  on day t (W1's headline months). Count kept.
- **News type**, by fixed keyword lists on those headlines (case-insensitive; a day can have several):
  earnings (earnings, EPS, revenue, results, quarter), guidance (guidance, outlook, forecast, raises, lowers, cuts),
  analyst (upgrade, downgrade, price target, initiates, reiterates), deal (acquire, acquisition, merger, to buy,
  buyout, takeover, deal), fda (FDA, approval, trial, phase), offering (offering, priced, dilution, private placement,
  warrants), contract (contract, award, partnership, agreement).
- **None:** no earnings window and no news.
- **Rates and oil:** daily change of the 10-year Treasury yield (FRED DGS10, percentage points) and daily % change of
  WTI crude (FRED DCOILWTICO), same calendar day.

Measures per sector, per driver group (pooled and per year): share of stock-days, mean absolute return, share of
|return| >= 8%, the up share of those 8% days, mean return, and the share of all the sector's 8% up days and 8% down
days that fall in the group (what drives up moves vs down moves). For rates and oil: correlation and beta of the
sector's equal-weight daily return to each, and the sector's mean return on the 10% of days with the largest rise and
the largest fall of each.

Time of day needs 5-minute bars for every stock-day and is left for a stage 3 (registered before it runs).

## Stage 2 result (2026-11-19, run 44598868569; descriptive, not counted)

Bug fix before reading any number: the first run flagged no earnings days because W1 stores 8-K items as lists;
`earn_days` now reads both lists and comma strings (test added). Filings were rebuilt from EDGAR because the W1
cache had been evicted. Summary JSON: `research/results/SECTOR1/SECTOR1_stage2.json`.

- **Earnings windows** are about 3% of stock-days but hold 8-20% of each sector's 8%+ days (Consumer nondurables,
  Business equipment, Shops ~20%; Utilities, Energy, Money ~8-10%). About one earnings day in 8 is an 8%+ move
  (Business equipment 19%, Utilities 3%). Their direction is a coin flip (47-53% up).
- **News days** (any Benzinga headline) are 15-22% of stock-days and hold 35-58% of 8% up days and 42-64% of 8%
  down days. Down moves lean on news more than up moves in every sector.
- **No news, no earnings** is ~80% of days and still ~40-65% of 8%+ days (highest in Utilities, Money, Energy,
  Other). Half the big moves have no company headline at all, which is why news-only models miss them. 2020 is the
  extreme: ~60% of big moves had no news (market-wide days).
- **News type sets the direction, not the size.** Deal headlines: +0.3% to +1.3% mean day, 63-85% of 8% days up.
  Offering headlines: -0.6% to -2.8% mean day, 0-38% of 8% days up. Earnings, guidance and analyst headlines: near
  50/50. FDA matters only in Health (10% of its 8% days). Utilities news skews down (39% of 8% news days up).
- **Rates and oil.** Energy is the most tied to both (corr 0.24 with the 10y change, 0.19 with oil; +2.9% on the
  top 10% oil days, -2.7% on the bottom 10%). Rising-yield days help Energy, Chemicals, Manufacturing and Money;
  Business equipment and Health don't care (corr ~0.03). On the 10% of days with the biggest yield drop, every
  sector averages -0.4% to -1.7% (these are risk-off days).

What this suggests (hypotheses only, each would be a new registration): direction can be read from the news type
(deal up, offering down) while earnings direction needs the content; half of big moves come without news, so a
boom finder needs market and sector inputs, not just headlines; Energy moves should be read against oil first.
