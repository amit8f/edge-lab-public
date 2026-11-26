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

## Amendment 2 (2026-11-26, before any stage-3 number exists): stage 3, how much of each move is explained by what

Source: Amit, 2026-11-26 ("compile a list of everything that makes a stock move ... come back with a % of each, for the
entire stock base and for each sector"). Still descriptive only, not counted. Same universe, sectors and years
(2019-2023; nothing from 2024 on is read, so the 2024-01..2025-09 test period and the holdout stay untouched).

**1. Common part (Shapley R-squared, so the blocks add up and order doesn't matter).** Per stock with at least 250
days, OLS of the daily return on four blocks: market (equal-weight universe return), big stocks (equal-weight return
of the 10 stocks with the highest previous-day 20-day dollar volume, the stock itself left out), sector (stage 1's
leave-one-out sector mean) and style (Fama-French SMB, HML, RMW, CMA and momentum, daily, Ken French library). Each
block's Shapley share of R-squared over all 16 subsets. Reported per sector and for all stocks as the
variance-weighted share (sum over stocks of block share x variance / total variance) and as the median over stocks.

**2. Stock-specific part, split by what happened that day.** The residual of the full model, each stock-day put in
the first group that applies: earnings window (8-K item 2.02, stage 2's rule); news by type in this order: deal,
offering, FDA, earnings or guidance, analyst, contract, other news (stage 2's keyword lists and window); other 8-K
(any other 8-K item, filed that day or the day before); offering filing (S-1, S-3, 424B*); insider filing (Form 4);
other SEC filing (any other form); heavy volume with none of the above (volume at least 2x its median of the 20
previous days); quiet (the rest). Per group: share of stock-days, share of the residual's sum of squares, share of all
the sector's |return| >= 8% days.

The answer table then splits 100% of each sector's return variance into: market, big stocks, sector, style, and the
stock-specific groups above. (In-sample per-stock fits; with an intercept the parts add to 100%.)

**3. Market-wide days.** Share of the equal-weight market's sum of squares that falls on its top 5% and top 10% of
days by |return| (how concentrated market moves are). Macro calendars (Fed days, CPI, jobs) are not in the data and
are left out (stated limit).

**4. Charts: how much of tomorrow is predictable from today's chart.** Features known at the close of day t-1 (1, 5,
20 and 60-day returns, distance to the 20-day high and low, 20-day volatility, day range, volume over its 20-day
median, the market's and the sector's 1-day return). Target: day-t return minus the day-t market return. LightGBM
(300 trees, learning rate 0.03, 31 leaves, seed 7), fit on 2019-2021, scored on 2022-2023: out-of-sample R-squared
(against a zero forecast) and the mean daily rank correlation, all stocks and per sector. Same-day volume vs |return|
is reported as a coincident link only (squared correlation of log volume ratio with |residual|).

**Limits stated in advance.** Shares describe 2019-2023 and the PICK main universe ($5+, $5M/day), not penny stocks.
News is Benzinga headlines only. Filing dates have no time of day. Keyword types are crude. In-sample R-squared
overstates the common part a little (9 regressors, about 1,000 days per stock). "Explained" means "moved together on
the same day", not cause; only part 4 measures prediction.

**Prediction (written first).** Common part 25-40% of variance (market 15-25%, sector 5-15% with Energy and Utilities
highest, big stocks 1-3%, style 3-6%); stock-specific 60-75%. Of all variance: earnings windows 8-12%, news days 20-30%,
filings 3-6%, heavy volume with no news 5-10%, quiet days 30-40%. Chart out-of-sample R-squared at or below 0.2% in
every sector, rank correlation near 0.01.

## Stage 3 result (2026-11-26, run 45776393346; descriptive, not counted)

Bug fix before reading any number: the first run (45775887902) crashed because the per-stock row stored the sector
block's share under the same key as the sector code; the code is now kept apart. Summary JSON:
`research/results/SECTOR1/SECTOR1_stage3.json`. 2,183 stocks have the 250+ full days the fit needs (stage 1 had
3,440 stocks in total; short-lived ones drop out).

Share of each sector's daily return variance, 2019-2023 (rows add to 100% up to rounding):

| sector | stocks | market | big stocks | sector | style | earnings | news | filings | heavy volume, no news | quiet days |
|---|---|---|---|---|---|---|---|---|---|---|
| All stocks | 2183 | 12% | 6% | 15% | 10% | 10% | 17% | 8% | 4% | 18% |
| Health | 298 | 7% | 4% | 11% | 8% | 10% | 28% | 8% | 4% | 19% |
| Telecom | 31 | 10% | 5% | 12% | 7% | 12% | 19% | 10% | 5% | 21% |
| Consumer nondurables | 84 | 11% | 5% | 12% | 9% | 15% | 15% | 8% | 3% | 21% |
| Shops | 168 | 12% | 6% | 15% | 9% | 12% | 18% | 8% | 2% | 18% |
| Other | 392 | 12% | 6% | 13% | 10% | 8% | 15% | 6% | 7% | 22% |
| Business equipment | 356 | 11% | 9% | 14% | 9% | 16% | 15% | 8% | 3% | 15% |
| Chemicals | 60 | 14% | 6% | 16% | 10% | 9% | 13% | 9% | 3% | 20% |
| Consumer durables | 49 | 13% | 7% | 16% | 10% | 10% | 19% | 6% | 3% | 15% |
| Manufacturing | 183 | 14% | 7% | 16% | 10% | 10% | 14% | 8% | 3% | 19% |
| Utilities | 75 | 13% | 5% | 24% | 8% | 5% | 16% | 8% | 3% | 19% |
| Money | 420 | 16% | 7% | 19% | 13% | 6% | 14% | 8% | 3% | 16% |
| Energy | 67 | 10% | 4% | 34% | 11% | 5% | 12% | 7% | 2% | 15% |

News is split by type in the JSON. All stocks: deal 2.0%, offering 0.5%, FDA 1.6% (Health 8.3%), earnings or guidance
headlines 3.7%, analyst 1.0%, contract 0.3%, other headlines 8.4%. Filings: insider (Form 4) 3.9%, other 8-K 1.8%, other
forms 1.9%, offering filings 0.2%. Quiet days are 58% of stock-days and hold 18% of variance; earnings windows are 3% of
days and hold 10% (Business equipment 16%, Consumer nondurables 15%, Energy and Utilities 5%).

- Common part 43% overall: Energy 59% (sector alone 34%, oil), Money 54%, Utilities 50%; Health 31%, Telecom 35%.
- Market moves are concentrated: the top 5% of days hold 50% of the equal-weight market's sum of squares, the top 1%
  hold 29%.
- Same-day volume vs |stock-specific move|: squared correlation 0.11 (coincident only).
- Charts: out-of-sample R-squared of next-day market-adjusted returns from t-1 chart features is negative in every sector
  (all -0.46%, worse than predicting zero), mean daily rank correlation 0.001. Telecom's 0.36 rank correlation is from
  very few days with 30+ telecom stocks and is noise. The model leaned mostly on yesterday's market return.

Prediction check: common part 43%, above the 25-40% predicted; market 12% (predicted 15-25%, lower) while big stocks
(6%) and style (10%) took more than predicted, since Shapley splits what they share with the market; sector 15% with
Energy and Utilities highest (right). Earnings 10% (right), news 17% (predicted 20-30%, lower), filings 8% (predicted
3-6%, higher), heavy volume with no news 4% (predicted 5-10%), quiet days 18% (predicted 30-40%, lower). Charts at or
below 0.2% (right: below zero).

## Amendment 3 (2026-11-26, before any stage-4 number exists): stage 4, what moves the market and the sectors, and which early signs see it coming

Source: Amit, 2026-11-26 ("you didn't say what moves the shared forces ... catch it early, see and notify and react ...
before or during the market day"). Still descriptive only, not counted; any trading rule from it is a new registration.
2019-2023 only, as before; fit or choose on 2019-2021, score once on 2022-2023.

**Data.** Alpaca SIP 5-minute bars, `adjustment=all`, 04:00-20:00 ET, zero-volume bars dropped, for: SPY, QQQ, IWM;
the sector ETFs XLE, XLF, XLK, XLU, XLV, XLI, XLB, XLP, XLY, XLRE, XLC (XLC from its 2018 launch); macro proxies USO
(oil), TLT (long Treasuries), UUP (dollar), HYG (credit), GLD (gold), VIXY (VIX futures); the seven largest stocks
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA. Per day: previous close (last regular bar at or before 16:00), pre-market
price (last bar before 09:30), open (first regular bar), 10:00, 15:30 and close prices; the day's realized variance
from regular-session 5-minute returns. Big-stock earnings nights: an 8-K item 2.02 filed by one of the seven on day
t-1 or t (W1 filings, as stage 2).

**Part A, what moves them (same day, not prediction).** Shapley R-squared of each ETF's close-to-close return on six
blocks: SPY (left out for SPY itself), the seven big stocks' equal-weight return, rates (TLT), oil (USO), dollar and
gold (UUP, GLD), credit and fear (HYG, VIXY). Also: share of each ETF's sum of squared returns on big-stock earnings
days, and the overnight (close to open) share of its squared moves.

**Part B, early signs (prediction, fit 2019-2021, scored 2022-2023), per target ETF.** Ridge regression (alpha chosen
by 5-fold time-series CV on 2019-2021 from 0.1, 1, 10, 100), features standardised on 2019-2021:
1. Size of the day: target log of the regular-session realized variance; features at 09:30: yesterday's, the last
   5 days' and the last 22 days' log realized variance, |pre-market move| of the ETF, SPY and VIXY, and the big-stock
   earnings-night flag. Report out-of-sample R-squared and the share of 2022-2023's top-20% days caught by the
   predicted top 20%.
2. Direction from the pre-market: target open-to-close return; features: the pre-market moves (previous close to
   09:29) of the ETF, SPY, QQQ, TLT, USO, VIXY.
3. Intraday momentum (Gao, Han, Li and Zhou 2018): target the last half hour (15:30-16:00); features: the overnight
   move and the first half hour (09:30-10:00) of the ETF and of SPY.
4. Lead-lag: target 10:00-16:00; features: every listed symbol's 09:30-10:00 return.
For 2-4: out-of-sample R-squared (against a zero forecast), correlation, sign hit rate, and the mean net return of
trading the predicted sign (cost 0.02% round trip for these ETFs, also shown at 0.20%), with a Newey-West t (lag 5).

**Limits stated in advance.** Macro calendars (Fed, CPI, jobs) are not in our data: part A sees their effect only
through rates, dollar and fear proxies. ETFs and the biggest stocks only, no single-stock early signs (that is NX3's
job). 2022-2023 is one rate-hike regime. Many cells: anything that looks good here is a hypothesis.

**Prediction (written first).** Part A: SPY plus the big stocks explain most of every sector ETF except XLE (oil),
XLU and XLRE (rates). Part B: size is predictable (out-of-sample R-squared 30-50% for log realized variance, about half
of the top-20% days caught); direction is not: pre-market and lead-lag R-squared below 1%, intraday momentum about 1%
for SPY with a hit rate near 52%, none clearly profitable after 0.02% except perhaps SPY's last half hour.

## Stage 4 result (2026-11-26, run 45778410647; descriptive, not counted)

Summary JSON: `research/results/SECTOR1/SECTOR1_stage4.json`. 1,237 days, 2019-2023; big-stock earnings days (an 8-K
2.02 from one of the seven on the day or the day before) are 16.6% of days.

**Part A, same day.** Shapley R-squared of close-to-close returns. SPY: big stocks 38%, credit and fear (HYG, VIXY)
43%, rates 3%, oil 3%, dollar and gold 1% (88% in all; VIXY and HYG move with the market the same minute, so they
mirror the move more than cause it). Sector ETFs: SPY 29-41%, big stocks 7% (XLE) to 34-39% (XLK, QQQ), oil 22% for
XLE and 1-3% elsewhere, rates 1-4% everywhere (daily TLT moves are mostly priced through the market itself). Least
explained: XLU 63%, XLE 65%. Big-stock earnings days hold 9-15% of each ETF's squared moves on 16.6% of days (less than
their share of days), except XLC 21% and XLY/QQQ/XLK 14-15%. Overnight (close to open) is 26% (XLU) to 53% (XLF) of
squared moves, 48% for SPY.

**Part B, early signs, scored once on 2022-2023.**
1. Size of the day is predictable: out-of-sample R-squared of log realized variance 0.49 (XLU) to 0.81 (XLY), SPY 0.71;
   the predicted top 20% catch 57-69% of the actual top-20% days (random would catch 20%).
2. Direction from the pre-market: out-of-sample R-squared negative for every ETF (-0.4% to -2.7%), hit rates 46-53%;
   no ETF has a positive net at 0.02% with t >= 2 (XLV and XLB are significantly negative).
3. Intraday momentum (first half hour and overnight -> last half hour): negative out-of-sample R-squared everywhere
   (SPY -0.85%, hit 52.4%), net negative after 0.02% for all 14.
4. Lead-lag (every symbol's first half hour -> 10:00-16:00): out-of-sample R-squared -2.5% to -15.8%, nothing positive.

Prediction check: big stocks and SPY dominate everything except XLE (oil) and XLU (right; XLRE is not rate-driven at
the daily level, wrong). Size predictable (right, better than predicted: 0.5-0.8). Direction not predictable (right);
intraday momentum, which we thought might survive for SPY, did not in 2022-2023 (consistent with Zhang-Hua 2024).

## Amendment 4 (2026-11-26, before any stage-5 number exists): stage 5, signs the day before that point to tomorrow's direction

Source: Amit, 2026-11-26 ("find what drives the direction ... what makes the 7 biggest stocks move a certain way ...
an edge the day before ... where the market will move the next day ... also for the oil and fear"). Still
descriptive only, not counted; any trading rule that comes out of it is a new registration tested forward.
2019-2023 only; fit or choose on 2019-2021, score once on 2022-2023. Nothing from 2024 on is read.

**Targets (17).** SPY, QQQ, IWM; AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA; XLE, USO (oil); VIXY (fear); TLT
(rates); HYG (credit); GLD; UUP (dollar). Same 5-minute bars and cache as stage 4.

**Family A, signs known at the 16:00 close of day t; target the close-to-close return of day t+1.**
- Market: today's SPY, QQQ, IWM returns; SPY's last half hour; SPY's gap and open-to-close split; SPY 5- and 20-day
  returns; SPY close vs its 20-day high; SPY log realized variance and its change vs its 22-day mean.
- Fear and credit: VIXY today and 5 days; HYG today; HYG minus SPY.
- Rates, oil, dollar, gold: TLT, USO, UUP, GLD today; TLT and USO 5 days.
- Big stocks: the seven's equal-weight return, minus SPY, and how many of the seven rose.
- Rotation: IWM minus SPY, XLU minus XLK, XLY minus XLP.
- Breadth (W1 universe, about 2,500 US stocks): share that rose; new 20-day highs minus new lows, as a share.
- News (Benzinga headlines in session t, i.e. 16:00 t-1 to 16:00 t): log count, all stocks and the seven,
  each minus its 60-day mean; a word-list tone (positive words minus negative words, per headline, averaged).
- Crypto: BTCUSDT (Binance public archive, hourly) from the previous US close to this one.
- Target's own: its return today, 5 days, last half hour, log realized variance change (big-7 and ETF targets).
- Calendar, about day t+1 (all public in advance): Monday, Friday, turn of month (last trading day or first three),
  monthly options expiry (third Friday), day before a market holiday, FOMC statement day, day before FOMC, CPI day,
  jobs-report day (dates in research/macro_dates_2019_2023.csv, from the Fed and BLS archives), a big-7
  earnings reaction day (8-K item 2.02, first trading day on or after filing), the target's own earnings day.

**Family B, signs known by 09:25 on day t+1; target the open-to-close return of day t+1.** After-hours moves
(16:00-20:00 close) of SPY, QQQ, the seven and the target; BTC from 16:00 to 09:00 ET; overnight headlines (created
16:00 t to 09:30 t+1): count and tone, all and the seven; pre-market moves (as stage 4) of SPY and the target.
Also, for after-hours signs only, a target from the 20:00 price to the t+1 close (what someone trading at 20:00
would get).

**Tests, per target, per sign.**
1. Alone: slope fit by OLS on 2019-2021, predicted sign traded on 2022-2023: correlation, hit rate, mean net per day
   after 0.02% (ETFs) or 0.05% (the seven) round trip, Newey-West t (lag 5).
2. Extremes ("when we see it"): the top and bottom 10% of the sign, cut on 2019-2021; the direction each tail had
   in 2019-2021; in 2022-2023, days in the tails, share that moved that way, mean net, t.
3. Calendar flags: mean next-day return on flagged vs other days, both periods, and trading the 2019-2021 sign.
4. Together: ridge (as stage 4) and LightGBM (small, fixed settings) on all family-A signs for close-to-close, all
   family-B signs for open-to-close; out-of-sample R-squared, hit rate, net, t.
Multiple testing: report how many sign-target cells reach t >= 2 in 2022-2023 with the same direction as 2019-2021,
against the about 2.5% one would see by chance.

**Limits stated in advance.** About 500 test days: a true 52% hit rate cannot be told apart from 50% (the 95% band
is about +/-4.4 points). 2022 was a bear year and 2023 a bull year. BTC has weekends, so Monday's BTC move spans
three days. Word-list tone is crude. Macro dates are release days without the surprise (consensus forecasts are not
free); we measure "on CPI day the market moves more / which way", not "a hot CPI".

**Prediction (written first).** Almost nothing survives: perhaps 5-10% of cells reach t >= 2 with the right sign,
near chance. The most likely survivors: short-term reversal of SPY and QQQ after big down days (bigger in high
volatility), FOMC-day and pre-FOMC drift up, turn-of-month up, VIXY's next-day fall after a spike, and the seven's
after-hours moves continuing into the next session's open but not past it. Combined models: out-of-sample
R-squared below 1%, hit rate 50-53%, nothing clearly profitable after costs. Size (stage 4) stays much easier than
direction.

## Stage 5 result (2026-11-26, run 45793794208; descriptive, not counted)

2,776 sign-target cells scored on 2022-2023. 58 reached t >= 2 (2.1%), against about 2.3% expected by chance: as a
group, the day-before signs carry no more direction than coin flips. Combined models (all signs together): ridge hit
rate 46-57% across the 17 targets, out-of-sample R-squared below zero for every one (-2% to -20%), no net t above 1.8;
LightGBM 44-55%, no net t above 1.9. The prediction (near chance, combined R-squared below 1%) holds.

What held in both periods (same direction 2019-2021 and 2022-2023; few days, so hypotheses only):
- Day before a market holiday: SPY +0.37% / +0.21% (19 test days, 68% up), QQQ +0.34% / +0.28%, VIXY -1.56% / -1.37%
  versus other days (t -2.7 / -2.3).
- Day before an FOMC statement: SPY +0.46% / +0.16%, VIXY -1.21% / -1.80% (t below 2 in each period).
- FOMC day: TLT +0.58% / +0.77% (16 test days, 81% up; t 1.8 / 3.3).
- Jobs-report Friday: XLE +0.79% / +1.10% and USO +0.73% / +1.39% (24 test days, 75% up; test t 3.8 / 3.1).
What did not hold: the strong next-day reversal of 2019-2021 (SPY today vs SPY tomorrow, correlation -0.28) is gone
in 2022-2023 (+0.02); VIXY, HYG, TLT, oil, dollar, gold, breadth, headline counts and tone, Bitcoin, after-hours and
pre-market moves all sit between 47% and 53% for SPY, QQQ and the seven.

Red flags, not edges: UUP's after-hours move reverses by the next close (t 6.7), most likely stale or wide after-hours
prints in a thin ETF (bid-ask bounce), not tradable. AAPL's big after-hours up moves continued next session (71% of
24 days), but none of the other six shows it, so it is likely luck.

## Amendment 5 (2026-11-26, before any stage-6 number exists): stage 6, surprises and slow followers

Source: Amit, 2026-11-26 ("there must be some underlying thing that makes them move" ... "do them", for options 2
and 3 of the reply that followed stage 5). Descriptive only, not counted; any rule that comes out of it is a new
registration tested forward. 2019-2023 only; directions chosen on 2019-2021, scored once on 2022-2023.

**Part A, the surprise on release days.** Consensus forecasts are not in our free data, so the surprise is measured
by the market's own first reaction (the standard event-study proxy): on CPI and jobs-report days (08:30 ET), the move
from 08:30 (close of the 08:25 bar) to 09:00 (close of the 08:55 bar) of TLT and SPY; on FOMC statement days
(14:00 ET), the move from 14:00 (close of the 13:55 bar) to 14:30 (close of the 14:25 bar). Dates from
research/macro_dates_2019_2023.csv (the unscheduled 2020 FOMC moves dropped, since they came outside 14:00). Targets
(same 5-minute SIP bars as stage 4, `adjustment=all`): SPY, QQQ, IWM, the seven, XLE, USO, VIXY, TLT, HYG, GLD, UUP.
For each event type and target: (1) does the rest of the day follow the first reaction (09:00 to the 16:00 close for
08:30 releases, 14:30 to 16:00 for FOMC)? (2) does the next day (close to close) follow it? Signals: the SPY
reaction, the TLT reaction, and the target's own reaction. Report the correlation in each period, the share of days
that follow the 2019-2021 direction in 2022-2023, the mean net after 0.02% (ETFs) or 0.05% (the seven), and the same
on the "big surprise" half (|TLT reaction| above its 2019-2021 median).

**Part B, slow followers (W1 universe, about 2,500 stocks, daily).** Each stock's group is its SIC 3-digit industry
when that industry has at least 6 stocks that day, else its Fama-French 12 sector. Leaders: the 3 stocks of the group
with the highest previous-day 20-day dollar volume; followers: the others. Leader move L = the leaders' mean return
that day (the stock itself left out); catch-up gap = L minus the follower's own return.
1. Sort: each day, followers ranked by catch-up gap; long the top 10%, short the bottom 10%; next-day and next-5-day
   returns of the spread, with Newey-West t, gross and after 0.20% round trip per side.
2. Events ("when we see it"): leaders move at least 2% (either way) and the follower moved less than half of that
   in the same direction, with no Benzinga headline naming the follower that session; trade the follower in the
   leaders' direction for 1 and 5 days. Share right, mean net, trades per day, t; also split by follower size (below
   and above the median dollar volume) and by whether the leaders had their own news.
3. Regression check: next-day return on L, own return and the market's equal-weight return, fit 2019-2021,
   correlation of its forecast with the outcome in 2022-2023.
Labels need the next 5 trading days, so feature days stop 5 trading days before 2023-12-31.

**Limits stated in advance.** About 24-36 release days of each type per period: these are hypotheses at best.
Pre-market prints at 08:30-09:00 are thinner than regular hours. Part B uses current SIC codes (as stages 1-3) and
daily closes, so a follower that catches up during the same afternoon is missed.

**Prediction (written first).** Part A: the first reaction continues through the rest of the day about 55% of the
time for SPY on CPI days and mostly for TLT, weakly; FOMC's first half hour reverses as often as it continues (the
press conference starts at 14:30); nothing passes t >= 2 in both periods. Part B: a small positive next-day catch-up
(Hou 2007 found it mostly in small, neglected stocks), below 0.10% a day for the spread, gone after 0.20% costs, and
event trades right 51-53% of the time.
