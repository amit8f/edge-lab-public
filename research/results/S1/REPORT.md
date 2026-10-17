# S1 result: the 5-minute VWAP snap-back scalp FAILED on both stocks and crypto

Pre-registration: [prereg/S1.md](../../prereg/S1.md). Data: `crypto_*.json/csv` (run 39014098151) and
`stocks_*.json/csv` (run 39015798896).
Running idea count at this result: phase 2 = 23, phase 1 + phase 2 = 38 (38 at registration).

## Primary verdict (as pre-registered)

| | Crypto (BTC, ETH) | Stocks (SPY, QQQ + top 20) |
|---|---|---|
| Period | 2021-01 .. 2025-09 | 2018-01 .. 2025-09 |
| Signal trades / days | 1,730 / 1,045 | 2,827 / 1,216 |
| Mean net per trade | **-0.14%** | **-0.24%** |
| Mean gross per trade | -0.04% | -0.04% |
| Daily signal minus control (mean) | -0.003% | +0.16% |
| HAC t (5 lags), bar 3.0 | **-0.10** | 7.40 |
| Bootstrap 95% CI | -0.05% .. +0.06% | +0.12% .. +0.21% |
| FF5 + momentum alpha (monthly), bar t 2.0 | not tested (crypto) | **-2.5% a month, t -9.2** |
| Deflated Sharpe (38 trials) | 0.00 | 0.00 |
| Hit rate | 43% | 29% |
| Average hold (bars of 5 min) | 11.4 | 9.7 |

**Both markets FAILED.** The prediction written before the run (both fail after costs, gross maybe
slightly positive) was right on the verdict; gross was slightly negative, not positive.

The stock column beats its control with t 7.4, but that is not an edge. The control is the same
symbol, same session, same side, and a long signal only fires on days that fell through the lower band
(short signals on days that rose through the upper one). A random long on a falling day loses: the
control lost -0.17% gross per trade, -0.37% after costs. The signal trades lost less than that and still
lost money, so the strategy itself has negative alpha (t -9.2) and fails the pre-registered rule.
This is a property of the registered control, not a data error; it is reported as a warning, not a result.

## Funnel

- Crypto: Binance archive reachable from the runner, so no Alpaca fallback. 499,147 five-minute bars per
  coin (1,734 days x 288 = 499,392; 245 bars missing in the archive). 1,956 raw signals, 1,730
  trades after "one position at a time".
- Stocks: 6,278 candidate common stocks (active and delisted) ranked each year by prior-year median
  dollar volume; 22 symbols a year, 45 distinct. Each symbol-year has the full session count
  (~19,500 bars, 252 x 78; 2025 through September 14,472 = 186 x 78).

## Data QA

- Pipeline check (stocks, SPY): last regular-session 5-minute close within 0.25% of the daily close on
  99.6% of 1,947 sessions (bar 99%); summed 5-minute volume 60-105% of daily on 99.8% (bar 95%). Pass.
- Trade samples (50 per market): largest single trade -5.6% (crypto), -2.0% (stocks); no implausible
  returns. Bar counts match the calendar exactly.
- Universe picks are the expected mega-caps (AAPL, NVDA, TSLA, AMZN ...); XYZ is Block (formerly SQ).

## Run log (bug fixes only, no method changes)

1. Crypto run 39013830940 crashed on a timezone comparison before computing anything (PR #15).
2. Stock run 39014302495 got 401 from Alpaca. PR #16 switched delisted lines to their base symbol (the
   convention every other study uses); the real cause was a rejected API key, replaced by Amit at
   08:01 UTC. Run 39015360708 failed on the same key. Run 39015798896 is the result above.

## POST-HOC / NOT PRE-REGISTERED (exploratory list, never findings)

| Variant | Crypto net / trade | Stocks net / trade |
|---|---|---|
| 1-sigma bands | -0.12% (7,070 trades) | -0.24% (13,757) |
| Exit at 6 bars | -0.12% | -0.22% |
| Exit at 24 bars | -0.14% | -0.25% |
| Long only | -0.11% | -0.25% |
| Short only | -0.17% | -0.23% |
| Cheap costs (crypto 0.5% round trip / stocks 0.02%) | -0.54% | -0.06% |

Every year is negative after costs in both markets. A handful of stock symbols are positive (SMCI, COIN,
ROKU, MRNA, AMC), each on 12-29 trades: too thin to read and picked after the fact.

Gross returns are negative before any cost, so no realistic fee level saves the rule. No paper test follows.
