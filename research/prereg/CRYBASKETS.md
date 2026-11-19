# CRY-BIG and CRY-MEME: the CRY-TREND rule on a big-coin basket and a meme-coin basket (crypto, outside the three tracks)

Registered 2026-11-19, before any number for either basket is computed. Source: Amit, 2026-11-19 ("we need here
also to split it one into the big coins and one into meme coins"), under his standing crypto go-ahead of
2026-11-14. Two ideas: phase-2 count 170 -> 172 (phase 1 + phase 2 = 187). Paper only.

Why: CRY-TREND (hold BTC and ETH while the close is above its close 28 days earlier) is the crypto rule still alive.
Amit wants the same idea run separately on the large coins and on meme coins, which move far more and on hype.
Trend rules are published for large coins; for meme coins there is no published test, and the bet is that their
long rallies and long collapses give a 28-day trend rule more to catch (and more to lose in whipsaws).

Mechanism (rule 10). 1. Counterparty: late, slow buyers and sellers who chase or dump after a coin has already moved
for weeks. 2. Big funds can't trade meme coins in size; small accounts can. 3. Meme spreads are wide, so costs are
set higher for that basket.

## Rule (all numbers fixed now)

- CRY-BIG coins: BTC, ETH, SOL, XRP, BNB (the five largest non-stablecoin coins by market cap that Kraken lists in
  USD; checked 2026-11-19 against Kraken's AssetPairs).
- CRY-MEME coins: DOGE, SHIB, PEPE, BONK, WIF, FLOKI (the six largest meme coins Kraken lists in USD, same check).
- Signal per coin: CRY-TREND's (`tracks/crypto/trend_study.py`): hold while the daily close is above the close 28
  days earlier, known at the close, applied to the next day's return. A coin with fewer than 28 days of history has
  no signal.
- Size: the book has one slot per coin (1/5 or 1/6). A coin is entered with 1/k of the book's value at that close
  (or the cash left, if less) and sold whole on exit (trend_study.book). The rest is cash.
- Costs (round trip, half on each side): CRY-BIG 0.5% (CRY-TREND's primary); CRY-MEME 1.0% (wider spreads). Each is
  also reported at 0.12% and, for CRY-MEME, at 0.5%.
- Benchmark (buy and hold): the same slots, each bought at the period start or on the coin's first day with 28 days
  of history if later, then never touched. Until a coin's slot is bought it sits in cash in both books.
- Data: Binance spot daily closes vs USDT from data.binance.vision (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT;
  DOGEUSDT, SHIBUSDT, PEPEUSDT, BONKUSDT, WIFUSDT, FLOKIUSDT), ending 2025-09-30. The Oct 2025+ holdout stays unread.
- Periods: CRY-BIG 2018-01-01 to 2025-09-30 and 2022-01-01 to 2025-09-30. CRY-MEME 2021-06-01 to 2025-09-30 (DOGE and
  SHIB first, the rest as they list) and 2023-06-01 to 2025-09-30.

## History check (it can only reject)

For each basket separately, at its primary cost, in both of its periods, the rule's daily net Sharpe ratio must be
above the basket's buy-and-hold and its largest drawdown smaller. If either period fails: that basket FAILED, no
book. Reported beside it: CAGR, round trips per coin, share of days held, and the Sharpe percentile against 1,000
random-timing copies of each coin's signal (CRY-TREND's twin test).

Caveats written first. The meme list is chosen today, from coins that survived and grew; dead meme coins are missing.
That flatters buy-and-hold more than the trend rule (the rule would have exited a dying coin), but it still means a
pass is weak evidence. BTC and ETH in CRY-BIG were read before (CRY-TREND), and the Jan-Sep 2025 window has been read
several times, so neither check is a test.

## Forward books (only for a basket that is not rejected)

A $1,000 ledger book per basket, next to a buy-and-hold twin, Kraken public daily candles (pairs XBTUSD, ETHUSD,
SOLUSD, XRPUSD, BNBUSD; XDGUSD, SHIBUSD, PEPEUSD, BONKUSD, WIFUSD, FLOKIUSD), same rule and cost as the primary
scenario. Ledger only: nothing is sent to an exchange. Judged at 2027-12-31: Sharpe above its buy-and-hold twin and
smaller drawdown, with at least 15 round trips in the book. Shown on the dashboard.

## Prediction (written first)

CRY-BIG not rejected: 0.6 (it is close to CRY-TREND, which passed). CRY-MEME not rejected: 0.45 (deep meme crashes
favour the trend rule; whipsaws and the 1% cost work against it).

## Result (run 44511119999, history check only; recorded 2026-11-19)

Verdict: **both not rejected.** At each basket's primary cost, in both periods, the rule beat its buy-and-hold on
Sharpe and had the smaller drawdown.

| Basket, period | Rule Sharpe | Hold Sharpe | Rule CAGR | Hold CAGR | Rule drawdown | Hold drawdown | Twin pct |
|---|---|---|---|---|---|---|---|
| CRY-BIG 2018-01..2025-09 (0.5%) | 1.41 | 1.02 | 81% | 63% | 55% | 82% | 1.00 |
| CRY-BIG 2022-01..2025-09 (0.5%) | 0.96 | 0.62 | 35% | 21% | 35% | 69% | 0.96 |
| CRY-MEME 2021-06..2025-09 (1.0%) | 0.90 | 0.63 | 46% | 21% | 56% | 77% | 0.96 |
| CRY-MEME 2023-06..2025-09 (1.0%) | 1.08 | 0.91 | 72% | 52% | 56% | 76% | 0.87 |

Round trips per coin: CRY-BIG 63-131 (2018-2025); CRY-MEME 24-61 (2021-2025). Cheaper-cost scenarios are better
still (CRY-MEME 2023-2025 at 0.12%: Sharpe 1.23). Predictions were 0.6 (CRY-BIG) and 0.45 (CRY-MEME): both passed.
This is not a test: the meme list was picked from survivors, BTC/ETH were read before, and 2025 has been read several
times. Next: the two $1,000 forward books (`tracks/crypto/basket_live.py`, branch `state-crybaskets`), judged
2027-12-31 against their buy-and-hold twins with at least 15 round trips each.
