# CRY-INTRA-BIG and CRY-INTRA-MEME: trend-gated, long-only crypto day models (crypto, outside the three tracks)

Registered 2026-11-19, before any number for either model is computed. Source: Amit, 2026-11-19, asked for "a model
that day trades them quickly based on several stuff"; after the research report
(`/mnt/project-files/reports/Crypto day trading big and meme coins.md`) he answered yes to every design question of
brainstorm round 1 (hold 4 to 48 hours, long only, trade only coins whose 28-day trend is on, order-flow / BTC /
time-of-day / volatility inputs with funding and open interest only as skip filters, separate big and meme models,
XGBoost / CatBoost / LightGBM / logistic, train to 2023, choose on 2024, forward book decides, Oct 2025+ unread).
Two ideas: phase-2 count 172 -> 174 (phase 1 + phase 2 = 189). Paper only.

Why: every short-horizon crypto model we tried (CRY-DAY, CRY-XS, CRY-FUND) failed or flipped in 2025, while every
slow long-only trend rule survived. This keeps the part that survived (long only, trend on) and asks whether a model
can pick the better hours to be in a coin that is already trending. The research found no published intraday signal
that beats costs, so the prior is low.

Mechanism (rule 10). 1. Counterparty: late buyers chasing a coin that is already in a trend, and sellers forced out by
liquidations, both visible as one-sided taker volume. 2. Large funds don't trade meme coins in size, and this
holds for hours, not seconds, so latency doesn't matter. 3. Costs are large against the edge: a trade must expect
more than about 1%.

## Data (fixed now)

- Coins: CRY-BIG's (BTC, ETH, SOL, XRP, BNB) for CRY-INTRA-BIG; CRY-MEME's (DOGE, SHIB, PEPE, BONK, WIF, FLOKI) for
  CRY-INTRA-MEME (`research/prereg/CRYBASKETS.md`).
- Binance spot 1-hour candles vs USDT from data.binance.vision (open, high, low, close, volume, quote volume, number of
  trades, taker-buy quote volume), ending 2025-09-30. Funding rates (monthly files) and the 5-minute metrics files
  (open interest, top-trader and taker long/short ratios) of the matching USDT perpetual (1000SHIBUSDT, 1000PEPEUSDT,
  1000BONKUSDT, 1000FLOKIUSDT for the coins quoted per 1,000). If the metrics files cannot be fetched for a coin, its
  open-interest inputs are missing (the models take NaN) and that is logged before any result.
- Candle times are UTC; an hour's candle is known at its close. The Oct 2025+ holdout is not downloaded.

## Decision points and the trend gate

Every 4 hours (00, 04, 08, 12, 16, 20 UTC, at the candle close), for each coin whose trend is on: the last finished
daily close (00 UTC) above the daily close 28 days earlier, CRY-TREND's signal. A coin with the trend off has no
decision. A coin with fewer than 28 days of daily history or fewer than 7 days of hourly candles has no decision.

## Inputs (all known at the decision time)

Coin returns over 1, 4, 24 and 72 hours; taker-buy share of quote volume (taker buys / all) over 1, 4 and 24 hours;
24-hour quote volume / its 7-day daily average; realized volatility (standard deviation of hourly returns) over 24
hours and 7 days; distance of the close below the 7-day high; BTC's 4- and 24-hour returns (for BTC itself, ETH's);
hour of day (the six slots) and day of week; last funding rate and its 3-day mean; open-interest change over 4 and 24
hours; the taker long/short ratio over the last hour. The meme model also gets the share of meme coins with the trend
on.

## Skip filter (not learned)

No entry when the coin's last funding rate is above the 95th percentile of its own 2020-2023 funding (pooled per
basket for coins that list later), or its 24-hour open-interest change is above the 95th percentile of the same. The
cut-offs are computed on training data only and frozen.

## Label, models and choice

- Label: the coin's return from the decision close to the close H hours later, minus the round-trip cost (CRY-BIG
  0.5%, CRY-MEME 1.0%, as in CRYBASKETS); positive = 1. H is 8 or 24 hours, chosen on 2024.
- Models: XGBoost, CatBoost, LightGBM (fixed moderate settings: depth 4, 300 trees, learning rate 0.05) and L2
  logistic regression on standardised inputs. One model per basket, all its coins pooled, no coin identity input.
- Train: from each coin's first decision through 2023-12-31. Choose on 2024: for each model, H and entry cut (the
  model score's top 5%, 10% or 20% of the training-period scores), take the version with the highest mean net return
  per trade among those with at least 60 trades in 2024. 24 versions per basket.
- Trades: enter at the decision close, exit at the close H hours later, no stop. One open trade per coin; at most
  one slot per coin, each 1/k of the book (CRY-BASKETS sizing).

## The one test (Jan-Sep 2025, a repeat look)

The chosen version, refit on data through 2024-12-31, trades 2025-01-01 to 2025-09-30 once. Pass needs all of:
1. at least 100 trades;
2. mean net return per trade > 0 after the basket's cost, day-clustered t >= 2.0;
3. above the random control: the same number of entries drawn at random from the same coin-decision points that
   passed the trend gate and skip filter, same H, 1,000 draws; the model's mean beats 95% of them;
4. above the gate alone: the mean net return of every trend-on, not-skipped decision with the same H.

The Jan-Sep 2025 window has been read three times before (CRY-DAY, CRY-XS, CRY-FUND), so a pass is not proof: it
earns only a $1,000 forward ledger book per basket (Kraken hourly candles, judged at 2027-12-31: 100+ closed trades
and mean net return > 0 with t >= 2). A fail is FAILED and stays logged.

## Prediction (written first)

CRY-INTRA-BIG passes: 0.15. CRY-INTRA-MEME passes: 0.15. The research found nothing that beats costs at these
horizons; the trend gate and skip filter are the only reasons to try.

## Choice on 2024 (run 44515119315; recorded 2026-11-19, before the 2025 test)

The first run (44513658917) stopped at fitting: zero volume or open interest gave x/0 inputs. Fix: those inputs are
treated as missing (a bug fix, made before any model result).

- CRY-INTRA-BIG: logistic regression, hold 8 hours, top 5% of scores. 2024: 70 trades, +1.44% net per trade, 64%
  winners. Every trend-on decision taken blindly lost -0.40% (8 h) / -0.21% (24 h) per trade after cost.
- CRY-INTRA-MEME: CatBoost, hold 24 hours, top 5%. 2024: 113 trades, +1.98% net per trade, 48% winners. The blind
  gate lost -0.87% / -0.68%.
- 2024 is the year the versions were chosen on (24 per basket), so these numbers flatter. Next: the one Jan-Sep 2025
  test. CRY-INTRA-BIG made 70 trades in all of 2024, so reaching the 100-trade minimum in nine months of 2025 is
  unlikely; the rule stays as registered.
