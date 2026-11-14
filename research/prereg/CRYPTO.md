# CRY-TREND and CRY-DAY: two crypto tests (crypto, outside the three tracks)

Registered 2026-11-14, before any crypto price was downloaded for them and before any number exists. Source:
Amit, 2026-11-14: "what if we build a model for crypto?", then "yes" to the trend test the research report
recommended (`/reports/Crypto trading for small accounts.md` in the project files), and "there is a lot of day
trading of crypto, I want to check and try to tap into this one also". Two ideas. Phase-2 count 160 -> 162
(phase 1 + phase 2 = 177). Crypto is not one of the three tracks in CLAUDE.md; like K1 these are separate tests at
the owner's request, sharing only `core/`. Paper only: no order is sent anywhere.

## Data (both)

Binance's public bulk archive (data.binance.vision, free, no key): spot daily candles for CRY-TREND; USD-M
perpetual-futures 15-minute candles and funding rates for CRY-DAY. Candle times are UTC. If the archive cannot be
reached from our runners, an amendment written before any return is computed names the replacement (Alpaca's
crypto bars). Nothing dated 2025-10-01 or later is downloaded or read (the phase-2 holdout).

## CRY-TREND: hold Bitcoin and Ether only while their trend is up

Mechanism (rule 10). 1. Counterparty: holders who sell late into falls and buy late into rises; trends in crypto
last because new money arrives slowly. 2. A $1B fund can and does trade this; nothing is reserved for a small
account. 3. Published: Liu & Tsyvinski (2021), Han, Kang & Ryu (to Aug 2023, 28-day rule, Sharpe 1.51 vs 0.85),
Zarattini, Pagani & Barbon (2025, ~18% a year net). The benefit is mostly smaller crashes; expect decay.

- Rule, fixed here from Han, Kang & Ryu and never tuned: each day at the 00:00 UTC close, a coin is held when
  its close is above its close 28 days earlier, otherwise that half sits in cash. BTC and ETH, half the book each,
  rebalanced to halves only when a coin is entered. Trades at that close (the next day's open).
- Costs: 0.5% a round trip (0.25% a side, Alpaca's spot taker fee) is the primary scenario; 0.12% (cheap futures)
  is reported beside it.
- History check, 2018-01-01 to 2025-09-30 (it can only reject): over the whole period **and** over 2022-01-01 to
  2025-09-30 alone, the rule's net Sharpe ratio must be above that of holding half BTC and half ETH, and its
  largest drawdown must be smaller. If either period fails: FAILED, no book.
- Control: 1,000 random-timing twins per coin with the same number of switches and the same share of days held
  (seed 0); the rule's net Sharpe is reported as a percentile of theirs. Reported, no rule.
- Forward test, if not rejected: a $1,000 ledger book (no orders) from the first day after the live job is
  built, with a buy-and-hold twin. The rule trades about a dozen times a year, so rule 20's 30 trades cannot be
  reached quickly; it is judged once, at 2027-12-31: pass if its net Sharpe is above the twin's with a smaller
  largest drawdown and at least 8 round trips; fewer round trips = no conclusion.
- Prediction: the history check passes (0.6); forward pass by the end of 2027 (0.3).

## CRY-DAY: a model that day-trades the most traded coins

Mechanism (rule 10). 1. Counterparty: leveraged retail traders in perpetual futures who are forced out by
liquidations and pay funding; a model may find bars where their flow overshoots. 2. Funds can take this away and
market makers already do; the only small-account advantage is size (no market impact). 3. Published: no verified
after-cost record for retail intraday crypto; fees were more than half of retail losses on one venue. Our S1
scalp lost 0.14% a trade on BTC and ETH.

- Universe, re-picked each quarter with no hindsight: the 10 USDT perpetuals with the highest quote volume in the
  last month of the previous quarter (stablecoin pairs and index products excluded; delisted coins included).
- Decisions at every 15-minute close, 2020-04-01 to 2025-09-30, for every universe coin, long and short. Entry
  at the next bar's open. Exit: a stop 1.5 swings away, a target 3 swings away, or after 16 bars (4 hours),
  whichever is first; if a bar touches both, the stop counts. One swing = the median 1-hour range over the prior
  7 days, as a share of the price.
- Costs: 0.12% a round trip (0.05% taker fee a side plus 0.02% slippage) plus the actual funding paid or received
  while the position was open.
- Inputs, all known at the decision bar's close: the coin's returns over 15 minutes to 7 days, realised
  volatility, place in its 24-hour range, volume and trade-count pace, taker-buy share, the swing, the last
  funding rate and minutes since funding, hour and weekday, the same returns for BTC, the coin's return minus
  BTC's, and the average 4-hour return of the universe.
- Target: the trade's net return divided by the swing, clipped to +-10; one model per side. Learners: XGBoost,
  CatBoost and LightGBM regressors with Huber loss (delta 1), PICK2's settings.
- Periods: train 2020-04 to 2023-12; choose on 2024; refit on 2020-04 to 2024-12; **one test on 2025-01-01 to
  2025-09-30**. The last 16 bars before each boundary are dropped from training.
- Book: a trade is taken when its predicted net return is at or above the bar, the coin has no open position and
  fewer than 3 positions are open; higher predictions first; each position is a third of the book. Bar
  candidates: the top 1%, 0.5%, 0.2% and 0.1% of the training predictions of both sides together. Learner and bar
  are chosen together on 2024: the highest mean net per trade with at least 200 trades.
- Gate: the chosen 2024 book's mean net per trade must be > 0. If not: "failed at validation" and 2025 stays
  unread.
- Primary test (2025): mean net per trade > 0 with Newey-West t >= 3 on daily book returns (rule 12), at least
  100 trades, mean (trade minus control) > 0 with t >= 2, and the daily book return's alpha on BTC's daily return
  > 0 with t >= 2 (rule 11's stand-in: there are no Fama-French factors for coins). Deflated Sharpe ratio reported
  with 177 trials.
- Control: for each trade, a random other universe coin and a random side at the same bar, same exits and costs
  (seed 0).
- Also reported, no rule: the same book long-only at 0.5% a round trip (what Alpaca spot would allow), results by
  side, by coin and by exit kind, and a hand check of 20 random trades against the raw candles.
- If it passes: a forward ledger book under its own registration. Until then no live job is built.
- Prediction: the 2024 gate passes (0.35); the 2025 test passes (0.10).

## Amendment 1 (2026-11-14, with the code and before any download): details the text left open

1. CRY-TREND: day d's return is earned with the signal from the close before it; a coin is entered with half the
   book's value at that moment (or the cash left, if less) and sold whole on exit. Sharpe ratios use daily returns
   and 365 days a year. A random twin keeps the coin's held runs and cash runs and shuffles each set.
2. CRY-DAY: a missing 15-minute candle is filled flat with no volume and is never a decision bar. A funding
   payment counts when its time is after the entry bar's open and no later than the exit bar's open. A position
   frees its slot at the end of its exit bar. When both sides of a row clear the bar the higher prediction is
   used. Daily book returns sum each trade's net return divided by 3, on its entry day, with 0 on other days.
   The universe candidates are every USDT symbol folder in the archive's futures listing.
3. The long-only 0.5% scenario reruns the book with shorts removed.
4. The "hand check" recomputes 20 random test trades from the raw candles with a separate slow routine.
