# CRY-XS: a daily model that ranks the 30 most traded coins (crypto, outside the three tracks)

Registered 2026-11-14, before any daily candle is downloaded for it and before any number exists. Source: Amit,
2026-11-14: "we need to work on the crypto like we did with regular stocks ... keep trying researching testing
inventing using different models and such until we crack it". One idea (the learners and book sizes are versions
of it, chosen on 2024). Phase-2 count 163 -> 164 (phase 1 + phase 2 = 179). Paper only: no order is sent anywhere.

Why this one first. CRY-DAY found a positive but thin result with too few trades to judge (34 in nine months).
This test trades every day, long and short at once, so it does not depend on the market's direction and reaches a
verdict in one pass. The Jan-Sep 2025 window was already read once for CRY-DAY; this is a second look at it with
a different strategy, and the deflated Sharpe ratio is reported with 179 trials.

Mechanism (rule 10). 1. Counterparty: retail traders who chase the coins that just rose and pay funding to hold
them; coins that lag the group catch up or keep lagging in ways a cross-sectional model may rank. 2. Funds trade
cross-sectional crypto factors (Liu, Tsyvinski & Wu 2022: size and momentum); nothing is reserved for a small
account except trading the smaller perpetuals that funds skip. 3. Published factor returns are before realistic
costs and funding; expect most of it to be gone.

- Data: Binance's public bulk archive, USD-M perpetual daily candles and funding rates, 2020-01 to 2025-09.
  Nothing dated 2025-10-01 or later is downloaded or read.
- Universe, re-picked each month with no hindsight: the 30 USDT perpetuals with the highest quote volume in the
  month before, listed at least 60 days (stablecoin pairs and index products excluded; delisted coins included).
- Decision each day at the 00:00 UTC close, for every universe coin. Entry at the next daily open, exit at the
  open after that (one day). A coin with a missing daily candle on the decision day, the entry day or the exit day
  is not ranked that day.
- Inputs, all known at the decision close: the coin's return over 1, 3, 7, 14, 28 and 60 days, volatility over 7
  and 28 days, place in its 28-day range, quote volume against its 28-day average, taker-buy share over 1 and 7
  days, the last day's funding and its 7-day mean, days since listing (capped at 365), the same day's rank of
  each of these within the universe, BTC's return over 1, 7 and 28 days and the universe's mean 1-day return.
- Target: the coin's next-day return minus the universe's mean that day, clipped to +-15%.
- Learners: XGBoost, CatBoost and LightGBM regressors with Huber loss (PICK2's settings) and a ridge regression.
- Book: each day long the K coins with the highest prediction and short the K with the lowest, equal weights,
  half the book on each side. K is 3 or 5. Costs: 0.06% of every dollar traded (a coin kept from the day before
  costs nothing) plus the actual funding paid by longs and received by shorts.
- Periods: train 2020-07-01 to 2023-12-31; choose on 2024; refit on 2020-07 to 2024-12; **one test on 2025-01-01
  to 2025-09-30**. The last 2 days before each boundary are dropped from training.
- Learner and K are chosen together on 2024: the highest daily net Sharpe ratio. Gate: its mean daily net return
  must be > 0. If not: "failed at validation" and 2025 stays unread.
- Primary test (2025): mean daily net return > 0 with Newey-West t >= 3 (rule 12), mean (book minus control) > 0
  with t >= 2, and alpha on BTC's daily return > 0 with t >= 2. Deflated Sharpe ratio reported with 179 trials.
- Control: each day, K random universe coins long and K others short (seed 0), charged the book's own cost that
  day.
- Also reported, no rule: the long side alone and the short side alone, results by month, average daily turnover,
  and the long side alone at 0.25% a side (spot).
- If it passes: a forward ledger book under its own registration. A result that is positive after costs but short
  of t 3 may get a watch-only forward book (the near-miss rule); it is still a FAIL here.
- Prediction: the 2024 gate passes (0.5); the 2025 test passes (0.05).

## Amendment 1 (2026-11-14, after the 2024 choice and before the 2025 test): a data check first

The 2024 gate passed with numbers too good to take at face value: XGBoost with K = 3 made +0.42% a day after
costs (Sharpe 3.3), and all three tree learners were above Sharpe 2. Under the project's rule for implausibly
large results, the test waits for a data check on the 2024 choice year only (`--stage qa`, no 2025 row is read):
how much of the profit comes from a few days or coins, moves above 50% in a day, the long and short sides apart,
which inputs the model leans on, and the same picks entered one day late. A data or code bug found this way is
fixed and the fit rerun, logged here. Nothing else about the method, the choice or the test changes.
