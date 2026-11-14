# CRY-DAY-FWD: the frozen crypto day-trading model, watched forward (crypto, outside the three tracks)

Registered 2026-11-14, before any candle dated 2026-11-15 or later exists. Source: Amit, 2026-11-14, chose "Watch
forward" on the decision card "Follow up the crypto day-trading model, which failed on too few trades?". One idea.
Phase-2 count 162 -> 163 (phase 1 + phase 2 = 178). Ledger only: no order is sent anywhere.

Why. CRY-DAY (`prereg/CRYPTO.md`) FAILED its one 2025 test: +0.85% a trade after costs and 74% winners, but only
34 trades (100 needed), daily t 2.08 (3 needed) and t 1.68 against its random control (2 needed). That result
stays failed. This is a new test of the same frozen model on data that does not exist yet. The history from
2025-10-01 to 2026-11-14 is not scored (it stays the crypto holdout); the candles of 2026-11-01 to 2026-11-14 are
read only to warm up the 7-day inputs.

Mechanism (rule 10): as in CRY-DAY. 1. Counterparty: leveraged retail traders in perpetual futures forced out by
liquidations. 2. Market makers and funds already trade this; the only small-account advantage is size. 3. No
verified after-cost record for retail intraday crypto exists; our own 2025 test is the only evidence and it is thin.

- Model, frozen: the two XGBoost files committed in `research/results/CRYDAY/models/` (refit on 2020-04 to
  2024-12) and the bar 0.005594582169120774 from `CRYDAY_fit.json`. Their SHA-256 hashes are logged with every
  entry. Nothing is refitted or re-chosen.
- Everything else as CRY-DAY and its Amendment 1: the 10 most traded USDT perpetuals re-picked each quarter by the
  last month of the quarter before, decisions at every 15-minute close, entry at the next open, stop 1.5 swings,
  target 3 swings, 16 bars, 0.12% a round trip plus actual funding, at most 3 positions of a third of the book
  each, one per coin, and the random coin-and-side control (seed 0).
- Data: Binance's public bulk archive, monthly files (15-minute USD-M candles, funding rates, and daily candles
  for the universe ranking). The archive publishes a month after it ends, so the job runs monthly and logs the
  trades of each finished month; this is a replay with a frozen model, not a live feed. The last 17 bars of the
  newest month wait for the next month (their exits are not known yet).
- Period: decision bars from 2026-11-15 00:00 UTC to 2027-12-31.
- Primary test, read once: when the book reaches 100 trades, or on the data to 2027-12-31, whichever is first.
  Pass = mean net per trade > 0 with Newey-West t >= 3 on daily book returns, at least 50 trades, mean (trade
  minus control) > 0 with t >= 2, and alpha on BTC's daily return > 0 with t >= 2 (BTC's perpetual, daily closes).
  Fewer than 50 trades by 2027-12-31 = no conclusion. Deflated Sharpe ratio reported with 178 trials.
- Until then the dashboard shows the running count and mean; those are not a result.
- Each run recomputes every earlier trade from the archive and stops if one differs from the ledger.
- If it passes: a paper book at a venue open to Israel residents under its own registration, and Amit's written
  sign-off before any real money, as for every idea.
- Prediction: 50 trades are reached by the end of 2027 (0.5); the test passes (0.07).
