# CRY-WIDE and CRY-MOM: two slow, long-only rules on the 30 most traded coins (crypto, outside the three tracks)

Registered 2026-11-14, before any number for these rules is computed. Source: Amit's standing instruction of
2026-11-14 to keep testing crypto ideas until one holds up. Why these two: the only crypto rule to survive so far is
the slow BTC/ETH trend rule (CRY-TREND), and in every short-horizon test the long side held up better than the
short side, which lost to squeezes and funding. Two ideas. Phase-2 count 165 -> 167 (phase 1 + phase 2 = 182).
Paper only.

The Jan-Sep 2025 window has now been read three times (CRY-DAY, CRY-XS, CRY-FUND), so these two are not chosen on
one period and tested on another. Like CRY-TREND, each is a fixed rule (nothing fitted or chosen) with a history
check that can only reject, followed by a forward ledger book if it survives.

Mechanism (rule 10). 1. Counterparty: holders who sell late into falls and buy late into rises; crypto trends last
because new money arrives slowly, and leaders keep leading for weeks (Liu, Tsyvinski & Wu 2022, crypto momentum
at 1-4 weeks). 2. Funds trade both; nothing is reserved for a small account. 3. Published returns are before
funding and realistic costs on small coins; the benefit of trend rules is mostly smaller crashes.

- Data and universe: CRY-XS's rows (the 30 most traded USDT perpetuals re-picked monthly, daily decisions at the
  00:00 UTC close, entry at the next open), 2020-07-01 to 2025-09-30. Longs pay the coin's actual funding. Costs
  0.06% of every dollar traded (the primary scenario) and 0.25% a side (spot, Alpaca) reported beside it.
- CRY-WIDE: each day, each universe coin whose close is above its close 28 days earlier is held with 1/N of the
  book, N being that day's universe size; the rest is cash. The twin holds every universe coin with 1/N.
- CRY-MOM: each Monday's decision, the 5 universe coins with the highest 28-day return are held with a fifth of
  the book each for the next 7 days (weights reset to a fifth every day: a fixed-weight approximation, stated
  here). The twin is the same equal-weight universe hold as above.
- History check for each (it can only reject): over the whole period and over 2022-01-01 on, the rule's daily net
  Sharpe ratio must be above the twin's and its largest drawdown smaller, at 0.06% a side. If either period
  fails: FAILED, no book.
- Control, reported: CRY-MOM against 200 random picks of 5 universe coins each Monday (seed 0), as a percentile.
- If a rule survives: a $1,000 forward ledger book with its twin, under a short amendment written before it starts,
  judged at 2027-12-31 against the twin (higher net Sharpe and smaller drawdown).
- Predictions: CRY-WIDE survives the history check (0.5); CRY-MOM survives (0.3).

## Amendment 1 (2026-11-14, after the history checks and before the forward books start)

Both rules survived (`research/results/CRYSLOW/CRYSLOW.json`). The forward books:

1. A $1,000 ledger book for each rule and one for the twin (equal-weight universe hold), all from the first day
   held, 2026-11-15 (decision at the 2026-11-14 close). No orders.
2. Data: Binance's archive monthly daily candles and funding, replayed once each month is published
   (`tracks/crypto/slow_live.py`, `crypto-slow.yml`), with xs_study's own code. Candles from 2026-08 warm up the
   28- and 60-day inputs; a coin's age is counted from its first candle in that window. The universe is ranked
   from 2026-11 on by the month before, as in the history check.
3. Each month is logged once in the hash-chained ledger; every earlier day is recomputed and must match.
4. Judged once on the days to 2027-12-31: a rule passes if its daily net Sharpe is above the twin's and its largest
   drawdown smaller. Nothing else changes.
