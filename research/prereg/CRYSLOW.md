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
