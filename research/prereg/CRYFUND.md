# CRY-FUND: trade against the side that pays the most funding (crypto, outside the three tracks)

Registered 2026-11-14, before any number for this rule is computed. Source: Amit's standing instruction of
2026-11-14 to keep testing crypto ideas until one holds up, and the post-hoc lesson of CRY-XS: its 2025 loss came
from shorting MYX while shorts were paying up to 10.7% a day in funding and the price kept rising. One idea (the
four versions below are chosen on history). Phase-2 count 164 -> 165 (phase 1 + phase 2 = 180). Paper only.

Mechanism (rule 10). 1. Counterparty: leveraged traders crowded on one side, who pay funding every 8 hours to
hold it and are forced out when the price turns; the other side is paid to wait. 2. Funds and market makers run
basis trades that cap funding on liquid coins; on smaller perpetuals the cap is weaker. 3. Published work on
funding mostly studies the carry (being paid), not the price move after extremes; the CacheCarti public screen
(Nov 2026) found its few surviving bots keyed on funding extremes. Our own evidence on it is one post-hoc trace.

- Data and universe: exactly CRY-XS's rows (`cache/crypto/xs_rows.parquet`): the 30 most traded USDT perpetuals,
  re-picked monthly, daily decisions at the 00:00 UTC close, entry at the next open, one-day hold, 2020-07 to
  2025-09. No model is fitted.
- Rule: each day, long the K coins whose funding is the most negative (shorts paying the most) and short the K
  whose funding is the most positive, equal weights, half the book a side; the long side receives and the short
  side pays the coin's actual funding over the day held. Costs 0.06% of every dollar traded.
- Versions: the funding measure is the last day's funding or its 7-day mean; K is 3 or 5. Four versions.
- Choice on 2020-07-01 to 2024-12-31: the version with the highest daily net Sharpe ratio. Gate: its mean daily
  net return must be > 0, else "failed on history" and 2025 is not computed.
- Primary test, 2025-01-01 to 2025-09-30: mean daily net > 0 with Newey-West t >= 3, mean (book minus random
  coins) > 0 with t >= 2, alpha on BTC's daily return > 0 with t >= 2. Deflated Sharpe ratio with 180 trials.
  This window has been read twice before (CRY-DAY and CRY-XS); it is a third look, which the trial count covers.
- Control: CRY-XS's random K long and K short each day (seed 0), charged the same cost.
- Also reported, no rule: each side alone, the funding part and the price part of the return apart, by month,
  and how often the book holds a coin that moved more than 50% in a day.
- If it passes: a forward ledger book under its own registration.
- Prediction: the history gate passes (0.55); the 2025 test passes (0.08).
