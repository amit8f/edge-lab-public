# CRY-TREND-VT: the BTC/ETH trend rule with position sizes cut when a coin is wild (crypto, outside the three tracks)

Registered 2026-11-16, before any number for it is computed. Source: Amit's standing instruction of 2026-11-14 to
keep testing crypto ideas until one holds up; this is the first idea of the next batch named in the 2026-11-14
report. Phase-2 count 169 -> 170 (phase 1 + phase 2 = 185). Paper only.

Why: CRY-TREND (hold BTC and ETH, half each, only while the close is above its close 28 days earlier) is the one
crypto rule still alive, and its benefit is mostly smaller crashes (drawdown 66% vs 88% for holding). The remaining
large losses come from holding full size through violent stretches. Volatility-targeted trend sizing is a standard,
published refinement (position size inversely proportional to recent volatility); nothing here is fitted.

Mechanism (rule 10). 1. Counterparty: the same slow, late sellers and buyers as CRY-TREND; the sizing only changes
how much is held when prices swing hardest. 2. Large funds already size by volatility; a small account can too, at
no capacity cost. 3. Published results are before realistic costs; here costs are charged on every dollar traded.

## Rule (all numbers fixed now)

- Data, universe and period: CRY-TREND's (`tracks/crypto/trend_study.py`): BTCUSDT and ETHUSDT daily closes, the
  same signal (close above the close 28 days earlier, known at the close, applied to the next day's return).
- Size: a coin with the signal on is held at weight (1/2) x min(1, 0.50 / v), where v is the coin's realized
  volatility: the standard deviation of its last 20 daily returns, times sqrt(365), known at the previous close.
  The rest is cash. The 0.50 target is the round number nearest each coin's long-run volatility, fixed before any run.
- Rebalance: at the entry close, at the exit close (sold whole), and at each Monday's close while held (weight reset
  to the formula). Between those days positions drift. Every dollar bought or sold pays 0.25% (0.5% round trip, the
  primary CRY-TREND scenario); the 0.12% futures scenario is reported beside it.
- Periods: 2018-01-01 to 2025-09-30 and 2022-01-01 to 2025-09-30. The Oct 2025+ holdout stays unread.

## History check (it can only reject)

At the primary cost, in both periods, the rule's daily net Sharpe ratio must be above CRY-TREND's and its largest
drawdown smaller than CRY-TREND's, and both above the buy-and-hold of the same two coins (Sharpe) as CRY-TREND already
is. If either period fails: FAILED, no book. Reported beside it: CAGR, average share of capital invested, trades per
year, and the Sharpe percentile against 1,000 random-timing copies of the plain signal (CRY-TREND's twin test).
2018-2025 is read again for BTC/ETH (CRY-TREND read it once), so a pass here is not a test: it earns only a $1,000
forward ledger book next to CRY-TREND's, judged at 2027-12-31 on the same two rules (Sharpe above CRY-TREND's,
drawdown smaller).

## Prediction (written first)

Not rejected: 0.4. The main risk is that CRY-TREND's Sharpe already counts its best stretches, where the sizing sells
exposure just before large gains.

## Result (run 43831535322, history check only; recorded 2026-11-19)

Verdict: **not rejected.** At the primary cost (0.5% round trip) the rule beat CRY-TREND on Sharpe in both periods
and had the smaller drawdown in 2018-2025 (52% vs 66%; hold 88%), and it stayed above buy-and-hold. 2018-2025:
Sharpe 1.02 (CRY-TREND 0.91, hold 0.72), CAGR 35% (CRY-TREND 39%). 2022-2025: Sharpe 0.69 (0.68, hold 0.56), the
margin over CRY-TREND is thin there. Futures-cost scenario (0.12%): 1.17 vs 1.03 and 0.87 vs 0.84. Average share
invested 41-45%; about 30 signal trades a year. Twin-test percentile 0.98 (2018-2025) and 0.89 (2022-2025).
Prediction was 0.4 for "not rejected": it passed. It gives up some return (CAGR) for the smaller drawdown, and
2018-2025 was read before, so this is not a test. Next: a $1,000 forward ledger book beside CRY-TREND, judged
2027-12-31 on Sharpe above CRY-TREND's and a smaller drawdown.
