# LONG1-LONG5: why NX3-HI's buys are weak, and four fixes for the long side

Registered 2026-11-28, before any LONG number is computed. Source: Amit, 2026-11-28 ("do the steps"), after HI3K
showed NX3-HI's buys lost -0.15% a trade on the 8-K feed and EXEC1 showed history longs at -0.07% a trade after
measured spreads. Amit's real account can't short until margin is approved, so a long-only edge is what he can use.
LONG1 is descriptive and not counted. LONG2, LONG3, LONG4 and LONG5 are four ideas: phase-2 count 194 -> 198
(phase 1 + phase 2 = 213). All five run together in `tracks/news/long_study.py` (nx3-study stage `long`).

## Data (no new download)

NX3-ALL-HI's walk-forward picks 2019-01 to 2025-09 (`models5_study` picks: NX3's XGBoost, retrained each year on
earlier years, confidence >= 1.5x the firing line). NX3-HI's buys are the first 3 of those longs each day in time order,
as `nx3_study.picks` with cap 3. Entry minute and price from `trades_v2`; COST1's quoted half-spreads at entry; EXEC1's
1-minute prices and quotes at 10:00 and 10:30; daily adjusted closes from NX3's chart panel; SPY from NX3's chart
folder. Only trades whose 5-session exit is on or before 2025-09-30 are used, the same set for every arm, so nothing
from the Phase 2 holdout is read.

## Costs (every arm)

Entry: the quoted half-spread at the entry minute (or at T for LONG2); a trade without a quote keeps the flat cost.
Exit: market-on-close at the official close, no spread (EXEC1 measured about 0.07% saved). Commission and fees as
`adaptive/costs.py` (IBKR Tiered on a $500 position). Longs pay no borrow.

## Books and the primary comparison

Each arm is a daily book: equal weight across that day's trades, 0 (cash) on a day it takes nothing. The baseline is
plain NX3-HI buys on the same days with the same costs. Plain NX3-HI already beat a same-day random control in history
(`research/prereg/NX3HI.md`), so beating it is stricter than beating random. Choose years 2019-2023; one look at
2024-01 to 2025-09.

Pass for each of LONG2-5 on the test window: (a) own net per trade > 0, (b) own daily mean Newey-West t (5 lags) >= 2,
(c) daily gain over plain NX3-HI buys mean > 0 with NW t >= 1.5. A pass gets a log-only forward copy on the live
NX3-HI buys (no orders) on the dashboard's Live tab; real money still needs PROTOCOL.md's 100 trades and Amit's
signature.

## LONG1 (descriptive, 2019-2023 only)

Net per trade, count, win rate and a plain t for NX3-HI buys, all NX3-ALL-HI buys and NX3-ALL-HI shorts, split by:
news time (before or at the open / 9:31-11:00 / after 11:00); the jump already made before entry (entry price vs
previous close, terciles); confidence; dollar volume; price; past 20-day return; volatility; SPY's previous day up or
down; the nine news-type flags; year; and gross vs net (how much is cost). Also the 3- and 5-session drift of the buys.
Cells under 30 trades are marked thin. Nothing in LONG1 changes LONG2-5, whose rules are fixed below.

## LONG2: wait for confirmation

A buy whose entry minute is before T waits to T. It buys at the open of the first 1-minute bar at or after T (within
10 minutes) only if that price is above its original entry price, the news move still holding. Otherwise it's skipped.
Buys entering at or after T are unchanged. T in {10:00, 10:30}, chosen on 2019-2023 by mean daily gain.

## LONG3: hold buys longer

Same buys, exit at the closing auction of the 3rd or 5th session after the news day instead of the 1st. The hold is
chosen on 2019-2023 by mean daily gain. Overlapping holds make neighbouring days correlated; the NW lags (5) cover a
5-session hold.

## LONG4: a model just for buys

For each year from 2020, an XGBoost regressor (NX3's settings) is fitted on the earlier years' NX3-ALL-HI buys, using
their walk-forward confidence, so no year sees its own outcomes. Inputs: NX3's 37 plus confidence. Target: each buy's
net after the costs above. Training ends 14 calendar days before the year starts. A buy is kept only if its predicted
net is above 0 (fixed, not tuned); NX3-HI's 3 slots are then filled from the kept buys in time order. 2020-2023 is
reported as a check.

## LONG5: buy only in a friendly market

Buys on day d are taken only if a market rule held at the previous SPY close; the 3 slots are filled from the buys
that remain. Rules: SPY above its 20-, 50- or 200-day average, or SPY's 20-day volatility below its 2016-2022 80th
percentile ("calm"). One rule is chosen on 2019-2023 by mean daily gain.

## Reported, no rule

Per arm: trade counts, share skipped or kept, missing data shares. LONG2 and LONG4 combined with COST1's spread skip
(half-spread > 0.5%) are hypotheses for later, not findings.

## Prediction (written first)

LONG1: most of the buys' loss is spread on trades in the first minutes; gross is positive. Buys on pre-open news with a
big jump already made do worst. LONG2: fails (EXEC1 showed the drift is in the first minutes), P(pass) 0.15. LONG3:
small gain, P 0.25. LONG4: fails, the buys have too little signal left after NX3, P 0.15. LONG5: P 0.25. Expected
passes about 0.8 of 4.
