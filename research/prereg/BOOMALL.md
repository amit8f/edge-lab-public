# BOOM-ALL: every confident BOOM pick, long the most bullish and short the most bearish (main track)

Registered 2026-11-20, before any BOOM-ALL or D7 number exists. Source: Amit, 2026-11-20: "it always pick 20 and
stop ... not cap it and just let it pick but make it pick just the ones he is most confidence", then "both ... why
only short if we see that more % went up than down why not both? and just read the market and try and build an
agent or a model to try and predict the direction". Two ideas: count 176 -> 178 (193 with phase 1).

Why: BOOM-SHORT has made almost no trades since 2026-11-09. Live D6 scores sit near 0, so its fixed cut (-0.119)
is almost never reached. Live picks Nov 12-19: 56% moved 8%+, 21% touched +8% first, 38% touched -8% first
(`research/results/boom_live_picks.json`, a falling market). Known before this registration (BOOM2 history): long
BOOM picks lost after costs in every D6 tenth; only the most bearish tenth made money short. Prior: low (about
15%) for the long side, moderate (about 30%) for the short side.

Mechanism (rule 10). 1. Counterparty: traders chasing or panic-selling a stock that is about to swing. 2. A fund
could do it but the names are small and jumpy. 3. Costs (~0.56% a trade here) eat most of the move; the test
decides.

## BOOM-ALL rules (fixed now)

- **Picks, uncapped:** every stock whose frozen BOOM1 score is at or above a line L, instead of the top 20. L is
  the BOOM1 score quantile in {top 1%, 2%, 3%} of each day's scored universe, chosen on 2019-2023 by the share of
  picks that move 8%+ with at least 10 picks a day on average. If fewer than 100 stocks score, no picks.
- **Side, by daily rank (no fixed cut):** among the day's picks, short the k with the lowest D6 score and go long
  the k with the highest, k in {3, 5}, chosen on 2019-2023 by mean net per side. Shorts need Alpaca shortable and
  easy to borrow today.
- **Exits and costs:** as BOOM-SHORT (entry at the first regular-session bar's open, exit at the +8% / -8%
  barrier touched first, else the close; NX3's cost model). Control: the same number of the day's other picks at
  random, same side.
- **Judged per side.** Longs and shorts are separate results.

## D7 arm: a direction model that reads the market (fixed now)

- Same picks, same target as D6 (which barrier, +8% or -8%, is touched first; neither = dropped from training).
- Inputs: D6's inputs plus market inputs known by 9:28 ET: SPY and QQQ pre-market move from the previous close,
  the pick's SIC-3 industry and its 10 most correlated stocks' pre-market moves (as ALIGN's related stocks),
  VIX-proxy (SPY 20-day realized vol), the previous day's SPY return, and the pick's own 5-day return.
- Model: XGBoost, same settings as D6, trained 2019-2022, its ranking cut chosen on 2023, then refitted on
  2019-2023 for the test. D7 replaces D6 in the BOOM-ALL rank rule for this arm only.

## Tests

The 2024-Sep 2025 years were already looked at in BOOM2's diagnosis, so for BOOM-ALL they are reported, not
deciding. D7 is a new model, so 2024-Sep 2025 is its first look.

- **BOOM-ALL (decides forward):** a $5,000 ledger shadow book (no orders), $500 a trade, max 10 a side a day.
  Passes per side at 100 closed trades: net > 0, NW t >= 2, beats the random control of the same side.
- **D7 (one look, 2024-01 to 2025-09):** passes if its rank rule's short side or long side has net > 0 after
  costs with NW t >= 2 on 100+ trades, and beats D6's rank rule on the same days. Pass: the D7 rank rule joins
  the BOOM-ALL book as a second ledger. Fail: written up.

BOOM-SHORT stays frozen and keeps running unchanged. Holdout from 2025-10-01 untouched for history.
