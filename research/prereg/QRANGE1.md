# QRANGE1: model-set take-profit and stop from predicted ranges

Registered 2026-11-24, before any number of this idea is computed. Source: Amit, 2026-11-24 ("for now do all 5", picking from his model-families list: GARCH, HMM regimes, survival models, Isolation Forest / CUSUM, quantile regression / NGBoost). One idea. Phase-2 count 183 -> 184 (phase 1 + phase 2 = 199).

## Mechanism (rule 10)

Fixed brackets lost (FAST1, STOP8; LESSONS 4): one size doesn't fit trades with very different ranges. A
quantile model predicts each trade's own next-day range (how far it may run its way, how far against), so the
target sits where the move is likely to stretch and the stop outside normal noise. Nobody is on the other side;
it's an attempt to keep winners' upside and cut the worst losers. Published as a method (quantile regression:
Koenker & Bassett 1978; NGBoost: Duan et al. 2020), not as a known edge.

## Rules

Base events: NX3-ALL-HI (prereg NX3ALLHI.md) as logged in `cache/nx1/nx3/trades_v2` (same models, seeds, confidence >= 1.5, uncapped, costs `N.cost_frac`), 2019-01 to 2025-09. Daily bars from Alpaca SIP (`adjustment=all` for returns, `raw` for prices). Choose years 2019-2023; one look at 2024-01 to 2025-09; nothing from 2025-10-01 on (holdout). Every input uses data known before the entry.

For each trade, predict the next session's favourable excursion F (best high or low vs entry, in the trade's
direction) at quantile q_tp and its adverse excursion A at quantile 0.9, with LightGBM quantile regression
(200 trees, depth 4, learning rate 0.05, seed 7) trained on 2019-2022 trades. Inputs: ATR(14)/price, GARCH-free
realised vol (20 days), price tier, side, NX3 confidence, gap at entry, SPY move at entry. Exit within the next
session: take profit at entry x (1 + predicted F), stop at entry x (1 - predicted A); a bar touching both counts
as the stop; otherwise the next close (NX3's normal exit). q_tp in {0.5, 0.7} chosen on 2023. Gaps fill at the open.

## Primary test (one look, 2024-01 to 2025-09)

Net per trade minus plain next-close NX3-ALL-HI on the same trades: mean > 0 with day-clustered t >= 2 and 300+
trades, and QRANGE1's own net per trade > 0. Pass: a $5,000 ledger shadow book.

## Reported, no rule

Hit rate of target and stop; calibration of the predicted quantiles on 2023 (share of trades beyond each);
an NGBoost version of the same rule (exploratory).

## Prediction (written first)

Stops still cut some winners. P(pass) about 0.15.
