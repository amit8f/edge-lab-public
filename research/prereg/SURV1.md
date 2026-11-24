# SURV1: survival-model exits for NX3-ALL-HI

Registered 2026-11-24, before any number of this idea is computed. Source: Amit, 2026-11-24 ("for now do all 5", picking from his model-families list: GARCH, HMM regimes, survival models, Isolation Forest / CUSUM, quantile regression / NGBoost). One idea. Phase-2 count 181 -> 182 (phase 1 + phase 2 = 197).

## Mechanism (rule 10)

Exit timing. FAST1 showed NX3's move keeps going after the first hour, and next-close beat every early exit;
NXE and STOP8 found no exit that beats next close. A survival model asks a different question: for this trade,
how likely is a further +3% move in its direction within the next 5 sessions, given it hasn't happened yet? If
some trades keep running for days, holding those longer captures the tail. Who is on the other side: slow
traders still adjusting to the news (post-news drift). Drift is published and has decayed in big stocks.

## Rules

Base events: NX3-ALL-HI (prereg NX3ALLHI.md) as logged in `cache/nx1/nx3/trades_v2` (same models, seeds, confidence >= 1.5, uncapped, costs `N.cost_frac`), 2019-01 to 2025-09. Daily bars from Alpaca SIP (`adjustment=all` for returns, `raw` for prices). Choose years 2019-2023; one look at 2024-01 to 2025-09; nothing from 2025-10-01 on (holdout). Every input uses data known before the entry.

Entry as NX3-ALL-HI. Horizon: up to 5 sessions. Event = the trade's direction-signed close-to-entry return reaches
+3% at a daily close; censored at 5 sessions. Model: a discrete-time hazard model (LightGBM classifier on
trade-day rows, fixed params: 200 trees, depth 4, learning rate 0.05, seed 7) with inputs known at each close:
days held, return so far, the NX3 confidence, side, price tier, ATR(14)/price, SPY return so far, day-of-week.
Trained on 2019-2022 trades. Exit rule: after the first close (NX3's normal exit) keep holding only while the
predicted probability of reaching +3% in the days left is >= p*, else exit at that close; p* from {0.3, 0.4, 0.5}
chosen on 2023 by net per trade. Costs: `N.cost_frac` for the days held (borrow included).

## Primary test (one look, 2024-01 to 2025-09)

Net per trade minus plain next-close NX3-ALL-HI on the same trades: mean > 0 with day-clustered t >= 2 and
300+ trades, and SURV1's own net per trade > 0. Pass: a $5,000 ledger shadow book.

## Reported, no rule

Share of trades held past day 1; mean days held; per side; Kaplan-Meier curve of time to +3% (descriptive);
the model's calibration on 2023.

## Prediction (written first)

Next close stays hard to beat (NXE, STOP8). P(pass) about 0.15.

## Amendment 1 (2026-11-24, code details, before any result)

`tracks/news/models5_study.py`. Closes 1..5 count from NX3's own exit close (close 1 = next close, using the
trade's logged return); later closes scale it by the all-adjusted close ratio. Trades without five unbroken sessions
after entry (gap, delisting, or past 2025-09-30) are dropped from SURV1 and from its next-close comparison alike.
A trade that reaches +3% at a close exits at that close. The hazard model's inputs at close k are held fixed for the
remaining days (only the day number changes) when it predicts the chance of reaching +3% later. The comparison is the
same trades' next-close net.

## Result (2026-11-24, run 45383535010): FAILED

`research/results/MODELS5/SURV1_result.json`. 12,649 NX3-ALL-HI trades with five unbroken sessions; 2023 chose
p* = 0.3 (+0.29% per trade, 20% held past day 1). Test 2024 to Sep 2025: 5,019 trades over 421 days. Gain over next
close +0.03% a day (NW t 0.48), well short of t >= 2. Per trade, SURV1 nets +0.11% against +0.19% for plain next close
on the same trades, so holding longer costs money on average (the by-day mean is slightly positive because a few
busy days weigh less). 22% of trades held past day 1, 1.34 days on average. Own net by day +0.12% (t 0.55).
Reported: holding helps longs (+0.39% per trade) and hurts shorts (-0.13%). Share reaching +3% by close 1..5: 32%,
39%, 43%, 46%, 49%. As predicted, next close stays hard to beat. No shadow book.
