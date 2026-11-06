# SHORT-ALL: shorts only, every learner, with what the failed short tests taught us

Registered 2026-11-06, before any SHORT-ALL number is computed. Source: Amit, 2026-11-06 ("you checked all the
models also for short only?" and "adapting, learning from what failed, applying different things"). Five ideas,
one per learner. Phase-2 count 121 -> 126 (phase 1 + phase 2 = 141). The NX3, NXK, NXK2, SHORT-FIX and bar-table
history was seen first and the lessons below come from it, so this can only reject; a forward book decides.

## Lessons carried in (all from earlier, already-failed or descriptive results)

- Weak shorts lose: in the XGBoost, CatBoost and KRR bar tables, shorts below confidence 2.0 lose 0.2% to 0.55% per
  trade after costs (NX3_fired_trades*.csv, NX3BARS_CAT/KRR_walkforward.json).
- Shorts do better when the market is already falling: SHORT-FIX's post-hoc weak-market arm (SPY down over the last
  5 days) beat random shorts by +1.14%/trade, t 2.57, positive in 6 of 7 years.
- Random shorts lose money, so a short rule must earn on its own after costs, not just beat random shorts.

## Rules

NX3's events, walk-forward (test years 2019 .. 2025-09, 5-session purge), news-effect models, seeds and costs.
Learners: XGB (NX3's), CAT and ET (NXK2), KRR (NXK), LGBM (LightGBM, 400 trees, 15 leaves, depth 4, rate 0.05,
Huber). Each day only shorts are taken: the first 3 fired events with a negative effect, in time order. Control:
as many random real events of the same day, shorted.
- plain (reported): every fired short.
- lessons (primary): only events whose confidence |e| / cut >= 2.0, on days when SPY's 5-day return at the
  previous close is below 0.

## Primary test (history)

Per learner, the lessons arm passes if pick-minus-control has mean > 0 with NW (5 lags) t >= 2.0 AND the shorts'
own net per day after costs has mean > 0 with NW t >= 1.5. Reported, no rule: the plain arm, trades, net per
trade, per-year net. Five learners are tested, so one pass near t 2 is weak evidence.

## Forward book

A learner that passes gets a short-only ledger shadow book (no orders), judged by PROTOCOL rule 20.

## Prediction (written first)

The lessons arm beats random shorts for most learners (about 0.6 each) but earns > 0 after costs at t >= 1.5 for at
most one (about 0.25 that any passes): a weak-market, high-confidence filter leaves few trades.
