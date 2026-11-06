# NXK2: CatBoost and Extra Trees on NX3's news events

Registered 2026-11-06, before any NXK2 number is computed. Source: Amit, 2026-11-06 (his list of model families:
CatBoost and Extra Trees are the "very high" and "tree" entries we had not run). NX3's history and NXM's were seen
first, so this can only reject (PROTOCOL rule 4); a forward book decides. Two ideas (NXK2-CAT, NXK2-ET). Phase-2
count 113 -> 115 (phase 1 + phase 2 = 130). Adds `catboost==1.2.8` to requirements.txt.

## Question

Do CatBoost (ordered boosting, symmetric trees, built to resist overfitting on noisy data) or Extra Trees (fully
random split points) pass NX3's gate, and do they beat XGBoost?

## Rules

NXK.md's design (NX3 unchanged except the learner; XGBoost rerun as the pairing baseline with the same seeds).
- CAT: CatBoostRegressor, 400 iterations, depth 4, learning rate 0.05, L2 10, Huber loss (delta 1), seed NX3's.
- ET: ExtraTreesRegressor, 200 trees, max_features 0.33, 200-event leaves, bootstrap sample 30%, median-filled inputs.

## Primary test (history)

NX3's gate per arm: pick-minus-control mean > 0 and NW (5 lags) t >= 2.0, and pick-minus-shuffled-news mean > 0 and
NW t >= 2.0. Reported, no rule: each arm minus XGBoost on shared days; per-year results.

## Forward book

An arm that passes gets a $5,000 ledger shadow book beside NX3 (no orders), judged by PROTOCOL rule 20.

## Prediction (written first)

Probability each arm passes the gate: about 0.4; probability either beats XGBoost with paired t >= 1.5: about 0.1.
