# NXK: other model families on NX3's news events

Registered 2026-11-06, before any NXK number is computed. Source: Amit, 2026-11-06 ("try different models, not just
what we have, also stuff other than XGB style, do your own research and come to a conclusion"). NXM already tried
LightGBM, a Huber linear model, a random forest, a small neural net and their average, and NXR the tamer trees; all
are trees, linear or neural nets on the same 37 inputs. NXK adds two families that work differently: remembering
similar past events, and a smooth kernel model. NX3's history was seen first, so this can only reject (PROTOCOL rule
4); a forward book decides. Two ideas (NXK-KNN, NXK-KRR). Phase-2 count 111 -> 113 (phase 1 + phase 2 = 128).

## Question

Does a model that scores a news event by the average outcome of its 100 most similar past events (KNN), or by an
RBF kernel ridge (KRR), pass NX3's gate, and does it beat XGBoost?

## Rules

NX3's design unchanged (NX3.md: events, 37 inputs, walk-forward models refitted each test year with the same seeds,
placebo baseline, firing cut at the 90th percentile, 3+3 slots, entry, next-close exit, costs), with the learner
replaced. Missing inputs are filled with the training median (plus an indicator), then standardised.
- KNN: 100 nearest neighbours, distance-weighted, on a fixed random sample of at most 100,000 training events.
- KRR: Nystroem RBF features (300 components, gamma = 1/37), Ridge alpha 10.
XGBoost is rerun as the pairing baseline with the same seeds.

## Primary test (history)

NX3's gate, per arm: pick-minus-control mean > 0 and NW (5 lags) t >= 2.0, and pick-minus-shuffled-news mean > 0 and
NW t >= 2.0. Reported, no rule: each arm minus XGBoost on shared days (paired NW t), per-year results.

## Forward book

An arm that passes gets a $5,000 ledger shadow book beside NX3 (no orders) judged by PROTOCOL rule 20.

## Prediction (written first)

Probability each arm passes the gate: about 0.4 (the news inputs carry the signal, so most learners find it);
probability either beats XGBoost with paired t >= 1.5: about 0.1.
