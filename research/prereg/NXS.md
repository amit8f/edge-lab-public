# NXS: agents that size NX3-ALL's trades between 5% and 10% (news track)

Registered 2026-10-31. Source: Amit, 2026-10-31: "I want it for an agent or whatever to pick the % based on the
confidence level ... between 5 to 10 % I dont want a default I want it to be flexiable", then he chose "Both"
(a trained sizer tested on history, and a Gemini sizer logged live). NX3-ALL's history result was already
computed when this was written (run 41187953628, 2026-10-31 09:17 UTC: its confidence formula, sized book
minus control +0.28% a day, NW t 4.87), so the formula's history is known and NXS's history can only reject
(PROTOCOL rule 4). Two ideas. Phase-2 count 89 -> 91 (phase 1 + phase 2 = 106).

## What stays NX3-ALL's

Picks (every fired event), entry, next-close exit, costs, the same-day random control with pick i's size on
control trade i, and the day rule (sizes adding up to more than the $5,000 book are all scaled down together).
Only the size of each trade, within 5% to 10% of the book, changes.

## NXS-ML (history, then forward)

For each test year Y from 2021 to 2025-09, an XGBoost regressor (200 trees, depth 3, learning rate 0.05,
min_child_weight 100, subsample and colsample 0.8, pseudo-Huber, seed 20261024) is fitted on NX3-ALL's
out-of-sample picks from the test years before Y (2019 on), target the pick's net return, inputs NX3's 37
inputs plus confidence |e|/cut, side and the pick's order in its day. A pick's size is 5% plus 5% times the
percentile of its predicted return among that year's training predictions. 2019 and 2020 only train it.
Code: `tracks/news/nxs_study.py`, stage `sizer` of nx3-study.yml.

Primary test: on the same picks and days (2021-2025-09), (a) the ML-sized book minus control, mean > 0 and NW t
(5 lags) >= 1.5, AND (b) the ML-sized minus the formula-sized book, mean > 0 and NW t >= 1.5, AND (c) rule 21:
out-of-sample Spearman correlation of predicted and actual net > 0 and in-sample no more than twice it. Pass:
a $5,000 ledger shadow book next to NX3-ALL's.

Reported, no rule: the formula and equal sizes on the same days, by year.

## NXS-LLM (forward only)

For each live NX3-ALL pick, Gemini (the SYS2 free model chain, temperature 0) gets the pick's side, confidence,
ATR/price, time, the day's picks so far and NX3's input values, and returns one size between 5% and 10%; out
of range or unreadable answers are logged as missing and the trade takes the formula size, counted as
missing. The prompt is fixed in the live code before the first call and logged by hash; each answer is logged
with the model id. LLMs know past prices, so no history test. Judged forward against NX3-ALL's formula on the
same picks: paired daily difference, NW t >= 2 after 60 trading days and 30 closed trades (rule 20).

## Prediction (written first)

NXS-ML will not beat the formula: per-trade returns are mostly noise, and confidence already carries most of
what is predictable. Out-of-sample Spearman about 0.02. Probability (a)-(c) all pass: about 0.2. NXS-LLM: no
better than the formula.
