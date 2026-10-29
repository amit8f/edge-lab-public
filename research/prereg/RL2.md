# RL2: RL1 with event inputs and a tamer model (main track)

Registered 2026-10-29, before any RL2 result exists. Source: Amit, 2026-10-29, "do all the fixes you
suggested", after RL1 failed (`prereg/RL1.md`: no signal in M4's eight factor inputs, t 0.01, and both
models scored +7-8% a month in-sample against about -1% out of sample). One idea: phase-2 count at
registration 66 (main 29, penny 10, news 16, cross-track 10, crypto 1). Phase 1 + phase 2 = 81.

Disclosed: the new inputs come from M5 (insider buying), M6 (analyst actions) and M7 (earnings surprise),
whose single-signal screens have results already (`research/IDEAS.md`), and the fix was chosen after RL1's
result. By PROTOCOL rules 4 and 23 this history can only reject RL2; a forward book decides.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** Investors slow to react to company events: insiders buying their own stock, analysts
   revising, an earnings surprise, a burst of news. Each was weak alone; RL2 asks whether a model that reads
   them together with the factor ranks finds what single sorts miss.
2. **Why us.** Weak, as for M4: liquid $5+ stocks.
3. **Published.** Combining many weak signals with trees (Gu, Kelly and Xiu 2020) beats single sorts;
   the event signals are each published (insider buying, analyst revisions, post-earnings drift).

## Data

RL1's month-end panel, label, rank label, walk-forward (2019-2025 test years) and costs
(`tracks/main/longterm/screen_rl1.py`), unchanged. Twelve inputs, each ranked within date:
RL1's eight, plus
- `insider`: open-market insider purchases filed in the last 182 days over market cap (M5's code, Form 4
  filing dates);
- `analyst`: M6's net analyst score over 90 days ((up - down) / sqrt(up + down), 0 without actions) from
  single-symbol Benzinga headlines (created time, New York date);
- `sue`: M7's latest standardized earnings surprise first filed within 120 days (missing otherwise);
- `news_n`: the number of single-symbol Benzinga headlines in the last 30 days.
The headline inputs are missing before 2017-01-01 (headlines start 2016-10). No new downloads beyond the
shared cache. Nothing dated from 2025-10-01 on is read.

## Model (fixed now, no tuning)

XGBoost regressor on the rank label: 300 trees, depth 2, learning rate 0.03, row and column sampling 0.5,
min child weight 200, L2 penalty 10, squared error, `hist`, seed 20261015.

## Primary test

Rule 21 first: the in-sample long-short mean (each year's fit on its own training months) more than twice
the out-of-sample mean fails RL2. Then the trading bar (PROTOCOL 11-12) on the monthly top-minus-bottom
decile, costs as RL1: mean > 0, HAC t >= 3, bootstrap CI > 0, FF5+momentum alpha t >= 2, deflated Sharpe at
the current idea count. A pass, or the near-miss lane (rule 19: control beaten with HAC t >= 1.5 and the
book's own mean after costs > 0), opens a forward shadow book, written as its own registration then.

Reported, no rule: the same model on RL1's eight inputs only (input attribution: RL2 minus it, NW t), the
random control, IC in and out of sample, per-year numbers, and gain importance by input.

## Prediction (written first)

Rule 21 passes (in-sample about 1.5x). Long-short about +0.3% a month, t about 1, so the trading bar fails
(~85%); about 0.3 that it reaches the near-miss lane.
