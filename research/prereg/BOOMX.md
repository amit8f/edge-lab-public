# BOOMX: open up the BOOM trees: which inputs make a stock boom, and which carry the direction

Registered 2026-11-19, before any number for it is computed. Source: Amit, 2026-11-19 ("dig into the trees and analyze
each thing... what parameters make a stock go boom more than others and what trees and parameters show the direction;
try different parameters, different models... don't hold back"). **Exploratory and descriptive: no trading rule, no
pass/fail.** Logged in IDEAS.md, not counted (count stays 170, 185 with phase 1). Any rule found here is registered as a
new idea and tested forward (or on data this study did not use) before any book uses it.

## Data (nothing from the 2025-10-01 holdout)
- Boom (part A): BOOM1's stock-days (`boom1_study.load`: $5+, $1M+ a day, 2016 to exits on or before 2025-09-30),
  label = the next session moves 8%+ either way from the open. Inputs: BOOM1's 73 (chart, market, pre-open news).
- Direction (part B): BOOM2's saved picks (the 20 a day BOOM1 chose, 2019 to 2025-09) with their 1-minute labels.
  Two labels: first touch (+8% before -8% vs the reverse; picks touching neither or both are dropped) and close vs open.
  Inputs: D6's (BOOM1's plus BOOM1's score, pre-market move and dollar volume, offering flag).

## What is computed (both parts)
1. One input at a time: its AUC on its own (0.5 = no information; shown as distance from 0.5 with the sign), the label
   rate in its lowest and highest tenth, and the AUC per year (does it hold every year?).
2. Inputs grouped (price trend, volatility and range, volume, gap and pre-market, news, market) and each group's
   share of the model's importance.
3. Models, all trained on 2019-2022 and scored on 2023, 2024 and 2025-01..09 separately: CatBoost (BOOM1/D6's
   settings), LightGBM, XGBoost, ExtraTrees, logistic regression (standardised), a depth-3 decision tree (printed as
   readable rules), and CatBoost on the top 10 inputs only. Reported: AUC per year; for part A also the share of each
   day's top 20 that boomed; for part B also the hit rate in the most confident tenth each side.
4. CatBoost's SHAP values on the 2023-2025 rows: mean absolute SHAP per input and the direction of its effect
   (correlation of input and SHAP).
5. The two strongest inputs together: a 5 x 5 grid of label rates.

## Limits (stated now)
Single-input AUCs on hundreds of thousands of rows will look "significant" even when tiny; read the size, not the
p-value. Many models and inputs are tried, so the best one is flattered; only a later, separately registered test
counts. 2023-2025 were already read by BOOM1/BOOM2.

## Prediction (written first)
Boom: volatility and range inputs (atr14, vol20, rng_atr) dominate, news adds a little, every model has a similar AUC.
Direction: every model's AUC stays under 0.56; pre-market move and BOOM1's score carry most of what there is.
