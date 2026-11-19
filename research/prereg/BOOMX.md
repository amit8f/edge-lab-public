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

## Result (run 44483068796, recorded 2026-11-19; full tables in research/results/BOOMX/BOOMX.md)

Boom (3.9M stock-days, base 3.9%): every model scores AUC 0.92-0.94 in 2023, 2024 and 2025; logistic regression is
within 0.01 of CatBoost, and one input alone (atr14, average daily range) reaches 0.89. SHAP: volatility and range 59%,
volume and size 18% (small, cheap, thin stocks boom more), price trend 11%, news 7% (more articles and stronger
negative headlines before the open), market 5%. Share of each day's top 20 that boomed: 0.50, 0.61, 0.68.

Direction, up-first vs down-first (18,752 picks): AUC 0.50-0.55 in 2023, 0.51-0.53 in 2024, 0.48-0.50 in 2025; no model
holds. Direction, close above open (33,770 picks, base 45%): AUC 0.51-0.54 every year, and the most bearish tenth closes
down 55-65% of the time. Every single input points the same way, every year: the jumpier the pick (vol20, range,
maxret20, bb_width), the bigger its pre-market gap up and the higher SPY above its 200-day average, the more likely it
closes below its open (vol20: highest tenth closes up 38%, lowest 48%; AUC 0.47-0.49 in each year 2019-2025).

Prediction check: boom right (volatility dominates, models alike). Direction: AUC under 0.56 right; the carriers are
volatility and the pre-market move, not BOOM1's score (1% of SHAP).

POST-HOC hypothesis (not a finding): "the wildest picks fade from the open" is one consistent effect, the same one
BOOM-SHORT trades. A rule built on it must be registered as a new idea and tested forward.

## Amendment 1 (2026-11-19, before any slice number exists): where is direction easier?

Amit, 2026-11-19: break the data into pieces and see which pieces carry direction. Same picks (2019 to 2025-09), same
labels. Slices: sector (FF12 from the current SIC), pre-market gap (down 5%+, down 1-5%, flat, up 1-5%, up 5%+),
news before the open (none, negative with FinBERT max negative > 0.5, other), volatility third (vol20), price band
($5-10, $10-30, $30+), SPY above or below its 200-day average, SPY up or down yesterday, the stock up or down
yesterday, weekday, year. Per slice: picks, close-up rate, up-first rate, mean open-to-close return, and the AUC for
close-up inside the slice of one CatBoost model (D6 inputs) trained on 2019-2022 and scored on 2023-2025/09.
Descriptive; many slices are compared, so any standout is a hypothesis for a separately registered, forward-tested idea.

### Amendment 1 result (run 44504568576, recorded 2026-11-19; research/results/BOOMX/BOOMX_slices.json)

No slice makes direction clearly easier: the model's close-up AUC inside every slice with 200+ test picks is 0.49-0.56
(largest: Utilities 0.563 on 250 picks, gap down 5%+ 0.557, SPY below its 200-day 0.547, wildest third 0.543).
What does differ is the base rate. Close-up share (all picks 45.2%): wildest third 42.4% vs calmest 47.6%; pre-market
gap up 5%+ 43.0% vs gap down 5%+ 47.6%; no news 44.1% vs negative news 48.1%; SPY above its 200-day 44.7% (mean
open-to-close -0.39%) vs below 47.1% (+0.07%); Tuesday 43.9% (-0.71%). Every sector closes up 42-48%. These lean the
same way as BOOMX's main result (wild, gapped-up, newsless picks fade); a rule combining them is post hoc and must be
registered and tested forward.
