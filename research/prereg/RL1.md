# RL1: bell-curve rank labels instead of raw returns (main track)

Registered 2026-10-24, before any RL1 result exists. Source: the weekly research scan
(`/mnt/project-files/research-scan/2026-10-24.md`, pick 1); Amit: "add all". Phase-2 count at
registration: 51 (main 25). Phase 1 + phase 2 = 66.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** None new. RL1 changes how a model is trained, not what it trades: a raw-return target
   lets a few huge winners and losers dominate the fit; a bell-curve rank (each stock's within-date return
   rank, mapped to normal scores) weights every stock the same. Any edge still comes from the inputs'
   known anomalies (momentum, low volatility, quality, value, issuance, growth).
2. **Why us.** Weak, as for M4: liquid $5+ stocks a large fund can also trade.
3. **Published.** Coqueret and Guida, "Label alchemy" (SSRN 7494298, Sep 2026): long-short decile Sharpe
   0.68 with the raw return, 1.69 with the Gaussianised rank, same 1,960 US stocks, 122 features and model.
   Not replicated yet; expect a smaller gain here (fewer inputs, shorter sample).

Disclosed: M10's exploratory table already holds an XGBoost regression on a *uniform* within-date rank of
the 21-day return (+0.24% a month over the universe, t 0.54). RL1 differs (normal scores, a monthly
long-short decile on M4's inputs, compared head to head with the raw-return twin), but that number was
seen before this registration.

## Data and walk-forward

M4's month-end panel and filters (`research/prereg/M4.md`): every input M4's six families use, each as a
within-date percentile rank. Label: the next month's return (entry at the next open, all-adjusted), no
label window reaching 2025-10-01. Yearly walk-forward 2019-2025 as M10 (train on every month-end whose
label closed before the test year's first date).

## Two models, identical except the label

XGBoost regressor with M10's fixed settings (400 trees, depth 4, rate 0.05, subsample 0.8, column sample
0.8, min child weight 20, `hist`, seed 20261015), squared error.
- **Raw:** label = the next month's return.
- **Rank (RL1):** label = Phi^-1((rank - 0.5) / n) of that return within its date (rank 1 = lowest).

## Primary test

Per test month: long the top decile by prediction, short the bottom decile, equal weight, 0.2% round trip
per side (`config/risk.yaml`). Head to head: monthly (Rank minus Raw) long-short return, Newey-West t
(3 lags). **RL1's label claim passes** if that difference is > 0 with t >= 2.0. Separately, Rank's own
long-short faces the trading bar (PROTOCOL 11-12: mean > 0, HAC t >= 3, bootstrap CI > 0, FF5+momentum
alpha t >= 2, deflated Sharpe with 66 trials); only that pass opens a forward shadow book. Rule 21: both
models report in-sample beside out-of-sample (the long-short mean, and the rank IC).

Reported alongside: a same-month random long-short control; Spearman IC of each model; per-year results.

Prediction: label claim ~55% fail; trading bar ~85% fail (M4's inputs had no edge as a top-20 book).
