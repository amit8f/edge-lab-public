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

## Clarification (2026-10-28, before any RL1 result exists; `tracks/main/longterm/screen_rl1.py`)

- M4's cached panel holds only 6- and 12-month returns, so RL1 builds its own month-end panel with M4's code
  (`prices.features`, `attach_fundamentals`, `eligible`, `composite`: price, market cap and liquidity
  filters, at least 4 of the 6 families, one row per company) on M3's month-end dates (2016-01 .. 2025-08);
  the label is M3's next-month return (next open to the next month-end close, inside one series). The 8 raw
  inputs behind the six families (momentum, volatility, gross profit and ROA, E/P and B/M, issuance, asset
  growth) are each ranked within date.
- Costs: the 0.2% round trip on each leg, plus a month of the 3% yearly borrow on the short leg
  (`config/risk.yaml`); the same for both models, so the head-to-head is unchanged by it.
- Rule 21 decides first: a model whose in-sample long-short mean (each year's fit scored on its own
  training months) is more than twice its out-of-sample mean is overfit, and then neither the label claim
  (for either model) nor Rank's trading bar can pass. The deflated Sharpe uses the 66 registered trials and
  is also shown with today's count.
- The random control draws both deciles at random each month (seed 20261015 + date).
