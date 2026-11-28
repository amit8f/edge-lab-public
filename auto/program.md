# AUTO1 agent instructions (frozen with the harness)

You improve a stock-buying model. You may change only `candidate.py`. Reply with the full new file in one ```python
block, preceded by one line starting `IDEA:` that says what you changed and why, in under 25 words.

The harness gives `fit(train)` a pandas DataFrame of past news events (one row each) and calls `score(model, frame)`
on a later year. Each row has the 37 input columns listed in the current file, plus `symbol`, `d` (day), `t` (time),
`price` (previous raw close), `entry` (price paid), `prev_ac` (previous adjusted close, same basis as `entry`),
`prev_c`, `dv20` (20-day dollar volume), and `net` (the return of buying at the event and selling at the next close, after costs). `net` exists in
`train` only for fitting; `score` never receives it. A row's score may use only that row and rows at or before
its own time `t`: the harness re-scores the frame cut at random times and rejects any candidate whose scores change
(so no same-day ranks or means that include later events). Each day, the harness buys the first 3 rows (by time) whose
score is above 0, and holds cash otherwise. Your score is the Sharpe ratio of that daily book over three years,
retrained each year on earlier years only.

Rules:
- Imports: numpy, pandas, sklearn, xgboost, lightgbm, scipy, math only. No files, no network, no exec/eval, no dunder.
- Training plus scoring must finish within 10 minutes on 4 CPUs for all three years.
- Make one focused change per try: a new input built from the existing columns, a different target, a different
  model or settings, a stricter buy rule, an ensemble. Learn from the trial history you're shown; don't repeat a
  change that was already thrown away.
- Buying fewer, better trades is fine. Costs are already in `net`.

## Idea menu (from research done before any trial, 2026-11-28; try them one at a time)

What we already know about these buys (2019-2023): they make +0.88% a trade before costs and +0.01% after, so a
better buy rule mostly means skipping trades that can't beat their costs. Buys on pre-open news did better (+0.34%)
than on news between 9:31 and 11:00 (-1.03%). Buys after a big jump before entry (`entry / prev_ac - 1`) did better
(+0.94%) than after a small one (-0.95%). Buys in 2020 lost.

Inputs worth building:
- The jump before entry, and the jump measured in units of the stock's own volatility (`vol20`).
- The news-window move (`r_react`, `r_open`) kept apart from the stock's older trend (`r5`, `r20`): news-driven moves
  tend to keep going, moves with no news tend to reverse (Jiang, Li and Wang 2021; Savor 2012).
- Neglect: a big jump after a weak past 20 days (`r20`) did best in one backtest of gap-up buys.
- Size and liquidity as conditions (`log_dv`, `dv20`, `log_px`): drift is stronger in small, thinly followed stocks,
  but their spreads are wider, so use them through the cost-aware target rather than as a straight preference.
- Staleness: `novelty` low means old news, which tends to reverse within a week (Tetlock 2011).
- Time of day: `mins_open` and `time_bucket` (pre-open news vs the first 90 minutes).
- Earnings and guidance flags interacting with sentiment strength.

Targets worth trying:
- `net` clipped, as now; `net` above a threshold as a binary label (one Kaggle news team found classifying
  "return > 0" beat regression); `net` divided by `vol20` (volatility-scaled).
- Ranking within each day (XGBoost `rank:pairwise` or `rank:ndcg`, grouped by `d`), since the book only buys the top
  3. Rank quality alone (IC) isn't the same as top-3 profit, so judge by the score you're given.

Models worth trying:
- LightGBM, XGBoost with monotone constraints, regularised linear models, small sklearn
  MLPs, averages of several seeds or models. Competition winners were mostly plain gradient boosting with good
  inputs; large ensembles added little.
- Stricter buy rules: buy only when the predicted net clears a margin, or only on pre-open news.

Things that usually fail: very deep trees, many near-duplicate inputs, anything tuned to one year.
