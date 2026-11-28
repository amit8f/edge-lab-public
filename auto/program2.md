# AUTO2-AUTO4 agent instructions (frozen with harness2)

You improve one of our news-trading models. You may change only `candidate.py`. Reply with the full new file in one
```python block, preceded by one line starting `IDEA:` that says what you changed and why, in under 25 words.

`fit(train)` gets a pandas DataFrame of past news events: real events and placebo events (column `kind`, "real" or
"placebo"; a placebo is a random stock-day with no news, used to learn what the stock would do anyway). Columns: the
37 inputs listed in the file, `symbol`, `d` (day), `t` (time), `price` (previous raw close), `entry` (price paid),
`prev_ac` (previous adjusted close, same basis as `entry`), `prev_c`, `dv20` (20-day dollar volume), and the labels
`mret` (return from entry to the next close minus SPY's) and `ret` (the same without SPY). `score(model, frame)`
gets only real events of a later year with no labels. A row's score may use only that row and rows at or before its
own time `t`: the harness re-scores the frame cut at random times and rejects a candidate whose scores change.

The book: each day, the first 3 events (by time) with score >= 1 are bought and the first 3 with score <= -1 are sold
short, each held to the next close, after costs (shorts also pay borrow). Your score is the Sharpe ratio of that daily
book over 2020, 2021 and 2022, retrained each year on earlier years only. So the score's scale matters: 1 is the bar.
A stricter bar (fewer, better trades) is fine; so is a different bar for buys and for shorts.

Rules:
- Imports: numpy, pandas, sklearn, xgboost, lightgbm, scipy, math only. No files, no network, no exec/eval, no dunder.
- Training plus scoring must finish within the time limit on 4 CPUs for all three years.
- Make one focused change per try, and learn from the trial history you're shown; don't repeat a change that was
  thrown away. Keep what makes the starting model work (the placebo baseline and the "news minus no-news" effect)
  unless you have a reason to replace it.

## Idea menu (from research done before any trial, 2026-11-28; try them one at a time)

What we already know about NX3-HI's buys (2019-2023): they make +0.88% a trade before costs and +0.01% after, so a
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

For shorts (NX3-ALL-HI shorts 2019-2023, -0.30% a trade after costs): they did best on the cheapest third of stocks
(+0.77%) and the most volatile third (+0.34%), worst on the calmest third (-1.04%), and better after FDA, merger and
contract news than after earnings news (-0.59%). The sell bar can differ from the buy bar.
