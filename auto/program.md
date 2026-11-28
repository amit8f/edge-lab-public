# AUTO1 agent instructions (frozen with the harness)

You improve a stock-buying model. You may change only `candidate.py`. Reply with the full new file in one ```python
block, preceded by one line starting `IDEA:` that says what you changed and why, in under 25 words.

The harness gives `fit(train)` a pandas DataFrame of past news events (one row each) and calls `score(model, frame)`
on a later year. Each row has the 37 input columns listed in the current file, plus `symbol`, `d` (day), `t` (time),
`price`, and `net` (the return of buying at the event and selling at the next close, after costs). `net` exists in
`train` only for fitting; never read it inside `score`. Each day, the harness buys the first 3 rows (by time) whose
score is above 0, and holds cash otherwise. Your score is the Sharpe ratio of that daily book over three years,
retrained each year on earlier years only.

Rules:
- Imports: numpy, pandas, sklearn, xgboost, lightgbm, scipy, math only. No files, no network, no exec/eval, no dunder.
- Training plus scoring must finish within 10 minutes on 4 CPUs for all three years.
- Make one focused change per try: a new input built from the existing columns, a different target, a different
  model or settings, a stricter buy rule, an ensemble. Learn from the trial history you're shown; don't repeat a
  change that was already thrown away.
- Buying fewer, better trades is fine. Costs are already in `net`.
