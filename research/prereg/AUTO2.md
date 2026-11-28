# AUTO2-AUTO4: self-improving loops for NX3-HI, the Ensemble and LightGBM, each on its own branch

Registered 2026-11-28, before any AUTO2-4 trial runs. Source: Amit, 2026-11-28 ("I want also the self-improving loop
for every good model we have to try and maximize it potential ... remember to do it like a branch", then "NX3-HI
ensemble and LGBM start with"). Three ideas: phase-2 count 199 -> 202 (AUTO2 200, AUTO3 201, AUTO4 202; 215-217
with phase 1). Trials inside a loop are not separate ideas; every one is logged and counted in the deflation.

Same method as AUTO1 (`research/prereg/AUTO1.md`), with three differences: each loop starts from a model we already
trade or shadow, its book trades both sides, and the bar to beat is that model's own starting recipe.

## The three loops

| Loop | Branch | Starts from (`auto/loops/<loop>/start.py`) |
|---|---|---|
| AUTO2 | `auto2` | NX3-HI: NX3's XGBoost news model, trade when confidence is 1.5x the firing line |
| AUTO3 | `auto3` | NXM Ensemble: mean of XGBoost, LightGBM, Huber linear, random forest and MLP effects, each over its own firing line |
| AUTO4 | `auto4` | NXM LightGBM arm |

Each start file rebuilds the model's design inside the loop (placebo baseline, news model on the excess label, effect
= news(x) - news(x with the news inputs at 0), firing line = 90th percentile of training |effect|). The agent edits
only `auto/loops/<loop>/candidate.py`, which begins identical to `start.py`. Live code never imports `auto/` (test
enforced) and NX3, NX3-HI, NX3-ALL and the NXM shadow books stay as they are.

## Harness (`auto/harness2.py`, frozen; a change starts a new idea)

- Events: NX3's real and placebo events 2016-10 to 2025-09 (exit on or before 2025-09-30), 37 inputs plus `entry`,
  `prev_ac`, `prev_c`, `dv20`. `fit()` gets real and placebo rows with the label input `mret`; `score()` gets real
  rows only and no outcome columns.
- Book: a candidate's score is in units of its bar. Each day the first 3 events by time with score >= 1 are bought and
  the first 3 with score <= -1 are sold short (NX3-HI's 3+3). Equal weight, cash on a day with none.
- Costs: flat IBKR-style cost per trade from `nx1_study.cost_frac`, shorts pay borrow.
- Score the agent sees: annualised Sharpe of the daily net book, walk-forward 2020, 2021, 2022 (train on events whose
  exit is 14+ days before Jan 1 of the year). Keep a change only if it beats the best kept score by 0.05.
- Same safety as AUTO1: import allow-list, no file/exec/dunder access, no secrets and no network while candidate
  code runs, look-ahead check (scores must not change when the frame is cut at 6 random times). Time limit per trial
  15 minutes (AUTO3: 30, five learners).
- Agent: free models only (Groq, then Gemini), up to 15 trials a night per loop, 200 trials maximum. Trials are
  appended to `research/results/AUTO<n>/trials.jsonl` on the loop's branch.

## Checks, in order (each looked at once per loop)

1. After 200 trials, or 21 nights, whichever comes first: the best kept candidate on 2023 against its own `start.py`
   (same harness, same costs). Pass: its 2023 Sharpe is above the start's AND the deflated Sharpe probability
   (Bailey and Lopez de Prado 2014, N = every trial run in that loop) is >= 0.95.
2. Only if 1 passes: one look at 2024-01 to 2025-09. Pass: own net per trade > 0, own daily mean NW t (5 lags) >= 2,
   daily gain over `start.py` mean > 0 with NW t >= 1.5.
3. Only if 2 passes: frozen into `models/auto<n>/` and run as a log-only forward copy beside the model it came from
   (no orders), on the dashboard. It replaces nothing until it beats the frozen original forward over 4 weeks and 30
   closed trades (Plan v3), and real money still needs PROTOCOL.md's 100 trades and Amit's signature.

A failure at 1 or 2 ends that loop's idea. Each loop passes or fails on its own.

## Prediction (written first)

Search finds 2020-2022 gains easily and most fade on 2023. The starting recipes already sit near the top of what
NX3's inputs can do (CatBoost, KRR and the Ensemble tied XGBoost in history), so the room is small. P(pass check 1)
about 0.25 per loop, P(pass 2 as well) about 0.08 per loop; expected full passes about 0.25 of 3.
