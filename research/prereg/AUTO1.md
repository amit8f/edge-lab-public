# AUTO1: a self-improving research loop for the buy side, kept apart from the live models

Registered 2026-11-28, before any AUTO1 trial runs. Source: Amit, 2026-11-28 ("I want to try the autoresearch but we
are not doing it on the main model ... open a branch test it and when we know its not a luck we could update it"),
after reading karpathy/autoresearch. One idea: phase-2 count 198 -> 199 (phase 1 + phase 2 = 214). The trials inside
the loop are not separate ideas, but every one is logged and counted in the deflation below.

## What it copies from autoresearch

An AI agent edits one file, runs a short fixed-budget experiment, keeps the change if the score improves and throws it
away otherwise, then repeats. The scoring code is read-only to the agent. Here:

- `auto/harness.py` (read-only to the agent): loads NX3's events and costs, trains whatever `candidate.py` defines,
  scores it. Frozen at registration; a change to it starts a new idea.
- `auto/candidate.py` (the only file the agent edits): inputs and transforms built from NX3's 37 event inputs, the model
  (numpy, pandas, scikit-learn, XGBoost or LightGBM only) and the rule that turns scores into buys.
- `auto/program.md`: the agent's instructions, frozen like the harness.
- Experiments live on their own git branch, `auto1`, made by the loop's workflow. Live code (`tracks/`, `paper/`,
  `adaptive/`, `models/`) never imports `auto/`, and a test enforces that. NX3, NX3-HI and every live book are unchanged.

## The score the agent sees (2019-2022 only)

A long-only book of at most 3 buys a day (Amit's real account can't short yet), walk-forward by year: train on years
before Y, score Y, for Y in 2020, 2021, 2022. Costs as LONG (quoted entry half-spread, closing-auction exit,
`adaptive/costs.py`). Score = annualised Sharpe of the daily net book. A change is kept only if it beats the best kept
score by at least 0.05. The agent never sees any 2023-2025 number.

## Budget and safety

Each trial: one call to a free model (Gemini or Groq, as SYS2B), then at most 10 minutes of training on an Actions
runner. Up to 40 trials a night in edge-lab-public (public repo, so no minute cap). The agent's code runs in a step with
no secrets in its environment and no network (`unshare -n`), and an import check rejects anything outside the allowed
libraries before it runs. Every trial (diff, score, kept or not, time) is appended to
`research/results/AUTO1/trials.jsonl`.

## Checks, in order (each looked at once)

1. After 200 trials, or 21 nights, whichever comes first: the best kept candidate is scored on 2023 against plain
   NX3-HI buys (the frozen model, same costs). Pass: its 2023 daily Sharpe is above NX3-HI's AND its deflated Sharpe
   ratio (Bailey and Lopez de Prado 2014, N = every trial run, variance of the trial Sharpes) gives probability >= 0.95
   that the true Sharpe is above 0.
2. Only if 1 passes: one look at 2024-01 to 2025-09, LONG's pass rule (own net per trade > 0, own daily NW t >= 2,
   daily gain over plain NX3-HI buys mean > 0 with NW t >= 1.5).
3. Only if 2 passes: the candidate is frozen into `models/auto1/` and runs as a log-only forward copy beside the live
   NX3-HI buys (no orders), on the dashboard's Live tab. It replaces nothing until it beats the frozen NX3-HI forward
   over at least 4 weeks and 30 closed trades (Plan v3's promotion rule), and real money still needs PROTOCOL.md's 100
   trades and Amit's signature.

A failure at 1 or 2 ends the idea; a new harness or a new score is a new registration.

## Prediction (written first)

The loop finds 2019-2022 gains easily (that's what search does), and most of them don't survive 2023. P(pass check 1)
about 0.3, P(pass 2 as well) about 0.1.

## Amendment 1 (2026-11-28, before any trial ran)

Amit asked (2026-11-28) for the buy search to try "different models, different inputs" informed by what traders,
books and papers use. Research notes: `/mnt/project-files/research_notes/Better news buy signals/` (practitioner
rules, academic evidence, ML setups). Changes, all made before trial 0:

1. `program.md` gets an idea menu from that research (inputs, targets, models to try one at a time).
2. The event table adds `entry`, `prev_ac`, `prev_c`, `dv20`, so a candidate can compute the jump before entry.
3. Safety fixes found while building: `score()` no longer receives the outcome columns (`net`, `ret`, `next`, `days`),
   and the harness rejects a candidate whose scores change when the frame is cut at a random time (a look-ahead
   within the day, such as same-day ranks that include later events).

Nothing else changes: same score, same keep margin, same checks.
