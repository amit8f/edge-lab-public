# NUM1: a Numerai Signals ranker, scored by Numerai forward

Registered 2026-11-24, before any NUM1 number is computed. Source: Amit, 2026-11-24 ("yes to both": the Numerai
Signals ranker from the ChatGPT review; account and model edgelab_signals made the same day). One idea. Phase-2
count 186 -> 187 (phase 1 + phase 2 = 202). Unstaked: no orders, no money in or out.

## Mechanism (rule 10)

Numerai pays its stakers for rankings that add to its hedge fund's book; unstaked, the payoff is a public,
timestamped score that nobody can edit (the credential path, report "Startup plan and best models"). Version 1 uses
only Numerai's own Signals features, so it tests whether we can build and run the pipeline and match Numerai's
example model; adding our news signal for US names is a later, separately registered version.

## Rules

Data: Numerai Signals `v3.1` (train 2003-2018, validation 2019 on). Model: LightGBM regressor (2,000 trees, learning
rate 0.01, depth 5, 31 leaves, 10% of features per tree, seed 7) on every numeric `feature_*` column for `target`,
fit once on train. Each live round: predictions on `live.parquet`, ranked into (0, 1), uploaded to edgelab_signals
(`tracks/numerai/signals.py`). The model is not refit while the test runs.

## Primary test (forward, Numerai's own scoring)

After 60 resolved daily rounds: mean of Numerai's main per-round score for edgelab_signals > 0 with t >= 2 (rounds
overlap by 20 days, so t uses Newey-West with lag 20). Reported beside it: Numerai's example model over the same rounds.

## Reported, no rule

Validation (2019 on) per-date Spearman correlation: mean, Sharpe, by year, US names only, and Numerai's example
predictions on the same rows. Validation can't pass or fail NUM1: Numerai has seen these years.

## Prediction (written first)

Validation correlation small and positive (about 0.01), close to Numerai's example model; forward score about zero.
P(pass) about 0.15.

## Amendment 1 (2026-11-24, before any NUM1 result)

The first training run (sector1 run 45437928633) lost its runner after 45 minutes with no log, most likely out of
memory: the full train file (4.33M rows) read as float64 does not fit in 16 GB. Change: read only the feature
columns (as float32), the date and `target`, and train on every 4th weekly date (Numerai's own example does the
same, since the 20-day targets of neighbouring weeks overlap). The job also gets 16 GB of swap. Model settings,
target, validation and the forward test are unchanged.
