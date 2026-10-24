# TB1: TabICLv2 as META1's take-or-skip model (main track)

Registered 2026-10-24, before any TB1 result exists. Source: the weekly research scan
(`/mnt/project-files/research-scan/2026-10-24.md`, pick 2); Amit: "add all". Phase-2 count at
registration: 52 (main 26). Phase 1 + phase 2 = 67.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** As META1 (`prereg/META1.md`): the sellers who push a quality stock down in a bad
   week. TB1 only changes the model that decides which R1 picks to take.
2. **Why us.** As R1: 20 liquid names a week; any fund could trade them. The point is model choice on a
   small sample (about 8,800 picks), where a no-tuning model may overfit less than gradient boosting.
3. **Published.** TabICLv2 (Qu et al., ICML 2026; `soda-inria/tabicl`, BSD-3) beats tuned gradient
   boosting on public tabular benchmarks; practitioners report the edge fades on very large tables. It is
   pretrained on synthetic data only, so it holds no market-data look-ahead.

Disclosed: META1 has already failed on these picks with LightGBM. TabICLv2 was chosen from an outside
benchmark, not from META1's returns (PROTOCOL 23), so a historical test is allowed.

## Everything as META1 except the model

Same picks, 18 inputs, label, walk-forward folds, take rule (at or above the median probability on the
model's own training rows), primary test and pass. Model: `TabICLClassifier` with default settings, the
package pinned to one commit of `soda-inria/tabicl` and the checkpoint `tabicl-classifier-v2-20260212.ckpt`
(sha256 logged), CPU, `random_state=20261024`. No tuning. Inputs as META1 (no extra scaling).

## Primary test and pass

META1's: per test week (2019-01 .. 2025-09) mean `fwd5` of taken picks minus mean of all picks, Newey-West
t (4 lags); pass to a forward shadow book (PROTOCOL 19) when mean > 0 with t >= 1.5 and the taken picks'
own mean after 0.2% costs > 0. Rule 21: in-sample AUC beside out-of-sample AUC; a gap > 0.10 fails it
first. META1's LightGBM numbers are shown beside TB1's on the same folds. Rule 22: the filter's effect on
the mean and on the top and bottom 5% of pick returns is reported.

Prediction: ~75% fail (META1 found little to learn in these inputs).
