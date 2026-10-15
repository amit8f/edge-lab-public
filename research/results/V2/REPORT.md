# V2 report: every finished screen re-judged on factors and luck

Pre-registration: `research/prereg/V2.md`. Runs 38762455822 (V2) and 38762460078 (blind second
implementation, written from the pre-registration without reading V2's code). Running idea count:
phase 2 16 (main 10), phase 1 + phase 2 31. V2 itself is not counted.

## Verdict

One idea technically survives the pre-registered rule (alpha > 0, t at or above its bar, Romano-Wolf
p < 0.05): **M5, insider buying plus buybacks**. It doesn't survive a look at where the alpha comes from.

| Idea | Raw excess | FF5+Mom alpha | t | Romano-Wolf p | Holm p | Deflated Sharpe | Survives |
|---|---|---|---|---|---|---|---|
| M4 six-factor | +1.2% / 6m | +1.8% | 1.93 | 0.002 | 0.16 | 0.03 | no (t < 2) |
| M5 insiders + buybacks | +2.5% / 6m | +8.7% | 2.61 | 0.000 | 0.03 | 0.002 | **yes (rule)** |
| M6 analyst actions | +3.0% / 6m | +0.6% | 0.56 | 0.72 | 1.00 | 0.07 | no |
| M7 earnings jumps | -0.7% / 6m | -1.8% | -1.94 | 1.00 | 1.00 | 0.001 | no |
| M10 boom model | +0.5% / 21d | +0.5% | 0.72 | 0.67 | 1.00 | 0.02 | no |
| M11 boom + quality | +0.5% / 21d | +0.3% | 0.52 | 0.79 | 1.00 | 0.03 | no |
| P3 penny boom | -2.2% / 21d | -2.1% | -3.96 | 1.00 | 1.00 | 0.00 | no |

## Why M5's pass isn't an edge

- The factor adjustment adds 6 points rather than removing them: M5's picks load -1.25 on momentum
  (insiders buy after falls) and +0.85 on value, and both factors moved against that mix.
- Two signal dates, 2020-08-31 and 2020-09-30, have excess returns of +103% and +137% (GameStop).
  Dropping the 2020 signal dates gives alpha +1.4%, t 0.93. Winsorising at 2/98% keeps t 2.58 but
  halves alpha to 3.9%; 2021-2025 alone gives t 1.36 (exploratory checks, run after the result).
- The deflated Sharpe ratio, which allows for the 28 ideas tried, is 0.002.

So M5 goes on the dashboard as "passed V2 on one event; deflated Sharpe 0.002", not as a pick
source. A forward test would need its own pre-registration.

M6's raw +3.0% shrinks to +0.6% once factors are removed: its picks were mostly small-cap and
momentum exposure. M4's raw edge was close to its factor exposure too.

## Checks

- Blind re-implementation: alphas, t-statistics, loadings and R² agree to at least 10 significant
  digits for all seven ideas; Romano-Wolf p and deflated Sharpe differ only in the third decimal
  (different bootstrap draws).
- Placebo (52-week misaligned calendar): loadings and R² collapse, as they should.
- French files: sha256 recorded in V2_results.json; factor days 1963-07-01 to 2026-08-31.
