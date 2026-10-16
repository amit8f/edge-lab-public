# V4 report: trading costs measured from quotes instead of assumed

Pre-registration: `research/prereg/V4.md`. Run 38762472323. Running idea count: phase 2 16, total 31.
Cost = half the median quoted spread in 09:31-09:40 ET on entry day plus half the median spread in the
last 10 minutes of exit day (SIP quotes).

| Idea | Picks costed | Assumed | Measured mean | Median | 90th pct | Share above assumption |
|---|---|---|---|---|---|---|
| M10 boom | 6,936 | 0.20% | 0.41% | 0.28% | 0.83% | 64% |
| P3 penny boom | 6,935 | 2.00% | 0.65% | 0.57% | 1.05% | 2% |
| M4 six-factor | 300 | 0.20% | 0.24% | 0.11% | 0.54% | 30% |
| M5 insiders | 300 | 0.20% | 0.36% | 0.26% | 0.79% | 58% |
| M6 analysts | 300 | 0.20% | 0.12% | 0.08% | 0.25% | 15% |
| M7 earnings | 300 | 0.20% | 0.36% | 0.26% | 0.73% | 58% |

## What changes

- M10 re-scored with its measured costs: +0.34% per 21 days, t 0.32 (was +0.55%, t 0.51). Still fails.
- P3 was charged too much: with measured costs it loses 0.85% per 21 days, t -1.54 (was -2.19%, t
  -3.99). The model is still worse than the average penny stock, but the loss is mostly not costs.
- The Shumway delisting bound changes M10 and P3 by 0.01-0.04 points (4 and 6 picks had no exit-day
  quotes).
- M12 is charged 0.41% (the measured M10 mean), per its pre-registration.
- The main-track 0.20% assumption is about right for large names (M4, M6) and too low for small,
  volatile picks (M5, M7, M10).

## Headline audit

200 headlines were sampled (100 classified as analyst actions out of 289,045; 100 unclassified with an
analyst word out of 148,924) in `headlines_to_label.csv`. Blind labelling and precision are next.
