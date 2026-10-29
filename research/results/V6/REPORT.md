# V6: why A1 and M6 nearly passed (post-hoc, not pre-registered)

Asked for by Amit on 2026-10-29 ("diagnose M6 and A1, don't drop them"). This is a diagnosis of two
finished screens. It changes neither verdict and adds no idea. It was not pre-registered: the A1 splits
below were first looked at on 2026-10-29 from the committed `A1_picks.csv` before this file was written.
Code: `validation/v6_near_miss.py` (study run 40960531271). Idea count unchanged: 66 (81 with phase 1).

## A1 (Beneish red-flag penny shorts): the near-miss came from two squeezes in the random control

A1's primary result was flagged shorts minus random penny shorts, +12.0% per monthly cohort, t 2.04.

1. **The flagged shorts lost money on their own.** Over 263 flagged shorts the mean 63-day short
   return was -1.7% before costs (median -2.1%, 48% winners, 33% stopped). The 2% round trip and about
   5% of borrow over 63 days come on top, so the book itself lost about 9% a trade. The "edge" was that
   random penny shorts lost more (-11.5% mean, 42% stopped).
2. **Two control trades carry most of the difference.** The best two months give 62% of the total.
   Dropping the best month leaves +6.7% (t 1.77), the best two +4.9% (t 1.16), the best three +3.1%
   (t 0.78), the best five +1.0% (t 0.25).

   | Month | Flag minus control | What happened |
   |---|---|---|
   | 2020-10 | +282% | Control short FPRX: trial news in November 2020; the first close above the stop was almost 4x the entry |
   | 2022-02 | +95% | Control short AGIG: closed 1.90 on 2022-03-04, 11.30 on 03-07 (raw SIP bars); the short lost 6.6x its size |
3. **These are real prices, not data errors.** Raw and split-adjusted SIP bars for the four largest
   losses (AGIG, FPRX, STML, VERU) show genuine one-day jumps (VERU 4.35 to 12.28 on 2022-04-11). The
   adjusted series differ from raw only by later reverse splits.
4. **The M-score itself barely sorts.** Flags with M between -1.78 and 2 lost (-3.7% and -6.9%); flags
   above 2 made +4% to +8% (71 trades). That's post-hoc and thin.

What this means: A1 never had a working short book. Its comparison is decided by whether a random
penny short happens to hit a squeeze, and a 35% stop checked at the close doesn't cap a gap. The same
weakness applies to A1F's forward test, which compares against the same kind of random short control.

## M6 (analyst actions): large-cap growth and momentum, not analyst information

V2 already computed this (`results/V2/V2_results.json`), and V6 only restates it. M6's +3.0% per 6 months over the
equal-weight universe drops to +0.6% (t 0.56) after FF5 plus momentum, with R² 0.61. The loadings are
SMB -0.42 (t -3.0), HML -0.44 (t -2.3) and momentum +0.26 (t 2.4). The picks were big, expensive
stocks that had been rising, and those beat the average stock in 2017-2024. 393 symbols filled the
1,980 pick slots, and 12 names (NVDA 52, MSFT 42, PANW 41, CRM 41, NFLX 39, AAPL, WMT, AMZN, NOW, CMG,
LULU, ADBE) took 23% of them. Against SPY the excess was +1.3% (t 0.91).

Correction: V2's report says M6's picks were "mostly small-cap". The SMB loading is negative, so they
were large-cap. The numbers in V2 are right; that sentence was wrong.

What this means: the analyst signal adds little beyond being long big growth winners. The registered
next step (a factor-neutral M6 twin after M8's 26-week read, failure review decision 7) is the right
test, because it removes exactly these exposures.

## Possible fixes (each would be a new, forward-only idea; none registered)

- A1 or A1F judged on the flagged book's own net result, or against a control with the same stop
  checked intraday, so one squeeze can't decide it.
- M6 ranked within size and industry groups (the factor-neutral twin), as already planned.
