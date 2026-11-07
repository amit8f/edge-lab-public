# BOOM2 (step 2 of the boom picker): which way will a BOOM1 stock move?

Registered 2026-11-07, before any BOOM2 number is computed and before any direction statistic on BOOM1's picks
exists. Source: Amit, 2026-11-07 ("for the direction we need to find the right way to approach and fix it so it
won't be a coin flip ... dig deeper ... build models just for this part"), and the research report
/mnt/project-files/reports/Direction of big intraday moves.md (its D0-D8 plan). BOOM1 passed on 2026-11-07 (55%
of its nightly 20 moved 8%+ from the open, up and down 50/50).

Six counted ideas: D1-NEWS, D1-NONEWS, D2, D3, D4, D6. D0 is descriptive and sets the baselines. Phase-2 count
127 -> 133 (phase 1 + phase 2 = 148). D5 (AI headline tone), D7 (meta-label skip layer) and D8 (squeeze tilt) are
registered later, only if these leave room. Main track only ($5+): BOOM1's universe; penny candidates need their
own BOOM1 first.

## Candidates and data

BOOM1's predicted short-list, never days that turned out to boom: the 20 highest-scored stocks per day from BOOM1's
own walk-forward (same code, seed and test years 2019 .. 2025-09; the `boom2_picks` stage re-runs it and saves the
picks). Label dates end 2025-09-29; the 2025-10-01 holdout is untouched.

For each pick, 1-minute SIP bars (`adjustment=raw`) from 4:00 to 16:00 ET of the session (`boom2_minutes` stage):

- pm_last: the last pre-market trade price at or before the 9:14 bar (what the 16-minute delay lets us see at 9:30);
  gap = pm_last / prior raw close - 1; pm_dv: pre-market dollar volume to 9:14, over the 20-day dollar volume.
- open: the first regular-session bar's open (the 9:30 entry, an opening order sent before the open).
- first5: the 9:30-9:34 bars' return from the open and volume over the day's 20-day average per 5 minutes
  (seen at 9:51); price9_51: the 9:51 bar's open (the 9:51 entry).
- Data check (BOOM1's open item, reported first): the minute bars' high and low against the daily bar's, both
  from the open; the share of picks where they differ by more than 2 points (bad prints in daily bars).

News (scores only, never headline text): NX3's real events decided before the open (BOOM1's `pre_open_news`),
plus the offering flag f_offering of any of them.

## Label and trade

From the 9:30 entry: up-first if the price reaches +8% before -8% (minute high/low), down-first if the reverse,
none if neither by the close. Both inside one minute bar: ambiguous, dropped and counted. A trade on side s exits
at the +8% or -8% barrier when touched first (filled at the barrier, a known optimism for stops, reported), else
at the 16:00 close. Net = s x return - N.cost_frac (NX3's cost model, one day, borrow on shorts); also reported
at 2x costs. D4 uses the 9:51 entry and +-4% barriers from it.

hit = the called side touched first (none counts as a miss). Same-day random control: the same names, random side.

## Walk-forward

Rules and models fitted on picks from earlier test years only: test years 2020 .. 2025-09 (2019 picks only
train). "Best D0 rule" for a test year = the D0 rule with the highest hit rate on all earlier picks.

## Arms

- D0 (descriptive): up-first / down-first / none shares overall, by gap sign, by news, by price bucket; hit of
  always-long, always-short, follow-gap, fade-gap; the share of booms already decided by 9:51.
- D1-NEWS: picks with pre-open news; follow the gap sign.
- D1-NONEWS: picks without; fade the gap.
- D2: picks with a pre-open offering flag; short.
- D3: 18 cells (price $5-10 / $10-30 / $30+, |gap| < 2% / 2-5% / 5%+, gap up / down); each cell takes the side that
  was touched first more often in earlier years (at least 30 earlier picks, else the best D0 rule).
- D4: at 9:51, follow the sign of the first 5-minute return; +-4% barriers from 9:51.
- D6: CatBoost three-class model (up-first / down-first / none; 500 iterations, depth 6, rate 0.05, L2 10, seed
  20261107) on earlier years' picks; inputs: BOOM1's inputs, BOOM1's score, gap, pm_dv, pre-open news scores and
  flags. Score = P(up-first) - P(down-first); trades the side of the score on picks whose |score| is in the top 30%
  of the training picks' |score| (coverage fixed at 30% in advance). The report's "train on candidates vs all
  stock-days" sub-test is dropped: minute labels exist only for candidates.

## Primary tests (each arm separately)

D1-NEWS, D1-NONEWS, D3, D6: hit minus the best D0 rule's hit on the same calls, paired by day, mean >= 4 points
(3 for D3) with Newey-West t (5 lags) >= 2.0, AND net pick-minus-control by day mean > 0 with t >= 2.0.
D2: down-first share on flagged picks >= the all-picks down-first share + 8 points, net short pick-minus-control
mean > 0 with t >= 2.0, and at least 150 flagged picks (else "too few, not judged").
D4: net pick-minus-control by day mean > 0 with t >= 2.0, and net mean > 0.

## Reported, no rule

Balanced accuracy; long and short calls separately; per year; net at 2x costs; mean net per trade; for D6 the
accuracy-coverage curve (10%-100%) and top inputs; the ambiguous count; each arm's number of calls.

## Prediction (written first)

Up and down near 50/50 overall, gap-up picks slightly more often down-first (fade). D1-NEWS passes: 0.35. D1-NONEWS:
0.25. D2: 0.3 (likely too few flagged). D3: 0.2. D4: 0.15 (the delay eats it). D6: 0.3. Most likely outcome: hit rates
in the mid 50s with net returns close to zero; a 70%+ hit rate would point to a bug.
