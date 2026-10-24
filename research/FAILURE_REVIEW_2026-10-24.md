# Why our ideas failed, and what to fix (2026-10-24)

Amit asked (2026-10-24) that a failed idea with a real reason behind it isn't dropped: find the cause,
learn from it, fix it and try again. This review covers every finished phase-2 idea, plus the phase-1 results whose lessons repeat.
Running idea count: 45 (60 with phase 1).

One rule runs through every fix below. A fix chosen by looking at an idea's results has to be tested on
data the fix never saw. Otherwise we keep adjusting until the history looks good, and that is luck we
fitted, not an edge. The free way to get unseen data is a forward shadow book: the idea logs its picks
from now on, with no money, and is judged after a fixed number of trades. Every fix here gets one.

## The seven reasons ideas failed

Most failures share a few causes. Fixing a cause helps every idea that has it.

### 1. A flexible model memorised noise (META1, META2, M10, P3, B0, NX1, phase 1 I-05)

This was the cause we diagnosed today. META2 scored 0.82-0.90 on its training years and 0.50 (a coin flip)
on new ones. The same thing happened to the phase-1 meta-labeling test (I-05), and we repeated it in
META1. That's a lesson we should have carried over.

Why it happens: a stock's return over a few days is mostly noise. In a week of R1 picks, returns spread
about 8.5% around the average, and the best single input explains under 1% of that. A model with hundreds
of trees and 20+ inputs finds patterns in the noise that don't repeat.

What to change for every model from now on:
- Report training score next to test score. A gap above 0.10 AUC means overfitting, and the result is
  thrown out before anyone reads the test number.
- Fewer inputs, chosen before fitting from a published mechanism; monotone constraints where the sign is
  known; heavier regularisation (fewer, shallower trees).
- Cleaner targets: return relative to the stock's own volatility (NX2 did this), or a longer horizon
  where the signal has more time to show over the noise.

### 2. Judging inputs by rank instead of money (found today in META3)

META2's diagnosis found three inputs whose rank correlation with next week's return kept the same sign
in 8-9 of 9 years. But when META3 kept the top half by those three, the money result was zero, even on the
years they were chosen from (-0.01% a week, t -0.17). Reason: returns have fat tails. Calmer stocks rank a
little better on average, but the few huge winners that make R1 profitable are the jumpy stocks. A filter
that cuts them gives up the mean.

Fix: every diagnosis checks means (and the top and bottom 5% of outcomes) before proposing a filter, not
only rank correlations. META3 keeps its free forward log, but I now expect it to fail (about 85%).

### 3. The model learned "risky", not "good" (M10, P3, B0, NX1, A1, M6, M5)

- M10 and B0 are good at finding stocks that will move a lot: M10's picks were 6.5x more likely than
  chance to land in the top 2%. But they fall just as often, so the mean is flat. Volatility was 42% of
  M10's model.
- NX1 fired on no-news days as often as on news days. Its top input was 20-day volatility.
- A1's accounting-flag shorts were mostly small biotechs with a first big revenue year, not fraud.
- M6's analyst ranking and M5's insider buying mostly picked small stocks and momentum exposure; M5's
  pass rested on GameStop in 2020.

Fix: take volatility, size and industry out of both the inputs and the target before training (M12 was
registered for this and hasn't run yet; run it). The other fix is to use what these models really do
well. M10 and B0 predict a big move, not its direction. That's worth something as a risk filter (don't
hold a P1/P2/R1 pick the boom model flags as about to swing hard) or for options straddles (phase 1
I-15). Both would be new registrations, tested forward.

### 4. Bad matching between data sources (G1, W1a, W1d, L1, NX3, I1)

Several ideas never got a fair test because the data join was wrong:
- G1 (customer links in 10-Ks): 15 of 50 links right. Single words matched companies ("Rocky Mountain" to
  Rocky Brands).
- W1a (leak reports): 23 of 50 right. It tagged buyers, sellers and denials as the rumoured target.
- L1: only 27 of 81 headlines matched to a filing were about that filing.
- NX3: 2023 and 2024 show nights of about -7% average, almost surely a few bad prices (the rerun with the
  fixed price basis is going now).
- I1: Whiting's post-bankruptcy shares under the old ticker showed +2,613% in a week.

These are the most worth fixing, because the ideas themselves were never tested. The mechanisms are
published (Cohen and Frazzini 2008 for G1), and the fix is engineering:
- G1: match only exact legal names and the XBRL customer tags in the filing, then hand-check 50 again
  before any return.
- W1a: start with the short-seller group alone (11 of 14 right already), and add a check that the
  company is the target of the rumour, not the buyer.
- Every study caps single-week returns for the QA pass and lists the 20 largest, as B0 and R1 did.

### 5. The edge is real but smaller than the trading cost (P3, I1, R1 penny, S1, P4)

- I1 beat its random control by +0.66% a week, but both lose after the 2% penny round trip. Held 20 days
  instead of 5, it was +1.92% (t 2.01), because the cost is paid once over a longer move.
- R1's rule on penny stocks: -0.57% a week after 2% costs.
- S1 lost money even before costs, so no cost fix helps it.

Fix: for penny ideas, longer holds and fewer trades, and measure our real cost with limit orders (X1 is
logging this) instead of assuming 2%. I1 with a 20-day hold goes into a forward shadow book.

### 6. The published effect has faded (M1, M7, M13, phase 1 I-07)

Lazy Prices (M1) was +0.10% a month before costs in 2017-2025, much less than the papers report.
Earnings surprise (M7, I-07) and the 150-day rule (M13) were flat. These are well known and traded away
in big stocks.

My view: these shouldn't be "fixed". Searching for the version that still works on the same data is how
a study fools itself. The one honest variant is testing them where arbitrage is hard (penny and small
stocks), as a new registration.

### 7. Not enough data to tell (E1, T1, near-misses)

E1 had 33 macro-day trades (needed 100). T1 has 8 years, so one year is one data point. And several ideas
landed between t 1.5 and 2.1, which means "maybe": R1 (1.89), L1 offering/penny (1.91), A1 (2.04 but
uneven), M6 (1.61), I1 20-day (2.01), NX2 vs random (2.77, but 1.55 vs shuffled news).

Fix: R1 and the two L1 cells are already in forward shadow books (FWD1). Add I1 (20-day hold), A1 (with
the biotech first-revenue cases removed by a rule fixed now), and M6 (factor-neutral version, already
logged forward as M8).

## Idea by idea

"Forward" means a free shadow book with its own pass rule. "New idea" means it counts.

| Idea | Cause | Fix | Next step |
|---|---|---|---|
| META1, META2 | 1: memorised noise | Simple score instead | META3 forward log from Oct 30 (expected to fail, see 2) |
| META3 | 2: ranks aren't money | None for now | Free forward log continues |
| M10, P3 | 1 and 3: finds big movers both ways | Use as a risk filter or for straddles | New idea, forward |
| B0 | 3: same as M10 | Same, as an "avoid" flag on P1/P2/R1 picks | New idea, forward |
| M12 | Not run yet | Run it (removes vol, size, industry) | Run the registered screen |
| NX1, NX2 | 3: fires on volatility; weak vs shuffled news | NX2's design was the fix; add capped returns | NX2 shadow book after the data rebuild |
| NX3 | 4: bad price nights | Price-basis bug fixed | Rerun running now; corrected gate soon |
| G1 | 4: 15/50 links | Exact names + XBRL tags | New idea; hand check first |
| W1a, W1d | 4: 23/50 events | Short-seller group first, target check | New idea; hand check first |
| L1 | 7: near-miss cells | Already forward | FWD1 book (first log Oct 28) |
| R1 | 7: t 1.89 | Already forward | FWD1 book (logging since Oct 23) |
| I1 | 5: cost eats it | 20-day hold | New idea, forward |
| A1 | 3 and 7: picks jumpy biotechs | Exclude first-revenue jumps by a fixed rule | New idea, forward |
| M6, M8 | 3 and 7: small-cap tilt | Factor-neutral ranking | M8 forward already; add a neutral twin |
| M5, P4 | 3: rested on GameStop; P4 flat | Insider buying after a drop (combine with R1) | New idea, forward |
| M1, M7, M13 | 6: faded | Don't refit; small-cap test only | Optional new idea |
| M4 | 3: mostly factor exposure | None; it's a factor portfolio | Keep the dashboard label |
| S1 | Loses before costs | None | Closed |
| E1, T1 | 7: too little data | Pool forward (T1 gets one more year each January) | T1 adds one year each January |
| K1, K2 | Stopped by choice | None | Closed |
| N1, M3, W1b, W1c, K3-K7, SYS1, SYS2, SYS2B, P1, P2 | Running or not finished yet | None yet | Diagnose if they fail |

## Suggested order

1. NX3's corrected result (running now).
2. M12, already registered, which tests fix 3 directly.
3. G1 with the fixed matcher. It's the best mechanism we haven't really tested.
4. Forward books for I1 (20-day), A1 (fixed rule) and the boom-model "avoid" flag.
5. W1a, short-seller group only.

Steps 3-5 are new counted ideas, so they'd go to the Nov 1 review unless you want them sooner.

## Decisions (Amit, grill session 2026-10-24, 13:00-13:08 UTC)

Amit said yes to every recommendation except one, which he changed (Q8 part 3, below).

| Q | Decision |
|---|---|
| 1 | Standing rule: training score next to test score; a gap over 0.10 AUC is a fail (PROTOCOL 21). |
| 2 | Standing rule: diagnoses check mean returns and the extreme 5%, not only ranks (PROTOCOL 22). |
| 3 | Run M12 now (registered 2026-10-15). Started 2026-10-24, run 40096838515. |
| 4 | Fix the matchers, with a hand check as the first gate: G2 and W2 registered. |
| 5 | I1 with a 20-day hold as a forward-only book: I1F (FWD2). |
| 6 | Leave M1, M7 and M13 closed; no re-tests on the same years. |
| 7 | A1 with the first-revenue exclusion as a forward-only book: A1F (FWD2). The M6 factor-neutral twin waits for M8's 26-week read. |
| 8.1 | One combined book of all near-miss books: COMBO1 (FWD2). |
| 8.2 | Re-score finished penny screens with measured costs once X1 has 30+ fills (a re-judgement, not a new idea). |
| 8.3 | Changed by Amit: instead of giving up on fast news, build a listener that reads the sources directly (SEC's live filing feed, press-release wires, the working outlet feeds) every ~20 seconds in chained GitHub jobs, 4am-8pm ET. Logging only for 2 weeks, then a speed report (our time vs Benzinga's vs the price move). Azure's $100 credit only if GitHub's gaps are too big. Not counted until a trade is tested. |
| 10 | No fast-trade idea is registered before NX3's corrected result and the 2-week speed report. |
| 11 | Order: (1) M12; (2) PROTOCOL 21-23; (3) the news listener; (4) forward books I1F, A1F, COMBO1; (5) G2 and W2 hand checks; (6) the cost re-check. All five new ideas registered now, not at the Nov 1 review. |

Idea count after these: 50 (65 with phase 1).

The listener (`core/listener.py`, `.github/workflows/listen.yml`) was checked from GitHub on 2026-10-24.
Answering: SEC's live filing feed, PR Newswire, Business Wire (its all-news feed was empty on a Saturday)
and 12 outlet feeds. Blocked for any GitHub client: GlobeNewswire (403), ACCESS Newswire (Cloudflare);
Newsfile has no feed. It logs from the first weekday after merge; the speed report follows two weeks later.

G2's matcher uses exact names and stricter sentences. The XBRL customer tags mentioned above are left
out of G2 to keep it buildable; they can be a later cross-check.

