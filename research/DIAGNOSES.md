# Why ideas lost, and what we could change (standing log, started 2026-11-01)

PROTOCOL rule 24: every FAILED or STOPPED idea gets an entry here in the same PR as its outcome. This file picks up
where `FAILURE_REVIEW_2026-10-24.md` stopped (it covers everything up to Oct 24 and its seven causes: 1 a flexible
model memorised noise, 2 ranks judged instead of money, 3 the model learned "risky" instead of "good", 4 bad
matching between data sources, 5 the edge is real but smaller than the cost, 6 the published effect has faded,
7 not enough data). Other places with diagnoses: `results/V6/REPORT.md` (A1 and M6), `results/META2/META2_diagnosis.json`,
`MECHANISMS.md` (who loses money to each idea). Numbers below are copied from `IDEAS.md` and the result files.
"Shown" means the stored numbers support it; "guess" means they don't.

A fix chosen from these results is a new idea that must be tested forward (rule 23). Nothing here is registered
until Amit says so; section C lists what Claude suggests.

## A. Failed since the Oct 24 review

### W2 (short-seller reports, 303 events, t -0.80) - cause: 3 and 5 together, shown

Post-hoc stage `diagnose` (`results/W2/W2_diagnosis.json`; per-trade means, so slightly different from the per-day
test number of -1.09%):
- The typical trade works. Gross median +0.67%, net median +0.20%. The mean is -0.95% because of the tails: the
  worst 5% of trades (15) lost 52% each, the best 5% gained 29% each. Without both ends the mean is +0.32%.
  2021 (the squeeze year) is the worst: -10.9% a trade. That is consistent with short squeezes in heavily shorted
  names, but the trade-level names weren't looked at, so which names is a guess.
- Costs are 0.46% a trade on average, larger than the typical net gain.
- The move before our entry matters. Where the stock had already fallen more than 3% from the prior close by
  our entry (144 trades) the net was +1.08% a trade (median +0.62%). Where it had risen (85 trades) the net
  was -4.07%. This split was chosen after seeing results, so it is a hint only.
- Article time doesn't matter (before open -0.8%, in session -1.1%, after close +1.8% on 9 trades).
- Not mainly the data: the hand check found 40 of 50 events right, so a fifth of the events are not real short reports; that dilutes the result but doesn't explain the squeeze tail.
What would show it's real: the same rule on days after registration, with a stop that cuts the squeeze tail.

### CH tree arms (chart-reading XGBoost and LightGBM on the 5-minute scanner) - cause: 1, shown for three of four

Scored 2026-11-01 (`results/CH/CH_result.json`; image arms and CHR-P in the next two entries). Under the registered order, rule 21 comes first.
- CHX-M, CHX-P, CHL-P are flagged overfit: their in-sample edge is 2.1x to 4.4x the out-of-sample one. By rule they
  fail before their t is read.
- CHL-M is not overfit but ends at t 2.92 with picks' own mean -0.09% a trade: the edge over random is smaller
  than the cost of trading the flagged stocks at all.
- The penny arms are the interesting ones: out of sample t 3.8 to 4.0, picks' own mean +0.6% a trade after costs,
  factor alpha t 3.6 to 4.4, ahead of the control in 6 or 7 of 7 years, shorts (t ~4) stronger than longs (t ~3).
  The registered rule still calls them failures because the model memorised too (+5% in sample).
- The ORB benchmark didn't reproduce the published result (main +0.01%, t 0.19; penny +0.27%, t 0.83), so any edge
  comes from the models, not from the opening-range idea.
- Pattern: this is NXM-RF again. NX3's tamer trees (NXR: depth 2, large leaves) kept the signal and lost the
  memorising. The same fix is the obvious one here.
Suggested fix (section C, item 3): tamer penny and main arms, history can only reject, a forward book for any that
pass. Odds Claude gives the penny arm: about 0.3 (the out-of-sample number is strong, but costs on penny shorts
depend on borrow, which P1 showed is often missing); the main arm about 0.1.

### CH picture arms (CHI-M, CHI-P: a small CNN reading the chart) - cause: 1 and 7, partly shown

Scored 2026-11-01 after Amendment 1 shrank the net (`results/CH/CH_result.json`). Rule 21 comes first.
- CHI-M: in sample +0.22% a day over the control, out of sample -0.04%, so it learned something in sample and nothing
  that carried over. t -0.60, picks' own mean -0.72% a trade. It made no picks at all in 2025 (why is a guess: the
  model's predictions may all have sat under the cut; the saved predictions weren't inspected).
- CHI-P: +0.82% in sample vs +0.36% out (2.3x, flagged), t 1.90, CI just under zero, picks' own mean -0.81% a trade.
  Its 2025 daily mean is +16%, so a few huge days likely carry the t of 1.90 (not checked).
- Why it lost: guess, two ways. (1) The net was cut down to fit free compute (100,000 sampled training rows a fit,
  3 passes, 8/16/32 filters), and the paper it copies used far more data and larger nets; a fair test of the picture
  idea wasn't possible here. (2) The tree arms on the same rows found the penny signal, so the signal exists in
  numbers; the picture adds nothing the 68 numbers don't already hold. Neither is proven.
- What would change it: more compute (a paid GPU, which Plan v3 says to flag) or a different picture. Odds Claude
  gives either: about 0.05, because the tree arms already hold the same information. Closed; no fix suggested.

### CHR-P (penny tamer LightGBM on the chart scanner) - cause: 5, shown

Scored 2026-11-01 (`results/CH/CH_result.json`), registered the same day as `prereg/CHR.md`.
- The fix worked on the thing it targeted: in sample +1.75% vs out of sample +0.97% (1.8x, under the 2x limit), where
  CHX-P and CHL-P were at 4.3x to 4.4x.
- It paid for that. The out-of-sample edge over the control fell from +1.2% a day to +0.97%, and t from about 3.9 to
  2.91 (rule 12 needs 3.0). Picks' own mean after costs fell from +0.6% a trade to +0.01%. The edge over random is
  real (CI +0.26%..+1.66%, alpha t 3.05) but is about the size of what trading these stocks costs.
- Longs carry it (t 2.80); shorts are weak (t 1.52), and penny shorts also depend on Alpaca having borrow, which P1
  showed it often doesn't.
- Rule 19 on paper: hand check passed, t >= 1.5, picks' own mean > 0 (by 0.01%), so it counts as a near-miss.
  CHR.md says a fail closes, so Amit decides whether it gets a near-miss shadow book (his Oct 24 rule).
- Not checked: the by-year means are 3% to 4% a day in 2021 and 2023 and 3% in 2025 against about 0.2% in other years,
  so a few extreme days may carry the result (rule 22 asks for means and the extreme 5%).

### G2 (customer-supplier links, exact-name matcher, t 0.10) - cause: 6 (faded), shown

The matcher was fixed (46 of 50 links right, 1,759 links) and the effect still isn't there: +0.04% a month, FF5 +
momentum alpha t 0.38, over 103 months with about 96 ranked suppliers a month. G1's 15 of 50 was not the reason it
failed. Same story as M1, M7 and M13: published effects from the 1990s-2010s have faded in 2017-2025 on free data.
Fix: none worth a registration. Closed.

### RL1 and RL2 (monthly rank-label XGBoost, long-short) - cause: 1 and 5, shown

Both flagged by rule 21 (in-sample +7.4% to +8.0% a month against -1.06% out of sample). RL2 cut the memorising
(+1.67% in sample) and added four event inputs (+0.15% a month, t 0.87), but its picks are still about even with
a random long-short before costs, and the 0.65% round trip does the rest. Closed: the monthly cross-section with
free data holds no signal that a model can find, whichever label it learns.

### TB1 (TabICLv2 as META1's take-or-skip model) - cause: 1, shown

Out-of-sample AUC 0.50 against 1.00 in sample; it takes 93% of picks. A bigger model memorises more. Same family as
META1, META2 and NXS-ML. Closed.

### SD1 (fresh vs stale accounting composite) - cause: 6, shown

The fresh composite lost -0.18% a month (alpha t -1.37), the stale one -0.08%. Nothing to replicate in $300M+
stocks. Closed.

### PD1 (post-earnings drift in penny stocks, t 1.37) - cause: 7, partly shown

+3.6% per 60 sessions after costs, but picks' own mean +2.3% (t 0.63) and factor alpha -1.1%. 967 events over 91
months is enough; the spread is wide (CI -1.5% to +9.2%). Exploratory 40-session hold +4.1% (t 1.73) was
read from the same results. Fix: a 40-session forward book would be a new idea with low odds (about 0.15).
Not suggested.

### EAP1 (earnings announcement premium) - cause: 6, shown

+0.01% a week (t 0.10). Only 65% of picks really reported in the hold week, but those that did beat the control
by just +0.06% (t 0.44). So getting the dates right doesn't bring it back. Closed.

### INS1 (insider buys, $5+ stocks, t 0.60) - cause: 5 and 6, guess

+0.19% per 20-session hold vs control, picks' own mean +0.52% against control +0.34%. Right size for a faded effect. Whether
clustering buys (2 or more insiders) helps is untested for $5+ stocks; P4 tested clusters in pennies and found
nothing (t -0.02). Fix: low odds (about 0.1). Not suggested.

### NXS-ML and EXQ (a second model on top of NX3's picks) - cause: 1, shown

NXS-ML: rank correlation 0.23 in sample, 0.022 out of sample. EXQ: overfit and -0.22% a day against EXA1. NX3's
few thousand useful days can't carry a second learned layer. Fixed by using the plain formula (NX3-ALL) and the
fixed ATR exit (EXA1). The successful version of this lesson: NXM-RF failed rule 21, and the tamer NXR models
(depth 2, large leaves) passed.

### AI bot C and its Groq copy (K4, K7) - cause: 5 and a pick bias, partly shown

-13.3% and -17.9% over 8 sessions against -3.4% and -7.4% for coin flips on the same picks. A coin flip losing 3-7%
is the cost of trading every 10 minutes (shown). C winning 21 of 96 trades against about half for the coin is more
than luck would give (shown at that size; 8 sessions is short). Why it picks the losing side is a guess (buying
surges that fade). Already in motion: K8 (once a day) and the proposed C-INV (registered 2026-11-01).

### P1 (shorts after bad penny news) - cause: the filter, shown

71% of signals can't be shorted on Alpaca (35 of 49), 16% hit the position cap. The idea is untested as much as
failed. Proposed in the monthly review: log skipped signals as hypothetical shorts (no idea count).

## B. Fixes named in the Oct 24 review that were never registered

- **B0 / M10 / P3 as an "avoid" flag.** M11 then tested the filter form: it halved the share of picks falling
  30%+ and added no mean. Low odds, not suggested.
- **M5 and P4 "insider buying after a drop, combined with R1".** INS1 and P4 cover insiders alone and found
  nothing. Combining two weak signals on a few hundred events won't have power. Not suggested.
- **W1a's big-outlet group.** Dropped at the hand check (1 of 21 right). A fix needs a parser for who the target of
  a rumour is, which we don't have. Parked.
- **M6 factor-neutral twin.** Waits for M8's 26-week read, as decided.
- **Registered but not run yet** (not failures): W1b, W1c, N1, COMBO1.

## C. Suggested adjustments (not registered)

1. **W2F** (forward only, 1 idea) - **skipped by Amit, 2026-11-01**; W2 stays closed: the W2 events, but enter only if the stock is already down more than 3% from the
   prior close at entry, and stop out if it rises 15% against us; shorts only in names Alpaca can borrow. Judged
   after 60 trading days and 30 closed trades, minus a same-day random control, NW t >= 1.5. Why: it targets
   the two things the diagnosis found (a squeeze tail and entries after the move). Odds Claude gives it: about 0.15,
   because the 3% rule was picked from these results and the stop may cost the best winners too. It also needs
   short sellers' reports to keep coming (about 40 events a year).
2. **P1 skipped-signal log** (no idea count), already in the monthly review.

3. **CHR-P** (registered 2026-11-01, `prereg/CHR.md`; Amit chose penny only, CHR-M not registered) (was: CHR-P and CHR-M, 2 ideas): CH's tree arms with NXR-style regularisation (LightGBM depth 2, 500-row leaves,
   L2 10, the same 68 inputs and rows), rule 21 first, then rule 12 and 11; history can only reject (as NXR); passing
   arms get a $1,000 (penny) or $5,000 (main) shadow book judged by rule 20. Cost: about 30 minutes of GitHub time.
   Odds: CHR-P about 0.3, CHR-M about 0.1. Shorts in penny stocks only where Alpaca allows borrowing.

Everything else above is closed with a reason.
