# FWD2: three forward-only books from the failure review (I1F, A1F, COMBO1)

Registered 2026-10-24, before any forward data exists, from Amit's decisions in the failure review grill
(`research/FAILURE_REVIEW_2026-10-24.md`, "Decisions"). Three new counted ideas: I1F (penny), A1F (penny)
and COMBO1 (cross-track). Phase-2 count after them: 48 (main 23, penny 9, news 7, cross-track 8, crypto 1);
G2 and W2 (`prereg/G2.md`, `prereg/W2.md`) bring it to 50. Phase 1 + phase 2 = 65.

Each change below was chosen after seeing the parent screen's results, so the history can't test it.
**Only forward data counts**, on Plan v3's path (PROTOCOL rule 16: no historical gate; rule 20's test
decides). No orders: each book is a hash-chained ledger (`core/ledger.py`) with a same-day random control.

## I1F: industry lead-lag, 20-day hold

I1 (`prereg/I1.md`) exactly, except the exit: the close of the 20th trading day from entry, not the 5th.

- Why: I1 beat its control by +0.66% a week, but both lost after the 2% penny round trip. Its 20-day
  variant (reported alongside, not a test) was +1.92% over control, t 2.01, because the cost is paid
  once over a longer move.
- Signal weeks overlap under a 20-day hold, so each signal week is a cohort; the weekly series is the
  mean over cohorts that close that week.
- **Pass (rule 20 form):** at the first date with >= 60 trading days and >= 30 closed picks, cohort
  pick-minus-control after costs > 0 with Newey-West t >= 2.0 (4 lags), and the picks' own mean after
  costs > 0.
- Prediction: about 70% fail (the 20-day number was the best of three holds).

## A1F: accounting red flags without first-revenue jumps

A1 (`prereg/A1.md`) exactly, with one exclusion fixed now and the borrow rule A1 already requires
forward:

- **Excluded:** flags where prior-year sales were under $10M, or SGI (sales growth index) is above 3.
  A1's flags were mostly small biotechs in their first big revenue year (IOVA sales x138, SCYX x27),
  which the M-score reads as manipulation.
- Shorts only when Alpaca marks the stock easy to borrow on the entry day (PROTOCOL rule 6).
- Monthly cohorts held 63 trading days with the 35% stop, so a pass needs time: the first judgement is
  at the first date with >= 60 trading days and >= 30 closed shorts.
- **Pass:** cohort flag-minus-control short return after 2% costs and 20% borrow > 0 with Newey-West
  t >= 2.0 (3 lags), and the flags' own mean after costs > 0.
- Prediction: about 70% fail (few flags a month: A1 had about 5, the exclusion cuts more).

## COMBO1: one book of all near-miss signals

Each near-miss book is weak alone. Their errors come from different events, so an equal-risk mix should
be steadier than any one of them. Square-root-of-N diversification is the textbook reason (Grinold and
Kahn's fundamental law); it holds only if the books' returns are close to independent, which the
forward data will show.

- **Members, fixed now:** the forward books R1, L1 offering/penny, L1 late filing/main (FWD1), NX2
  (FWD1, once its live log starts), I1F and A1F. A book joins on its first logged trade. Nothing else
  joins without a new registration.
- **Daily return:** the mean, over member books with at least one open trade that day, of each book's
  pick-minus-control return that day after costs, each scaled to 1% daily volatility by its own
  trailing 60-day realised volatility (equal weight until a book has 60 days). No other weights, no fitting.
- **Pass:** after >= 60 trading days with at least two members open and >= 60 closed trades in total,
  mean daily COMBO1 return > 0 with Newey-West t >= 2.0 (5 lags). Correlations between members are
  reported alongside.
- Prediction: about 60% fail.

## Shared

Holdout and history: no historical screen is run for these three. Forward data starts with the first
signal day after this registration is merged. A missed day is logged as missed, never back-filled from
later data.

## Live book details (2026-10-24, before any forward data)

How the I1F and A1F logs are built, fixed before their first signal day (Fri 2026-10-30). Both run inside
the nightly M4 job (`tracks/main/longterm/live.py`, `tracks/penny/fwd2_live.py`) and write to its ledger,
kinds `I1F_picks` and `A1F_picks`.

- **Rows:** the weekly panel's row for t (`weekly_panel.symbol_rows`) for every common stock, from the
  job's ~420-day bar window. The penny flag is I1's formula. The main flag is that night's M4 universe
  (one listing per company, $300M+ cap) with I1's bar floors ($5+, 20-day dollar volume $2M+, 253+ bars).
  CIKs come from SEC's current ticker map; industries from EDGAR's current SIC per company (as in I1),
  cached in `state/longterm/sic/`.
- **I1F control:** as many penny names as picks, drawn from the penny universe minus the picks with
  `default_rng(I1 SEED + t's ordinal)`.
- **A1F:** each 10-K is scored from that night's companyfacts zip, a filing enters once (the record lists
  every flagged accession, including skipped ones). Prior-year sales are this year's sales divided by
  SGI. Easy to borrow means Alpaca's `shortable` and `easy_to_borrow` that night; a flag that isn't is
  logged as skipped. The control is drawn from the easy-to-borrow penny names that aren't flagged, so
  flags and control face the same borrow rule, with `default_rng(A1 SEED + t's ordinal)`.
- **Returns** are read later from SIP all-adjusted bars: entry at the next session's open; I1F exits at
  the close of the 20th session from entry; A1F at the first close 35%+ above entry or the 63rd session's
  close. Costs: I1F 2% round trip; A1F 2% plus 20% a year borrow.
- A day the job misses stays missed. Days before 2026-10-26 are built and printed only (a dry run), never
  logged.
