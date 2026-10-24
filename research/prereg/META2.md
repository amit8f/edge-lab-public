# META2: take-or-skip for R1's picks, with the news and short-selling behind each drop (main track)

Registered 2026-10-24, before any of this idea's news or short-selling inputs were joined to R1's picks or
any model was fit. Amit's request after META1 failed (2026-10-24, "do that", answering the proposal to retry
with new information the first filter doesn't see). Phase-2 count at registration: 44 (main 22, penny 7,
news 7, cross-track 7, crypto 1). Phase 1 + phase 2 = 59.

## Why a second try is a new idea

META1 (`prereg/META1.md`) failed: its 18 price, volume and filing-count inputs couldn't tell R1's winners
from its losers (out-of-sample AUC 0.504). R1 already picks on price, so those inputs mostly repeat what R1
knows. META2 adds information R1 never sees: what the news said during the drop, and whether short sellers
were piling in. Everything else (picks, label, model settings, walk-forward, rule, test, pass bar) is
META1's, unchanged, so the only difference is the new inputs.

- **Mechanism:** Chan (2003): drops that come with news keep falling, drops without news reverse. Short
  sellers are on average informed (Boehmer, Jones and Zhang 2008; Engelberg, Reed and Ringgenberg 2012:
  shorts trade on news); a drop where short selling rises is more likely to be information than forced
  selling, which is what R1 is paid for (Nagel 2012).
- **Prediction written now:** about 70% this fails too. The new inputs are coarse (headline counts and a
  word-list tone, short interest twice a month), and META1 showed how little a model finds in this set.

## Data

- Picks, `fwd5`, label, the 18 META1 inputs and the data rules: exactly as META1 (R1's 8,790 picks,
  week-ends 2017-01-06 .. 2025-09-19; nothing from 2025-10-01 on).
- News: Benzinga articles from Alpaca's news API (all symbols, fetched month by month, 2015-12 .. 2025-09),
  the same files B0 used. An article counts for session s when created after session s-1's close and up to
  session s's close (UTC times, early closes included). Tone per headline: Loughran-McDonald word lists
  (`tracks/news/lm.py`: NEG, POS or NEU). Headline text stays in the job's cache; nothing of it is committed.
- Short interest: FINRA consolidated short interest (twice a month), usable from 8 trading days after its
  settlement date (FINRA publishes about 7 business days after), as in B0. Short volume: FINRA Reg SHO daily
  files (off-exchange short volume over off-exchange total), the day after, as in B0.
- Not used: borrow fees and lendable shares (no free history; the IBKR logger only runs forward) and SEC
  fails-to-deliver (posted about a month late, so stale for a 5-day drop).

## Inputs (26, fixed now): META1's 18 plus 8 new

News, all up to t's close:
1. `news5`: articles tagging the stock in the 5 sessions to t.
2. `neg5`: of those, headlines labelled NEG.
3. `pos5`: of those, headlines labelled POS.
4. `news60`: articles in the 60 sessions before those 5 (normal coverage).

Short selling, all known by t:
5. `dtc`: latest usable short interest (at most 45 days old at t) over the 20-day mean raw volume (days to
   cover); missing when a split happened in the last 30 days, as B0.
6. `si_chg`: log of that short interest over the one before it (same symbol, previous settlement).
7. `shvr5`: mean daily Reg SHO short-volume ratio over the 5 sessions to t (at least 3 days).
8. `shvr_chg`: `shvr5` minus the mean ratio over the 60 sessions before those 5 (at least 20 days).

Missing values stay missing (LightGBM handles them); a stock with no article gets 0 for the news counts.

## Model, walk-forward, rule, primary test, pass bar, control

All exactly as META1: LightGBM with META1's fixed settings, expanding window with test years 2019 .. 2025-09
and a 2-week embargo; take a pick when its probability is at or above the median on the model's own
training rows; primary test = weekly mean `fwd5` of taken picks minus all picks, Newey-West t (4 lags) and
block-bootstrap CI (4-week blocks, 10,000 draws, seed 20261024); same-week random control with the same
seeds. **Pass to a forward shadow book** (PROTOCOL rule 19): mean > 0 with t >= 1.5, and the taken picks' own
mean after 0.2% costs > 0. A failure ends META2.

## Forward (if it passes)

As META1's forward section, with the live inputs read the same way: Alpaca news for the week to Friday's
close, the latest FINRA short-interest file published by Friday and Reg SHO files to Friday (posted that
evening). Judged once by PROTOCOL rule 20 (>= 60 trading days and >= 30 taken picks closed; taken minus all
after costs > 0 with NW t >= 2.0; taken picks' own mean after costs > 0).

## Reported alongside (not tests)

META1's list (per year, top-5 variant, gain importance, taken share, market-cap split), plus: the same
walk-forward on the 8 new inputs plus `ret5` only (does the new information carry anything alone?), and the
weekly mean `fwd5` of picks with `news5` = 0 vs > 0 and with `shvr_chg` above vs below that week's median.

## Data QA (equity-data-qa)

Coverage of each new input (share of picks with a value, per year); min, max and percentiles; the share of
picks with any news per year (coverage of Alpaca's Benzinga feed grew over time, so a rise is expected and
reported); 10 random picks' `news5` recounted from the raw article files; short interest dates checked to be
before t minus the 8-day lag on every row.
