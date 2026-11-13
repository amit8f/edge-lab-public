# DIRECTION: four tests of new inputs for calling up or down

Registered 2026-11-13, before any data for these tests is pulled or any number computed. Source: Amit, 2026-11-13
("direction is the biggest problem ... dig deeper, build a model or several"), and his choice on the decision card
the same day: register tests 1-4 of the research report `/mnt/project-files/reports/Predicting the direction of
moves.md` (notes in `research_notes/Predicting the direction of moves/`). Four ideas: DIR-LLM, DIR-GATE, DIR-PEER,
DIR-FLOW. Phase-2 count 141 -> 145 (phase 1 + phase 2 = 160).

What the research says first (written so a pass or fail is read against it): professional hit rates sit near
50-55%; free-data direction calls are capped near 55%; break-even accuracy is 0.5 + cost / (2 x move captured),
so BOOM at its observed ~0.56% cost and an 8% move needs 53.5%, and the $1-5 tier at 2% needs 62.5% (closed). Six
BOOM direction arms already failed (BOOM2). None of these four repeats them: each adds an input BOOM2 did not have.

## Shared rules

- History screens use label dates before 2025-10-01 only. Forward paper data is the confirmatory test.
- BOOM candidates: BOOM1's predicted 20 a day as in BOOM2 (`research/prereg/BOOM2.md`), main track ($5+), with
  BOOM2's first-touch label, ambiguous minutes excluded, and BOOM2's cost model (the ~0.56% observed round trip),
  also reported at 2x. The penny tier is not tested (2% cost needs 62.5%).
- Control: the same names on the same day with a random side, `default_rng(seed = yyyymmdd + idea number)`.
- Statistics: day-clustered bootstrap (2,000 draws, seed 0) of the arm's per-day mean net minus the control's.
  Every pass rule also needs the arm's own mean net > 0. Hit rate, balanced accuracy and per-side results are
  reported, no rule. Fewer than the stated minimum trades = "too few", not a pass.
- Each result is reported with the running idea count. Negative results stay.

## DIR-LLM: forced up/down answer with the model's own confidence

- Model: Gemini 2.5 Flash on the free key, pinned by model ID (training cutoff January 2025 per Google; checked on
  the model card before the first call and logged). Confidence = the answer token's log-probability where the API
  returns it; otherwise the vote share over 5 samples at temperature 1 (logged which).
- Prompt (fixed now, hashed in the log): company names and tickers masked as "Company X"; the headlines; "Will the
  stock trade higher (A) or lower (B) at the close than at 9:51 ET? Answer with one letter." A date-only probe asks
  the same with no headline, only the date and a masked firm, on 300 random candidate-days.
- Window: dates 2025-02-01 to 2025-09-30 (after the cutoff, before the holdout); then forward.
- BOOM arm: candidates with at least one single-symbol headline from the prior 16:00 to 9:14 ET. Entry 9:51 (first
  bar's open), exit at +-4% from the entry or the close (BOOM2's D4 barriers). Trades: the top third of calls by
  confidence over the window (cut fixed from the confidence values alone, before any label is joined). Minimum 200.
  Pass: arm minus control >= +0.25% a trade with the bootstrap lower bound > 0, and the probe's hit rate within
  50% +- 3 points.
- NX3 arm: NX3's long history picks in the window, same prompt asking about the next close. Trades: the top third
  of picks the model calls A, by confidence. Minimum 150. Pass: their mean net minus all NX3 longs in the window
  >= +0.10% a trade, bootstrap lower bound > 0, and the probe rule above.
- Forward: once built, scores (model ID, prompt hash, raw probabilities, time) are logged to the hash-chained
  ledger before the open; no trades until a pass.
- Prediction: BOOM 0.15, NX3 0.20.

## DIR-GATE: NX3 longs filtered by news type and novelty

- No fitted model. Novelty = the highest TF-IDF cosine similarity (word 1-2 grams, vocabulary fitted on 2019-2022
  headlines) between the pick's headline and the same stock's headlines of the prior 20 sessions; novel < 0.8.
- Gate (fixed now): keep NX3 long picks whose headline is novel AND carries neither NX1's earnings flag nor its FDA
  flag (the research says these reprice too fast to drift).
- Period: all NX3 walk-forward history picks before 2025-10-01; shorts are not gated (they stay off).
- Minimum 500 gated trades. Pass: gated mean net minus all NX3 longs >= +0.10% a trade with a day-clustered lower
  bound > 0, and the gated longs beat NX3's same-day random control. Reported: each NX1 flag and the novelty
  deciles separately.
- Prediction: 0.20.

## DIR-PEER: is the move the stock's own, or its sector's?

- Inputs: the stock's move since the prior close minus its sector ETF's (SIC 2-digit from EDGAR to the 11 SPDR
  sector ETFs, map committed before any data), and the mean move of its same 3-digit-SIC peers in BOOM1's universe
  (at least 3 peers, else no call). At 9:30 entry the moves are to the last pre-market bar visible (9:14); at 9:51
  to the 9:35 bar.
- Rule (fixed now): if the peers moved the same way as the stock, follow the stock-minus-ETF move's sign (a sector
  move with the stock leading); otherwise fade it (a move the peers don't share).
- Entries 9:30 (barriers +-8% from the open) and 9:51 (+-4% from the entry), each its own result; period
  2020-01 to 2025-09; minimum 1,000 calls each.
- Pass (either entry): balanced accuracy >= the better of follow-gap and fade-gap + 3 points, and arm minus control
  with bootstrap lower bound > 0.
- Prediction: 0.10.

## DIR-FLOW: buyers or sellers pushing before 9:35

- Data: Alpaca SIP trades and quotes (`feed=sip`) for BOOM candidates, 4:00 to 9:35 ET, 2022-01 to 2025-09 (a
  fixed window to keep the download small), in 500-symbol batches, deduplicated on symbol, exchange and trade ID;
  trades with odd-lot, out-of-sequence or derivatively-priced conditions handled as listed in the code's condition
  table, committed before any result. Each trade signed by Lee-Ready against the quote 1 second earlier (tick rule
  when inside the spread or no quote).
- Input: signed-volume imbalance (buy - sell) / (buy + sell) from 4:00 to 9:35, opening cross print excluded.
- Rule: side = the imbalance's sign; entry 9:51, +-4% barriers or the close. Minimum 1,000 calls.
- Pass: balanced accuracy >= the gap-sign rule + 3 points, and arm minus control with bootstrap lower bound > 0.
  Reported: large-print share, odd-lot share and exchange-D (off-exchange) share deciles. A fail closes the
  order-flow family for BOOM.
- Prediction: 0.10.

## After the four

A combined model (report test 5) is registered only if at least one of these is non-negative, in a new file.
Options snapshots (test 6) were not chosen.

## Amendment 1 (2026-11-13, written before any DIR result was read and before any DIR-PEER data exists; the DIR-GATE items describe code committed before its run)

Details the code needs that the text above left open. None changes a pass rule.

- DIR-GATE: a pick's similarity is the highest cosine between any of its event headlines (previous close to NX3's
  decision time) and any of the stock's single-symbol headlines in the 20 sessions before (previous close of the
  session 20 back to the previous close). The vocabulary drops terms seen in only one 2019-2022 headline
  (`min_df=2`). The control is k of the day's real NX3 events at random, taken long, seed yyyymmdd + 2.
- DIR-PEER: the SIC-to-sector-ETF map is the one already committed in `tracks/main/ch_data.py` (`SECTORS`, SIC
  ranges to the 11 SPDR sector ETFs, SPY where no range matches), in place of a new 2-digit map. Peers are BOOM1
  universe stocks of that session with the same first three SIC digits (EDGAR). The previous close is the
  all-adjusted close scaled to the session's raw prices. Follow-gap and fade-gap use the sign of the stock's own
  move at the same observation time (9:14 for the 9:30 entry, 9:35 for the 9:51 entry). No call when the stock's or
  the peers' move is missing or exactly zero.
