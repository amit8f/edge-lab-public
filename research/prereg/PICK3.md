# PICK3: the same-day picker with a moving bar (registered 2026-11-15, before any number for it exists)

Count 169 (184 with phase 1). Amit, 2026-11-15, decision card "Build PICK3", after PICK2-MAIN and PICK2-OOF made almost
no trades in 2023 ("I expected 0-1 trades a day, more toward 1"). The data check (`PICK2.md`) found two causes: one
fixed bar for every day, set mostly by 2020, and scores multiplied by the stock's swing, which shrinks every score in
a calm market.

## Question

Does PICK2's committed COMBINED model, ranked on its raw score and bought against a bar that moves with recent
sessions, make money after costs at about one trade a day?

## Rules (all fixed now; nothing is chosen on data)

- Universe, decision rows (every main row, $5+ and $5M/day, 09:35-14:00 every 5 minutes), entry at the open of the bar
  starting at the decision time, exits (+8% target, PICK2's frozen COMBINED tier stops, the 15:50 bar's close) and
  PICK's 0.2% cost: as PICK2.
- Model: PICK2's committed COMBINED refit models (2019-2023: XGBoost, CatBoost, LightGBM in
  `research/results/PICK2/models`). Score r = the mean of the three models' raw predictions (net return in units of
  the stock's swing). No refit, no learner choice.
- Moving bar for session d, from the previous 60 sessions only: M = each session's highest r over all its rows.
  Normal rows clear at the 30th percentile of M; bad-market rows (PICK's frozen market cut) at the 70th percentile
  (stricter, Amit's bad-day rule). A row also needs r > 0.
- Buys: a stock's first clearing row is its only candidate that day; time order, at the same minute the higher r
  first; at most 2 a day; $1,000 each (20% of a $5,000 book; fixed, not confidence-sized, so the check counts trades
  alike).
- Control: one random universe stock bought at the same minute with the same rules (PICK's `control_pick`).

## History check (can only reject)

Rows 2023-10 to 2023-12 only warm the bar (scores, no returns). The check: 2024-01-02 to 2025-09-30 with the 2019-2023
models. 2024-2025 was read by PICK's test with PICK's own models and gave the bad-day hint, so this is a seen period:
it can reject PICK3, never pass it. Rejected unless all hold: at least 150 trades; mean net per trade > 0; Newey-West
t >= 2 on daily book returns (days with trades); mean of (pick net - control net) > 0. Reported: trades per day, share
of days with a trade, per-year and bad/normal splits, exits.

## Forward test

If not rejected: a $5,000 ledger book (no orders) logged after each close like PICK2-BAD, from the first session
after it is built, with the bar carried from live scores. Pass by PROTOCOL rule 20 (60 trading days, 30 closed
trades, net > 0 with NW t >= 2, beats random buys with t >= 2), judged then or at 2027-12-31.

## Prediction (written first)

Trades per day 0.8-1.6 (chance 0.7). Not rejected by the history check: 0.3. Passes forward: 0.1.

## Result (2026-11-15, run 43741342553, `research/results/PICK3/PICK3_check.json`)

**Rejected.** 2024-01-02 to 2025-09-30, 438 sessions: 357 trades, 0.82 a day, a trade on 46% of days (the rate Amit
expected was met). But the average trade lost 0.22% after costs (median +0.03%), the daily book's NW t was -1.69, and
the picks did worse than the random same-minute buys (-0.22% vs -0.12%). 10.4% ended +3%. Exits: 83% the 15:50 close,
15% the stop, 2% the +8% target. Of the four rules only "at least 150 trades" held. By year: 2024 -0.15% (193 trades),
2025 -0.31% (164). Count 169 (184 with phase 1).

POST-HOC / NOT PRE-REGISTERED: the 48 trades on bad-market minutes averaged +0.29%, the 309 on normal minutes
-0.30%. That is the same pattern PICK's own test showed (bad days +1.0%, normal days negative) and is what PICK2-BAD's
forward book already tests, so it is not a new lead. The model has a rate-of-trades fix (the bar) but no edge on
normal days.
