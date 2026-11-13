# BOOM-SHORT: short the most bearish of BOOM1's nightly 20, forward paper test (main track)

Registered 2026-11-07, before any BOOM-SHORT day is logged. Source: Amit, 2026-11-07, chose "Shadow book" on the
decision card after BOOM2's post-hoc diagnosis. Phase-2 count 134 -> 135 (phase 1 + phase 2 = 150).

## Where it comes from (post hoc, disclosed)

BOOM2 (`research/prereg/BOOM2.md`) failed all six direction tests. Its post-hoc diagnosis
(`research/results/BOOM2/BOOM2_diagnosis.json`, run 42414448693 after a leak in the diagnosis code was fixed)
found that BOOM1's picks drift down from open to close (median -0.9%, every year 2020-2025), and that the tenth of
picks the D6 side model scored most bearish earned +0.34% a trade short after costs, out of sample by year.
Because that rule was found by looking at the results, history cannot confirm it: only this forward test decides.

## Rule (frozen before the first day)

- Models: the 2025 fold of BOOM1's and BOOM2's walk-forward, exactly as tested (`tracks/main/boom_short.freeze`):
  BOOM1 trained on stock-days up to the purge before 2025-01-01, D6 on the 2019-2024 picks. Cut = the 10% quantile
  of D6's scores on its out-of-sample 2025 picks. Files, sha256 and the cut in `models/boom_short/manifest.json`.
- Each session before 9:30 ET (`tracks/main/boom_short.py`): BOOM1's universe at the previous close (frozen
  symbol list, $5+, $1M+ 20-day dollar volume), its chart, market and pre-open news inputs; the 20 highest
  BOOM1 scores; each pick's pre-market gap and dollar volume from SIP minute bars up to 9:14 (no pre-market trade:
  0 and 0); D6 score; short every pick with score <= cut that Alpaca marks shortable and easy to borrow, lowest
  first, at most 10, $500 each on a $5,000 ledger book. The decision is logged before the open; a day not decided
  by 9:30 is logged as missed with no trades.
- Trade, as BOOM2's history: entry at the first regular-session minute bar's open, exit at the +8% (stop) or
  -8% (target) barrier touched first, filled at the barrier, else the close. Net = short return - NX3's cost model
  (one day, borrow). Settled from SIP minute bars after the close.
- Control: each day, as many of the other picks drawn at random (default_rng(20261109 + yyyymmdd)), shorted
  the same way.

## Forward test

At 100 closed trades (or 2027-12-31, whichever first): mean net per trade > 0 and the per-day book mean's
Newey-West t (5 lags) >= 2.0, AND book minus control per day mean > 0. Reported, no rule: the control alone,
net at 2x costs, hit rate (down-first share), missed and blocked days. Ledger only; real money needs PROTOCOL's
sign-off.

## Known differences from history (logged, not fixed)

- Live universe: only stocks active today (history kept delisted names).
- News: the decision at 9:28 reads headlines up to then (history up to the open); pre-market bars end ~9:12
  because of the 16-minute delay (history 9:14).
- No pre-market trade: gap and dollar volume set to 0 (history used the open's gap).
- Barrier fills at exactly -8% / +8%: optimistic for the stop side, as in history.

## Prediction (written first)

Post-hoc edges usually shrink out of sample. Pass: 0.2. Most likely: a mean near zero, positive before costs.

## Amendment 1 (2026-11-13, before any trade): decisions may finish until 9:35 ET

Why: no trade has been logged yet. 11-09 never ran (timers skipped), 11-11 crashed (FinBERT meta bug, PR #161),
and 11-12 finished at 9:30:39 ET and was logged missed (all overnight headlines were scored after the 9:28 wait;
PR #166 now scores them before the wait). Amit asked on 2026-11-13 to let the decision run until 5 minutes after
the open so a slow morning isn't lost.

Change: a decision that finishes after 9:30 but by 9:35 ET is kept. Its shorts and control enter at the open of
the first whole minute after it finished (logged as `entry_from`; the -8%/+8% barriers are measured from that
price), and later than 9:35 is missed as before. Inputs are unchanged: news up to 9:28, pre-market bars up to
then minus the 16-minute delay. Days that entered after 9:30 are flagged in the report and the pass rule is
also read without them. The pass rule itself is unchanged.
