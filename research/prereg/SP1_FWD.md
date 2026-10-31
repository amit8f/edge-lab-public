# SP1 forward log (main track, near-miss under rule 19)

Written 2026-10-31. Source: Amit, 2026-10-31: "lets continue with the NX2 and SP1 also add them to the
dashboard we will monitor them for now and decide later". SP1's history screen (2026-10-30) failed rule 12
on event count but met rule 19. Not a new idea: the count stays 91 (106 with phase 1).

## What is logged (`tracks/main/sp1_live.py`, nightly in leads.yml, no orders)

- Every Form 10-12B and 10-12B/A filed from 2025-10-01 on, from EDGAR's daily form index.
- Each company's first one, flagged as a candidate when its EDGAR submissions show no 10-12B in the 365
  days before (SP1's event rule).
- Its first regular-way trading day, entered by hand in `research/checks/SP1_live_dates.csv` from the
  parent's or its own completion release (SP1's round-1 fix: when-issued days don't count). Then, once its
  22nd session counting that day has traded, the entry: that session's raw SIP open and the exit session
  250 sessions later.
- The 5 same-day random $5+ controls are drawn at judging time by SP1's seeded rule on point-in-time data.

## How it is judged

SP1's own frame (entry, hold, costs, controls). Because there are about 1.5 spin-offs a month and each
holds a year, rule 20's 30 closed trades would take until about 2029. Amit chose to watch it on the
dashboard (candidates seen, entries open) and decide later; no account and no money until he does.
