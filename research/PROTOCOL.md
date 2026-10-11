# Research protocol (phase 2)

Carries over phase 1's rules (`breakout-scanner/RESEARCH_PROTOCOL.md`) with three changes: a fresh idea
count, shorting allowed, and per-track ledgers.

1. **Rules before results.** Each idea gets a file `research/prereg/<ID>.md`, committed before any result
   of that idea is computed. It fixes: track, universe, data, signal, entry, exit (from
   `config/risk.yaml` unless stated), costs, period, one primary test, the success criterion, the
   control, and the stop rule. Any change after results exist is a new idea.
2. **Ledger.** Every idea is a row in `research/IDEAS.md` with its ID (`M1`, `P1`, `N1`, ...), date,
   source and outcome. Rows are never deleted or rewritten; outcomes are appended.
3. **Counting.** Two counts are reported with every result: the track's idea count and the phase-2
   total. Deflated Sharpe ratios use the phase-2 total. Phase 1's 15 ideas are reported alongside
   ("phase 1 + phase 2") so a reader sees the full number of attempts.
4. **Screens and confirmation.** A historical screen on phase-2 training dates (before 2025-10-01) can
   only reject. A screen pass goes to forward paper trading under its own pre-registration; only the
   forward test can confirm.
5. **Controls.** Same rules on a same-day random selection from the same universe (seed fixed in the
   pre-registration). Long-short ideas also report a beta-matched control.
6. **Costs and shorting.** Costs from `config/risk.yaml`. A short counts only if the stock was
   easy-to-borrow on Alpaca's asset list at the signal date; borrow fees are charged daily.
7. **Data quality.** Each study runs the equity-data-qa checks and hand-checks a random sample of its
   trades against raw bars.
8. **Post-hoc work is labelled** and feeds `IDEAS.md` only as hypotheses.
9. **Going live.** Real money needs, per track: at least 100 closed paper trades, the idea's forward
   pre-registered test passed after costs, and the owner's written sign-off in this file.
