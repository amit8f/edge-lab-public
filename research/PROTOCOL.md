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

## Added 2026-10-15 (Amit approved the "deeper" plan, `rethink/DEEPER_2026-10-15.md` in the project files)

These apply to every idea registered from 2026-10-15 13:00 UTC on, and to the screens M1, M3 and N1,
which had no result then. Screens with results keep their verdicts; M11 keeps its own registered rules
(it already passes only at t >= 3).

10. **Mechanism first.** A pre-registration opens with three answers, written before any data is
    touched: who is on the other side of the trade and why they keep accepting a bad price; why a fund
    with $1B can't take the trade away while a $5k account can; and whether it is published, with the
    expected decay. `research/MECHANISMS.md` answers them for the ideas registered before this rule.
11. **Factor alpha.** Every screen and forward test also reports its monthly return's alpha on the
    Fama-French 5 factors plus momentum (Ken French's files, sha256 saved; Newey-West errors,
    `core/stats.py`). A screen passes only if its pre-registered test passes **and** that alpha is > 0
    with t >= 2.0. Event ideas report a market-model CAR test with the Kolari-Pynnonen correction.
12. **Higher bar for the count.** A historical screen passes only at a HAC t >= 3.0 (Harvey, Liu & Zhu
    2016), not 2.0, and reports the deflated Sharpe ratio with the phase 1 + phase 2 idea count as the
    number of trials. The forward test keeps its own pre-registered bar.
13. **Outside timestamps.** Each pre-registration's sha256 is stamped with OpenTimestamps (anchored in
    the Bitcoin blockchain; only the hash leaves the repo) when it lands on main. The `.ots` proofs and a
    manifest live on the `state` branch under `timestamps/` (`core/timestamps.py`).
14. **Pipeline replication.** Before a new data pipeline's screen counts, it reproduces a known result
    to a pre-set tolerance (for the month-end stock panel: Ken French's momentum and market factors; see
    `research/REPLICATION.md`). A failed replication stops the screens that use that pipeline.
15. **Project deadline.** If no idea has passed its forward test by 2027-12-31, the project's output is
    the research write-up, and no real-money track opens without a new written decision here.

## Added 2026-10-20 (Amit approved Plan v3, `docs/PLAN_V3.md`; his merge of this change is his signature)

16. **No historical gate for Plan v3 strategies.** Ideas that enter the Plan v3 system skip the historical
    t >= 3 screen of rules 4 and 12. They get a bug and look-ahead check, then start in the shadow book, and
    the forward record is their test. Historical screens already registered keep their own rules.
17. **Rules the system makes itself.** The weekly coach may create new rules. Each is logged in
    `research/IDEAS.md` (and counted) before it starts in the shadow book, with no orders and no money. It is
    promoted only under `research/prereg/SYS1.md` (beat the frozen copy over at least 4 weeks and 30
    trades after costs, at the confidence that SYS1 sets for the number of rules tried). The learning rule
    itself is frozen in SYS1, and any change to it is a new registration.
18. **Going live is unchanged.** Rule 9 still applies to every Plan v3 book.
