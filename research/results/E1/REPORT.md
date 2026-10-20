# E1 result: macro-release blackout filter (run 2026-10-20)

Running idea count: phase 2 = 32, phase 1 + phase 2 = 47.

**FAILED.** P4's 261 trades (2017-2025-09) include only 33 entered on a CPI, jobs-report or FOMC day, below
the registered minimum of 100. Macro-day entries did worse (-4.97% vs -0.38% net a trade, difference
-4.6 points), but the date-clustered t is -1.35, far from -3. No blackout filter; no paper test.

Secondary (descriptive only, not a finding): CPI days n 18, t -0.36; jobs days n 6, t -0.03; FOMC days n 9,
difference -12 points, t -4.10; day before a release n 28, t -0.87. The FOMC number rests on 9 trades and
is one of four secondary cuts, so it is noise until shown otherwise.

Only P4 had a trade-level file when E1 ran, so the test could not reach its minimum sample. If later screens
save trade lists, a re-run needs a new registration.

Files: `macro_days.csv` (frozen before the join, see `CALENDAR_NOTES.md`), `run_e1.py`, `E1_output.txt`.
