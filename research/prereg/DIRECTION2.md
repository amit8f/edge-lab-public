# DIRECTION2: five more ways at up vs down, from the outside brainstorm

Registered 2026-11-13, before any data for these tests is pulled or any number computed. Source: Amit asked Gemini,
ChatGPT and Perplexity (prompt written in the thread, 2026-11-13); the triage is
`/mnt/project-files/reports/Direction brainstorm triage 2026-11-13.md`. Amit chose "All five" on the decision card the
same day. Four new ideas (DIR-BRKT, DIR-SSTOP, NX3-EXIT, DIR-AGE) plus a second arm of DIR-FLOW (DIR-OFI, written into
`DIRECTION.md` as Amendment 2). Phase-2 count 145 -> 150 (phase 1 + phase 2 = 165).

Shared rules are `DIRECTION.md`'s: label dates before 2025-10-01 only; BOOM candidates are BOOM1's predicted 20 a day
from the walk-forward, main track ($5+ at the open), 2020-01 to 2025-09 unless stated; 1-minute SIP bars
(`adjustment=raw`, one session at a time); BOOM2's cost model (`boom2_study.net`, about 0.56% a round trip), also
reported at 2x; same names, same day, random side as the control, `default_rng(seed = yyyymmdd + idea number)`;
day-clustered bootstrap, 2,000 draws, seed 0. Every pass also needs the arm's own mean net > 0. Fewer trades than
the minimum = "too few". Numbers below: DIR-BRKT 6, DIR-SSTOP 7, NX3-EXIT 8, DIR-AGE 9 (DIR-OFI is 5).
A barrier touched in the same minute as the opposite barrier is ambiguous and dropped, in the arm and its control.

## DIR-BRKT: let the price pick the side

- Reference P = the close of the last bar before 9:35. A buy-stop at P x 1.02 and a short-stop at P x 0.98. Bars from
  9:35 on: the first bar whose high reaches the buy-stop goes long, whose low reaches the short-stop goes short (both
  in one bar = no trade). Fill = the stop price, or that bar's open when it opened past the stop. No trigger by 15:00 =
  no trade.
- Exit: stop back at P (about 2% from the fill), target P x 1.08 for a long (P x 0.92 for a short), else the last bar's
  close. Control: the same trigger minute and fill, a random side, the mirrored stop and target.
- Minimum 1,000 trades. Pass: arm minus control >= +0.25% a trade with the bootstrap lower bound > 0. Reported: k = 3%
  (same rules), share of picks that trigger, per-side results.
- Prediction (chance of a pass): 0.12.

## DIR-SSTOP: short every pick, with a tight stop

- Short every candidate at the first regular-session bar's open. Stop at +4% from the entry, target -8%, else the
  close. A stop that gaps is filled at that bar's open.
- Compared with (1) the plain short (stop +8%, target -8%, BOOM2's barrier) on the same picks, and (2) a random side
  with the mirrored rule (long: stop -4%, target +8%).
- Minimum 1,000 trades. Pass: mean net > 0 with lower bound > 0, AND tight minus plain >= +0.25% a trade with lower
  bound > 0, AND arm minus control lower bound > 0. Reported: stops at +3% and +6%, D6's most bearish tenth with the
  +4% stop, the 1st/5th/50th/95th/99th percentiles of the open-to-close move.
- Prediction: 0.15.

## NX3-EXIT: when should an NX3 long be sold?

- NX3's walk-forward long picks (3 a day, same models and seeds as `dir_gate.nx3_longs`), 2019 to Sep 2025, same
  entry as NX3. Exits: +15, +30 and +60 minutes after entry, the same session's last bar, the next session's open,
  and the next session's close (NX3's own exit, the baseline). An exit time past the same session's close uses the
  close. Costs: NX3's (`nx1_study.net`), one day held for the next-open and next-close exits, none for same-day ones.
- Primary: each of the five alternatives minus the next-close baseline, per trade, with day-clustered bootstrap and
  Holm correction across the five. An alternative passes if its gain >= +0.05% a trade and its Holm-adjusted lower
  bound > 0. A pass makes a shadow book with that exit; NX3's own book is unchanged.
- Reported, no rule: the overnight leg (next open minus same-day close) and the day leg; the same table for NX3's
  shorts; MAE/MFE (worst and best move before the exit).
- Minimum 1,000 trades. Prediction: 0.15.

## DIR-AGE: fresh news versus old news at the open

- Candidates with at least one single-symbol Benzinga headline from the previous close to 9:29. Age = 9:30 minus the
  newest such headline's time. Fresh: newest at 8:30 or later. Stale: newest before 7:30. Between: no call.
- Rule: fresh = follow the sign of the stock's move from the previous close to the last bar at or before 9:35; stale
  = fade it. Entry 9:51 (the first bar's open at or after 9:51), barriers +-4% from the entry or the close (BOOM2's
  D4 label).
- Minimum 500 calls. Pass: balanced accuracy >= the better of follow-the-move and fade-the-move on the same calls + 3
  points, AND arm minus control with lower bound > 0. Reported: fresh and stale separately, earnings-flag split.
- Prediction: 0.10.

## Order and costs

DIR-SSTOP, DIR-BRKT and DIR-AGE run on BOOM2's minute bars (being rebuilt now; the rebuild also stores each pick's
minute path from 9:30 to the close in the cache). NX3-EXIT reads NX3's event minute bars and daily bars. DIR-OFI is the
largest download and runs after DIR-FLOW's trade download.

## Implementation notes (2026-11-13, written with the code, before any data)

- 1-minute bars don't show whether a bar's high or low came first. A bar that reaches both a trade's stop and its
  target drops that trade (arm and control alike). DIR-BRKT starts checking exits on the bar after the fill bar.
- DIR-SSTOP's reported "D6's most bearish tenth" uses D6's walk-forward score (each year scored by a model trained
  on earlier years' picks, `boom2_study.side_d6`), never the frozen 2025 model on earlier years.
- DIR-AGE's move is the last bar at or before 9:35 against the previous session's raw close (BOOM2's `c`).
