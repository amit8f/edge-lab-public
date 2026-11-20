# FAST1: fast trades on strong news calls, out within an hour (news track)

Registered 2026-11-20, before any S1 number on 2024-2025 exists. Source: Amit, 2026-11-20 (a trade that enters
expecting a big move within about an hour, with a take-profit and stop), then "both" (BOOM-ALL and S1). Settings
come from S0 (`research/prereg/S0.md`, 2019-2023 only). One idea: count 178 -> 179 (194 with phase 1).

The name S1 was already taken (the 2026-10-17 VWAP scalp), so this is FAST1, the "S1" of S0.md.

Why: S0 found strong NX3 calls move +0.41% their way in 30 minutes and +0.46% in 60 (control about 0), right 52% of
the time, and that waiting loses most of it. Costs are 0.2% round trip (main) and 2% (penny). Prior: low (about
15%): the move is small against the costs.

Mechanism (rule 10). 1. Counterparty: traders who react to the headline slowly. 2. Funds do this faster; we only
get what is left after a minute. 3. A same-day exit dodges overnight risk but pays costs on a small move.

## Rules (fixed now)

- **Events:** NX3-ALL-HI's events (NX3 walk-forward, confidence c >= 1.5, uncapped), side from NX3, main track
  ($5+) and penny ($1-5) apart.
- **Entry:** the open of the first 1-minute SIP bar (raw) at or after NX3's entry minute (pre-open news at 9:31).
  Intraday longs wait 15 minutes (S0: entered at once they lose).
- **Exit, one of (chosen on 2019-2023 by mean net per trade, at least 300 trades):** time exit at 30 or 60
  minutes; or a bracket TP/SL of (2%, 1%) or (3%, 1.5%), first touch within 60 minutes, else out at 60 minutes.
  Fills: target at its price, stop at its price or the first bar's open beyond it, a bar touching both is the stop.
- **Costs:** config/risk.yaml. Shorts only where easy to borrow today; history ignores borrow, as NX3 does.
- **Control:** the same day's random real events, same sides, same entry and exit (S0's control).

## Primary test (one look, 2024-01 to 2025-09)

Pass, all of: mean net > 0 a trade after costs; day-clustered t >= 2; at least 300 trades; beats the random
control's 95th percentile over 200 draws. Reported beside it, not deciding: the same events held to the next close
(NX3's own exit), longs / shorts, main / penny, each year, trades per day.

Pass: a $5,000 ledger shadow book (no orders). Pattern-day-trader rule: a real account under $25k allows 3 day
trades in 5 days, so S1 stays on paper until that is settled. Fail: written up. Holdout from 2025-10-01 untouched.

## Amendment 1 (2026-11-20, same day, code details before any FAST1 number exists)

`tracks/news/fast1_study.py`. Same NX3 walk-forward refits and seeds as S0. The 200 control draws are the day's
real events at random, as many as the day's FAST1 trades, with their sides. Costs are NX1's `cost_frac` for a
$500 position held one day (main or penny by entry price, borrow on shorts). The choice and the test run in one
job: the test reads only the exit the choice years picked. "Right way" (reported) = net above minus the cost.
