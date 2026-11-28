# JUMP1: NX3-HI's pre-open buys that already jumped, on data LONG1 never saw

Registered 2026-11-28, before any JUMP1 number is computed. Source: Amit, 2026-11-28 ("All three"), testing LONG1's
post-hoc cell (research/prereg/LONG.md: buys on news by the open whose price had already risen most before entry
made +0.94% a trade in 2019-2023). One idea: count 203 -> 204 (219 with phase 1).

## Data (independent of LONG1)

NX3-HI's buys as HI3 replays them (arm XGB-HI, research/prereg/HI3.md): the 2026 Benzinga replay and NX3's logged live
days (cache hi3), plus HI3K's 8-K press-release replay Oct 2025-May 2026 (cache hi3k). Settled nets as HI3 (NX3's
training code, flat costs). No download.

## Rule

A JUMP1 buy is an XGB-HI buy with minutes from the open <= 1 (news by 9:31, entry one minute after) and jump >= the
bar, where jump = (1 + r_open)(1 + r_react) - 1 = entry price / previous close - 1. The bar is fixed before the test:
the top-third boundary of jump among NX3-HI's buys in 2019-2023 (`long_study --stage jumpcut`, written to
research/results/LONG/jump_cut.json).

## Primary test (pooled HI3 + HI3K, one look)

Pass if all hold, with at least 30 JUMP1 buys (fewer: no verdict, wait for forward data): (a) JUMP1 net per trade > 0,
(b) the daily mean of JUMP1 buys has Newey-West t (5 lags) >= 2, (c) JUMP1 net per trade minus the other XGB-HI buys'
> 0 with Welch t >= 1.5. Reported by source and the 2 x 2 cells (pre-open or later, high jump or not), no rule.
A pass gets a log-only forward copy on NX3-HI's live buys.

## Prediction

P(pass) 0.2: the cell was the best of about 40 LONG1 cells, so most of it is likely luck; HI3K's buys lost overall.

## Amendment 1 (2026-11-28, before any JUMP1 result)

The first `jumpcut` run (46143805444) found the 2/3 quantile of jump among 2,137 NX3 buys in 2019-2023 is exactly
0.0: 75% of those buys came from pre-open news, and many enter at a price equal to the prior close, so jump = 0 ties
sit on the boundary. With `>=`, every zero-jump buy would count as "high jump", which is far more than the
registered top third. Fix: a buy is high-jump only when jump > cut (strictly above). The cut itself stays the
registered 2/3 quantile. `jump_cut.json` now also records the share of zero jumps and the share above the cut.
That run's job also failed to commit (the VETO1 results folder was missing from the mkdir list); fixed. No JUMP1
number had been computed when this was decided.

## Amendment 2 (2026-11-28, before any JUMP1 result)

The re-run (46144278250) showed every 2019-2023 buy has price / prev_c - 1 = 0 exactly (share_zero 1.0): in that
cache `price` is the reference close, not the entry. So the bar was measuring nothing. Bug fix: the cut now uses the
same jump JUMP1 reads from the HI3/HI3K inputs, (1 + r_open)(1 + r_react) - 1, for the same 2019-2023 buys, still
the 2/3 quantile, still strictly above. No JUMP1 number exists yet.

Side effect found: LONG1's descriptive "jump_before_entry" cells (low / mid / high, run 46130873541) used the same
zero column, and its tertiles were cut by row order (rank method "first"), so those cells carry no information about
jumps. That LONG1 hint (big jump +0.94% vs small -0.95%) is void. LONG1's other cells are unaffected.

## Amendment 3 (2026-11-28): forward shadow book under the near-miss rule

JUMP1 failed narrowly on history (daily t 1.94 vs 2, gap t 1.32 vs 1.5; IDEAS.md). Amit said continue (2026-11-28)
to the offered forward book, so under the near-miss rule (PR #65) it gets a log-only shadow book; this is the same
idea (count stays 204), not a new one, and no real money.

- Book JUMP1-FWD: every NX3-HI ledger buy (role picks, side > 0) whose decided inputs have mins_open <= 1 and
  (1 + r_open)(1 + r_react) - 1 > 0.0932 (the fixed 2019-2023 bar), $500 a trade at NX3-HI's own settled net,
  from 2026-11-30. Built on the dashboard from state-nx3 (no new orders, no new account).
- Verdict at 30 such buys (expected to take months: about one a week in the replays): pass = own net > 0 and
  JUMP1-FWD minus NX3-HI's other buys > 0 per trade with Welch t >= 1.5. Reported with the running idea count.
- Prediction: 0.25.
