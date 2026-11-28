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
