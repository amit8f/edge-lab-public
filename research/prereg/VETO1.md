# VETO1: skip buys after a late-filing notice

Registered 2026-11-28, before any VETO1 number is computed. Source: Amit, 2026-11-28 ("All three"; "Wires + 10-K
veto" from the pre-news plan, /mnt/project-files/reports/News before the news plan 2026-11-28.md). One idea: count
204 -> 205 (220 with phase 1).

## Idea

A company that can't file its 10-K or 10-Q on time files a notice (Form 12b-25: NT 10-K, NT 10-Q). Papers find
these firms do badly afterwards. Buying them on good news may be buying into trouble.

## Data

EDGAR quarterly form indexes 2018-Q4 to 2025-Q3 (forms NT 10-K, NT 10-Q and their /A), mapped CIK -> ticker with SEC's
current ticker file (company_tickers_exchange.json; firms no longer listed are missed, so the veto set is a lower
bound). Trades: NX3-ALL-HI's walk-forward buys 2019-01 to 2025-09, LONG's cost model (measured entry half-spread,
closing-auction exit), as `long_study.load`.

## Rule

A buy on day d is vetoed if the company filed a notice dated d-60 to d-1 (calendar days). Fixed, nothing tuned.

## Primary test (uncapped NX3-ALL-HI buys, all years, one look)

Pass if at least 30 buys are vetoed, the vetoed buys' net per trade is below 0, and vetoed minus kept net per trade
is below 0 with Welch t <= -1.5. NX3-HI's capped buys and the daily book gain from the veto are reported, no rule.
A pass gets a log-only veto on NX3-HI's live buys (forward check: 4 weeks, as Plan v3).

## Prediction

P(pass) 0.25. Notices are rare (likely under 2% of buys), so the main risk is too few vetoed trades.
