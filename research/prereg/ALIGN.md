# ALIGN: strong news calls that their related stocks and the market already agree with (news track)

Registered 2026-11-20, before any ALIGN number exists. Source: Amit, 2026-11-20: "maybe see the sector trend for the
day if the sector ... is running up than most probably if there is a news that is up / neutral its a sign to go up
also ... find just a small amount of trades that we fill pretty sure", then yes to Q1-Q4 with changes: Q1 "make it
NX3HI", Q3 "look more than that like sector that are connected to it also and ... companies its connected to or
related", and his target of at least 5-10 trades a day on average, with moves of 3% or more and a take-profit and
stop ready. One idea: phase-2 count 175 -> 176 (phase 1 + phase 2 = 191).

Why: the sector study (SECTOR1 stage 2) found that news sets the direction of a move only for some news types, and
that half of big moves come with no company news (sector and market days). DIR-PEER (2026-11-14) found that the
sector's move alone says nothing about which way a big mover goes. ALIGN asks the narrower question: when our
strongest news call and the stock's related stocks point the same way, is the call right more often? Prior: low
to moderate (about 20%).

Mechanism (rule 10). 1. Counterparty: traders who react to the headline alone, late. 2. A fund can do this, but
only on a few names a day at our size. 3. If related stocks are already moving, part of the news may already be
priced, so the filter could also hurt; the test says which.

## Rules (fixed now)

- **Base events:** NX3-ALL-HI's events, uncapped: NX3's fired events (NX3.md: walk-forward models refitted each test
  year, same seeds, firing cut, entry 1 minute after the headline, pre-open news at 9:31) with confidence
  c = |e| / cut >= 1.5. Side = NX3's side. NX3-HI's 3+3 cap is not used (it would cap the day at 6 trades; Amit
  wants 5-10 a day).
- **Related stocks**, point in time, union of:
  (a) same SIC 3-digit industry (EDGAR SIC codes, as SECTOR1);
  (b) the 10 stocks with the highest correlation of daily returns over the 250 trading days before January 1 of the
  event's year (PICK main universe, at least 200 shared days);
  (c) the stock's Fama-French 12 sector, equal weight.
  Related move = equal-weight mean of (a) and (b) members' returns from the previous close to the entry minute (all
  adjustment=all, SIP), each member needing a trade in the last 5 minutes. Sector move = the same for (c).
- **Aligned** when: related move has the trade's sign and |related move| >= k x its own 60-day daily sigma; the
  sector move does not have the opposite sign beyond 0.5 sigma; and SPY's move from the previous close does not go
  against the trade by more than 0.25%.
- **Exit:** a bracket from the entry price: take-profit TP and stop SL, else out at 15:55 the same day. Fills: target
  at its price; stop at its price or the first bar's open beyond it (gaps hurt); same bar touching both counts as
  the stop.
- **Choices made on 2019-2023 only:** k in {0, 0.5, 1.0}; (TP, SL) in {(3%, 1.5%), (5%, 2%), (8%, 3%)} or the
  volatility version TP = 2 x, SL = 1 x the 30-minute range average of the last 20 days. The pick is the highest
  mean net per trade with at least 300 trades in 2019-2023.
- **Costs:** config/risk.yaml (main $5+, penny cost for $1-5). Shorts only where Alpaca says easy to borrow today;
  history ignores borrow, as NX3 does.

## Primary test (one look, 2024-01 to 2025-09)

ALIGN trades with the chosen k and bracket. Pass, all of:
1. at least 60% of trades end in the trade's direction (exit price beyond entry in the trade's favor, before costs);
2. mean net >= +1.0% a trade after costs, day-clustered t >= 2;
3. at least 30 trades;
4. beats the same entries with a random side (same days, 200 draws, 95th percentile) and beats NX3-ALL-HI's own
   events with the same bracket (the filter has to add something).

Reported beside it, not deciding: trades per day (Amit's target is 5-10), the share of trades that reach +3% in
their direction, longs and shorts apart, main and penny apart, each year.

Pass: a $5,000 ledger shadow book next to NX3-ALL-HI (no orders), sized as NX3-ALL; a paper account only after
PROTOCOL rule 20. Fail: written up, no book. Holdout from 2025-10-01 untouched.
