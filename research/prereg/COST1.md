# COST1: the real spread at each news trade's entry, and skipping the expensive ones

Registered 2026-11-24, before any COST1 number is computed. Source: Amit, 2026-11-24 ("yes to both", accepting the
cost model from the ChatGPT review). One idea. Phase-2 count 184 -> 185 (phase 1 + phase 2 = 200).

## Mechanism (rule 10)

NX3 enters one minute after news, when spreads are widest (C0: the large-stock half-spread is 19 bp at 09:35
against 2 bp at 15:59), and our history charges every trade the same 0.41% spread. A trade whose quoted spread at
entry is far above that pays more than the history says, and the edge is only about 0.10% a trade. Paying the market
maker less is the whole gain; nobody else is on the other side. Published (trading-cost-aware execution: Frazzini,
Israel & Moskowitz 2018).

## Measurement (not tested, reported)

For every NX3-ALL-HI trade 2019-01 to 2025-09 (`cache/nx1/nx3/trades_v2`, as `models5_study` picks), the NBBO in
force at the entry minute and at 15:59 of the exit session, read as in C0 (prereg C0.md: last SIP quote in the 2
seconds before, else the 60 seconds before; bid > 0, ask > bid, spread under half the midquote). Measured round trip
= half-spread at entry + half-spread at exit. Reported: its distribution by price tier and minutes after the open,
coverage, and NX3-ALL-HI's net per trade with measured instead of flat spreads (commission and borrow as
`adaptive/costs.py`).

## Rule

Skip a trade when its quoted half-spread at entry is above c, with c in {0.25%, 0.5%, 1.0%} chosen on 2019-2023 by
net per trade with measured costs. Trades with no valid entry quote are kept and charged the flat cost.

## Primary test (one look, 2024-01 to 2025-09)

Per day, sized NX3-ALL-HI return with the skip rule minus without it, both with measured costs: mean > 0 with
Newey-West t >= 2, and the rule's own net per trade > 0. Pass: the live NX3-ALL-HI ledger logs the entry quote and
a COST1 shadow copy that skips the wide ones ($5,000, no orders).

## Reported, no rule

Share of trades skipped; the skipped trades' own net (are they the bad ones?); per side; a LightGBM model of the
entry half-spread from price, dollar volume, vol20, ATR and minutes after the open (trained 2019-2022), for live
use where no quote is read in time.

## Prediction (written first)

Measured spreads are close to the flat 0.41% on average, the widest ones sit on small stocks near the open, and
skipping them removes about 10% of trades without a significant gain. P(pass) about 0.2.
