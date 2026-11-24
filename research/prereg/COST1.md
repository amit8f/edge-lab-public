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

## Result (2026-11-24, run 45424062638): PASSED

`research/results/MODELS5/COST1_result.json`. 12,673 NX3-ALL-HI trades 2019-01 to 2025-09; entry quote found for 95%,
exit quote for 99.8%. Measured round trip: median 0.55%, mean 0.88%, 75th percentile 1.11%, against the flat 0.41%
(median 0.50% at $20+, 0.58% at $5-20, 0.62% under $5; 0.58% in the first 30 minutes, 0.20% later). With measured
spreads NX3-ALL-HI's net per trade 2019-2025 is -0.35%, against +0.08% with the flat cost.

2019-2023 chose c = 0.5% (+0.07% per trade, against +0.01% at 0.25% and +0.05% at 1%). Test 2024-01 to 2025-09, 5,022
trades over 421 days: the rule skipped 51% of trades. Sized daily gain over taking every trade +0.35% a day, NW t
5.08; the rule's own net +0.65% per trade (longs +1.16%, shorts +0.22%), against -0.28% for every trade and -1.20%
for the skipped ones. Reported: a LightGBM of the entry half-spread from price, dollar volume, vol20 and minutes after
the open ranks the 2023-2025 spreads at Spearman 0.72.

The prediction (spreads near the flat cost, no significant gain) was wrong in both parts. The size of the gain is a
red flag for checking: the kept trades earn more than every trade did with the flat cost, so wide-spread entries
have worse returns as well as higher costs. The forward copy decides.

## Amendment 1 (2026-11-24, after the result above, before any forward data)

The forward copy: from 2026-11-25, each settled NX3-ALL-HI day gets its picks' quoted half-spreads read after the
fact from Alpaca's quote history (as in the study: entry minute, and one minute before the exit session's close),
nets with measured spreads, and the skip at c = 0.5% (`tracks/news/cost1_live.py`, kind `cost1_settle`). No orders:
the paper accounts keep trading NX3-ALL as registered. Forward test after 100 kept trades: the sized daily gain of
the copy over NX3-ALL-HI, both with measured spreads, has mean > 0 with NW t >= 2, and the kept trades' own net > 0.
Also logged: NX3-ALL-HI's own P&L with measured against flat spreads, which is the cost check real money needs.
