# NXM-VOTE: trade a news event only when several model families agree

Registered 2026-11-24, before any vote result exists. Source: Amit, 2026-11-24 ("did we try to merge different
models ... NXM-LGBM and NXM-ENS and NXM-MLP and NXR-RFR because they are showing promising signs", then "yes").
One idea. Phase-2 count 187 -> 188 (phase 1 + phase 2 = 203).

## Mechanism (rule 10)

NX3's edge is thin and a few big trades carry it (LESSONS 11). Each NXM/NXR arm is a different kind of model (boosted
trees, a neural net, a forest, and ENS, the average of five), so their errors differ in part. An event that several
of them call the same way is less likely to be one model's noise. Precedent: ensembles of unlike learners beat their
parts when errors are weakly correlated (Gu, Kelly & Xiu 2020 report the same for return prediction). Risk: the arms
share NX3's inputs and labels, so they may agree on the same false calls (live, 30-56% of their trades overlap with
ENS), and requiring agreement cuts the trade count.

## Data and models

NX3's events (`cache/nx1/nx3/trades_v2`), walk-forward as NXM (`tracks/news/nxm_study.year`: train on everything up to
the purge before each test year, test years 2019-2025, label dates before 2025-10-01 only). Voters, with the settings
fixed in NXM.md and NXR.md: LGBM, MLP, ENS (needs XGB, LIN and RF fitted as well) and RFR. Each voter's confidence is
its effect over its own fire cut (|e|/cut >= 1 = fired).

## Rule

Per real event: a voter votes long if its confidence is >= 1 with e > 0, short if <= -1. The event is traded on a
side when at least k voters vote that side and more vote it than the other. Uncapped, like NX3-ALL (no 3+3 slots,
so no alphabetical fill at the open). Size: NX3-ALL's 5% to 10% of the book by the mean confidence of the voters on
that side (`nx3_study.conf_sizes`). Exit at the next close, NX3's costs (flat spread, borrow on shorts).
k in {2, 3, 4} chosen on 2019-2023 by mean sized daily net.

## Primary test (one look, 2024-01 to 2025-09)

Per day, sized vote book minus a same-day random control (as many of the day's real events, same sides and sizes,
`nx3_study.picks`): mean > 0 with Newey-West t >= 2, and the vote book's own sized daily net > 0.

Pass: a $5,000 NXM-VOTE ledger book (no orders), live from the day its models are saved; forward test after 100
trades with the same rule.

## Reported, no rule

Per voter alone (uncapped, same sizing) and NX3-ALL (XGB alone): trades, net per trade, median net per trade, sized
daily net; vote minus ENS alone and vote minus NX3-ALL per day (NW t). Share of net from the top 3 trades. Vote
trades per day. Net with COST1's measured spreads for vote trades that are also NX3-ALL-HI trades with a quote.

## Prediction (written first)

The vote book trades less than any arm and wins a little more per trade, but the arms agree mostly on the same
events NX3 already takes, so it ties NX3-ALL. P(pass) about 0.35; P(beats NX3-ALL by NW t >= 2) about 0.1.

## Result (2026-11-24, run 45441301081): PASSED

`research/results/NXMVOTE/NXMVOTE_result.json`. 2019-2023 chose k = 4, all four voters agreeing (+0.06% a day; k = 2
-0.03%, k = 3 -0.01%). Test 2024-01 to 2025-09, 433 days: 1,881 trades (4.3 a day), +0.48% per trade, median +0.13%,
longs +0.58%, shorts +0.39%; the top 3 trades are 24% of the profit. Sized book minus same-day random control +0.36% a
day, NW t 3.29; own sized daily net +0.15%, NW t 1.86 (the rule asks only > 0).

Reported, same period and sizing, each model alone and uncapped: XGB (NX3-ALL) -0.02% a trade, LGBM -0.21%, MLP
-0.32%, ENS -0.16%, RFR -0.17%. Vote minus each per day: XGB +0.17% (t 1.68, not significant), LGBM t 2.31, MLP t 4.21,
ENS t 2.49, RFR t 3.50. 88% of vote trades are also NX3-ALL trades. With COST1's measured spreads, the 68% of vote
trades that have a quote go from +0.75% (flat) to +0.15% a trade, and +1.27% for those COST1 would keep.

The prediction (ties NX3-ALL) holds for the head-to-head; the pass is against random. Caveats: k = 4 won 2019-2023 by a
small margin, and real spreads take most of the edge unless COST1's skip is applied.

Next (as registered): the $5,000 NXM-VOTE ledger book from 2026-11-25 (`tracks/news/nxvote_live.py`, kinds
nxvote_day / nxvote_settle), using the version 1 LGBM, MLP, ENS and RFR models the NXM/NXR books already run, k = 4,
uncapped, a short only where easy to borrow. Forward test after 100 trades.
