# BOOM1 x NX3: NX3's long calls on the stocks BOOM1 expects to move big

Registered 2026-11-24, before any BOOM1 x NX3 number exists. Source: the ChatGPT review Amit pasted on 2026-11-24
("BOOM1 x NX3: magnitude prediction combined with independent direction"), and Amit's "All three" on the card that
followed. One idea. Phase-2 count 188 -> 189 (phase 1 + phase 2 = 204).

## Mechanism (rule 10)

NX3's profit comes from a few big winners (LESSONS 11) and its longs carry it. BOOM1 picks, the night before, the 20
stocks most likely to move 8%+ the next session (55% hit against 3% at random) but cannot tell the way. NX3 reads
the news direction. If big news moves on stocks already primed to move (high volatility, recent range) run further,
an NX3 long on a BOOM1 pick has a larger expected move with the same direction skill. Counterparties: the slower
investors post-news drift studies name (Chan 2003). Risk: BOOM1's picks are the most volatile stocks, where
spreads are widest (COST1) and GARCH sizing showed volatility alone does not pay.

## Data and rule (nothing fitted)

BOOM1 exactly as registered (BOOM1.md walk-forward: CatBoost per test year, scores for every stock-day, the 20
highest per night), rerun to get each night's top 20 and top 100 (scored on day d's close and session d+1's pre-open
news, both known before d+1's open). NX3's walk-forward fired trades (`research/results/NX3/NX3_fired_trades.csv`,
every event the walk-forward models fire, after NX3's costs, exit at the next close). The rule: take an NX3 long
(confidence >= 1) on session s only when the stock is in BOOM1's top 20 for s. Same entry, exit and costs as NX3.

## Primary test (one look, 2024-01 to 2025-09)

Pass if, with at least 30 trades: the intersection longs' net per trade is > 0 with Newey-West t >= 2 on daily mean
net (days with a trade), AND it beats all NX3 longs over the same period by at least +0.20 points per trade (the
review's hurdle). Pass: a $5,000 ledger book (live NX3-ALL longs on that night's live BOOM1 top 20), forward test
after 100 trades.

## Reported, no rule

2019-2023 the same way (nothing is chosen, so it is a second look, not a test). Trade count and trades a day; share
of intersection trades that boomed (8%+ range) against NX3 longs off the list; BOOM1 top 100 in place of top 20;
NX3 shorts on BOOM1's list (the BOOM-SHORT idea seen from NX3); NX3-ALL-HI (confidence >= 1.5); the top 3 trades'
share of profit; net with COST1's measured spreads where a quote exists.

## Prediction (written first)

The intersection is small (around one trade every few days) and wins more per trade on gross moves, but spreads
eat the gain. P(pass) about 0.25.

## Descriptive check run with it (not counted): NX3 against matched controls

The review asked whether NX3 only picks a kind of stock (volatile, cheap) rather than adding information. For
every NX3 fired trade 2019-2025: controls = the same day's real news events NX3 did not fire on, same side, same
price tier (<$5, $5-20, $20+) and same vol20 quintile (cut within the year). Reported: pick net minus matched
control mean, by day, NW t; per side; 2024-Sep 2025 and 2019-2023 separately. No rule, no book.
