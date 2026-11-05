# NX3-BARS: a separate confidence bar for longs and for shorts

Registered 2026-11-05, before any NX3-BARS number is computed. Source: Amit, 2026-11-04/05: most of NX3's bad trades
look like shorts, so "raise the bar only for shorts"; then "have just the data of the result of each stock ... and
then you could do an analysis, go over all the results and find where is the best to place the bar for long and
for short". NX3's capped history, its first live days and NX3-HI's registration were seen first, so this history
can only reject (PROTOCOL rule 4); a forward book decides. This replaces the shorts-only bar (NX3-SHORTBAR) offered
on 2026-11-04. One idea. Phase-2 count 104 -> 105 (phase 1 + phase 2 = 120). NX3 itself is unchanged.

## Question

NX3 trades the first 3 long and 3 short events of the day that clear the firing line. If longs and shorts each
need their own confidence c = |e| / cut to take a slot, and the two bars are chosen without looking at the year
being scored, does the book still beat the same-day random control after costs, and does it beat NX3?

## Rules

Everything is NX3's (NX3.md: events, inputs, walk-forward models refitted each test year with the same seeds,
firing cut, entry minute and price, next-close exit, costs, 3+3 slots in time order) except the bars:
- Grid for each side: 1.0 (NX3's own), 1.25, 1.5, 2.0, 2.5.
- For test year Y the long bar and the short bar are each the grid value with the highest mean daily
  pick-minus-control, for that side alone, pooled over the EARLIER test years (their results come from their own
  walk-forward models). A bar needs at least 30 pick days to be eligible; with none eligible the bar is 1.0. The
  first test year therefore uses 1.0 for both.
- Year Y is then scored with its two bars. No bar is chosen using year Y or later.

## Primary test (history)

Pooled daily pick-minus-control over the test days under the chosen bars: mean > 0 and Newey-West (5 lags)
t >= 1.5 (the near-miss bar, grill 2026-10-30). Pass: a $5,000 ledger shadow book beside NX3 with the bars that
were chosen for the last test year (or the most common choice if stable), no orders; an Alpaca paper account only
under PROTOCOL rule 20.

## Reported, no rule

- The chosen bars per year and whether they are stable.
- The same pooled series for NX3's own rule (1.0, 1.0) on the same days, and the paired difference on shared days.
- `NX3_fired_trades.csv`: every event the models fire in the test years, with date, symbol, side, confidence
  and the net result until the exit (no headlines), so the confidence cut-offs can be studied by anyone. The table
  and the result by confidence bucket and side are descriptive and post hoc; picking a bar from them is not a
  result (it uses the data it is measured on).
- Live trades (NX3 and NX3-ALL ledgers, Alpaca fills) in the same layout, as they settle.

## Prediction (written first)

Shorts need a higher bar than longs (history: shorts -0.32% a trade, longs +0.16% in the NXE 2-day test). The
chosen short bar will be 1.5 or higher in most years and the long bar 1.0 or 1.25. Probability the primary test
passes: about 0.5; probability it beats NX3 with paired t >= 1.5 on shared days: about 0.2.
