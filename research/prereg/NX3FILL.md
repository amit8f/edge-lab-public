# NX3-FILL: score NX3's live trades at the real Alpaca fill, and measure the gap

Registered 2026-11-05, before any NX3-FILL number is computed. Source: Amit, 2026-11-05 ("yes"), after ALIT showed
+11% on the dashboard and a loss in Alpaca: the ledger enters at the IEX last trade (13.75) while Alpaca filled at
about 15.2-16.1 (thin, stale IEX print on a gappy open). One idea. Phase-2 count 105 -> 106 (phase 1 + phase 2 =
121). No trading rule changes: NX3, NX3-ALL and their entries stay frozen. This is a measurement and a scoring basis.

## Question

How far is the ledger's entry price (`entry_iex`) from what Alpaca actually filled, per trade and by side and gap size, and
what do the NX3 and NX3-ALL books earn when every trade is scored at its real fill?

## Rules

- Data: the logged decisions (side, `entry_iex`, exit) and the logged Alpaca orders (`filled_avg_price`) of EL-NX3 and
  EL-NX3ALL, joined by client order id. Exits are scored at the real exit fill when there is one, else the ledger exit.
  Trades without an entry fill (rejected, skipped for buying power) are counted and left out.
- Gap per trade = side-signed (fill - entry_iex) / entry_iex, so a worse fill is negative for the book.
- Costs: the config/risk.yaml costs, unchanged.

## Primary test

The mean gap over all filled entries (both accounts, longs and shorts together), day-clustered bootstrap 95% CI. The
gap is "material" if the mean is worse than -0.10% a trade with the CI below zero. Nothing else is a pass/fail.

## Reported, no rule

- Gap by side, by day, by |open gap vs previous close| bucket, and by price bucket; the worst 10 trades.
- NX3 and NX3-ALL book net per trade on the ledger basis and on the fill basis, and the difference.
- If the gap is material and concentrated in gappy opens: a follow-up idea (a different entry price rule, for example
  the first SIP trade after the decision) is registered separately and tested on a forward book. It is not tested here.

## Prediction (written first)

Mean gap about -0.15% a trade for NX3-ALL (shorts worse than longs), with a handful of gappy-open trades carrying most of it.
Probability the primary test calls it material: about 0.6.
