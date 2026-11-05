# ROUTER1: send each situation to the model with the best record on that kind of case

Registered 2026-11-05, before any ROUTER1 number exists. Source: Amit, 2026-11-05 ("build a model that if a specific
thing is better in model A it will go there, and if it is better in model B it will go there"; "yes"). One idea.
Phase-2 count 106 -> 107 (phase 1 + phase 2 = 122). Every model it routes between stays unchanged.

## Question

If each trade opportunity is handled by whichever specialist (NX3, NX3-ALL, CHR-P, and the other forward books) has
the best past record on that kind of case, does the routed book beat the best single specialist and an even split?

## Rules

- Situation key: side (long/short) x universe (main $5+ / penny) x kind of trigger (news, chart, filing, other as each
  specialist logs it). Keys with fewer than 30 closed trades per specialist use the overall best specialist.
- Choice rule, fixed now: for a situation, route to the specialist with the highest mean net return per trade
  (book minus same-day control) over its closed trades logged BEFORE that day, shrunk toward the specialist's overall
  mean with weight 30/(30+n). Ties and a negative best go to "no trade".
- Uses only past records: history before 2025-10-01 for the first fit (read-only), then forward books only. The rule
  is refit weekly, never edited after a result.
- Shadow ledger book of $5,000, no orders. An Alpaca paper account only under PROTOCOL rule 20.

## Primary test (forward)

At 60 trading days and 100 trades: routed book minus the best single specialist chosen in advance (the one with the
highest overall mean at the first fit), same days, Newey-West (5 lags) t >= 1.5 and mean > 0. Also reported against an
even split across specialists and against the same-day random control.

## Reported, no rule

How often each specialist is picked per situation, how often the choice flips week to week, and the routed book
before and after costs.

## Prediction (written first)

The router does not beat the best single specialist at 60 days: with few trades per situation, choices are noisy.
Probability it passes: about 0.25. Its main value is showing where each specialist is strong.
