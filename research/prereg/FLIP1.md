# FLIP1: do our AI agents read the news? (cross-track)

Registered 2026-10-28, before any flipped answer exists. Source: Amit asked for a sweep of new ideas on
2026-10-28 and said "add all of those"; this was pick 2. Phase-2 count at registration: 57 (cross-track 10).
Phase 1 + phase 2 = 72.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** None: FLIP1 measures our own agents. The old news-signal model failed because it
   ignored the news (its placebo test). An LLM can also answer from the stock's numbers or its prior about
   the company, whatever the headline says.
2. **Why us.** Not applicable (a measurement).
3. **Published.** Ye et al. (2026), "The Alpha Illusion" (arXiv 2605.16895), test P3: a counterfactual
   reversal of the input should reverse the agent's answer.

## Measurement (forward only, logging only)

Once a trading day in each system (`adaptive/flip.py`):
- SYS2 (`adaptive/watch.py`): the first flagged stock whose analyst view has a non-zero tone and whose
  headlines contain words in `flip.PAIRS` is sent to the analyst again, same model and prompt, with each
  such word swapped for its opposite ("beats" for "misses", "raises" for "lowers", ...). Logged: tone before
  and after.
- SYS2B (`adaptive/scouts.py`): the first scout answer of the day with a long or short pick and at least one
  listed word in its headline and filing blocks is asked again on the same model and budget, with the words
  swapped only inside those blocks. Logged: each
  pick's side before and after.
The flipped answer never trades, never counts as an agent error, and its text stays in the evidence branch.

"Moved" means the answer went the other way: a lower tone after a positive one (a higher one after a
negative one), or a long that is no longer long (a short no longer short).

## Primary test

At the first date with >= 40 logged flips in a system: the share that moved. **The agent reads the news**
if the share is >= 70% and its exact binomial test against 50% has p < 0.01. Below 50%, the agent is
reported as ignoring the news.

What happens next is fixed now: the result goes in the Sunday report. A system whose agent ignores the news
keeps running unchanged (its books are judged on their own record), and any fix is a new registration.

Prediction: about 65% that both agents pass (headline tone is the easiest thing for an LLM to read; scouts
also weigh price and lane context, so they may move less).
