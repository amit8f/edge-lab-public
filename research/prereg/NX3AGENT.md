# NX3-AGENT: a final decision agent on the tree's fired events

Registered 2026-11-04, before any NX3-AGENT number exists. Source: Amit, 2026-11-04: "should we add an idea that
we add a final decision bot (agent) that determines if to invest (long / short) the stock based on the finding the
tree gave us?" (yes). One idea. Phase-2 count 103 -> 104 (phase 1 + phase 2 = 119). Forward only: an LLM knows how
past news turned out, so history can't test it (PROTOCOL: LLM outputs validated after the model's training cutoff).
NXS-LLM (sizing) and EXL (exits) already log Gemini choices on NX3's picks; this one decides the trade itself.
META1 and META2 (machine-learning take/skip filters) failed on history.

## Question

Given the events NX3's tree fires, does an LLM that chooses TAKE, SKIP or FLIP for each earn more, after costs, than
simply trading what the tree says?

## Rules

- Events: every event the NX3 tree fires on a day (NX3-ALL's set; a short only where easy to borrow at decision time).
- Agent: Gemini (gemini-3-flash-preview, fallback gemini-3.5-flash-lite), temperature 0, asked once per day after the
  close with the fixed prompt in `tracks/news/nx3agent_live.py` (logged by its sha256 with the model id and raw
  answer). It sees only numbers logged at decision time: the tree's side and confidence, the news kinds found,
  the headlines' average positive/negative tone, article count, 20-day volatility, price, minutes from the open and
  the move to the news. v1 does not show it the headline text (the rows keep none; Benzinga terms).
- Answers: TAKE trades the tree's side, SKIP does not trade, FLIP trades the other side (a flipped short only where
  easy to borrow). A missing or unreadable answer counts as TAKE. Sizes are NX3-ALL's confidence rule on the tree's
  fired events (5% to 10% of a $5,000 book, scaled down together past 100%); the agent only changes the decision.
- Exit and costs as NX3 (next close, NX3's cost model). Same-day random control: as many of the day's decided events as
  the tree fired, with the tree's sides and sizes (seed offset 300).
- Ledger shadow book `nx3agent` next to NX3 in NX3's live job, no orders, from 2026-11-05.

## Primary test (forward)

Daily agent book minus the tree-only book on the same events and sizes (`agent_minus_tree`): mean > 0 and
Newey-West (5 lags) t >= 1.5 over at least 60 trading days and 100 taken trades. An Alpaca paper account only under
PROTOCOL rule 20.

## Reported, no rule

Agent minus control; share of TAKE, SKIP, FLIP; each action's mean net on its own; long and short separately;
the days the model fell back to gemini-3.5-flash-lite; missing answers.

## Prediction (written first)

Gemini will mostly TAKE and will SKIP a minority; flips will lose money on average because the tree's own side
already beats random. Probability the primary test passes: about 0.2; probability the agent is worse than the tree
(negative mean): about 0.5. Free-tier Gemini quota may fail some days; failed days count as TAKE and are listed.
