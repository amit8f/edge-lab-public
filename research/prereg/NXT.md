# NXT: trailing stops on NX3's trades (news track)

Registered 2026-10-31, before any NXT number is computed and before NXE's results exist (NXE's run
41166203095 was still going). Source: Amit, 2026-10-31: "if we see it going good we can move the takeprofit
more and put the stoploss at a win ... we will need to monitor it and update if needed", and stops "according
to the level of confidence". NX3's history and first live days were seen first, so this history can only
reject an arm (PROTOCOL rule 4); forward books decide. Two arms, each counted. Phase-2 count 86 -> 88
(phase 1 + phase 2 = 103).

## Question

Does a stop that starts at a fixed distance, moves to breakeven once the trade is up, and then trails the
best price do better than NX3's next-close exit on NX3's own entries?

## What stays NX3's and NXE's

Everything in NXE.md's "What stays NX3's" (entries, picks and same-day random controls 2019-01..2025-09,
entry minute and price, side, $500, costs with borrow for the days held, holdout drop) and its ATR. The frame
is NXE's exit-agent frame: day d after the entry, then sessions d+1..d+5 (`frame_sessions`), daily SIP bars,
`adjustment=all`.

## Arms (settings fixed now)

Both arms walk the frame session by session with a stop level S and the best price B so far (B = entry at
the start; for a short, "best" is the lowest price and every level mirrors):
1. A session after d that opens past S fills at its open.
2. Else, if the session's low reaches S (high, for a short), the exit is S.
3. Else B is updated with the session's high (low, for a short), and S is raised for the next session, never
   lowered: once B >= entry + 1 ATR, S = max(S, entry, B - k x ATR).
4. No target. Still open after d+5: exit at d+5's close.

Checking the stop before updating B with the same session means a session's high never protects that same
session's low (the order inside a session is unknown, so the unfavourable order is assumed).

- `NXT-TR`: k = 1.5 for every trade; the starting stop is entry - 1.5 ATR.
- `NXT-TC`: k from NX3's confidence c = |e| / cut (the fired score over the year's firing cut; c >= 1 for
  picks; a control's c is clipped up to 1): k = 1.5 x min(c, 2), so between 1.5 and 3 ATR. The starting stop
  and the trail both use that k. More confident picks get more room.

## Primary test (per arm, history)

Same as NXE: daily pick-minus-control under the arm's exit (the control traded the same way), pooled over
test days, mean > 0 and Newey-West t >= 1.5 with 5 lags. An arm that passes gets a $5,000 ledger shadow book
on NX3's live picks, with the stop updated during the session from the live job's price polls; an Alpaca
paper account only after rule 20 (60 days and 30 closed trades, NW t >= 2).

## Reported, no rule

Each arm minus H1 (NX3's next-close exit) and minus NXE-EXA1 on the same picks (paired daily mean, NW t);
NXT-TC minus NXT-TR; share of trades exiting at the starting stop, at breakeven, on the trail, and at time;
worst 1% of trades and worst month; long and short; under $5 vs above; per year.

## Prediction (written first)

Both arms will be worse than H1 after costs: news drift is mostly in the first 1-2 days and the trail mostly
cuts trades on normal noise. NXT-TC will be a little better than NXT-TR. Probability either passes: about
0.25.
