# NXE: when and how to exit NX3's trades (news track)

Registered 2026-10-30, before any NXE number is computed. Source: Amit's grill 2026-10-30 (round 1: NX3-DAY,
NX3-HOLD5, NX3-HOLD20; round 2: an exit study on NX3's own trades adding day 2 and 3, and an exit agent in
three levels) after the research report `/mnt/project-files/reports/Live strategies and adaptive exits.md`.
NX3's history (gate passed 2026-10-24) and its first live days were seen first, so this history can only
reject an arm (PROTOCOL rule 4); forward books decide. Eight arms, each counted. Phase-2 count at registration
74; after NXE, NXF and INS1: 86 (phase 1 + phase 2 = 101).

## Question

NX3 exits at the next close. Published news-drift evidence says the move lasts 1 to 3 days (longer at small
stocks) and that tight stops usually cost money. Does a different exit on NX3's own entries do better?

## What stays NX3's

The entries: NX3's walk-forward picks and same-day random controls, 2019-01 to 2025-09
(`results/NX3/NX3_walkforward.json`'s run, trades_v2 cache), the same entry minute and price, side, $500
size and costs (adaptive/costs.py, borrow for the calendar days held). Only the exit changes. A pick whose
exit would fall on or after 2025-10-01 is dropped from that arm (holdout).

ATR = the 14-session average true range on adjustment=all SIP daily bars ending the session before d.

## Arms (settings fixed now)

Hold arms (exit at a session's close, adjustment=all daily close; NX3 itself is H1 = d + 1):
- `H2`, `H3`: exit at the close of session d + 2, d + 3. No stop.
- `H5`, `H20`: exit at the close of d + 5, d + 20, or earlier at the first session close (from d's close on)
  that is 3 x ATR or more against the trade; that close is the exit.

Intraday arm:
- `DAY`: stop 1.5 x ATR, target 3 x ATR from the entry price, watched on 1-minute SIP bars of day d from the
  entry minute; a bar that reaches both counts as the stop; else exit at the close of the 15:55 bar.

Exit agent (one 5-session frame, daily SIP bars from d's entry: a session's low/high touching a level is a hit;
both in one session counts as the stop; a session opening past the stop fills at its open; else exit at the
close of d + 5):
- `EXA1` (level 1): stop 1.5 x ATR, target 3 x ATR.
- `EXQ` (level 2): stop and target per trade from two quantile LightGBM models (4.7.0, 200 trees, depth 3,
  at least 200 trades a leaf, learning rate 0.05) on NX3's 37 inputs plus side, e/cut and ATR/price. One
  predicts the adverse excursion (worst move against the trade over the frame, in ATRs) at its 80th
  percentile: that is the stop. The other predicts the favourable excursion (best move for the trade) at its
  50th percentile: that is the target. Each clipped to [0.5, 6] ATR. Walk-forward like NX3: fitted on all of
  NX3's fired events (not only picks) in the years before each test year, 5-session purge.
- `EXL` (level 3): Gemini (the SYS2 free model chain) is given each live NX3 pick's inputs, ATR and the 5-day
  chart numbers and returns a stop and a target in ATRs, temperature 0, prompt fixed in `tracks/news/nxe_live.py`
  before the first call. Forward only (LLMs know past prices, so history can't test it); logged per pick with
  the model id, answer and API cost.

## Primary test (per arm, history)

Amit's bar for a quick history check (grill round 1 Q2): daily pick-minus-control under that arm's exit (the
control traded with the same exit rule), pooled over test days, mean > 0 and Newey-West t >= 1.5 (lags = the
arm's longest hold in sessions). `EXQ` must also pass rule 21 (in-sample mean no more than twice out of sample)
and beat `EXA1` on the same picks (paired daily mean > 0, NW t >= 1.5); else it's reported as failed and EXA1's
rule is used. An arm that passes gets a $5,000 ledger shadow book on NX3's live picks; an Alpaca paper account
only after rule 20 (60 days and 30 closed trades, NW t >= 2).

## Reported, no rule

Each arm minus H1 on the same picks (paired daily mean, NW t); worst 1% of trades and worst month; hit rates of
stop and target; long and short separately; price under $5 vs above; per year. Every arm is counted, so the
report shows 8 tries.

## Prediction (written first)

H2 and H3 will be close to H1 (+/- 0.05% a trade); H5 and H20 will be worse after borrow on shorts; DAY and
EXA1 will be worse than H1 because tight levels catch noise; EXQ won't beat EXA1. Probability any arm passes
and beats H1 by NW t >= 1.5: about 0.3.
