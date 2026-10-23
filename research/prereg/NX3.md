# NX3: fast NEWS-XGB, entry 1 minute after the news (news track)

Registered 2026-10-23, before NX3's data are built and before any NX3 return is computed. NX1 and NX2
results were read first (that is why NX3 exists), so NX3 is a new idea, counted. Phase-2 count at
registration: 38 (news 6). Phase 1 + phase 2 = 53. NX1 keeps running; NX2 failed and has no book.

## Why

NX2 entered at the next open after a 10:00pm batch, when most of the reaction to the news is already in the
price, and only about half of its edge came from what the news said. NX3 enters minutes after the
headline and sees how the price has started to react. Amit asked for the XGBoost to stay central and to act as fast as possible.

## Mechanism (PROTOCOL rule 10)

Prices take minutes to hours to absorb company news, longer at small stocks and for news that is hard to
read (Chan 2003; Tetlock 2010; Ke, Kelly & Xiu 2019). A $1B fund can't take $500 positions in thin names
at the speed of a headline; we can, with free data (Alpaca's IEX feed live, SIP minute bars on history).

## Events

- Headlines: NX1's (Benzinga, single-symbol, the same 3-day dedup, the same 23 news inputs per headline
  from NX1's tone files).
- Event (symbol, trading day d): the stock's first headline after the previous session's close and no
  later than 2 minutes before d's close. Its decision time t = that headline's time, or d's open when it
  came before the open. The news inputs combine every headline from the previous close up to t.
  Universe and dates: NX1's universe flag on the previous trading day; d from 2016-10-03 to 2025-09-23, so
  every exit ends before the 2025-10-01 holdout.
- Entry: the open of the first 1-minute SIP bar (adjustment=all) starting at least 1 minute after t and
  within 10 minutes of it; else the event is dropped. One minute is the finest wait the minute bars can
  show without assuming a fill inside the bar the headline arrived in. Exit: the close of the next trading day after d
  (adjustment=all daily close).
- Label: stock return entry to exit minus SPY's over the same minute and close (beta 1), divided by
  vol20 x sqrt(1.5), clipped to [-6, 6] (NX2's scaling).
- Placebo events: for each real event, a stock drawn at random (seeded) from the same 500-symbol batch's
  universe with no Benzinga article on d or the 3 sessions before (NX1's quiet rule), at the same t,
  with NX1's no-news values for every news input.

## Inputs (37, fixed)

NX1's 32 (stock and market inputs taken at the previous session's close; time bucket 0 before the open, 1
in the session) plus 5 intraday: minutes since the open at t; the stock's return from the previous close to
t; its return from t to the entry price (the first reaction); its share volume in the 30 minutes before t
over 30/390 of its 20-day median daily share volume (dv20 / price); SPY's return from the previous close to t.

## Models, effect and picks

NX2's design and settings (seed 20261024): a price-only baseline B on placebo events (the 6 stock, 2
market and 5 intraday inputs); a news model N on real events' z - B; effect e = N(x) - N(x0), x0 being x with
the 23 news inputs at their no-news values. Fired: |e| at or above the training years' 90th percentile.
Each day, fired events are taken in time order: up to 3 longs (e > 0) and 3 shorts. Control: as many of
that day's real events at random (seeded), same sides. Costs: NX1's (adaptive/costs.py on a $500 position at
the previous raw close, borrow for the calendar days held).

## Historical gate (walk-forward 2019 to 2025-09, before the holdout)

NX2's two tests on daily pick-minus-X differences, pooled over all test days: picks minus control, and
picks minus the same models fed that day's news shuffled among its real events. Both need mean > 0 and
Newey-West t (5 lags) >= 2. Also reported, no rule: per year numbers, the picks' vol20 percentile, the
share of picks before the open, and input-group importance.

If both hold, a $5,000 ledger shadow book starts with a live scorer on IEX real-time quotes (headline to
decision within the 1-minute wait; Alpaca's news and IEX streams are real time), version 1 trained on all of 2016-10 .. 2025-09-23. Its forward test is
NX2's (book minus control, NW t >= 2 at 100 closed trades or 2027-12-31). If either fails, NX3 is reported
as failed and gets no book.

## Prediction (written first)

Entering early should keep more of the reaction than NX2, but a 1-minute wait is slow next to machines that
trade on headlines in milliseconds, and costs at small stocks are high. I expect the gate to fail with
probability about 0.55; if it passes, the edge sits in pre-open news at stocks under $2B and the first
reaction input matters most.
