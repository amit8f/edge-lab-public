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

## Live book details (2026-10-24, after the gate passed and before any forward data)

How the registered shadow book reads the rules above with live data (`tracks/news/nx3_live.py`):
- Headlines arrive from Alpaca's Benzinga feed, read every 30 seconds. An event is decided on the first
  read at or after t + 1 minute; one not decided within 10 minutes of t is logged as missed.
- Intraday inputs come from Alpaca's free IEX real-time feed, the only free real-time source. The price
  at t is the first IEX minute bar's open (news before the open) or the last IEX close before t. The entry
  input (`r_react`) uses the latest IEX trade at decision time, the price a 1-minute entry would pay.
  `rv30` counts IEX volume divided by the stock's IEX share of SIP volume over the previous 5 sessions,
  since IEX carries only a few percent of trading.
- Stock and market inputs, universe, novelty, the 3-day dedup and FinBERT's commit are training's code
  and settings. The model is version 1 (`research/results/NX3/nx3_v1_news.json`) with its frozen cut.
- A short is taken only where Alpaca marks the stock shortable and easy to borrow (as NX1's book);
  otherwise the slot passes to the next fired short. The control is drawn as in training from all of the
  day's decided events, after the close.
- Every return is settled with training's code on SIP bars once they are free (entry = open of the first
  SIP minute bar at or after t + 1 minute, exit = next close, minus SPY), with training's costs on a
  $500 trade. Each decided event logs both its live inputs and, at settlement, the SIP entry, so the gap
  between live and training inputs is measured.

## Bug fixes (2026-10-28, before any live day settled; no rule above changes)

- News names symbols that Alpaca doesn't list as US stocks (foreign listings such as `TSX:CAE`, codes such
  as `005380`). The scorer sent them to Alpaca's bar API with the batch's other new symbols; Alpaca refused
  the whole request (a 400, logged on 2026-10-28 for `TSX:CPX` and `TSX:CAE`), and the scorer stopped,
  restarted on the same events and stopped again until those events passed their 10-minute limit. Every
  pre-open batch held such symbols (60 to 83 events a day), so every pre-open event on 2026-10-26 (when the job
  also started late), 10-27 and 10-28 was logged as missed (1,432, 1,244 and 1,522 events); the first
  decisions came at 09:40 and 09:41, just past the limit, and those books held only intraday picks. Fix: only
  symbols in Alpaca's active US stock list get stock inputs; the rest are logged "not in universe", as in
  training, where they have no bars.
- The settlement passed training's code no exit date, so no day settled. Fix: the next session is passed.
- 2026-10-26, 10-27 and 10-28 did not run the registered event set. They are settled and kept in the ledger
  (append-only), marked `excluded`, and left out of the forward test and its count of closed trades.

## Paper account (2026-10-29, before any forward trade has settled; Amit chose "Start now")

The live book's picks are also sent to the Alpaca paper account EL-NX3 (fake money, `tracks/news/nx3_orders.py`):
a market day order at each pick's decision, $500 in whole shares at the IEX price the scorer read (a pick
above $500 a share is skipped), and a market-on-close exit on the next session. Fills are logged after the
close as ledger kind `nx3_orders`. This changes nothing in the forward test: it stays the ledger book minus
its same-day control, settled on SIP bars. An order error never stops the book.
