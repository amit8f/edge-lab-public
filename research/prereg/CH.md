# CH: chart-reading models on a live unusual-activity scanner (main and penny)

Registered 2026-10-30, before any data for this idea was downloaded and before any model was fit. Source:
Amit, 2026-10-30 ("use XGB or other models ... that will see a stock graph use all of the data volume VWAP
pattern all the lines"), settled in a three-round grill the same day; he replied "confirmed" to the design
summary at 09:01 UTC. Six arms, each counted: phase-2 count at registration 74 (main 34, penny 13, news 16,
cross-track 10, crypto 1). Phase 1 + phase 2 = 89.

Arms: `CHX-M`, `CHL-M`, `CHI-M` (main) and `CHX-P`, `CHL-P`, `CHI-P` (penny). X is XGBoost, L is LightGBM,
I is a small image model that reads a picture of the chart. The main and penny arms never share a model, a
cut or a scale (CLAUDE.md, tracks).

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** Traders who react late or by habit to stocks that suddenly move: retail buyers chasing
   gaps and breakouts (Barber and Odean 2008, attention), and holders who sell into strength or hold on to
   losers near reference prices such as the 52-week high (George and Hwang 2004). Published intraday patterns
   on such stocks are the opening-range breakout on "stocks in play" (Zarattini, Barbon and Aziz 2024) and the
   first half hour predicting the last (Gao, Han, Li and Zhou 2018). A model reading many chart numbers at
   once may find which of these setups continue and which reverse.
2. **Why us.** Partly. These are small, fast trades in names that move today; a $1B fund can't put size into
   a $3 stock on a gap, and a $5k book can. Against that, intraday chart patterns are the most crowded retail
   and HFT ground there is, and our data is 16 minutes late.
3. **Published.** The single setups, yes (2004 to 2024), and they decay; chart-image CNNs too (Jiang, Kelly
   and Xiu 2023, on daily charts). Expect little after costs, especially in penny stocks.

## Question

Among stocks the scanner flags as unusual during the day, can a model reading the chart up to 16 minutes ago
choose long, short or skip, and a stop/target pair, so that its trades beat same-day random flagged stocks
traded the same way, after costs?

## Data and universe

- Daily: W1's cache (`tracks/news/w1_data.py`): Alpaca SIP daily bars, raw for price, gap and dollar volume,
  all for returns; >15-day gaps end a series; zero-volume bars dropped; delisted stocks kept.
- Intraday: Alpaca SIP 5-minute bars (`feed=sip`; `adjustment=raw` for prices, VWAP and fills; the day's split
  factor from the daily all/raw ratio for anything compared across days), 4:00 to 20:00 ET, downloaded only
  for candidate days (below), in ~500-symbol batches with atomic writes and resume.
- Main universe on day t: previous raw close >= $5 and 20-day median raw dollar volume >= $5M (stands in for
  the $300M cap, as in EAP1). Penny: previous raw close $1 to $5 and 20-day median dollar volume >= $1M. One
  row per company (highest dollar volume).
- **Candidate days (download filter).** A symbol-day is downloaded when, on its daily bar, volume >= 1.5x the
  20-day average, or |gap| >= 3%, or the high is above the prior 20-day high, or the low is below the prior
  20-day low, or the close is within 2% of the prior 252-day high or low. This uses the whole day and only
  limits what is downloaded; a trigger can only fire inside it. Live, the scanner has no such filter, so a
  few triggers live would never have been seen in history (an early volume spike on a quiet day). The forward
  book flags trades whose day fails this filter and reports them separately. Disclosed now.
- Dates: entries 2017-01-01 to 2025-09-23, so no exit reaches 2025-10-01 (holdout untouched). 2016 is used
  only for warm-up (moving averages, the 52-week range).

## Scanner (the triggers)

At each 5-minute bar close s from 9:45 to 15:30 ET, using bars ending at or before s, a universe stock
triggers when any of these holds (VWAP, volume and range are the regular session's from 9:30):

1. Relative volume at time: today's volume to s >= 2x the expected volume to s (20-day average daily volume
   times the share of a day's volume normally done by s; that share curve is fit per track on 2016 data).
2. Gap: |today's open / previous close - 1| >= 3%.
3. New 20-day high or low: price at s above the prior 20 sessions' highest high, or below their lowest low.
4. VWAP cross on volume: the last bar closes on the other side of VWAP from the previous bar, and its volume
   is >= 3x the 20-day average daily volume / 78.
5. Opening-range break (s >= 10:00): price at s outside the 9:30-10:00 high/low, with relative volume >= 1.5.
6. Strong first half hour (s = 10:00 only): |price at 10:00 / previous close - 1| >= 2% (main) or 4% (penny).
7. Near the 52-week extremes: price at s within 2% of the prior 252-day high or low, with relative volume
   >= 1.5.

A stock is sent to the model at its first trigger of the day only (one decision per stock per day; exploratory
below covers later triggers).

## What the model reads

**Timing.** Decision at s + 16 minutes (bars end 16 minutes before now, CLAUDE.md). The model reads bars to
s only. Entry is at the open of the 5-minute bar starting at s + 15 minutes, the stand-in for the live price.

**XGB and LGBM inputs** (68, computed at s; the list is fixed now and is not pruned after any result):
- Trend (10): distance from session VWAP, VWAP +/- 1 and 2 standard deviation bands (%B form), from the 9 and
  20 EMA on 5-minute bars, from the 20, 50 and 200-day SMAs, 20-day slope of the 50-day SMA, ADX(14) daily.
- Momentum (9): RSI(14) on 5-minute and daily bars, MACD histogram (12, 26, 9) on 5-minute and daily, returns
  over the last 30 minutes, today to s, 1, 5 and 20 days.
- Volume (10): relative volume at time, last bar's volume / average bar, OBV 20-day slope, CMF(20) daily,
  up-volume share today, the 20-day volume profile's point of control and value-area high and low (distance),
  dollar volume (log), 20-day volume trend.
- Volatility (7): ATR(14) %, 5-minute ATR %, Bollinger width (20, 2) daily, Bollinger %B daily and 5-minute,
  today's range / ATR, 20-day realised volatility.
- Levels (14): distance from previous day's high, low and close, from the opening range high and low, from
  the 52-week high and low, from the nearest classic pivot (P, R1, S1) and the nearest Fibonacci level
  (38.2/50/61.8%) of the last 60-day swing, gap %, gap filled share, today's open position in yesterday's
  range, round-number distance, days since the 52-week high.
- Candles (8, by code on daily and 5-minute bars): engulfing, hammer/shooting star, doji, inside bar, outside
  bar, three-bar run, gap-and-go (no fill by s), long upper/lower wick share.
- Market and context (10): SPY return today to s and 5-day, SPY above its VWAP, the sector ETF's return today
  (SPDR sector of the stock's SIC, else SPY), VIX previous close and its 5-day change, minutes since the open,
  day of week, which trigger fired (one code), price (log).

Missing inputs stay missing (both learners handle it).

**Image model input.** A 3-channel picture rendered by code at s, 96 x 64 pixels per panel, two panels side by
side: today's and yesterday's 5-minute bars to s (OHLC bars, session VWAP line, 9 EMA), and the last 60 daily
bars (OHLC, 20 and 50-day SMAs), each with a volume strip below; prices scaled to each panel's range, as in
Jiang, Kelly and Xiu (2023). No numbers are added to the picture.

## Labels and decision

Each candidate gets six labels: net return of a long and of a short under three stop/target pairs, with the
stop at 1.0, 1.5 or 2.0 times the daily ATR(14) from entry and the target at twice the stop (1x/2x, 1.5x/3x,
2x/4x). Exit at the first barrier touched or at the close of the 5th session (entry day counts as the first).
The entry day uses the 5-minute bars after entry; later days use daily high and low; when a day touches both
barriers, the stop counts (conservative). Net is after the cost below, divided by entry.

Each arm fits one model per label (six outputs; the image model has one network with six outputs), loss
Huber (delta at the training labels' 90th percentile of |label|), to predict that net return. For a
candidate, the arm takes the largest of the six predictions. It trades that side and pair when the largest
prediction is above the cut, else skips.

**Cut and size, fixed on training years only.** For test year Y: fit on 2017 to Y-2, predict year Y-1, and set
the cut at the 80th percentile of Y-1's largest predictions. Size is 2% of the book at the cut, rising
linearly with the prediction's percentile in Y-1 to 10% at the 99th percentile and above. Then refit on 2017
to Y-1 (5-session purge before Y) and apply that cut and scale to year Y. Test years 2019 to 2025 (to
2025-09-23).

**Settings, fixed now, no tuning:**
- XGB: XGBoost, 400 trees, depth 4, learning rate 0.05, row and column sampling 0.8, min child weight 200
  (tame, after NXR), L2 10, seed 20261030 + year.
- LGBM: LightGBM, the same mapped settings (15 leaves, at least 200 events a leaf), deterministic.
- IMG: a 3-layer convolutional net (32, 64, 128 filters, 5x3 kernels, batch norm, leaky ReLU, max-pooling), a
  dropout 0.5 dense layer, 6 outputs; Adam 1e-3, batch 256, at most 10 passes, early stop on a random 10% of
  the training rows, CPU, seed 20261030 + year. If one fit can't finish inside a GitHub job, the image arms
  train on a fixed random half of the training rows (seeded), and that is reported.

## Costs

`adaptive/costs.py` (Plan v3's IBKR-style model) for a $250 position (5% of a $5,000 book, the middle of the
size range): half the measured spread each side, commission, sale fees; shorts add borrow at the track default
(main 3%, penny 20% a year) for the calendar days held. History can't tell which penny stocks were shortable;
the forward book skips shorts Alpaca won't allow (as P1). Disclosed now.

## Primary test (per arm)

1. **Rule 21 first.** Each year's model scored on its own training rows with the same decision rule and
   control: mean daily pick-minus-control, weighted by days, beside the pooled test mean. In-sample more than
   twice out-of-sample marks the arm overfit and failed before its test number is read.
2. **Control.** For each pick, a random stock from the same day's triggered candidates of the same track (not
   the pick), drawn with `numpy default_rng(20261030 + date ordinal)`, entered at its own first trigger's
   entry time, the same side and the same stop/target multiples of its own ATR, the same costs.
3. **Statistic.** Daily cohorts 2019 to 2025-09: picks' mean net minus controls' mean net (unweighted). Mean,
   Newey-West t (5 lags, holds overlap up to 5 sessions) and a 95% block bootstrap (20-day blocks, 10,000
   draws, seed 20261030).

An arm passes when rule 21 holds and PROTOCOL rule 12 holds (HAC t >= 3.0, bootstrap CI above 0, picks' own
mean net > 0, at least 300 trades and 100 trading days) and rule 11 holds (monthly sums of the daily
difference: FF5 + momentum alpha > 0 with t >= 2). The deflated Sharpe is reported with 89 trials. Each arm is
judged on its own; with six arms, one pass at t >= 3 is still worth a forward look, and the six results are
shown together.

A failure that meets rule 19 (hand check passed, picks beat the control at HAC t >= 1.5, picks' own mean net
> 0) gets a near-miss shadow book. Otherwise the arm closes.

**Hand check before any return is read.** Draw 20 random triggers per track. For each, check on a separate
Alpaca pull (and a chart site for the look) that the trigger fired at the logged time, the inputs at s match
(VWAP, relative volume, gap, 20-day high/low), and the logged entry is the open 15 minutes after s. At least
19 of 20 per track must be right. A matcher or timing bug may be fixed and checked again on history (rule 23).

## Forward (passes and near-misses)

A $5,000 shadow book per arm (log only, no orders): the scanner runs every 5 minutes in market hours on bars
to now - 16 minutes, the frozen final model (fit on 2017 to 2025-09) decides, entry at the live quote, size
2% to 10% by the fixed scale, no position count cap, total open exposure at most 100% of the book (a trade
that would pass it is skipped and logged), no leverage, shorts only where Alpaca allows. Each pick logs a
same-day random control as above. Promotion to an Alpaca paper account follows rule 20 (60 trading days and 30
closed trades, beats the control at NW t >= 2, own mean > 0). Real money is rule 9, unchanged.

## Reported, no rule (exploratory, labelled now)

- Size-weighted returns (2% to 10% by confidence) and the book's equity curve with the 100% exposure cap.
- Every trigger (not only the first per day), and each trigger type on its own.
- Entry at s (as if real-time data were bought), against s + 15 minutes: how much the delay costs.
- Long and short separately; each stop/target pair's share and result; time of day; by year.
- The best single published setup on its own (opening-range break long above, short below, 1.5x/3x) against
  the same control, as a benchmark the models must beat to be worth their complexity.
- How often the arms agree (same stock, same day, same side), and the mean of the three main or three penny
  arms' predictions as a combined arm (exploratory, not counted).

## Prediction (written first)

Main: picks beat the control by +0.05% to +0.15% a trade before costs, HAC t below 2. At $250 a trade, costs
are about 0.7% a round trip in main and 1% in penny (spread plus the $0.35 minimum commission), so picks' own
mean net is below 0 in both tracks. The image arms do no better than the tree arms. Penny moves are larger
but noisier. Most likely all six fail, with at most one main arm reaching the near-miss lane.

## Amendment 1 (2026-11-01): the image arms, shrunk to fit free compute

Written before any image-arm result exists; the tree arms' results were seen first and are unaffected. Reason: the
registered net (32/64/128 filters on 64 x 192 pictures, about 3.5 GFLOP a picture a pass) cannot be trained in a
GitHub job, and the first run died of memory building every picture at once (36.8 GiB). Amit, 2026-11-01:
"Shrink". CH.md already allows training on a random sample if a fit cannot finish; this goes further, so it is
written here. Changes, for CHI-M and CHI-P:
- The net reads the picture at half height (32 x 192; each pair of rows becomes its maximum). The drawing, the
  inputs and the panels are unchanged.
- Filters 8, 16, 32 (5 x 3 kernels, batch norm, leaky ReLU, (2,1) max-pooling, dropout 0.5, 6 outputs).
- Each fit sees a fixed random sample of at most 100,000 training rows (seeded), at most 3 passes; the early stop
  on a random 10% stays. Adam 1e-3, batch 256, seed 20261030 + year, as before.
- Pictures are drawn only for the rows a fit or a prediction needs.
- Rule 21 for the image arms scores the same 100,000-row sample each final model was fitted on, not every
  training row.
- Everything else (labels, cut, size, control, tests, pass rules, the count of 89 trials) is unchanged.
