# NX1: NEWS-XGB, the headline model (news track)

Registered 2026-10-22, before the model is trained and before any return of this idea is computed. From the
Plan v3 model round (`docs/PLAN_V3.md` section 13, decisions M1-M14). Phase-2 count at registration: 35
(news 4). Phase 1 + phase 2 = 50.

## Mechanism (PROTOCOL rule 10)

- Other side: traders who under-react to company news in the days after it, most of all for small stocks
  and news that is hard to read.
- Why a $1B fund can't take it: $250-500 positions in thin stocks, a few a day, held 3 days.
- Published: news tone predicts returns for a few days (Tetlock 2007; Tetlock, Saar-Tsechansky &
  Macskassy 2008), and machine-read news adds to it (Ke, Kelly & Xiu 2019). Expected effect: small after
  costs. The old news-signal model failed its placebo test (it fired almost as often on no-news times),
  so this registration makes that test a gate.

## Question

Does an XGBoost model on headline tone and type, plus a few stock features, pick 3-day long and short
trades that beat a same-day random pick from the same news, after costs, while firing much less on
no-news times?

## Events

- Headlines: Benzinga via Alpaca, articles that name exactly one symbol. The same headline for the same
  symbol within 3 days counts once.
- Nightly batch: the run after 10:00pm ET on trading day d takes headlines created after 10:00pm ET on
  the previous trading day, up to 10:00pm ET on d (weekend and holiday headlines join the next trading
  day's batch). All of one stock's headlines in a batch make one event.
- Universe on d: common stock (`core.instruments.classify`), raw close at least $1, median 20-session raw
  dollar volume at least $1M, no gap over 15 calendar days in the last 20 sessions. Point in time
  (delisted stocks kept).
- Entry at the open of the first session after d; exit at the close of the third session (entry day = 1).

## Label (training only)

The return from entry open to exit close (`adjustment=all`) minus beta times SPY's return over the same
window. Beta comes from 250 daily returns ending on d (at least 120, else 1). Classes are long (top 20% of
the training years' labels), short (bottom 20%) and none. The edges are set on the training data only.

## Inputs (32, fixed)

- News (23): article count; Loughran-McDonald negative, positive and net word counts (`tracks/news/lm.py`);
  FinBERT (`ProsusAI/finbert`, pinned revision logged) mean positive, mean negative, max negative, max
  positive and mean neutral; mean headline length; 12 regex type flags (earnings, guidance, analyst up,
  analyst down, offering, merger, FDA or trial, contract, legal or SEC, bankruptcy or delisting, dividend
  or buyback, executive change; patterns in the code); time of the last headline (before the open, in the
  session, after the close).
- Novelty (1): days with news for the stock in the 30 calendar days before d.
- Stock (6): returns over 1, 5 and 20 sessions to d's close, 20-day volatility, log dollar volume, log price.
- Market (2): SPY's 1- and 5-session returns to d's close.

No AI-model reading is an input here. AI-read inputs may join only through forward retraining (the cutoff
look-ahead), as a new version.

## Training

- Data: batches from 2016-10 through 2025-09-24, so every exit ends before the 2025-10-01 holdout.
- XGBoost multi-class, fixed settings: depth 4, 400 trees, learning rate 0.05, subsample 0.8, column
  sample 0.8, min child weight 50, balanced class weights, seed 20261022. No tuning.
- Walk-forward, reported and not a gate (Plan v3 lets forward data decide): test years 2019 through
  2025 (January to September), each trained on all earlier years with the last 5 sessions before the test
  year dropped. The report shows the importance of each input, the picks' after-cost return against the
  random control, and the placebo numbers below.
- The live model (version 1) is trained once on all of 2016-10 .. 2025-09-24. Weekly retrains add paper-
  period data only; version 1 keeps running in the ledger beside each retrain.

## Picks and the shadow book

- Each night: score the batch. Events whose most likely class is long or short are ranked by p(long) -
  p(short); the top 3 longs and the top 3 shorts are picked.
- Control: the same number of events drawn at random from that night's batch (seeded), each given the
  same side as the pick in its slot.
- Book: $5,000 in the ledger (SYS1 shadow rules, `adaptive/sizing.py`, `adaptive/costs.py`: 1% risk to a
  10% stop, 10% name cap, 5% per short, 1x gross). Shorts only where Alpaca marks the stock shortable.
  Fills at the entry open and exit close from SIP daily bars (raw), plus costs.

## Placebo gate (must pass before any Alpaca account)

Each night, the same number of placebo events as real events: random universe stocks with no Benzinga
article in the batch or the 3 sessions before it, scored with their real stock and market inputs and the
no-news values for every news input. The gate reads after 60 trading days of paper data. It passes only
if both hold:
1. The share of placebo events whose most likely class is long or short is at most half the share for
   real events.
2. Real fired events beat placebo fired events: mean signed 3-day market-adjusted return after costs,
   difference > 0 with Welch t >= 2.

The walk-forward report shows the same two numbers on history.

## Primary test (read at 100 closed trades, and at 2027-12-31)

Daily return of the pick book minus the control book, after costs, Newey-West t (5 lags). Pass: t >= 2 with
a positive mean, and the placebo gate passed. Then the book moves to its own paper account under SYS1's
promotion rule. Fail: dropped, or re-registered as a new version.
