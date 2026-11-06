# SHORT-ML: a model built only to find shorts

Registered 2026-11-06, before any SHORT-ML number is computed. Source: Amit, 2026-11-06 ("a model dedicated to short",
pick up specific news fast on bad days, "be creative"). NX3, NX3-BARS, SHORT-FIX and SHORT-FIX2 were seen first
(NX3's shorts earn about zero on their own; they only beat random shorts), so this history can only reject
(PROTOCOL rule 4); a forward book decides. One idea. Phase-2 count 109 -> 110 (phase 1 + phase 2 = 125).

## Question

NX3 is one model for both sides, with a placebo baseline. If one model learns only "which bad-news events keep
falling until the close", fed with the market-stress inputs too, do its shorts make money after costs?

## Rules

- Events: NX3's real events with negative tone (fb_neg > fb_pos), same entry minute, price and costs as NX3 (NX3.md),
  exit at the next session's close (NX3's rule), short side only.
- Model: one XGBoost regressor (NX3's settings) on the same 37 inputs (the SPY 1- and 5-day returns and the SPY move
  at the entry are the market-stress inputs), trained per test year on events up to the purge date (NX3's
  walk-forward), target = minus the market-adjusted next-close return scaled by the stock's volatility.
- Fire: predictions above the 90th percentile of the training predictions (and above 0). First 3 per day in time
  order are shorted. Control: as many of the day's negative-tone events at random, also shorted.
- No borrow data in history: ignored; live trading uses Alpaca's easy-to-borrow flag.

## Primary test (history)

All of: pick-minus-control mean > 0 and NW (5 lags) t >= 1.5; the shorts' own net per day after costs > 0 with NW
t >= 1.5; at least 300 short trades.

## Reported, no rule

The same on days when SPY's prior-day return was below -1% (bad days) and on calm days; net per trade; per year.

## Forward book

If it passes: ledger shadow book `nx3sml` next to NX3, no orders, $5,000, 3 shorts a day, from the next trading
day; rule: NW t >= 1.5 on its own net at 30 closed trades; an Alpaca paper account only per PROTOCOL rule 20.

## Prediction (written first)

Probability the primary passes: about 0.25. A single model on shorts alone has less data than NX3's two-sided
model, and the bad-news drift may be priced by the next close.
