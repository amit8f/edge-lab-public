# NX3TA: chart indicators added to CatBoost on NX3's news events

Registered 2026-11-06, before any NX3TA number is computed. Source: Amit, 2026-11-06 ("using the graph and volume,
all the indicators there are"). One idea. Phase-2 count 115 -> 116 (phase 1 + phase 2 = 131). NX3's and NXK2's
history were seen first, so this can only reject (PROTOCOL rule 4); a forward book decides.

## Question

Does adding classic chart and volume indicators to NX3's 37 inputs make CatBoost (NXK2-CAT, the best learner so
far) pick better trades on the same news events?

## Data

NX3's trade files unchanged (same events, entries, exits, costs). New stage `chart_bars`: daily SIP bars
2015-01-01 .. 2025-09-30 (raw and all-adjusted) for every symbol NX3 ever traded and SPY, inputs computed by
`core/chart_features.py`. Inputs for an event on session d use the bars up to the previous session's close, so they
are known before any news time on d. A >15-day gap ends a series; zero-volume bars dropped; ratios from all-adjusted
bars, price level and dollar volume from raw bars.

Added inputs (28): r60, r120, 12-1 month momentum, vol60, vol20/vol60, RSI(14), RSI(2), MACD histogram / price,
Bollinger %b and width, ATR(14) / price, distance from the 20/50/200-day averages, distance from the 52-week high and
low, gap, open-to-close, close location in the day's range, range / ATR, volume / 20-day average, 5-day / 20-day
volume, 20-day signed-volume balance (OBV slope), largest and smallest daily return in 20 days, up/down streak; SPY
distance from its 200-day average and SPY 20-day volatility. NX3 already has r1, r5, r20, vol20, log dollar volume,
log price, SPY r1 and r5.

## Rules

Two arms, same walk-forward (test years 2019 .. 2025-09, 5-session purge), seeds, picks (3 long + 3 short in time
order above the 90th-percentile fire cut), random control and shuffled-news test as NX3:
- CAT: NXK2-CAT exactly (37 inputs).
- CAT_TA: the same CatBoost with the 28 inputs added to both the placebo baseline and the news model. The news
  effect is still N(x) - N(x with news set to placebo), so the chart inputs can change which events fire and which
  side they take, never count as news by themselves.

## Primary test (history)

Passes if CAT_TA passes NX3's gate (pick-minus-control and pick-minus-shuffled-news, each mean > 0 and NW 5-lag
t >= 2.0) AND CAT_TA minus CAT on shared pick days has mean > 0 with NW t >= 1.5 (the near-miss bar).

Reported, no rule: each arm's gate numbers and per-year results; share of events with every chart input present.

## Forward book

If it passes: a $5,000 ledger shadow book beside NXK2-CAT (no orders), judged by PROTOCOL rule 20.

## Prediction (written first)

CAT_TA passes the gate: about 0.6 (CAT already does). Beats CAT at t >= 1.5: about 0.2, because the news model
already sees recent returns and volatility and the effect term cancels most of what charts say about the stock.
