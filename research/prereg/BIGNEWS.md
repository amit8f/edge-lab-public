# BIGNEWS: rare, huge news (or chart shocks) traded long or short

Registered 2026-11-11, before any BIGNEWS number is computed. Source: Amit, 2026-11-11 (grill-me, 12 questions):
find the rare news that is groundbreaking for a company, for better or worse (his example: a biotech jumping on
cancer-vaccine results), flag it as soon as it happens, run it through our models and agents, and decide long,
short or nothing. Almost every headline must NOT trigger it. Two ideas: BIGNEWS (this file) and BIGNEWS-CAL
(end of this file, registered only, built later). Phase-2 count 139 -> 141 (phase 1 + phase 2 = 156).

Only plumbing is shared with older work (news feed, wire listener, 1-minute SIP bars, ledger, costs). The label,
the reader model, the agent prompt, the triggers and the exits are new and were not tuned on NX3, BOOM1 or SYS2b
results.

## Triggers (any one wakes the system)

1. A headline in the Benzinga/Alpaca feed or in our wire listener (PR Newswire, Business Wire, SEC current
   filings; `core/listener.py`) naming a stock in the universe.
2. A chart shock with or without news: in a 1-minute bar, volume at least 10x the stock's median 1-minute volume
   for that time of day (20 prior sessions) AND a move of at least 3% from the prior close (main) / 5% (penny).
   A shock wakes a news search over the prior 60 minutes; with no news found it can still trade only if stage 3
   clears its bar on chart inputs alone.

Universe: $1+ price and $1M+ median daily dollar volume (prior 20 sessions, raw SIP), main = $5+, penny = $1-5,
reported apart. Shorts only when easy to borrow (live) / modelled with borrow cost (history).

## Label (what "huge" means for training and grading)

For an event at entry time e: the move after entry, not the jump before it. Up event if the best long exit
(Q7 exits below) gains at least 4% net of costs before the stop is hit; down event the same for a short. Everything
else is "not huge". So a headline whose whole jump happened before our entry counts as a miss.

## Pipeline

- Stage 1, filter (every headline and shock): event type (new keyword classifier: trial results, FDA decision,
  buyout/merger, guidance change, fraud/short report, bankruptcy, offering, large contract, delisting, other),
  novelty (no same-type headline for the stock in 3 days), first-minutes volume ratio, gap from prior close,
  market cap, time of day. Keeps the top ~2% of headlines by a gradient-boosted model fitted on 2019-2022.
- Stage 2a, our own reader: a FinBERT-size model fine-tuned on 2019-2022 headlines (title + first 400 chars),
  target = the label above (up / down / not huge). Fine-tuned on CPU in GitHub Actions; frozen after 2022.
- Stage 2b, LLM judge (forward only, PROTOCOL LLM-cutoff rule): a free Gemini/Groq model given the goal and a
  strict rubric (new? material for a company this size? already priced in? direction?). Logged from day 1 of the
  forward book; it enters the decision only through a challenger (rule at the end) once 60 forward events exist.
- Stage 3, decision: a CatBoost model on stage 1 + 2a inputs plus 1-minute price/volume since the trigger,
  fitted on 2019-2022, cut chosen on 2023 so that at most ~0.05% of headlines fire. Long if up-score clears the cut,
  short if down-score clears it, nothing otherwise. No cap on trades a week.

## Entry and exit grid (choices made on 2019-2023 only)

- Entry: 1 minute after the trigger, 5 minutes after, or the regular open (for news before 9:30). Pre-market entry
  (from 4:00am ET, Alpaca extended hours) for news before 9:30 is a fourth option, priced at the first pre-market
  1-minute bar after the trigger + 1 minute plus a 1% extra cost (thin books).
- Exit: same-day 15:55, next close, or 5 sessions; each with a stop and take-profit set by volatility tier (calm /
  normal / jumpy by 20-day realised volatility, tiers and widths fitted on 2019-2022).
- One entry x exit combination is picked on 2023 (highest net per trade with >= 30 trades), then frozen.

Costs: `config/risk.yaml` (main round trip, penny 2%), plus borrow on shorts.

## Primary test (once, 2024-01 to 2025-09; holdout from 2025-10 untouched)

All of these:
1. Net per trade after costs >= +2.0%.
2. Newey-West t >= 3 (trades clustered by day).
3. Winners >= 60% of trades (target 65%).
4. At least 30 trades.
5. Beats random trades in the same stocks on the same days with the same entry/exit (paired, t >= 2).
6. Beats NX3's highest-confidence trades over the same months (net per trade, t >= 2).

Main and penny are tested separately; each may pass alone.

Pass: a $5,000 ledger book (no new Alpaca accounts), $1,000 a trade, at most 3 open; real money is not discussed
before 30 forward trades that also clear 1-3. Fail: no book; the result stays and the diagnosis is a new idea.

## Self-improvement (only after a pass)

Each week the system may propose a challenger (new inputs, the LLM judge, a stricter or looser cut). It runs as a
shadow ledger next to the frozen book and replaces it only after beating it over 8 weeks and 15+ trades; Amit is
told each time.

## Reported, no rule

Event-type mix; hit rate per type; shocks with no news; how much of the jump happened before each entry time;
listener vs Benzinga speed on fired events.

## Prediction (written first)

About 0.25 that main or penny passes: +2% a trade with 60% winners is far above anything we have (NX3 ~+0.1%).
Most likely failure: too few trades clear the cut, or the drift after entry is small once costs are taken.

# BIGNEWS-CAL: positions before scheduled catalysts (registered, built later)

Public, scheduled binary events (FDA decision dates, announced trial readouts) taken before the date, long or short
by a model's guess of the outcome. Registered now so its count is honest; its rules, data source (a free catalyst
calendar) and test are written before any data is pulled, in a later amendment to this file. No result exists.

## Amendment 1 (2026-11-11, before any data is pulled or any number computed)

Stage 2a reads the headline only. Our cached history (`news_history.fetch_range`, 2015-10 to 2025-09) kept the
headline and no summary, so "title + first 400 chars" cannot be trained or tested on history; live uses the same
headline-only input so training and live match. Found while mapping the cached data; no result exists.

## Amendment 2 (2026-11-11, before any data is pulled or any number computed)

Amit's choice (2026-11-11): the history test finds chart shocks on 5-minute bars first. A shock is a 5-minute
bar (9:30-16:00, raw SIP) whose volume is at least 10x the stock's median volume for that same 5-minute slot over
the 20 prior sessions, with a move of at least 3% (main) / 5% (penny) from the prior close; the trigger time is
the bar's end. The 1-minute scan registered above runs later, as data allows, and is reported as a check of the
5-minute version; it does not change the primary test.
