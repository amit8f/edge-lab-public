# PICK-MAIN and PICK-PENNY: a few very confident same-day longs

Registered 2026-11-09, before any data for it is downloaded or any number computed. Source: Amit's grill-me answers,
2026-11-09 17:27-18:27 UTC (thread "Full rethink and deep research"; Q1-Q19, every recommendation accepted or changed
by him as written here). Two ideas: PICK-MAIN (main track) and PICK-PENNY (penny track), with separate data,
models and settings. Phase-2 count 136 -> 138 (phase 1 + phase 2 = 153).

## Question

Can a model that watches every liquid stock through the day, and buys only when it is very sure, pick longs that
end the day +3% or better more often than random buys at the same minutes, and make money after costs?

## Universe (point in time, rebuilt each day from the prior session)

- PICK-MAIN: prior close >= $5 and median dollar volume over the prior 20 sessions >= $5M (about 2,500 names).
- PICK-PENNY: prior close $1 to $5 and median dollar volume over the prior 20 sessions >= $1M (about 800 names).
- Delisted stocks kept. A >15-calendar-day gap ends a series. Zero-volume bars dropped.

## Data

5-minute SIP bars (`feed=sip`): `adjustment=raw` for prices, fills, dollar volume; `adjustment=all` for return
inputs. News: the Benzinga headlines and the scores NX3 already computes (time-stamped, first tradable after
publication). Daily bars and SPY 5-minute bars for market inputs. Only bars complete at the decision time are used.

## Trading rule

- Decision times: every 5 minutes from 9:35 to 14:00 ET. A stock clearing the bar is bought at the next bar's open
  (one buy per stock per day). Long only.
- Exit: take profit +8% (main) or +15% (penny) above the fill; else the stop; else the 15:55 bar's close.
- Stops by tier (Amit Q10, Q17): each stock's tier comes from its usual 30-minute swing over the prior 20 sessions
  (main: calm / normal / jumpy terciles; penny: one tier). Candidate stop widths: 1x, 1.5x, 2x the 30-minute swing
  (penny 1.5x, 2x, 3x), plus any rule found in the published research on stops before the 2019-2023 fit, added to
  the candidates and listed in an amendment before any result. The width per tier is chosen on 2019-2023 only.
- Size (Amit Q7, Q15): 10% to 40% of a $5,000 ledger book, rising linearly with confidence from the bar to the
  top 1% of training scores above the bar; highest confidence first until the book is fully used. No fixed
  number of picks. This overrides `config/risk.yaml` `max_name` (0.20) and `risk_per_trade` for this ledger
  book only; no orders, and real money needs its own size decision under PROTOCOL rule 20.
- Costs: `config/risk.yaml` round trip, main 0.2%, penny 2%.

## Model

- Target: does the trade (rule above) end at +3% or better after costs (1/0).
- Three versions per idea (Amit Q2, Q14): NEWS (news inputs since the open and the prior close, plus the stock's
  price level and time of day), CHART (price and volume shape over 5-minute and daily bars), COMBINED (both, plus
  the market inputs). COMBINED is the primary test; NEWS and CHART show where any edge comes from.
- Learners tried in each version (Amit Q3): XGBoost, CatBoost, LightGBM, Extra Trees, logistic regression, KRR.
  Chosen by validation log-loss.
- Bad days (Amit Q4): a market score from SPY's gap, its move since the open, the last 5 sessions and the prior
  20-day volatility. On days in the worst fifth of training days by that score, the bar is raised; never a skip.
- The bar: from candidates {top 0.10%, 0.05%, 0.02%} of training decision scores (normal days), and for bad
  days {same, twice as strict, four times as strict}; the pair with the best validation share of +3% trades with
  at least 100 validation trades.

## Periods

Train 2019-01 to 2022-12, validate 2023 (learner, stop width, bar). Then refit on 2019-2023 and test once on
2024-01-02 to 2025-09-30 (5-session purge). Label dates from 2025-10-01 stay untouched (phase-2 holdout).

## Primary test (per idea, COMBINED version, test period)

All three must hold, with at least 30 test trades (fewer: "too few", not a pass):
1. The share of trades ending +3% or better beats random buys (one random universe stock bought at the same
   minute with the same exit rules for each pick): day-clustered bootstrap, 95% interval of the difference > 0.
2. Mean net per trade after costs > 0, Newey-West t >= 2 on daily book returns.
3. Mean net minus the random buys > 0, Newey-West t >= 2.
Pass: a 60-day forward ledger book per idea (Amit Q8), then PROTOCOL rule 20 for any real money.

## Reported, no rule

NEWS and CHART versions; share of days with a pick; the share of trades reaching +3%, +8%/+15%, and the stop;
per year; bad days vs other days; per stop tier; the book's daily % (Amit's 3% goal is reported, not a pass bar).

## Prediction (written first)

Most days have no pick. The share ending +3% or better is well above random (picks are jumpy stocks), but after
costs the mean net is near zero for main and negative for penny. Probability PICK-MAIN passes: about 0.15;
PICK-PENNY: about 0.05.

## Amendment 1 (2026-11-11, after the rows download, before any model is fitted or any label statistic is seen)

Details the registration left open, fixed now:
- News inputs per row (from NX1's tone cache, single-symbol headlines with time < the decision time, since the
  prior session's 16:00): count, mean and max FinBERT positive and negative, LM net, any of each NX1 flag, and
  minutes since the latest headline (capped at 1,000; no headline = cap). NEWS = these + log_px + mins_open;
  CHART = the chart inputs already in the rows; COMBINED = both + SPY's move since the open and the market score.
- Market score (bad days): z-scored on 2019-2022 days, the mean of SPY's gap, SPY's move since the open at the
  decision time, SPY's 5-session return, and minus its 20-day volatility. Bad = the lowest fifth of 2019-2022
  values (that cut is frozen).
- Learner choice: each learner is fitted on the middle stop width's label (main 1.5x, penny 2x) for each version;
  the lowest 2023 log-loss wins. KRR and Extra Trees are fitted on a fixed 20,000-row random sample of training rows
  (memory); the others on all training rows. Then the winning learner is fitted per stop width, and the width per
  tier (terciles of swing30 on 2019-2022 rows) and the bar pair are chosen on 2023 as registered.
- Test: the frozen 2019-2023 refits score every decision row of 2024-01 to 2025-09 in a second download pass
  (all rows, not the 3% sample), keeping only rows above the lowest candidate bar plus one random universe stock
  at the same minute per kept row. The book (one buy per stock per day, sizing as registered) is built from those.
Clarification (same day, before any fit): the market score is computed per decision row (SPY's move since the open
changes through the day), z-parameters from 2019-2022 (one value per day for gap, 5-session return and volatility;
rows for the move since the open), and "bad" is a row score below the 20th percentile of 2019-2022 row scores. SPY's
gap uses adjusted open over adjusted prior close (the only SPY daily cache stores adjusted prices).

## Amendment 2 (2026-11-13, after the fit, before any test-period row is downloaded or scored)

How the one test is run, fixed before it runs. Nothing about the models, widths or bars changes (all frozen in
`research/results/PICK/PICK_fit.json` and `models/`).
- Test rows (`tracks/main/pick_test.py`, stage rows): 2023-12 is a warm-up month (the 20-session swing only), then
  every decision row of every universe stock, 2024-01-02 to 2025-09-30, gets the same chart inputs as pick_data
  (a vectorised copy, tested equal to `pick_data.inputs`), NX1 news inputs and the market score with the frozen
  parameters and cut. Each version scores every row with its frozen per-width models and its tier's width. A row
  is kept if any version's score clears that version's frozen normal-day bar (the bad-day bar is never lower).
- Random control: for each kept row and each version it clears, one other stock of the same track with a decision
  row at the same minute, drawn with `default_rng` seeded by (day, minute, symbol, version); it trades by the
  same rule with its own tier's width for that version. Trades are simulated only for kept and control rows.
- Picks (stage test): a row is a pick if its score clears the day's bar (bad-day bar on bad rows). One buy per
  stock per day: the first time it clears. A pick whose trade is undefined (no bar to fill) is dropped with its
  control.
- Size: 10% + 30% x (score - bar) / (top - bar), clipped to 10-40%, where top is the 99th percentile of the
  2019-2023 normal-day scores (refit models, sampled rows) at or above the normal bar. The book is $5,000 per day:
  picks are taken in time order, and at one decision time highest score first; a pick takes the room left if it
  is under its size, and is skipped if under 10% is left. Room is released only at the close (exit times are not
  modelled; this is the cautious reading of "until the book is fully used").
- Statistics: rule 1, 2,000 day-resampled bootstrap draws (seed 0) of the pick share minus the control share of
  trades ending +3% or better; rule 2, NW t (5 lags) of the daily book return (sum of size x net) over days with a
  pick, also reported over all test days; rule 3, NW t of the daily mean of pick net minus its control's net. At
  least 30 picks. Everything is reported per version; only COMBINED decides.
