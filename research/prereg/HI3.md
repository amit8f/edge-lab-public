# HI3: LightGBM, CatBoost and Ensemble news models, normal and high bar, on 2026 data (news track)

Registered 2026-11-27, before any HI3 number is computed. Source: Amit, 2026-11-27: "for those [LightGBM,
CatBoost, Ensemble] I want you to run the test on 2025-2026 current and I want for each to test them all on HI
meaning make the bar higher for the three of them and tell me how they did". One idea with six arms. Phase-2
count 192 -> 193 (phase 1 + phase 2 = 208).

## What was seen before this was written

The three models' 2019-Sep 2025 walk-forward history (NXM.md, NXK2.md) and their live ledger books since
2026-10-29 (LGBM, ENS) and 2026-11-09 (CAT): book P&L % on the dashboard 2026-11-27 (LGBM +17.3%, ENS +15.2%,
CAT -3.3%, capped 3+3 books). The live part of this test therefore only describes; the replay part (below) is new.

## Data window

The Phase 2 holdout (label dates from 2025-10-01) is opened here at the owner's request, for these six arms only.
Alpaca now serves Benzinga news only from about May 2026 (probe run 45996703595, 2026-11-27: one window per month,
none Sep 2025 to mid April 2026, full pages from 2026-05-15 on; the window seems to roll forward) and our backup
stops at Sep 2025. Oct 2025 to April 2026 is not used here unless the same Benzinga feed is found elsewhere; if it
is, it gets its own amendment before any of its numbers exist.
- **Replay (decides):** the first calendar month from 2026-04 on in which Alpaca returns articles on every trading
  day is the news start N. Novelty needs 30 calendar days of earlier news, so replay sessions run from the first session at
  least 31 days after N through 2026-10-28 (the day before the live books began).
- **Live (reported, not deciding):** the same six arms re-made from NX3's logged decided events, 2026-10-29 to the
  last settled session.

## Arms

The version 1 models exactly as the live books load them (`tracks/news/nxm_live.load`, sha256-checked), trained on
data before 2025-10-01 and never refit: LGBM, CAT, ENS, each
- **plain:** fired when |e| >= its frozen cut; 3 longs and 3 shorts in time order (NXM books' rule);
- **HI:** the same, but an event only takes a slot when its confidence |e| / cut >= 1.5 (NX3-HI's rule, NX3HI.md).

## Replay rules (written so a replayed day matches a live day)

Every replay session d: Benzinga articles from Alpaca from the previous close; headline rows, FinBERT at NX1's
pinned commit, the 3-day dedup and events exactly as `nx3_live.poll` makes them (`nx3_study.session_events`);
stock and market inputs at the previous close with `nx3_live.stock_context`; the intraday inputs with
`nx3_live.intraday` at decision time t + 1 minute, from SIP 1-minute bars (live uses IEX, which no longer has a
history we can trust; the IEX share is set to 1); the price at decision = the close of the last SIP bar starting at
or before t. Shorts: history ignores borrow, as NX3's history did. Control: as many of the day's decided events at
random with the picks' sides, default_rng(SEED + yyyymmdd + arm number), arm numbers 1-6. Trades settled with NX3's
training code (`nx3_study.month_trades`: entry at the first SIP minute bar >= t + 1 minute, exit at the next close,
minus SPY) and NX3's costs ($500 a trade, `nx1_study.net`).

## Primary test (per arm, replay sessions)

Daily pick book minus control, mean > 0 with Newey-West (5 lags) t >= 1.5, the near-miss bar used for NX3-HI and
the live variants. With about 50 sessions this is a weak test; it can show a clear failure, not prove an edge.
The forward rule of each live book is unchanged (NXM.md, NXK2.md, NX3HI.md).

## Reported, no rule

Per arm, replay and live: trades, win rate, mean net per trade after costs, book P&L in $ and %, days with no pick,
longs and shorts apart, HI minus plain on shared days, and the top 5 trades' share of the profit.

## Prediction (written first)

Plain arms: pick minus control positive but under t 1.5 (few days). HI arms: fewer trades, higher net per trade,
similar t. CatBoost below LightGBM and Ensemble, as in the live books.
