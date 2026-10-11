# Plan v2: three tracks, fresh start

Written 2026-10-11 from Amit's answers. It replaces section 5 and section 8 of `RETHINK.md`; the rest of that file (evidence, legal line, tools) still stands. Nothing is built until Amit approves.

## What changed with your answers

- **Shorting is allowed.** This opens the side of the market where the evidence is strongest. In the GPT headline paper the short leg earned 26 bp a day against 8 bp for the long leg, and every robust pattern in your PENNY and catalyst studies points down (offerings, reverse splits, delisting notices, late filings).
- **Fresh start.** A new phase with its own idea ledger and its own untouched holdout. Phase 1's 15 ideas stay on record as phase 1. The data code is correct, so it gets copied in, not rewritten.
- **CV wins for now.** That favours a clean, readable monorepo with a public results page, and an LLM component that's done properly (post-cutoff validation, agreement between two models).
- **Little time.** Everything runs unattended on GitHub Actions. Your part is one weekly report (about 15 minutes) and approving each pre-registration before it runs.

## Layout: one repo, three tracks that don't share models

```
edge-lab/                       (name is a placeholder)
  core/      data layer copied from breakout-scanner: SIP bars incl. delisted, EDGAR, Form 4,
             Reg SHO, Benzinga news, options cache; point-in-time universe; cost models
  tracks/
    main/    $5+, $300M+ cap          own ideas, models, ledger, paper sleeve
    penny/   $1-5, liquid             own ideas, models, ledger, paper sleeve
    news/    event trades, any price  own ideas, models, ledger, paper sleeve
  paper/     Alpaca paper orders + an internal ledger that also charges borrow fees and spreads
  research/  pre-registrations, idea ledger (per track and global count), reports
  site/      GitHub Pages results page (free)
```

Tracks share data, never models or thresholds. They may only use each other's *outputs* as inputs when that is written into a pre-registration (for example, news events as an avoid filter for penny longs).

**Where it runs.** GitHub Actions: a nightly job after 10pm ET (needed for same-day Form 4s), plus a news job every 30 minutes during market hours. News trades are held for days, so 30 minutes of lag costs little, and this removes the need for the Oracle VM. Finnhub is dropped (too slow); news comes from Alpaca's Benzinga feed. The repo should be public: public repos get unlimited Actions minutes, private ones get 2,000 a month.

**Phase 1's forward tests (I-01 shadow, I-12, I-13, I-14, I-15 forward) keep running where they are** until their scheduled analyses. Moving them risks breaking clean data. Their results feed the new tracks when they land.

## The first ideas per track

Each one is pre-registered before any result: one primary test, random control, costs, DSR with the new phase's count.

| ID | Track | Idea | Hold | Side |
|---|---|---|---|---|
| M1 | main | LLM + embedding read of 10-K / 10-Q changes ("Lazy Prices"): long unchanged filers, short changers | 1 month | long-short |
| M2 | main | Options data picks direction among the big-mover list (IV spread, skew, call/put volume); trade the stock | 5 days | long-short |
| M3 | main | Residual momentum, 20-30 names, forward-only core | 1 month | long |
| P1 | penny | Short after offerings, reverse splits, delisting notices and late filings, only when easy to borrow | 20 days | short |
| P2 | penny | Long after guidance raises (the I-13 rule) with dilution and pump filters | 5 days | long |
| N1 | news | Two LLMs label each Benzinga headline (Gemini Flash free tier + a Llama model on Groq); trade only where both agree; primary test on the negative-news short leg in liquid stocks | 1-3 days | short first, long secondary |

P1 and P2 start from the phase-1 descriptive results. That is in-sample evidence, so their tests are forward-only, like I-12 and I-13. M1, M2 and N1 get a historical screen first: if a screen fails, the idea dies in weeks, not months.

## Models, and how new ones get in

- **Tabular:** XGBoost stays the baseline; CatBoost and a logistic model are the challengers. A November 2025 study (Rahimikia, Ni & Wang, arXiv 2511.18578) found CatBoost beat zero-shot time-series foundation models on stock returns: Chronos R² -1.37%, TimesFM -2.80%. So I'd skip Chronos and TimesFM.
- **Text:** a small local embedding model for filing similarity (no future knowledge to leak), and two LLMs as labellers. The two must agree, and their labels are checked on a hand-labelled sample and only on dates after each model's training cutoff.
- **Sizing:** calibrated probabilities (isotonic, as in news-signal) set position size within the risk limit.
- **Adding a model:** a new model enters only as a pre-registered challenger to the track's baseline, on training dates, with one comparison. At most one survives per track. That keeps "use many models" from becoming many hidden tests.

## Risk rules (I picked them; you can change them)

| | Main | Penny | News |
|---|---|---|---|
| Risk per trade (to the stop) | 0.5% of account | 0.25% | 0.5% |
| Long stop | 3 x ATR(20), max -15% | 3 x ATR(20), max -25% | 2.5 x ATR(20), max -12% |
| Short stop (squeezes) | 3 x ATR(20), max +20% | 3 x ATR(20), max +35% | 2.5 x ATR(20), max +15% |
| Take profit | none | none | none |
| Time exit | day 5 (M2), month end (M1, M3) | day 5 longs, day 20 shorts | day 3 |
| Max open positions | 10 | 5 | 5 |

Stops are checked on the close and filled at the next open (gaps go through intraday stops anyway). If a track's paper sleeve falls 10%, it pauses for review. If the whole paper account falls 15%, everything pauses.

**Account notes for later, when real money starts below $5k:** shorting needs a margin account with at least $2,000 in it, and you can't short fractional shares. Swing holds aren't day trades, so the pattern-day-trader rule only bites if a position is opened and closed the same day.

## What I'd pay for, if anything

Nothing yet. Free Alpaca data is enough for nightly and 30-minute jobs. The first paid item would be Alpaca's real-time SIP + options plan, around $99 a month, and only if a track passes and needs intraday entries. That's not likely before 2027.

## Order of work

1. **Week 1:** create the repo, copy in the data layer with its tests, build the paper ledger and the GitHub Actions jobs, start the news job logging (no trades yet).
2. **Week 2:** register M1, N1, P1, P2 and M3; run the M1 and N1 historical screens.
3. **Weeks 3-4:** register and screen M2; any idea that passed its screen starts paper trading; the first weekly report goes out.
4. **Dec 9:** phase 1's I-12 interim feeds P1.

Paper trading runs until you're confident. I'd also hold you to a written bar: at least 100 closed paper trades per track and a pre-registered test passing after costs, before any real money.
