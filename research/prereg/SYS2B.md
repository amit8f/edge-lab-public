# SYS2B: the scout picker (cross-track)

Registered 2026-10-23, before any result exists. This is Amit's design from the planning thread (Q1-Q14,
2026-10-22/23), with a second AI's review folded in. The lane list is
https://claude.ai/artifact/X1ypAMa5eHYjm7KH9gomv1. Phase-2 count at registration: 36 (cross-track 7). Phase 1 +
phase 2 = 51.

SYS2B runs beside SYS2 (`research/prereg/SYS2.md`) and changes nothing in it. A changed lane, model or prompt
is a new version and counts as a new idea.

## Mechanism (PROTOCOL rule 10)

- Other side: traders who are slow to price public, verifiable company events: filings, insider and activist
  disclosures, dilution, accounting and legal problems, index changes, unusual trading. Most of all in small
  stocks that few analysts follow.
- Why a $1B fund can't take it: positions of $50-500 in thin stocks, a few a day.
- Published: post-earnings drift (Bernard & Thomas 1989), insider purchases (Lakonishok & Lee 2001; Cohen,
  Malloy & Pomorski 2012), activist 13Ds (Brav, Jiang, Partnoy & Thomas 2008), equity issuance (Loughran &
  Ritter 1995), borrow fees (Engelberg, Reed & Ringgenberg 2018), restatements (Palmrose, Richardson & Scholz
  2004), LLM reading of news (Lopez-Lira & Tang 2023). Expected effect: small after costs. This tests whether
  scouts with research beat a coin on live data only, so no model's training cutoff can leak.

## Question

After costs, does each scout-picker book beat its coin twin: the same stocks at the same moments, the same
size and stop, on a random side?

## Universe and shared checks (code, before any AI call)

- SYS2's universe: active common stocks, last raw close at least $1, median 20-session raw dollar volume at
  least $1M, no gap over 15 calendar days, no split that day. $5 and up is MAIN; $1-5 is PENNY.
- Each candidate must also pass these checks at its pass:
  - a quote no older than 15 minutes;
  - not halted (an IEX trade in the last 30 minutes during the session);
  - quoted spread at most 2% (MAIN) or 5% (PENNY), where a quote exists;
  - no reverse split, ticker change or delisting notice in the last 5 trading days or announced.
- One ticker map joins every source: Alpaca symbols, SEC CIKs (DERA and company tickers), and names for
  RSS, FDA and social text. Text that names no mapped company is dropped.
- Alerts are grouped into one event per stock and catalyst. Two alerts are one event when they share a
  stock, fall within 2 trading days of each other, and either share a source item or both come from the
  fundamental group below. An event lists every lane that saw it; extra lanes count as confirmations, never
  as extra trades.
- Each event keeps three times: the source's publication time, our detection time, and the first price after
  detection.
- Context passed to every agent (never a lane of its own):
  - market mood: SPY, IWM and the sector ETF's return today and over 5 days, VIXY's 5-day change as the
    volatility proxy, and the share of the universe above its 20-day average;
  - a catalyst calendar: earnings dates (Finnhub), FDA dates where public, and CPI and FOMC days (no new
    entries in the 30 minutes either side of a release, per the registered blackout filter).

## The lanes

Code watches every lane's source every 5 minutes during the session, and also before the open and after the
close where the source publishes then.

Trigger lanes (can start a trade):

| # | Lane | Group | Code watches |
|---|---|---|---|
| 1 | Earnings surprise | fundamental | Headlines and Finnhub actuals vs estimates, EPS and revenue (same-sign surprises flagged) |
| 2 | Guidance change | fundamental | Raised, cut or withdrawn guidance in headlines and 8-K item 2.02/7.01 |
| 3 | Fresh filing | fundamental | EDGAR live feed: 8-K items (1.01, 1.02, 2.01, 2.05, 2.06, 3.01, 4.01, 4.02, 5.02, 8.01), 10-Q, 10-K, DEF 14A |
| 4 | Insider buying | ownership | Form 4 code P only (no A, F, M or 10b5-1-marked trades); flags for 2+ insiders, CEO/CFO, $100k+, after a 20% drawdown |
| 5 | Activist stake | ownership | New SC 13D and 13D/A with a purpose change (13G is ignored) |
| 6 | Analyst moves | fundamental | Upgrades, downgrades, initiations and target changes in headlines; estimate revisions if a free source covers them |
| 7 | Share supply | ownership | S-1, S-3, 424B, resale registrations, ATM programs, convertibles, warrants, lockup ends; repeated raises flagged |
| 8 | Biotech catalyst | fundamental | FDA approvals and rejections (openFDA), trial results (headlines, ClinicalTrials.gov), with cash runway from the last 10-Q |
| 9 | Contract win | fundamental | Contract and award headlines, with size against last year's revenue |
| 10 | Takeover | fundamental | Deal and bid headlines and 8-K 1.01/2.01, confirmed and rumoured kept apart, target and acquirer apart |
| 11 | Volume with no news | trading | Volume pace 3x+ normal for the time of day, with no mapped headline or filing in 3 sessions; the AI must find the reason or answer no trade |
| 13 | Chart breakout | trading | New 52-week highs and lows and 20-day range breaks on 2x volume (all-adjusted closes) |
| 14 | Oversold bounce (experimental) | trading | Drops of 3+ daily standard deviations with no adverse filing in 3 sessions |
| 15 | Sympathy move | trading | Peers, customers and suppliers of the day's biggest movers, from 10-K customer lines (G1) and the industry code |
| 17a | Squeeze long | ownership | IBKR borrow fee in the top decile plus price up 5%+ on 2x volume; FINRA short interest as context |
| 17b | Hard-to-borrow short | ownership | IBKR borrow fee in the top decile plus an adverse event or a down move |
| 21 | Adverse event | fundamental | Material weakness, restatement, non-reliance, auditor exit, going concern (EDGAR full-text search), SEC/DOJ/FTC/state actions, recalls, covenant breaches |
| 22 | Buyback and dividend | fundamental | New or accelerated buybacks, dividend starts, raises and cuts |
| 23 | Index change | trading | S&P, Nasdaq and Russell add/delete announcements |

Confirm-only lanes (shown to the agents as context; they never start a trade):

| # | Lane | Code watches |
|---|---|---|
| 12 | Gap and hold | Opening gaps of 4%+ and whether the first hour holds |
| 16 | Sector leader | Strongest and weakest stock in the strongest and weakest sector ETF |
| 18 | Social buzz | Jumps in StockTwits and Reddit mentions from public pages, with repeat-account filtering |
| 19 | Small-outlet scoop | The RSS collector's small and foreign outlets; needs a second public source or a filing |
| 20 | Politician trades | House periodic transaction reports, with transaction and disclosure dates kept apart |

Lanes whose data isn't working on 2026-10-27 join later. Each lane's start date is logged in the ledger, and
the lane is judged only from that date. Not built in this version (no free data): options flow, 13F holdings,
earnings-call transcripts.

## Agents

- Scouts: a lane's AI wakes when its code finds an event it hasn't seen, and also gets one wider turn about
  once an hour. New events always go first; when the day's budget runs low, the hourly turns are dropped first.
  - Each scout returns fixed fields only (`adaptive/guards.py`): stock, side (long, short or no trade),
    horizon (1, 5 or 10 trading days), catalyst, the evidence for and against, what would prove it wrong, the
    source times, and confidence (0-100). At most 3 stocks per wake.
  - Models, in fallback order: Groq `openai/gpt-oss-20b`, Groq `qwen/qwen3.8-27b`, Google
    `gemini-3-flash-preview`.
- Checker (Google `gemini-3.5-flash-lite`): sees each finalist event with its lanes, the numbers and the
  evidence. It may run one Google search per finalist through `gemini-2.5-flash` while that model's free
  quota lasts. It answers agree, disagree or unclear, and names the facts it checked. A web page can only
  lower a case. Rumours from lane 19 need a second public source.
- Picker (Groq `openai/gpt-oss-120b`): sees the checked finalists, the context and both books. It answers
  long, short or none per finalist and close or none per held stock, with confidence, a stop and a horizon.
- Risk (code): SYS2's rules. Confidence at least 60 (a gate only, until lanes have track records); a stop of
  at least 1% and at most 15% (MAIN) or 25% (PENNY); shorts only where Alpaca marks the stock shortable and
  easy to borrow. Size: 1% of the book at risk to the stop, at most 10% per stock (5% per short), 1x gross,
  and the liquidity caps.
- Budget: SYS2 keeps first call on the shared free models. SYS2B uses at most 600 calls a day per Groq model
  and 300 per Gemini model, and each lane at most 40 scout calls a day. When every model is out, scouting
  stops for the day; stops and horizon exits still run.
- Every call logs the model, the prompt's SHA-256, the answer's SHA-256 and the parsed fields. Raw evidence
  (headline and filing text, the web sources) goes only to the private `amit8f/edge-lab` repository, branch
  `evidence`. The public repository never holds headline text.

## Books and fills (ledger only)

- SCOUT-MAIN $5,000 and SCOUT-PENNY $500, each with a coin twin, with SYS2's fills and costs
  (`adaptive/costs.py`).
- Exits: the stop (checked every pass, a gap fills at the worse price), the picker's close, the horizon, or 10
  trading days, whichever comes first. Positions are kept overnight.
- Safety, as in SYS2: a 10% drawdown from peak stops entries until Amit restarts the book. More than 20 orders
  in a day, or an answer the code can't read, stops entries for the day.

## Primary test (read at 100 closed trades per book, and at 2027-12-31)

For each book: daily return of the scout book minus its coin twin, after costs, Newey-West t with 5 lags.
Pass: t >= 2 and a positive mean.

Reported, not tested (no lane is dropped during the test). For each lane:
- events found, finalists, checker verdicts and trades;
- every candidate, including rejected ones, with its 1-, 5- and 10-day market-adjusted return after costs;
- the win rate and the mean of the trades it started or confirmed.

Also reported: the scout books against SYS2's books over the same days, and against SPY (MAIN) and IWM
(PENNY).

## What a result does

As in SYS2: a book moves to a paper account under SYS1's promotion rule against its twin (at least 20
trading days, 30 closed trades and confidence of at least `adaptive/promotion.required(n)`). The EL-PICK
accounts are shared with SYS2; whichever book passes first takes them, and the other waits. Fail: dropped or
re-registered as a new version.

## Build clarifications (2026-10-23, before any result)

Written while building `adaptive/scouts.py`, `adaptive/lanes.py` and `core/edgar_live.py`, before the first
pass. They fill in details the text above leaves open and change nothing it fixes.

1. Start. The code does nothing before 2026-10-27. Lanes running that day: 1-14, 17a, 17b, 21-23 and the
   confirm-only lane 12. Not yet built, each to be logged with its own start date when it runs: 15, 16, 18,
   19, 20, lane 8's openFDA and ClinicalTrials.gov part, and lane 6's estimate revisions. Lane 1's
   Finnhub actual-vs-estimate part and the earnings calendar run only once a `FINNHUB_API_KEY` secret exists.
2. Sources. Headlines: Alpaca (Benzinga) news naming 1-3 stocks, sorted into lanes by the word lists in
   `adaptive/lanes.py`. Filings: EDGAR's current-filings feed every pass, for the company the filing is about
   (filer, issuer or subject), accepted in the last 18 hours. An 8-K with item 1.01, 2.02, 7.01 or 8.01 has
   its press release's first 600 characters run through the headline lists (guidance only with 2.02 or 7.01).
   A 13D/A counts for lane 5 only when its text names Item 4 or the purpose of the transaction. Lane 21 also
   runs EDGAR full-text searches for "going concern" and "material weakness" once an hour.
3. Number lanes, all on the day's live price and volume against the universe row:
   - 11: volume pace 3x or more, no headline from 3 sessions back (Alpaca news), and no fundamental,
     insider, activist or supply lane on the stock in 3 sessions;
   - 13: volume pace 2x or more and a price above the 52-week high close or below the low, else beyond the
     20-day high or low close;
   - 14: today's move at or below -3 daily standard deviations, with no lane 21 event in 3 sessions;
   - 17a / 17b: a borrow fee in the universe's top 10% (IBKR's file, read once an hour); 17a needs +5% or
     more on 2x pace, 17b a lane 21 event in 3 sessions or -5% or worse;
   - 12 (confirm-only): an opening gap of 4% or more; after the first hour, "held" means at least half the gap
     is still there.
4. Shared checks: the halt test starts 30 minutes after the open; the spread is the delayed SIP quote's. An
   8-K item 3.01 counts as a delisting notice and fails the stock.
5. Scouts. One call shows at most 5 events, ranked by the number of lanes, then today's absolute move. An
   event counts as new for 60 minutes after detection. An answer with extra keys is read without them; one
   with a missing or invalid field is unreadable.
6. Budget pacing. Each model gets a daily target (SYS2B's share after SYS2): gpt-oss-20b 40, qwen3.8-27b 60,
   gpt-oss-120b 60, gemini-3-flash-preview 250, gemini-3.5-flash-lite 120, gemini-2.5-flash 60 (all under
   the caps above). New-event calls may run up to 25% of the target ahead of an even pace from 9:00 to
   15:30 ET; hourly turns only on pace. A model that answers "quota used up" is skipped for the day.
7. Checker. It answers twice per finalist, without and with the web; the final verdict is the lower of the
   two. "Disagree" ends the finalist, and so does no checker answer at all. "Unclear" goes to the picker,
   which sees the verdict. The web search is one Gemini 2.5 Flash call with Google-search grounding per
   finalist; Groq's own search models aren't available on our key (checked 2026-10-23).
8. Picker. It runs on any pass with checked finalists, and at most once an hour for the held stocks alone.
9. Exits. A position closes at the first pass after it has been held `horizon` trading days (10 at most);
   its coin twin uses the same horizon and stop.
10. FOMC statement days (2:00pm ET, from federalreserve.gov on 2026-10-23): 2026-10-28, 2026-12-09, then the
    2027 meetings. No entries from 1:30 to 2:30pm ET on those days. CPI and jobs come out at 8:30am,
    before the session.
11. Evidence. One JSON-lines file per pass in `amit8f/edge-lab`, branch `evidence`, under `sys2b/<date>/`.
12. More free providers (2026-10-23, before the start and before any result). Keys checked from Actions on
    2026-10-23. Each list is in fallback order; a later model is used only when every earlier one is out of
    budget or quota.
    - Scouts: Groq `openai/gpt-oss-20b`, Groq `qwen/qwen3.8-27b`, Google `gemini-3-flash-preview`, NVIDIA
      `nvidia/nemotron-3.5-lightning-30b-a3b`, Cloudflare `@cf/zai-org/glm-5.3-flash`.
    - Checker: Google `gemini-3.5-flash-lite`, Cloudflare `@cf/deepseek-ai/deepseek-v4.1-flash-1017`, NVIDIA
      `deepseek-ai/deepseek-v4.1-flash`.
    - Picker: Groq `openai/gpt-oss-120b`, NVIDIA `nvidia/nemotron-3-ultra-550b-a55b`, Cloudflare
      `@cf/moonshotai/kimi-k2.6`.
    - Web search: Gemini 2.5 Flash with Google grounding, then Tavily news search (5 results from the last 7
      days; its snippets go to the checker as the web text).
    - Daily caps / pacing targets: NVIDIA Lightning 600/400, Ultra 300/150, DeepSeek 300/150; Cloudflare
      GLM 100/80, DeepSeek 100/80, Kimi 60/40 (Cloudflare's free tier is 10,000 "neurons" a day); Tavily
      40/30 (1,000 free searches a month). NVIDIA and Cloudflare calls time out after 60 seconds with one
      retry. Their answers are cut to the JSON object (reasoning and code fences removed) and then read by
      the same strict parsers.
    - Every call logs which model answered; results are reported per model as well as overall.
13. Fixes from the live dry run (2026-10-23, before the start and before any result). (a) Events from
    confirm-only lanes (no owner, so no shared check) crashed the pass log; they now log with an empty check.
    (b) The "no quote in 15 minutes" and "halted?" checks used one feed's trade time, often a stale IEX print
    on thin stocks while the delayed SIP feed showed trading (64-70 of about 260 events per pass were
    rejected this way). They now use the freshest trade on either feed, counting delayed SIP trades as 15
    minutes later than their stamp; a stock with no trade on either feed for 30 minutes is "halted?", for 15
    minutes "no quote". Steady passes took 22-34 seconds; the first pass of a day, which builds the universe
    and reads the overnight backlog, took 18 minutes.
