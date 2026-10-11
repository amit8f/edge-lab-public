# Rethink: news-signal and breakout-scanner

Written 2026-10-11 after reading both repos end to end and searching papers, Reddit, X, YouTube, Hacker News, GitHub and quant blogs. No code was changed. Nothing here is a result; every direction below still has to be pre-registered before it is tested.

> **Update 2026-10-11:** Amit answered the questions. The plan is now in [PLAN_V2.md](PLAN_V2.md), which replaces sections 5 and 8 below.

## The short version

Your process is better than the market's average retail researcher by a wide margin, and that's why the results are negative. Fifteen ideas failed for the same three reasons that kill almost everything people post about online:

1. **Costs beat short-horizon signals.** Gross edges of 0.1-0.5% per trade disappear at 0.1-0.4% round trips.
2. **News is priced before you can act.** Your own catalyst study found 91% of news lands outside market hours and the move is in the gap. Lopez-Lira & Tang's GPT-4 headline strategy fell from a Sharpe of 6.5 (late 2021) to 1.2 (early 2024) before costs, and it was unprofitable at a 0.2% round trip.
3. **Predicting "big move" is easy, predicting direction is not.** The scanner's 15.3% hit rate vs 1.6% base is real, but its picks also crash 20% at 12.4%. Option prices already know this: in I-15 the implied move was larger than the realized move on the model's picks.

So I would stop adding variations of "end-of-day features → 5-day direction" and spend the next two months on three directions that bring **new information** or a **slower horizon**, where a solo researcher with free data has a real chance. Details in section 5.

## 1. What to keep and what to drop

| Item | Verdict | Why |
|---|---|---|
| Pre-registration, idea ledger, random controls, DSR, holdout discipline | **Keep, unchanged** | This is the most valuable thing you've built, and it's the CV story. A Reddit user who tested 700 strategy versions in Oct 2026 reached the same conclusions you did, but with a weaker process. |
| Data pipeline (SIP bars incl. delisted, point-in-time EDGAR, Form 4, Reg SHO, Benzinga news, options cache) | **Keep** | It's the expensive part and it's correct. Every new direction below reuses it. |
| Nightly forward tests on GitHub Actions (I-01 shadow, I-12, I-13, I-14, I-15 forward) | **Keep running, don't touch** | They're free, pre-registered and the only clean data you'll get. The Dec 9 I-12 interim stays as planned. |
| Scanner as a *direction* model | **Drop** | I-01 to I-05 settled it. Lesson L-01: end-of-day features carry no direction information among the picks. |
| Scanner as a *volatility / risk* tool | **Keep, repurpose** | It's a good input for position sizing, for avoiding names, and as the universe for direction signals that use new data (direction 2). |
| Selling options on the picks instead of buying | **Don't pursue** | I checked this from your I-15 phase 2 numbers. Backing out the implied move from the two cost settings, the gross edge for an option *seller* is about +0.2% of the stock price at 1 day and +0.5% at 5 days for MAIN picks, against spread costs of 0.6-1.4%. PENNY picks look better gross (about +2.6% at 5 days) but only survive if the half-spread is under ~6%, which PENNY options never are. Approximate, derived from cell means, not a test. |
| news-signal 2-hour large-cap model | **Drop as a trading model, keep as a CV project** | Its own placebo test shows the Buy class fires just as often at random no-news timestamps, so the edge came from technical context, not news. FinBERT sentiment was null (r = -0.019). Large-cap 2-hour news is the most competitive slice of the market. |
| news-signal Section 9b wider-universe check | **Run it once, as registered** | It's already registered and costs no new idea. Report it either way, then archive the live loop. No need for a new server. |
| Stop/target grids, ORB, PEAD, weekly ranking, insider buying, short volume | **Stay dropped** | Each failed its own pre-registered test. The internet agrees: the October Reddit sweep found insider clusters at +0.21% (t 0.7) once bought at the first tradable price, ORB dead, options-flow-as-stock-signal dead on 597 stocks. |

## 2. What the internet actually says

I separated evidence from marketing. YouTube is almost all marketing ("INSANELY profitable AI bot", "85% win rate"); one Reddit user backtested a 400k-view YouTube strategy over 16 years and it lost. The useful material came from papers and from a few long Reddit post-mortems.

**News and LLMs**
- Lopez-Lira & Tang (arXiv 2304.07619, v6): GPT-4 scored headlines for 4,123 stocks, Oct 2021 to May 2024, after its training cutoff. Long-short Sharpe 2.97 before costs, but it decays every year (6.54 → 3.68 → 2.33 → 1.22), it's carried by small stocks and bad news (the short leg earns 26 bp/day vs 8 bp for longs), and it's unprofitable at 20 bp round trip. Long-only retail can't take the part that works.
- Gao, Jiang & Yan (arXiv 2512.23847): LLMs memorize history. A Llama model predicted next-day returns from the *date alone* (t = 3.53) in its training window, and the effect vanished after its cutoff. Any LLM backtest on pre-cutoff dates is contaminated. Use post-cutoff dates, or similarity measures that don't rely on the model knowing the future.
- r/algotrading "LLMs are not the right tool" (141 upvotes) and the Oct 2026 700-version sweep agree: an LLM reading post-cutoff 8-Ks "understood them fine, but by the first tradable price the move had already happened."
- Robeco, "The term structure of ML alpha" (SSRN 4474637): one-month ML signals were profitable mainly before 2004; after costs they're near zero since. Three- to twelve-month horizons still work because they trade less.

**Big moves and penny stocks**
- WallStreetBets sentiment follows price and then reverts (Semenova & Winkler 2021): +0.2% on the day, -0.05% the next, reversal over four days. Social chatter is a lagging signal.
- Your own PENNY study matches the literature: offerings, reverse splits, delisting notices and late filings predict *declines*. These are avoid filters, because shorting $1-5 stocks is often impossible or costs 20%+ a year to borrow.
- Stops don't protect small caps: the Reddit sweep saw worst trades of -59% to -91% through stops on gaps. Your I-04 grid (0 of 144 variants positive) said the same.

**Options**
- Retail loses 5-9% around earnings on options, 10-14% on high-volatility names; spreads average ~20% of the option price (Quantpedia summary of an "Anatomy of Retail Option Trading" study).
- Cao & Han: the volatility premium is real (high-volatility stocks' options are overpriced, ~1-2%/month at mid prices) but paying 50-100% of the spread takes it to about zero. Matches what I backed out of I-15.
- Augustin, Brenner & Subrahmanyam: before M&A announcements, out-of-the-money call volume, implied vol and spreads rise abnormally for targets over the prior 30 days. That's public data and it leaks information about direction.

**What practitioners say survived** (r/algotrading, Oct 2026, delisted names included, costs doubled as a check): SPY/QQQ dip-buying in uptrends (+0.25%/trade over 804 trades), a month-end SPY trade, and a residual-momentum + SPY blend (18.1% vs 13.8% a year, t = 1.8, "suggestive, not significant"). Small, slow, boring. Also: "84% of my dip-buying edge happened between the close and the next open," and SPY lost ~3.9% a year from 10:30am to the close in 2023-2026.

The best framing I found is from a r/algotrading post: "an edge is a reason someone pays you." Start from who is forced or slow (index funds rebalancing, retail chasing, insiders hiding bad news in long filings), then test the cheapest version first.

## 3. Getting ahead of news, legally

### Where the line is (US; not legal advice)

Trading on information nobody else has is legal **unless** you got it through a breach of duty or deception. That covers:

- **Illegal:** anything from someone who owes the company confidentiality (an employee, a friend at the company, a consultant, a leaked document), or information you got by lying, hacking or bypassing access controls. Tippees are liable too (*Dirks*, *Salman*). Using your own employer's secrets to trade a *competitor* also counts: a jury agreed in *SEC v. Panuwat* (2024, "shadow trading"); the Ninth Circuit appeal was argued in 2025 and I couldn't confirm the outcome.
- **Legal:** combining public, individually non-material pieces into a conclusion (the "mosaic theory"). Public web data scraped while logged out is generally fine (*Meta v. Bright Data*, 2024); scraping behind a login or defeating CAPTCHAs is where lawyers get nervous. The SEC's main alt-data case, App Annie (2021, $10M+), punished the *vendor* for lying about how it got the data.
- **Also off-limits:** posting hype to move a stock you hold (pump-and-dump), even as a joke.

"Read between the lines" is legal when the lines are public. So the plan stays on public filings, public market data and public registries.

### Public leading signals, and where you already stand

| Signal | Evidence | Free data | Your status |
|---|---|---|---|
| Form 4 insider buys | Weak once you buy at the first tradable price | EDGAR | Tested twice, null |
| Earnings-date shifts (Johnson & So) | Decent in papers | Benzinga notices | Too thin in your data |
| NT 10-K / 10-Q late filings | Strong for PENNY declines | EDGAR | Forward test I-14 running |
| Offerings, reverse splits, delisting notices | Strong declines | EDGAR, Benzinga | Forward test I-12 running |
| Options activity before events (OTM calls, IV, skew) | Good for M&A targets; mixed as a general stock signal | Alpaca options since Feb 2024 | Cached for picks; not tested for direction → **direction 2** |
| 10-K / 10-Q language changes ("Lazy Prices") | Strong in the paper, weaker in replications | EDGAR full text | Not tested → **direction 1** |
| DoD daily contract announcements (published 5pm on defense.gov, often before small contractors' press releases) | Plausible, I found no study | Free | Hypothesis only |
| clinicaltrials.gov record changes (status, enrollment, completion date) | Plausible for small biotech, no strong study found; one thesis found no pre-FDA run-up | Free API | Hypothesis only |
| Court dockets (CourtListener), USAspending awards | Anecdotal | Free APIs | Hypothesis only |
| Google Trends / Wikipedia attention | Predicts a short rise, then reversal (Da, Engelberg & Gao 2011); a r/quant user found attention reversal dead recently | Free | Low priority |
| Reddit / X / StockTwits chatter | Follows price, then reverts | Composio (works in this project) | Low priority; useful as an avoid filter for pumped names |

## 4. Take profit and stop loss

You asked me to set these. Your own evidence and the practitioners' agree, so I'd fix them like this and pre-register them with whichever direction goes first:

- **No take-profit.** Every fixed target you tested (I-03, I-04, I-05, the holdout's +20% exit) did no better than holding.
- **Time exit is the main exit.** Swing trades: sell at the close of day 5. News/event trades: sell at the close of day 3 (or day 20 for the slower filing-driven directions).
- **One wide disaster stop, checked at the close, filled at the next open:** 3 x the 20-day ATR, capped at -15% for MAIN and -25% for PENNY. It won't stop gaps; its job is to cut the rare stock that keeps falling for days.
- **Size by risk, not by conviction:** each position risks 0.5% of the account to its disaster stop, at most 10 open positions, at most 2 in one industry.
- **Account-level brake:** stop opening new paper trades if the paper account falls more than your drawdown limit (question 3), and review before restarting.

## 5. The three directions I'd test next

Each is one idea in the ledger with one primary test. Running all three takes the count from 15 to 18, so the deflated-Sharpe bar rises for each; that's why I'd stop at three.

### Direction 1: an LLM reads filings for slow information (monthly, long-only)

**Idea.** When a company quietly rewrites its 10-K or 10-Q (MD&A, risk factors, litigation, executive language), the market under-reacts for months, because few people read 150-page filings. Cohen, Malloy & Nguyen ("Lazy Prices", 1995-2014) found a long non-changers / short changers portfolio earned up to 188 bp a month. Much of that was in the short leg and small firms, and a 2020-2026 long-only replication on liquid stocks only matched SPY's Sharpe (0.56 vs 0.53, in-sample). So the honest prior is "small but plausible."

**Why it fits you.** Holding a month means costs of ~0.1-0.2% a month instead of per trade. The data is free and you already parse EDGAR (the Financial Statement and Notes text you used in D-21). It uses LLMs where they help: reading long text, not guessing prices. And it's the strongest CV piece of the three.

**Avoiding the look-ahead trap.** The primary signal is a text-similarity score from a small local embedding model (no world knowledge to leak). LLM labels (what changed: new risks, going-concern language, customer losses) are a secondary feature validated only on filings after the model's training cutoff.

**Test.** Historical screen on 2019-2025 training dates in MAIN, long the top decile of "unchanged" filings vs the universe, then a forward ledger. Fast to kill: a few weeks of work.

### Direction 2: options tell the direction of the scanner's big movers (swing, 5 days)

**Idea.** The scanner knows *which* stocks will move but not which way. Informed traders often use options first (Augustin et al. on M&A; Cremers & Weinbaum and Xing, Zhang & Zhao on implied-volatility spreads and skew). Among the daily top-20 picks, buy only those whose options lean bullish (call IV above put IV, low put skew, call-heavy volume) and trade the **stock**, not the option.

**Why it fits you.** It attacks the exact weakness in L-01 with information the scanner never saw. Stock costs (0.1-0.4%) are tiny next to these names' ±20% moves, so even a small shift in the up/down ratio pays. You already cached Alpaca option data for the picks in I-15.

**Risks.** Only ~18 months of option history before the spent holdout, so the historical screen will be thin and the forward test does the real work. The Oct 2026 Reddit sweep found options flow useless as a general stock signal; the bet here is narrower (direction among stocks already expected to move a lot).

### Direction 3: a boring core that probably works (monthly, long-only)

**Idea.** Your I-11 controls found 12-1 momentum alone made +1.48% a month net (DSR 0.52), and L-23 already says it needs forward data. Residual momentum (momentum after removing market and factor moves) has a long record (Blitz, Huij & Martens) and practitioners report it survives delisted names and costs. Register a long-only residual-momentum portfolio of 20-30 MAIN stocks, monthly, against SPY, as a forward paper ledger.

**Why.** It's the only candidate with a decent chance of beating SPY after costs, and it's cheap to run on your existing pipeline. Expect a few percent a year over SPY at best, with deep drawdowns in momentum crashes. It's the "something real to paper trade" while directions 1 and 2 are judged.

### What I'd skip
Short-horizon LLM headline trading (decayed and needs shorting), social-chatter signals (lagging), selling option premium (spreads), PENNY long models (costs and dilution), and anything intraday (paid real-time data and fills).

## 6. Using many models without fooling yourself

- **Where more models help:** LLMs as readers that turn filings and news into features (direction 1, catalyst labels), checked against a hand-labelled sample like your catalyst classifier check. An XGBoost + logistic ensemble is a cheap add-on; it rarely changes the verdict.
- **Where they don't:** asking LLMs to pick stocks or read charts. Multi-agent "AI hedge fund" repos (TradingAgents, ai-hedge-fund and similar) publish demos, not cost-adjusted out-of-sample records.
- **The council is for decisions, not trades:** use it for go/no-go calls like "which direction first" or "is this result real."
- **Free LLM capacity:** Gemini's free tier (~1,500 requests/day on Flash models; free-tier inputs may be used for training, which is fine for public filings) and Groq (~1,000 requests/day on Llama-class models) are enough for nightly filing labels. A local embedding model on the laptop handles bulk similarity. Check current limits before building, since they change often.

## 7. Skills, MCPs and connectors

| Tool | What it is | Recommendation |
|---|---|---|
| **superpowers** (obra/superpowers) | Process skills: brainstorm, TDD, systematic debugging, verify before done | Keep; your CLAUDE.md already uses it |
| **graphify** | Local code knowledge graph for cheaper lookups | Keep |
| **council** | Several AI advisors + chairman | Keep, for big decisions only |
| **ponytail** (DietrichGebert/ponytail) | Makes the agent write less code | Optional. JetBrains measured -10% cost and -15% code (half the advertised claims), and it only works with its SessionStart hook |
| **caveman** (JuliusBrussee/caveman) | Terse output to save tokens | Optional for coding sessions (~25% real savings); turn it off for write-ups |
| **Everything Claude Code (ECC)** | Huge bundle of agents, skills, hooks | Skip. It overlaps with superpowers and loads a lot of context |
| **UI UX Pro Max** | Design system skill | Only if you build a public results page for the CV (a good idea, see below) |
| **Alpaca MCP** (alpacahq/alpaca-mcp-server) | Market data, options chains, paper orders from Claude Code | Add on the laptop when paper trading starts |
| **SEC EDGAR MCPs** | Filing search | Skip; your own parsers are better |
| **Composio (Reddit, plus X and YouTube search)** | Already connected here | Enough for research; X and YouTube search work through it, so no X API key is needed |
| **vidIQ** | YouTube analytics | Out of credits; not needed |

**For the CV:** a free GitHub Pages site that shows the idea ledger, each forward ledger's hash chain and live paper results would turn the repos into something a recruiter can read in two minutes.

## 8. What happens next

Weeks 1-3: register and screen direction 1, register direction 3 as forward-only, run the news-signal 9b check. Weeks 3-6: register and screen direction 2 on the cached option data. Dec 9: I-12 interim as planned.

A straight answer on timing: a historical screen can kill an idea in weeks, but a pass still needs months of forward data before any money is risked. Two months gets you "these are dead, this one is alive and being paper traded," not "this is proven."
