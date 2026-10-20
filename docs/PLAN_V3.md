# Plan v3: a self-improving paper trading system (APPROVED by Amit 2026-10-20, with the 12 additions in section 12)

Built from the 49 answers in `DECISIONS_2026-10-20.md` (Q1-Q49). Nothing here is built until Amit approves.
Paper trading only. Real money still needs 100+ closed paper trades that pass after costs, beating the
random control, and Amit's signature in `research/PROTOCOL.md`.

## 1. What changes from Plan v2

- Old backtests no longer decide what gets traded. New ideas get a quick check for bugs and look-ahead, then
  go straight into a shadow book. The forward record is the test (Q19).
- The system learns from its own paper trading every week, and may build new rules (Q1-Q3).
- AI agents watch the market all day and make decisions, checked by code (Q9, Q23, Q27).
- Kept as before: P1/P2 run untouched as the frozen baseline (Q11), M8's forward test continues, the holdout
  stays untouched, and every trade log stays append-only and hash-chained.

## 2. The money (paper)

Updated 2026-10-20 after approval: main is a $5,000 book and penny a $1,000 book, and each strategy
holding paper money has its own Alpaca paper account (`research/prereg/SYS1.md`, clarification 1). The
table below is the original split.

| Book | Share | Alpaca paper account |
|---|---|---|
| P1/P2 baseline (unchanged rules) | 30% | the existing one |
| Main system ($5+ stocks) | 50% | new account A (Amit creates it) |
| Penny system ($1-5 stocks) | 20% | new account B (Amit creates it) |
| Shadow rules | 0% (logged, no orders) | none |

Main and penny each have their own learner, models and money. They share the data layer and the agents (Q29).

## 3. Rules every trade follows

- At most 10% of the book in one stock. No sector or position-count limits (Q30). No leverage: 1x, and shorts
  count toward it (Q49).
- Regular hours only, 9:30-16:00 ET (Q42). Limit orders, cancelled after 15 minutes unfilled (Q41).
- A stop-loss on every trade: the system picks it, at most 15% (main) or 25% (penny) (Q43).
- Holding period is the system's choice, from minutes to weeks (Q5). There's no day-trade limit, since the
  PDT rule ended 2026-06-04 (Q35). The first run must confirm the paper accounts no longer apply it.
- Earnings: no special rule (Q44).
- Shorts: only where Alpaca allows. Shorts Alpaca refuses go to a shadow book charged at IBKR's real borrow
  fee (we already log these). If those make money, we move to IBKR (Q20, Q22).
- Costs taken off every paper trade: the spreads measured in V4 (0.41% main, 0.65% penny, refreshed monthly
  from quotes), plus an IBKR-style commission ($0.0035/share, $0.35 minimum, 1% maximum), plus SEC and FINRA
  sale fees, plus borrow fees on shorts (Q40).

## 4. Safety

- The book pauses at a 10% paper loss from its peak. Amit restarts it by hand. This applies during world shocks
  too (Q4, Q13).
- A world shock is Amit's call. The risk agent also reads the news and raises the alarm if it sees one (Q14).
- Kill switch: more than 20 orders in a day, or an agent answer the code can't read, stops the book for the
  day and messages Amit (Q45).
- If the reviewer agent and the model disagree, the trade is skipped and logged (Q26).

## 5. How a trading day works

1. Before the open: refresh the data, the borrow list and the news. Each book's strategies post their candidates.
2. Every 5 minutes, 9:30-16:00: plain code scans every stock (free IEX real-time prices, Alpaca's Benzinga
   news stream, new SEC filings) for unusual moves, volume or news (Q33, Q34, Q47).
3. Flagged stocks go to the analyst agent (reads the news and filing) and then the picker agent (decides long,
   short, close or nothing, with size and stop) (Q27).
4. The risk agent sizes the trade within the rules, the reviewer agent checks it, and code places the order.
5. After the close: log everything (agent inputs and outputs too), and update the frozen copy and the random
   control books.

The watcher runs as two chained GitHub Actions jobs per day in the public repo (each job is capped at 6 hours).
Only summaries and decisions are committed; no raw headline text.

## 6. The agents (Q23-Q27)

| Agent | Job | Runs |
|---|---|---|
| Analyst | Reads news and filings for flagged stocks; returns a short structured view | all day |
| Picker | Decides long / short / close / nothing for flagged stocks; one strategy among the others | all day |
| Risk manager | Sizing, the 10% caps, the pause, world-shock watch | all day |
| Reviewer | Checks each order against the rules and the picker's reason | each order |
| Coach | Runs the weekly learning rule, proposes new rules into the shadow book | weekly |
| Researcher | Searches new papers and posts, adds Claude's own ideas; Amit approves each one | weekly |

Analyst, picker, risk and reviewer run on free Gemini Flash and Groq Llama. Claude does the researcher and
coach work in this project. After 4 weeks we compare a sample of free-model decisions with a paid model (about
$10-40 a month) and Amit decides with numbers (Q24, Q10).

## 7. How it learns (Q2, Q3, Q18, Q37, Q46, Q48)

- Strategies at the start (Q16): main gets residual momentum (M3), the M4 composite, analyst actions (M6/M8),
  and the picker. Penny gets P4 insider clusters, P1/P2-style event signals, the B0 burst model when it's done,
  and the picker. The AI trader bots join as strategies (Q21).
- Every Sunday the coach shifts money toward strategies with better recent results after costs. It keeps some
  money on the rest and moves at most 10 points per strategy per week (a "bandit" method).
- Models inside strategies (XGBoost and others) are retrained weekly on forward data only, with a fixed seed
  and a logged version, so the training always lags what it predicts.
- New rules, from the coach, the researcher or Amit, start in the shadow book. A rule is promoted after
  beating the frozen copy over at least 4 weeks and 30 trades after costs. The confidence needed starts at about
  80% and rises toward 90% as more rules are tried. Every shadow rule counts in the idea log.
- The learning rule itself is frozen in a pre-registration. Each week we compare the learning book with a
  frozen copy of its starting weights and a same-day random control. That comparison tells us whether the
  learning helps.

## 8. What Amit sees (Q31)

A short report every Sunday in this project: each book's profit and loss against the frozen copy and the random
control, what the coach changed and why, and new ideas waiting for his yes. It fits on one phone screen.

## 9. Build order (Q32)

Each stage starts paper trading as soon as it passes its tests (synthetic-data tests, then one dry-run day
with orders logged but not sent).

1. Stage 1: the shadow book, cost model, frozen-copy and random-control books, the learner on existing
   strategies, the two new paper accounts, the 10% pause and the kill switch.
2. Stage 2: the risk and reviewer agents, plus the all-day 5-minute watcher.
3. Stage 3: the analyst and picker agents.
4. Stage 4: the researcher and coach routines, and the Sunday report.

## 10. What Amit needs to do

- Sign two dated amendments in `PROTOCOL.md`: new ideas skip the historical t >= 3 screen, and the system may
  create rules that go through the shadow book (Q36).
- Create two more Alpaca paper accounts and add their keys as GitHub secrets (I'll give the exact names).
- Approve this plan.

## 11. Things I'll check while building, and tell you about if they matter

- Whether the paper accounts still apply the old day-trading limit.
- Whether IEX-only real-time prices are too noisy for the watcher (Q15, Q34).
- Free AI limits under all-day load (Q24).
- Better free data, or a paid option that would clearly help (Q47).

## 12. Additions from the professional review (approved 2026-10-20, details in PRO_REVIEW_2026-10-20.md)

1. Bracket orders: every stop-loss (and take-profit) is held by Alpaca. The watcher only moves stops.
2. Shorts at most 5% of the book at entry, each with a broker-side buy-stop. Any position past 10% of the
   book is trimmed back to 10%. No penny shorts held overnight into a known event.
3. Everything is judged market-adjusted: return minus beta times SPY (main) or IWM (penny). The learner
   ranks strategies on that number, and net exposure is reported.
4. Volatility sizing: each trade risks about 1% of the book down to its stop, with the 10% cap on top.
5. Fill realism: orders are capped at 1% of 5-minute dollar volume (penny) or 0.5% of daily dollar volume
   (main). A limit order counts as filled only when the price trades through it. Measured spread added on top.
6. Data limits: the scan uses multi-symbol snapshots (about 10 calls per pass). The 30-symbol IEX stream is
   kept for open positions and top flags.
7. Accounts: Amit has 2 Alpaca logins, so 6 paper slots; 4 are used (the existing book and the AI-bot
   thread's 3 mirrors). Main and penny each get one of the 2 spare slots, and Amit may add more logins.
   The mirrors stay with the AI bots until those bots fold in as strategies (Q21).
8. Paper equity stays above $25k (old day-trade check in paper); our ledger sizes trades as a $5k book.
9. Agents return fixed fields only (direction, confidence, one-line reason), and code rejects anything else.
   News text is wrapped and labelled as data, and the reviewer never sees raw headlines.
10. Every agent decision logs the exact model and version, at temperature 0. A model change counts as a new
    strategy version.
11. The Sunday report shows how many more weeks each book needs to reach the real-money bar at its current pace.
12. Daily theme-exposure report, flagged above 40% of the book on one theme (a report, not a limit).
Later, before real money: Israeli tax and the W-8BEN form with an accountant.
