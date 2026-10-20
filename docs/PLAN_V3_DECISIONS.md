# Deep planning phase: decisions log (started 2026-10-20)

Format: question, Amit's answer, date. Open items are at the bottom.

## Already decided earlier (from memory, re-confirm only if Amit wants)
- Paper trading only until 100+ closed paper trades pass after costs and Amit signs PROTOCOL.md.
- An idea must pass a forward paper test by end of 2027, else the project becomes a research write-up.
- Historical screens need t >= 3 after costs; holdout from 2025-10-01 untouched.
- Stay free; three separate tracks (main, penny, news).

## Decisions
- Q1 (2026-10-20): goal is mainly A+B: a strategy that works AND makes paper profit. Markets change, so old backtests don't decide what works. The strategy must self-improve from live paper trading as days go by (online learning), not from backtests. Claude's condition: freeze the learning rule (what it may change, how fast, with what limits) and test that rule forward against a fixed copy and a random control.

- Q2: it may change anything, including building new rules (c). Claude's condition: new rules start in a shadow book (logged, no orders) and earn paper money only after beating the frozen copy forward.
- Q3: update weekly.
- Q4: pause at a 10% paper drawdown, unless a world-wide shock (e.g. COVID) is happening. Meaning of the exception still open (Q13).
- Q5: no fixed holding period; same-day to a few days, the system decides.
- Q6: US stocks only.
- Q7: weekly search routine + Claude's own ideas; Amit approves each.
- Q8: as much time as needed, mostly little.
- Q9: LLMs may give inputs AND make decisions.
- Q10: ask Amit before any spending.
- Q11: P1/P2 keep running unchanged as the frozen baseline.
- Q12: real money rule stays: 100+ closed paper trades, profit after costs, beats random control, Amit's sign-off.
- Q13: the 10% pause still applies in a world-wide shock; Amit restarts by hand.
- Q14: Amit decides a shock is happening, but the system must also read news and detect one itself if he forgets.
- Q15: stay with free ~15-min delayed data; use free real-time data if any is better; if delay clearly hurts results, Claude tells Amit with options, paid ones included.
- Q16: learner starts from existing signals + well-known published ones.
- Q17: P1/P2 keep their share; new system 70% of paper account; shadow rules 0% until promoted.
- Q18: promotion = beat frozen copy over >= 4 weeks and >= 30 trades after costs.
- Q19: new ideas skip the t>=3 historical screen; quick bug/look-ahead check, then shadow book.
- Q20: Amit wants to keep trying shorts, open to leaving Alpaca. Broker choice open (Q22).
- Q21: AI trader bots become strategies inside the new system.
- Amit (2026-10-20): use agents in the system; keep XGBoost and other models where they fit.
- Q22: keep Alpaca for orders; Alpaca-refused shorts tracked as shadow trades at IBKR borrow fees; move to IBKR if they pay.
- Q23: agent team approved (researcher, analyst, risk manager, reviewer, weekly coach); Amit asked about a stock-picker agent (Q27).
- Q24: free Gemini/Groq; Claude must say if other agents (free or paid) would do much better.
- Q25: agents run in the public repo on Actions, all inputs/outputs logged.
- Q26: reviewer vs model disagreement -> skip trade, log it.
- Q27: picker agent watches the market all day (price, chart, news), can buy/sell/long/short any time, across all stocks.
- Q28: paper sized as a $5,000 account.
- Q29: main and penny separate learners (they behave differently).
- Q30: only limit = 10% of account per position; no sector or position-count limits.
- Q31: Sunday report, short and plain, for Amit's comments.
- Q32: build order approved (shadow book + learner, risk/reviewer, analyst/picker, researcher/coach).
- Q33: all-day watcher: code scans every stock every 5 min; flagged stocks go to the picker agent; chained GitHub jobs in market hours.
- Q34: free Alpaca IEX real-time for the watcher; delayed SIP for checking results.
- Q35: Amit rejects the 3-day-trade limit; wants no limit. Fact (checked 2026-10-20): SEC approved scrapping the PDT rule on 2026-04-14, effective 2026-06-04; Alpaca stopped enforcing PDT on live accounts that day (intraday margin framework, $2,000 minimum for 4x intraday). So no day-trade limit; verify on the first run that the paper account no longer applies PDT.
- Q36: PROTOCOL.md gets dated amendments, signed by Amit, for Q2 and Q19.
- Q37: count every shadow rule; promotion bar rises with the count, but not too strictly (Q48).
- Q38: separate Alpaca paper accounts for P1/P2, main, penny.
- Q39: new system 70% = main 50% + penny 20%.
- Q40: realistic costs: measured spreads (V4) plus IBKR-style commission and fees, plus borrow fees on shorts.
- Q41: limit orders, cancel after 15 min.
- Q42: regular hours only.
- Q43: stop-loss on every trade, max 15% main, 25% penny.
- Q44: earnings: no special rule; the system decides size.
- Q45: kill switch (>20 orders/day or unreadable agent output) stops the day and messages Amit.
- Q46: bandit coach, max 10 points shift per strategy per week; models inside strategies retrained weekly on forward data.
- Q47: Alpaca Benzinga news stream + SEC filings. Standing rule: Claude flags better free sources and any paid option that would clearly improve results, for anything we use.
- Q48: promotion bar starts ~80% confidence, rises to ~90% (not 95%) as rules pile up; adjustable later.
- Q49: no leverage: 1x, shorts count toward it.
- 2026-10-20 (after approval): Amit set paper money: main system $5,000, penny system $1,000; P1/P2 unchanged (frozen). He chose ONE PAPER ACCOUNT PER STRATEGY that has earned paper money (option b), so he can follow each one in Alpaca; the learner works by changing each account's trade size. Shadow rules need no account. He'll add Alpaca logins as needed (3 paper accounts each). P1/P2 account = secrets ALPACA_API_KEY_ID/SECRET (to be named EL-P1P2).

## Open
- 2026-10-20: Amit confirmed the summary and APPROVED PLAN_V3_2026-10-20.md. Pro review (PRO_REVIEW_2026-10-20.md, 12 items) awaits his yes before Stage 1.
