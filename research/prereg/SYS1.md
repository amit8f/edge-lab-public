# SYS1: the Plan v3 self-improving paper system (cross-track)

Registered 2026-10-20, before any result of the system exists. Amit approved the design in the planning
thread (49 answers in `docs/PLAN_V3_DECISIONS.md`, the plan in `docs/PLAN_V3.md`, and 12 additions in
`docs/PLAN_V3_REVIEW.md`). Phase-2 count at registration: 33 (cross-track 5). Phase 1 + phase 2 = 48. Each
rule the system later creates is counted separately (PROTOCOL rule 17).

## Mechanism (PROTOCOL rule 10)

- Other side: whoever each strategy trades against (written in that strategy's own registration). SYS1
  adds no edge of its own. Its claim is that moving money toward strategies that are working, slowly,
  beats keeping the starting weights.
- Why a $1B fund can't take it: the books trade $1-2.5k positions in small and penny stocks.
- Published: weighting strategies by recent results is common practice. Expected effect: small. Strategy
  returns are noisy, and a slow bandit mostly avoids losers rather than finding winners.

## Question

After costs and market exposure, does the learning book beat (a) its frozen copy, which keeps the starting
weights, and (b) the same-day random control?

## Books

Main ($5+, 50% of a $5,000 book, paper account EL-MAIN) and penny ($1-5, 20%, EL-PENNY), each with its own
learning book, frozen copy and random control (`adaptive/books.py`). Starting strategies and equal starting
weights are listed in `docs/PLAN_V3.md` section 7. A strategy joins with weight 0 in the shadow book.

## Frozen learning rule (`adaptive/bandit.py`, `adaptive/promotion.py`)

- Every Sunday: score = mean daily market-adjusted return over its standard error, over the trailing 60
  trading days (return minus beta times SPY for main, IWM for penny). Target = 0.7 x softmax(score) + 0.3 x
  equal weight. Each weight moves toward the target by at most 10 points a week.
- Promotion out of the shadow book: at least 20 trading days and 30 closed trades. One-sided confidence that
  the rule's daily market-adjusted return beats the frozen copy's must be at least 0.80 + 0.10 x
  min(n - 1, 49) / 49, where n is the number of shadow rules tried so far.
- Sizing, caps, costs and safety: `adaptive/sizing.py`, `adaptive/costs.py`, `adaptive/guards.py`,
  `adaptive/orders.py` as committed with this registration.

## Primary test (read at 100 closed trades per book, and at 2027-12-31)

Daily market-adjusted return of the learning book minus the frozen copy, Newey-West t (5 lags). Pass: t >= 2
and the learning book's market-adjusted return beats the random control at t >= 2. Reported every Sunday,
with the number of weeks left to the 100-trade read at the current pace.

## What a result does

Pass: the learning rule may go to real money under PROTOCOL rule 9, with Amit's sign-off. Fail: the frozen
copy and the strategies are judged on their own, and the learning rule is dropped or re-registered.

## Clarification 1 (2026-10-20, before any result; Amit's choice in the planning thread)

- Book sizes: the main system is a $5,000 book and the penny system a $1,000 book (replacing the 50% / 20%
  of $5,000 above). P1/P2 are unchanged.
- Accounts: each strategy that holds paper money gets its own Alpaca paper account, so Amit can follow it
  on its own. The learner's weights set each account's trade budget (weight x the system's book), and
  nothing moves between accounts. Every account keeps over $25,000 of paper equity (PLAN_V3 addition 8).
  The frozen copy and the random control are ledger-only books with no account. Shadow rules are
  ledger-only too.
- The learning rule, promotion bar, caps and primary test are unchanged.
