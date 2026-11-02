# C-INV: the opposite of AI bot C, once a day (main track)

Registered 2026-11-01, before the book exists. Source: the monthly review of 2026-11-01
(`/mnt/project-files/reviews/MONTHLY_2026-11-01.md`, idea B); Amit: "do both and we will see". By Oct 29, C
(K4) was down 13.3% over 8 sessions while a coin flip on the same names at the same moments was down 3.4%;
C won 21 of 96 closed trades and the coin flip 46. That was seen first, so the test uses only days after
registration (PROTOCOL rule 23). One idea. Phase-2 count 92 -> 93 (phase 1 + phase 2 = 108).
K4, K7 and K8 keep running unchanged. No real money: fills are simulated in our own ledger.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** If C's picks carry a stable bias (for example buying after a surge that then fades),
   the other side of that bias is whoever buys the surge late. If C's losses are noise, there is no
   counterparty and C-INV should do no better than a coin flip.
2. **Why us.** Nobody else has C's answers.
3. **Published.** No.

## Rules (fixed now)

- One $10,000 book, `C-inv`, in `tracks/main/ai_wide.py`, using C's own answers from the same runs (Gemini):
  no extra AI calls, no change to C's prompt or parser.
- Once per ET day, at C's first readable answer with a price time at or after 10:00 ET, from the first trading
  day this code runs: the target is every stock C wants, on the opposite side (C long -> C-INV short, C short
  -> C-INV long). A new short that isn't on that day's shortable list is dropped and logged. C wants all
  cash -> C-INV goes to cash.
- Sizing, costs, prices and the $500 floor are C's (K4): up to 5 positions, a fifth of equity each, 0.1% a
  side, 3% a year borrow on shorts. Positions are held until the next day's 10:00 answer; positions still
  wanted on the same side are kept.
- An unreadable or failed answer doesn't use up the day. With no readable answer from 10:00 ET to the close,
  the book holds. If C stops, C-INV gets no more answers and holds.
- Coin twin `C-inv-coin` takes each new C-INV position with a coin-flip side (K4's `coin_side`) and drops it
  when C-INV does.
- C-INV's holdings are priced every run but stay off C's shortlist, so C sees exactly what it saw before.

## Primary test

Daily net return of C-INV minus `C-inv-coin`, each from equity at the day's last run, Newey-West t (5 lags).
**Pass:** mean > 0 and t >= 1.5 (the near-miss bar) after at least 60 trading days from C-INV's first trade
and at least 30 closed trades. Before then, no conclusion.

## Secondary (no pass rule)

C-INV's own net return; C-INV minus SPY (the HOLD book); trades, turnover and costs paid; C-INV minus C-slow
(K8).

## What happens next (fixed now)

A pass goes in the Sunday report and gets a PROTOCOL rule 20 forward test with its own written rule before any
Alpaca paper account. A fail changes nothing elsewhere; C-INV stops and stays in the log.

## Prediction written now

About 0.65 that C-INV makes money before costs over 60 days, 0.4 after costs, and 0.2 that it passes. Eight
sessions is little, and once-a-day trading cuts the cost C pays but also changes which of C's answers count.
