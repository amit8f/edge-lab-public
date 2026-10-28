# SYS2: the all-day picker (Plan v3 stages 2 and 3, cross-track)

Registered 2026-10-22, before any result exists. Amit chose to build it next in the planning thread
("Day scanner first", 2026-10-22). Design: `docs/PLAN_V3.md` sections 3-6 and review items 5, 6, 9 and 10.
Phase-2 count at registration: 34 (cross-track 6). Phase 1 + phase 2 = 49. A changed model or prompt is a new
version, logged and judged on its own, and counts as a new idea.

## Mechanism (PROTOCOL rule 10)

- Other side: traders who react slowly or wrongly to a fresh move, volume burst or headline in the first
  hour after it. The picker claims it can tell continuations from fades better than a coin.
- Why a $1B fund can't take it: positions of $50-125 in thin small and penny stocks, a few a day.
- Published: LLM reading of news predicts next-day returns (Lopez-Lira & Tang 2023), but results after a
  model's training cutoff are much weaker. Expected effect: small or none. This tests it on live data only, so
  the model's cutoff can't leak.

## Question

After costs, does each picker book beat its coin twin (same stocks, same moments, same size and stop, random
side)?

## Universe (built once a day before the open, `adaptive/watch.py`)

Active common stocks (`core.instruments.classify`), traded in the last session, last raw close at least $1,
median raw dollar volume over 20 sessions at least $1M, 20 daily returns with no gap over 15 calendar days.
Stocks closing at $5 or more go to MAIN, $1-5 to PENNY. Volatility is the standard deviation of the last 20
daily log returns from `adjustment=all` closes; normal volume is the 20-session median share volume. Stocks with a
split effective that day (Alpaca corporate actions) are left out for the day.

## The 5-minute pass (9:35 ET to the last 5-minute mark before the close)

1. Prices: the latest IEX trade (Alpaca's free real-time feed), else the latest 15-minute-delayed SIP trade.
   Volume: today's SIP volume from the delayed-SIP snapshot.
2. Flags, by plain code:
   - move: |ln(price / previous close)| at least 3 daily standard deviations;
   - volume: today's volume at least 3x normal volume times the share of the session elapsed at the volume's
     time (from 5% of the session on);
   - news: a Benzinga article tagging the stock (and at most 3 stocks) published in the last 10 minutes.
3. Flagged stocks are ranked by number of flag types, then the largest move. Up to 6 MAIN and 4 PENNY stocks
   that aren't held and weren't shown in the last 30 minutes go to the agents (slots one book can't fill go to
   the other, 10 in all).
4. Analyst (Groq `openai/gpt-oss-20b`, temperature 0): for each flagged stock reads up to three headlines
   from the last 24 hours, wrapped as untrusted data, and returns event type, tone (-2..2) and confidence.
5. Picker (Groq `openai/gpt-oss-120b`, temperature 0): sees the numbers, the analyst's fields, the wrapped
   headlines and both books, and answers long, short or none for each flagged stock and close or none for each
   held one, with confidence (0-100), a stop (percent) and a one-line reason.
6. Risk (code, `adaptive/sizing.py`): an entry needs confidence at least 60, a stop of at least 1% and at most
   the cap (15% MAIN, 25% PENNY), shorts only where Alpaca marks the stock shortable and easy to borrow. Size:
   1% of the book at risk to the stop, at most 10% of the book per stock (5% for a short), 1x gross, at most
   0.5% of daily dollar volume (MAIN) or 1% of average 5-minute dollar volume (PENNY).
7. Reviewer (Google `gemini-3.5-flash-lite`, temperature 0, a different model family): sees the numbers, the analyst's fields
   and the proposal, never a headline. It names the direction it would take. If that differs from the picker's,
   the trade is skipped and logged.

Agents return fixed fields only (`adaptive/guards.py`). Every pass logs the flags and numbers, the model names,
the prompt's SHA-256, each raw answer's SHA-256, the parsed fields, the risk and reviewer outcomes and the
fills. No headline text is logged.

## Books and fills (ledger-only shadow books, SYS1)

- PICK-MAIN $1,250 and PICK-PENNY $500 (SYS1 clarification 2), each with a coin twin.
- Fills are simulated at the price the pass used, plus `adaptive/costs.py` (half the measured spread a side,
  IBKR-style commission, sale fees, borrow on shorts at the track default).
- Exits: the stop, checked every pass at that pass's price (a gap through the stop fills at the worse price);
  the picker's close; or the first pass after 10 trading days held. Positions are kept overnight.
- Twin: when the picker book enters, its twin enters the same stock at the same price, size and stop, on a
  seeded random side (long when a short isn't allowed). It exits at its own stop, when the picker exits that
  stock, or after 10 trading days.
- Safety: a book 10% below its peak takes no new entries until Amit restarts it. More than 20 orders in a day,
  or an agent answer the code can't read, stops new entries for the day (stops and the 10-day exit still run).
  A missing answer (network) skips that pass's agents and is logged; it isn't a kill. When a model's free
  daily quota runs out, the agents stop for the rest of the day (exits by stop and age still run).

## Primary test (read at 100 closed trades per book, and at 2027-12-31)

For each book: daily return of the picker book minus its coin twin, after costs, Newey-West t (5 lags). Pass:
t >= 2 and a positive mean. Secondary, descriptive: each book against SPY (MAIN) or IWM (PENNY).

## What a result does

The books start with no Alpaca account. A book moves to its paper account (EL-PICK-MAIN, EL-PICK-PENNY, once
Amit creates them) under SYS1's promotion rule against its twin: at least 20 trading days, 30 closed trades and
confidence of at least `adaptive/promotion.required(n)`. Pass at the primary read: it may join SYS1's learner
as a funded strategy. Fail: it is dropped or re-registered as a new version.

## Clarification 1 (2026-10-22, after a one-hour test run that afternoon and before any closed trade; Amit's choice)

- PICK-MAIN is a $5,000 book, so a 10% position can hold one share of a stock up to $500. At $1,250 the test
  run sized CRS and FCFS at zero shares. PICK-PENNY stays $500 ($1-5 stocks fit). EL-PICK-MAIN would hold $5,000.
- Both books and twins start fresh at the first pass of 2026-10-23. The 2026-10-22 passes stay in the ledger
  as a test and are left out of every result.

## Clarification 2 (2026-10-28, Amit's choice; no rule on entries, sizing, exits or the test changes)

- An agent answer that can't be read is asked once more, with the same prompt plus one line saying the last
  answer could not be read (`adaptive/watch.py` `RETRY_NOTE`). Only an answer still unreadable after that one
  retry counts as an agent error under the kill switch (`adaptive/guards.py`), which is unchanged. The retry's
  model, prompt hash and answer hash are logged with the first call.
- Why: one unreadable answer stopped new entries in both books for the rest of the day on 2026-10-23, 10-26
  and 10-27 (once each from the picker, the reviewer and the analyst). Raw answers aren't logged here, so the
  retry is the fix rather than a parser change.
