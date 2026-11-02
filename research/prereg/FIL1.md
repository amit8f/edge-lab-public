# FIL1: buy after a buyback-authorization 8-K (cross-track, a DAY1 shadow sleeve)

Registered 2026-11-02 with DAY1 (`prereg/DAY1.md`), before any FIL1 return exists. Source: Amit's yes to the
day-bot plan (filings specialist, Q14). One idea; count 94 -> 97 together with DAY1 and AIR1.

## Question

Does buying a stock soon after its company files an 8-K announcing a new share-repurchase authorization beat
a same-day random stock over 5 sessions, after costs, in forward paper data?

Why this filing type: open-market buyback announcements are followed by positive returns beyond the
announcement day in the literature (Ikenberry, Lakonishok & Vermaelen 1995; Peyer & Vermaelen 2009), and the
first-minute filing-lag study L1 found no buyback cell to reuse, so this is not a fix chosen from our results.
Honest odds: about 0.2, because the announcement jump is mostly priced before our entry.

## Event

- An 8-K (not 8-K/A) on EDGAR whose items include 7.01 or 8.01, and whose main document or an EX-99 exhibit
  contains "repurchase" or "buyback" within 200 characters of "authoriz" (case-insensitive), and does not
  contain "completed" or "concluded" within the same 200 characters (completion notices are not new programs).
- Universe: DAY1's (main $5+ with $300M+ market cap, or pennies $1-5 with 20-day median dollar volume >= $1M,
  previous raw close).
- Timing: acceptance time from EDGAR's live feed (the news listener's SEC source). Entry at the first DAY1
  decision at or after acceptance + 1 minute, inside 9:31 to 15:30 ET; filings accepted outside those hours
  enter at the next 9:31. Entry price: IEX last trade (ledger), fill logged.
- One event per company per 30 calendar days.

## Trade and control

Long only, held 5 sessions, exit at the 5th session's close. Size $100 in the shadow ledger. Control: one
random stock from DAY1's universe on the same day, same entry minute and exit, drawn with
`default_rng(20261102 + date ordinal + 1)`. Costs: DAY1's cost model.

## Primary test (forward only)

PROTOCOL rule 20's numbers: after at least 60 trading days and 30 closed trades, picks minus control after costs
at Newey-West t >= 2.0 and picks' own mean after costs > 0. Pass -> FIL1 gets a DAY1 budget (DAY1.md). Fail ->
stops, DIAGNOSES.md entry. No history screen (the forward record decides; PROTOCOL rule 16 lane for system
sleeves).

## Hand check before the first counted day

20 detected events read by hand against the filing: is it a new authorization? Fewer than 18 of 20 right -> fix
the matcher (a data-check fix, rule 23) and redraw before any return is read.
