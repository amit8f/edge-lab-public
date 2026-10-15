# Pipeline replication (PROTOCOL rule 14)

Written 2026-10-15, before the month-end panel (`tracks/main/m3_screen.py`) has been built or compared
with anything. If our data and return code are right, factors rebuilt from our own panel should track
Ken French's published factors closely. If they don't, every screen on this panel (M1, M3) stops until
the cause is found.

## Month-end stock panel (M3 and M1)

On each signal date t (2016-01-29 .. 2025-08-29), among the eligible universe with a 12-1 momentum
value, a market cap and a 1-month forward return:

- Size split at the median market cap; momentum split at the 30th and 70th percentiles.
- Six value-weighted portfolios; our UMD = 1/2 (small high + big high) - 1/2 (small low + big low).
- Our market = the value-weighted mean forward return minus Ken French's RF for that month.
- Each is compared with Ken French's Mom and Mkt-RF for the month after t.

Passes only if all three hold: correlation with Mom >= 0.80, correlation with Mkt-RF >= 0.95, and the
mean monthly difference from Mom within 0.5 percentage points.

Why not tighter: French uses NYSE breakpoints, all listed stocks including micro-caps, and month-end
to month-end returns. Our universe starts at $300M market cap and enters at the next day's open. Those
differences lower the correlation but shouldn't push it below these levels; a sign error, a stale
price, a look-ahead or a broken split adjustment would.

Not possible here: Lazy Prices on the paper's own years (1995-2014) needs data before 2016, which
Alpaca doesn't have. CRSP through WRDS would make it possible.
