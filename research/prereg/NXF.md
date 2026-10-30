# NXF: filters on NX3's picks (news track)

Registered 2026-10-30, before any NXF number is computed. Source: Amit's grill 2026-10-30 round 2 Q4 ("mix
and match": test filters on NX3 one at a time, each with a random control), after the research report
`/mnt/project-files/reports/Live strategies and adaptive exits.md`. NX3's history was seen first, so this
history can only reject a filter (PROTOCOL rule 4). Three filters, each counted (count in NXE.md).

## What stays NX3's

Everything but which fired events may become picks: NX3's walk-forward, fired events, effect, picks rule (first
3 fired longs and 3 fired shorts in time order, now among the events that pass the filter), same-day random
control drawn from all that day's real events, next-close exit, costs, test years 2019 to 2025-09.

## Filters (fixed now)

- `NOV` (fresh news): the event's novelty input is at most 2 (at most 2 days with a Benzinga article on the
  symbol in the 30 calendar days before d).
- `RV` (unusual volume at entry): the entry minute bar's SIP share volume is at least 2x the median of the same
  minute's volume over the stock's previous 20 sessions (same minute of the session; a missing bar counts 0).
- `NO8K` (no bad 8-K, longs only): a long is skipped when the company filed an 8-K with item 1.03, 2.06, 3.01,
  4.01 or 4.02 with a filing date in the 5 sessions before d up to d (EDGAR acceptance before t on d). Shorts
  are untouched.

## Primary test (per filter)

Daily pick-minus-control, pooled, mean > 0 and Newey-West t (5 lags) >= 1.5 (Amit's quick-check bar), and the
filtered picks minus NX3's own picks on the same days reported beside it (no rule). A filter that passes gets
a $5,000 ledger shadow book on NX3's live events; an Alpaca paper account only after rule 20.

## Prediction (written first)

NOV helps a little (fresh news drifts more, Chan 2003); RV is close to NX3 (the first-reaction input already
holds most of it); NO8K changes few trades. Probability at least one filter passes and beats NX3's picks by NW
t >= 1.5: about 0.25.
