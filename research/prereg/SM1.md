# SM1: which news sources carry the moves? A source track record from the listener (news track)

Registered 2026-10-24, before the listener's first live day (2026-10-26). Source: the weekly research scan
(`/mnt/project-files/research-scan/2026-10-24.md`, pick 5); Amit: "add all". Phase-2 count at
registration: 55 (news 9). Phase 1 + phase 2 = 70.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** Traders who read a slower source (or none) and keep trading at the old price for
   minutes after a filing or release lands on the faster source.
2. **Why us.** A source track record is cheap to keep and needs no paid feed; the edge, if any, is in
   knowing where to look, not in speed.
3. **Published.** "Point-in-Time Financial RAG with Frozen LLM Readers and Market-Feedback Adaptive
   Retrieval" (EMNLP 2026 industry track): a memory of which sources' items later paid raised portfolio
   Sharpe from 0.52 to 0.84 (89 stocks; no code). This is measurement only; per the failure-review
   decision Q10, no fast trade is built before NX3's forward result and the listener speed report.

## Data (forward only)

The news listener's logs (`core/listener.py`): items with exactly one ticker that maps to a common stock
priced >= $1. Collapse duplicates: the same ticker within 30 minutes counts once, credited to the source
that saw it first. Move: the absolute market-adjusted return from the first IEX minute bar after
`first_seen_utc` (pre-open items: 9:31) to that session's close (SIP all-adjusted, read the next night).
Control: one random stock from the same price group (L1's groups) over the same minutes
(`default_rng(20261024 + date ordinal)`). Hit = the item's move exceeds its control's.

Source memory: Beta(1, 1) per source, updated only with outcomes whose close has passed.

## Primary test

At the first date with >= 60 trading days and >= 30 items for at least 3 sources: a likelihood-ratio test
that the sources' hit rates are equal. **Sources differ** if p < 0.01. Reported alongside: each source's
posterior mean hit rate and 95% interval, median move minus control, and the listener speed report's
first-seen lead per source.

Use is fixed now: SM1's table is an input the SYS2 picker may read only under a new registration.

Prediction: ~50% "sources differ" (EDGAR and wires should beat outlets; outlets have few single-ticker items).
