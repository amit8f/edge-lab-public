# CR1: do our AI lanes crowd into the same picks? (cross-track)

Registered 2026-10-24, before any lane has logged a pick (SYS2B starts 2026-10-27). Source: the weekly
research scan (`/mnt/project-files/research-scan/2026-10-24.md`, pick 4); Amit: "add all". Phase-2 count
at registration: 54 (cross-track 9). Phase 1 + phase 2 = 69.

## Mechanism (PROTOCOL rule 10)

1. **Counterparty.** None: CR1 measures our own system. If many lanes pick the same names, the books are
   one bet counted many times, and COMBO1's "close to independent" assumption fails.
2. **Why us.** Not applicable (a measurement).
3. **Published.** Chincarini, Falvey and Moneta, "Crowding in an Artificial World" (SSRN 7479638): four
   independent AI agents given the same 209 signals built six strategies from five signals; overlap about
   7 standard deviations above chance, and return correlations rose out of sample.

## Measurement (forward only)

Each trading day, for every pair of SYS2B trigger lanes and the SYS2 picker that each logged >= 3 picks:
the Jaccard overlap of their picked symbols (intersection over union). Null: for each pair, 1,000 redraws
of the same pick counts from that day's scan universe (`numpy default_rng(20261024 + date ordinal)`).
Daily z = (observed mean overlap - null mean) / null sd, averaged over the days.

## Primary test

At the first date with >= 40 trading days logged: mean daily z. **Crowding confirmed** if it is >= 3.0.
Reported alongside: overlap per pair, correlation of the pairs' daily shadow-book returns, and COMBO1's
member correlations.

What happens next is fixed now: a confirmed result is reported in the Sunday report with the pairs that
crowd. It changes no rule by itself; using it (for example, counting crowded lanes as one bet in SYS1's
promotion test) is a new registration.

Prediction: ~60% confirmed (lanes share the same 5-minute scan and context).
