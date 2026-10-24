# FWD1: forward shadow books for near-miss screens (PROTOCOL rules 19-20)

Registered 2026-10-24, before any forward data exists. Not a new idea: each book trades a rule already
registered and counted, unchanged. Idea count stays 42 (57 with phase 1).

Which failed screens qualify (from their committed results, checked 2026-10-24):

| ID | Rule traded forward | vs random, HAC t | Picks' own mean after costs | In? |
|----|---------------------|------------------|-----------------------------|-----|
| R1 | prereg/R1.md, weekly, main universe | 1.89 | +0.33% a week | yes |
| NX2 | prereg/NX2.md, nightly, its frozen v2 model | 2.77 | not reported by the screen; the book must show > 0 | yes |
| L1 offering/penny | prereg/L1.md, that cell only | 1.91 | +1.35% a trade | yes |
| L1 late filing/main | prereg/L1.md, that cell only | 1.57 | +0.78% a trade | yes |
| L1 late filing/penny | | 1.46 | | no (t < 1.5) |
| A1 | | 2.04 | -5.9% a trade (control -17.9%) | no |
| I1 | | 1.42 | picks -1.54% a week | no |
| M6 | already has an account (EL-M6) | | | n/a |

The two L1 cells were picked as the best of 14 after results, so they are post-hoc hypotheses (rule 8);
the forward test is their first real test.

**Each book.** Same entry, exit, universe and costs as its registration, on live data (IEX real-time or
SIP bars ending now - 16 minutes). Same-day random control drawn as in its registration. Logged
append-only through `core/ledger.py` in the shadow book, no orders.

**Most-confident subset (added before any forward data, at Amit's request 2026-10-24).** Each book also
logs a second, smaller variant: only its 5 most confident picks per signal date. Confidence is fixed now:
R1, the most negative 5-day return (deepest loser); NX2, the largest |news effect|; L1 cells, the
largest first-minute move against the trade (the most overreaction). The variant is judged by the same
rule 20 test on its own trades; both variants count toward the Holm adjustment (eight tests). When the
META take/skip model (Plan v3) is trained, it is added as a third variant under its own registration.

**Promotion (rule 20).** Judged once, at the first date with >= 60 trading days and >= 30 closed trades:
forward pick-minus-control after costs > 0 with Newey-West t >= 2.0 (lags as in its registration) and
the picks' own mean after costs > 0. Four books are judged, so each is also reported with a Holm
adjustment over all variants. A book that misses stops and is logged; it doesn't get a second window.
