# NX3-HI: NX3 with a higher confidence bar for its 3+3 slots (news track)

Registered 2026-11-02, before any NX3-HI number is computed. Source: Amit, 2026-11-02: "since NX3 is stricter with
3 to each why there isn't the bar higher so if he picks one of the slots it should take one that is more
confident about?" (answered yes to testing it on history first). NX3's capped history and its first live days were
seen first, so this history can only reject (PROTOCOL rule 4); a forward book would decide. One idea. Phase-2
count 100 -> 101 (phase 1 + phase 2 = 116). NX3 itself is unchanged.

## Question

NX3 fills its 3 long and 3 short slots with the first events of the day that clear the firing line. If a slot
only goes to an event whose confidence c = |e| / cut is at least 1.5 (so strong news, not just over the line),
does the picked book still beat the same-day random control after costs, and does it beat NX3's own picks on
the same days?

## Rules

Everything is NX3's (NX3.md: events, inputs, walk-forward models refitted each test year 2019..2025-09 with the
same seeds, firing cut, entry minute and price, next-close exit, costs, 3+3 slots in time order), except:
- Fired events with c < 1.5 are not taken (they don't use a slot). The control is as in NX3: as many of the
  day's real events at random, same sides.
- Slots still fill in time order among the events that clear c >= 1.5. A short only where easy to borrow in live
  trading; history ignores borrow as NX3's does.

## Primary test (history)

Daily NX3-HI pick book minus control, pooled over test days: mean > 0 and Newey-West (5 lags) t >= 1.5, the
near-miss bar Amit set for live variants (grill 2026-10-30). Pass: a $5,000 ledger shadow book next to NX3 in
NX3's live job (no orders); an Alpaca paper account only after PROTOCOL rule 20 (60 days and 30 closed trades,
NW t >= 2).

## Reported, no rule

- NX3-HI minus NX3 on the days both have picks (paired daily difference, NW t): whether the higher bar helps,
  not only whether it works.
- The same at c >= 1.25 and c >= 2.0 (exploratory: only 1.5 is the registered variant).
- Picks per day, days with no pick, share before the open, per year.

## Prediction (written first)

Fewer, stronger picks will have a higher mean per trade but fewer days and a noisier t; the paired difference with
NX3 will not be significant (the NXE exit tests also found no rule that clearly beat NX3's own). Probability the
primary test passes: about 0.5; probability NX3-HI beats NX3 with paired t >= 2: about 0.15.
