# NX3-FRESH: skip NX3's picks decided on a stale price

Registered 2026-11-28, before its forward test starts. Source: Amit, 2026-11-28 ("All three" on the next-steps card),
after AIOK on 2026-11-27: NX3 decided it at 09:39 on an IEX print from 2026-11-25 (8.98) while the stock traded
near 5 after a 45% gap. One idea: phase-2 count 202 -> 203 (218 with phase 1).

## Why

NX3's live scorer reads the latest IEX trade at decision time. For about 10% of decided events that trade is from
an earlier day (the stock hasn't traded on IEX yet that morning), so two price-move inputs (r_open, r_react) are
built from a stale price or left blank, a state training never saw (training used the first SIP bar after t).
About 2 picks a day are decided this way.

## Rule

NX3-FRESH = the NX3 ledger book's settled picks, minus each pick whose decision's `entry_iex_t` falls on an earlier
New York date than the decision day. A skipped slot is not refilled. NX3-HI-FRESH is the same on NX3-HI's book.
Computed from the logged decisions and settlements (no live code changes): `dashboard/build.py` `fresh_books`.

## Test (forward only)

From 2026-11-30. Checked once at 4 weeks and 30 closed NX3-FRESH trades: NX3-FRESH's daily book minus NX3's daily
book (same days, $500 a trade) mean > 0 with Newey-West t (5 lags) >= 1.5, and NX3-FRESH's own net per trade > 0.
NX3-HI-FRESH vs NX3-HI is reported the same way, no rule. A pass makes the skip a candidate change to NX3, which
would still go through Plan v3's promotion rule.

## Disclosed (seen before registering)

A back-look over NX3's logged days 2026-10-29 to 2026-11-25 (descriptive, not part of the test): NX3's 15 stale picks
made +$127 together (NX3-FRESH -$57 vs NX3 +$70), while NX3-HI's 27 stale picks lost -$73 (NX3-HI-FRESH +$773 vs
+$700). Mixed, and small samples.

## Prediction

P(pass) 0.3. Stale picks are mostly pre-open news on thin stocks, which LONG1 found to be NX3's better buys, so
skipping them may cost as much as it saves.
