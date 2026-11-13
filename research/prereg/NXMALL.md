# NXM-ALL: the NXM, NXR and NXK model books without the 3+3 cap

Registered 2026-11-13, before any -ALL book exists or any number for it is computed. Source: Amit, 2026-11-13: "can
we add to all the models the dosent have all an all one so they dont have a limit", with a switch on the dashboard
to see each one (for example NXM-ENS -> NXM-ENS-ALL). Eight ideas, one per arm: NXM-LGBM-ALL, NXM-LIN-ALL,
NXM-MLP-ALL, NXM-ENS-ALL, NXR-XGBR-ALL, NXR-RFR-ALL, NXK-KRR-ALL, NXK2-CAT-ALL. Phase-2 count 150 -> 158 (phase 1 +
phase 2 = 173). The capped books are unchanged.

## Question

Each arm's live book takes only its first 3 longs and 3 shorts a day. Do all the events each arm fires, sized by its
confidence the way NX3-ALL is, beat the same-day random control after costs?

## Rules

Everything is the arm's own capped book's (`tracks/news/nxm_live.py`: version 1 models checked by sha256, the inputs
NX3 logged at each decision, the arm's frozen firing cut, entry at decision, next-close exit, NX3's costs and
settlement), except:
- Picks: every event the arm fires (e > 0 long, e < 0 short; a short only where the stock was easy to borrow at
  decision time), no cap.
- Size: NX3-ALL's rule (`nx3_study.conf_sizes`): confidence c = |e| / the arm's cut; 5% of a $5,000 book at c = 1,
  rising in a straight line to 10% at c >= 2; a day adding up to more than the book is scaled down together.
- Control: as many of the day's decided events at random, `default_rng(SEED + yyyymmdd + 200 + arm number)`;
  control trade i takes pick i's side and size.
- Logged after the close as kind `nxmall_day`, settled as `nxmall_settle`. First session: Mon 2026-11-16.
  Ledger only, no orders.

## Forward test (per arm)

Daily sized book minus control (sum of size x (pick net - control net)). Pass: PROTOCOL rule 20, as NX3-ALL: 60
trading days and 30 closed trades, mean > 0 and Newey-West t (5 lags) >= 2; judged at that point or 2027-12-31.
Holm across the eight arms is reported beside each verdict.

## No history test

These are forward-only books, decided by Amit. Known before registration and recorded here so it can't be read as a
surprise later: in the NXK2 history (2026-11-06), every KRR or CatBoost trade taken alone lost money net
(about -0.18% and -0.15% a trade), and only XGBoost's high-confidence trades made money alone. The test above is
against the random control, so an arm can pass it while losing in absolute terms; the dashboard shows both.

## Prediction (written first)

Like NX3-ALL, the uncapped books will earn less per trade than the capped ones (later events get worse entries).
Probability any one arm passes: about 0.25; that NXK-KRR-ALL or NXK2-CAT-ALL ends positive in absolute terms: about 0.3.
