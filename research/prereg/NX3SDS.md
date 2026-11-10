# NX3-ALL-SDS: shorts bought back the same day

Registered 2026-11-10, before any NX3-ALL-SDS number is computed. Source: Amit, 2026-11-10, after NX3-ALL lost
about $208 on Friday 2026-11-06's trades (252 shorts held over the weekend into Monday's rally): "we need to sell
the short ... that same day not the long, the long can be sold next day". One idea. Phase-2 count 138 -> 139
(phase 1 + phase 2 = 154).

SHORT-FAST (2026-11-06) already measured NX3's capped picks with a same-session exit: shorts -0.02% a trade
after costs (3,363 trades), about NX3's next-close shorts; longs -0.19%. So this rule is expected to change risk,
not the average. NX3's live days were seen first, so history can only reject (PROTOCOL rule 4).

## Rules

NX3-ALL unchanged (prereg NX3ALL.md): same models, seeds, fired events, easy-to-borrow rule, confidence sizing
5%..10% of $5,000, costs (`config/risk.yaml`, borrow for 1 day on a same-day short). One change: every short, pick
or control, is bought back at 15:55 on the day it was opened, at the close of the raw SIP 1-minute bar ending at
15:55 (5 minutes before an early close). Longs keep the next session's close. A short without a 15:55 price drops
with its pair.

## Primary test (history, 2019 to 2025-09, NX3-ALL's walk-forward)

All four, per day, Newey-West (5 lags):
1. Sized pick-minus-control mean > 0 with NW t >= 1.5 (NX3-ALL's near-miss bar).
2. Own sized net per day (after costs) mean > 0.
3. Paired daily own-net gain over NX3-ALL (same days, same picks) mean >= -0.02% of the book (about -$1 a day):
   no clear loss of average.
4. Worst 1% of days (own sized net) better than NX3-ALL's.

Pass: the $5,000 ledger shadow book keeps running and is promoted over NX3-ALL only under the standard rule (it
beats NX3-ALL over 4 weeks and 30 trades, then Amit decides). Fail: the shadow book stops and the result stays.

## Forward (from 2026-11-10)

Ledger only, no orders: NX3-ALL's logged picks and control each day (kind nx3all_day), settled after the next
close with shorts at 15:55 (kind nx3sds_settle, `tracks/news/nx3sds_live.py`). On the dashboard as NX3-ALL-SDS.

## Reported, no rule

- Per year own net; shorts' own net per day; day-to-day spread.
- Days when 80%+ of the money is short (like 2026-11-06): the gain over NX3-ALL on those days.

## Prediction (written first)

Average about the same as NX3-ALL (test 3 passes), worst days smaller (test 4 passes), so the history check passes
with probability about 0.55; it fails mainly if the shorts' overnight move is part of the edge.
