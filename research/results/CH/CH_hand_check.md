# CH hand check (2026-10-31)

Pass rule (prereg): at least 19 of 20 random triggers per track right. Result: **main 19/20, penny 20/20. Passed.**
No return or label was read for this check.

The 40 rows are `research/checks/CH.csv`, drawn by `stage=check`. Each was checked against separate Alpaca SIP pulls
from `tracks/hand_check.py` (plain HTTP; it shares no code with the study):

- `CH_bars.json`: the day's 5-minute bars, raw and all-adjusted, with each bar's VWAP.
- `CH_daily_bars.json`: about 275 prior sessions of daily bars, raw and all-adjusted.

What was compared at the logged bar close s:

| Field | How it was checked | Result |
|---|---|---|
| entry | Open of the raw 5-minute bar at s + 15 minutes | 40/40 exact |
| vwap_dist | Close at s over the session VWAP built from the bars' own vw | 40/40, error < 0.001 pts |
| gap, d_prev_c | Against yesterday's close, rebased to today's raw prices from the independent daily bars | 40/40 within 0.025 pts |
| vol20, 20-day high/low, 252-day high/low | Rebuilt from the independent daily bars | used in the trigger replay below |
| rvol | Independent vol20 and cumulative volume | see below |
| trigger and s | All seven triggers replayed in the logged order, from 9:45 on | 40/40 same trigger; 39/40 same bar |

On rvol: the volume-share curve is a fixed 2016 input, not something this check tests. So rvol was replayed with the
curve values implied by the logged rows. Within each track and slot, the ratio of independent to logged rvol is
constant (main 0.998 and penny 0.727 at 9:45, with a local curve estimate), which shows vol20 itself matches.

The one miss is OSW on 2023-12-20 (main). It's the same trigger (new 20-day high), but the check fires it one bar
later. The 9:45 close was 13.5500 and the independent 20-day high was 13.5504, 0.003% apart, so the gap is a
rounding-level difference in the adjusted high. Entry, gap and VWAP all match.

## Bug found while checking (fixed before any result, PROTOCOL rule 23)

`build_month` compared today's raw bars with yesterday's raw close (`ctx["prev_c"] = r.prev_c`). The 20-day and
52-week levels and vol20 were already moved to today's raw basis. On a split day this made a fake gap: a 2:1 split
read as -50%, so the gap trigger fired, and `gap`, `gap_filled`, `ret_today` and the first-30-minutes trigger were
wrong. None of the 40 sampled rows was a split day, so the sample didn't show it; it came from reading the code.

- Fix: `ctx["prev_c"] = r.prev_ac / f0` (yesterday's adjusted close on today's raw basis).
- Test: `tests/test_ch.py::test_build_month_split_day_gap_uses_todays_basis` failed before the fix with `['gap'] != ['hl20']` and passes after it.
- Rebuild: `stage=rebase` deletes every month holding a candidate whose price basis moved by more than 1% overnight (splits and large distributions), then rebuilds those months in the same run. Its counts go to `CH_rebase.json`.
- Not rebuilt: ordinary dividend days (basis move under 1%). Their gap input is off by the dividend yield, which never comes near the 3% gap threshold.

The 40 checked rows aren't split days, so the rebuild doesn't change them, and the check stands.
