import numpy as np
import pandas as pd

from tracks.news import nxe_study as E

D = pd.Timestamp("2024-03-04")


def _dates(n):
    return [D + pd.Timedelta(days=i) for i in range(n)]


def test_hold_exit_at_the_k_th_close_or_the_close_checked_stop():
    closes = [100, 101, 99, 95, 102, 103]
    assert E.hold_exit(100.0, 1, 2.0, closes, _dates(6), 2) == (99, _dates(6)[2])
    # long, 3 ATR = 6: the first close at or below 94 ... none; a close of 95 is only 2.5 ATR down
    assert E.hold_exit(100.0, 1, 2.0, closes, _dates(6), 5, stop=3.0) == (103, _dates(6)[5])
    # short: a close 6 or more above the entry stops it out at that close
    assert E.hold_exit(100.0, -1, 2.0, [100, 104, 106.5, 90], _dates(4), 3, stop=3.0) == (106.5, _dates(4)[2])
    assert E.hold_exit(100.0, 1, 2.0, [100, np.nan, 101], _dates(3), 2) is None
    assert E.hold_exit(100.0, 1, np.nan, closes, _dates(6), 5, stop=3.0) is None


def test_path_exit_counts_a_bar_with_both_levels_as_the_stop():
    assert E.path_exit(100, 1, 97, 106, [101, 107], [99, 96], [100, 105]) == (97, "stop")
    assert E.path_exit(100, 1, 97, 106, [101, 106.5], [99, 98], [100, 105]) == (106, "target")
    assert E.path_exit(100, -1, 103, 94, [101, 102], [99, 98], [100, 99.5]) == (99.5, "time")
    assert E.path_exit(100, -1, 103, 94, [101, 102], [95, 93.9], [100, 94]) == (94, "target")


def test_frame_exit_gap_through_the_stop_fills_at_the_open():
    s = [(np.nan, 101, 99, 100, D), (95, 96, 94, 95.5, D + pd.Timedelta(days=1))]
    assert E.frame_exit(100, 1, 97, 106, s) == (95, s[1][4], "stop")
    s = [(np.nan, 101, 99, 100, D), (100, 107, 96, 105, D)]
    assert E.frame_exit(100, 1, 97, 106, s)[2] == "stop"          # both in one session
    s = [(np.nan, 101, 99, 100, D), (100, 107, 98, 105, D), (105, 108, 104, 107, D)]
    assert E.frame_exit(100, 1, 97, 106, s)[:2] == (106, D)
    s = [(np.nan, 101, 99, 100, D), (100, 102, 98, 101, D)]
    assert E.frame_exit(100, 1, 97, 106, s) == (101, D, "time")


def test_excursions_are_in_atrs_and_follow_the_side():
    s = [(np.nan, 104, 99, 100, D), (100, 106, 96, 105, D)]
    assert E.excursions(100, 1, 2.0, s) == (2.0, 3.0)
    assert E.excursions(100, -1, 2.0, s) == (3.0, 2.0)
    assert all(np.isnan(E.excursions(100, 1, 0.0, s)))


def test_atr_uses_the_true_range_of_the_last_n_sessions():
    h = [11, 12, 13]
    l = [9, 10, 11]
    c = [10, 11, 12]
    assert E.atr(h, l, c, n=2) == 2.0
    assert np.isnan(E.atr(h, l, c, n=3))


def test_bad_8k_window_and_same_day_acceptance():
    d = pd.Timestamp("2024-03-08")
    before = [pd.Timestamp("2024-03-01")]
    fl = {7: [(pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-05 17:00"), {"4.02"})]}
    assert E.bad_8k(fl, 7.0, d, d + pd.Timedelta(hours=9, minutes=30), before)
    fl = {7: [(d, d + pd.Timedelta(hours=8), {"2.06"})]}
    assert E.bad_8k(fl, 7, d, d + pd.Timedelta(hours=9, minutes=30), before)
    assert not E.bad_8k(fl, 7, d, d + pd.Timedelta(hours=7), before)          # accepted after t
    fl = {7: [(pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-05 17:00"), {"2.02", "9.01"})]}
    assert not E.bad_8k(fl, 7, d, d + pd.Timedelta(hours=9, minutes=30), before)
    assert not E.bad_8k(fl, None, d, d, before) and not E.bad_8k(fl, float("nan"), d, d, before)


def test_daily_diff_and_costed_drop_the_holdout():
    rows = pd.DataFrame({"d": [D, D, D, D + pd.Timedelta(days=1)], "role": ["pick", "pick", "control", "pick"],
                         "net": [0.02, 0.0, 0.005, 0.1]})
    dd = E.daily_diff(rows)
    assert len(dd) == 1 and abs(dd.iloc[0] - 0.005) < 1e-12
    t = pd.DataFrame({"d": [D, pd.Timestamp("2025-09-29")], "side": [1, 1], "price": [20.0, 20.0]})
    out = E.costed(t, np.array([0.01, 0.01]), [D + pd.Timedelta(days=2), pd.Timestamp("2025-10-02")])
    assert len(out) == 1 and out["days"].iloc[0] == 3 and out["net"].iloc[0] < 0.01


def test_after_entry_keeps_the_path_to_1555():
    t0 = D + pd.Timedelta(hours=9, minutes=30)
    b = pd.DataFrame({"t": [t0 + pd.Timedelta(minutes=i) for i in range(0, 390)], "o": 10.0, "h": 11.0, "l": 9.0,
                      "c": 10.0, "v": 100.0})
    r = E.after_entry(b, t0 + pd.Timedelta(minutes=1), D, keep_path=True)
    assert r["v_entry"] == 100.0 and len(r["path_h"]) == 385 and r["post_h"] == 11.0
    assert "path_h" not in E.after_entry(b, t0 + pd.Timedelta(minutes=1), D, keep_path=False)


def test_arm_trades_on_synthetic_rows():
    r = {"symbol": "AAA", "d": D, "entry": 100.0, "side": 1, "atr": 2.0, "price": 100.0, "role": "pick",
         "post_h": 101.0, "post_l": 99.5, "post_c": 100.5, "path_h": [100.5, 107.0], "path_l": [99.5, 99.0],
         "path_c": [100.2, 106.5], "_i": 0}
    for n_ in range(0, 21):
        r[f"date{n_}"] = D + pd.Timedelta(days=n_)
        r[f"c{n_}"] = 100.0 + n_ * 0.1
    for n_ in range(1, E.FRAME + 1):
        r |= {f"o{n_}": 100.0, f"h{n_}": 101.0, f"l{n_}": 99.0}
    t = pd.DataFrame([r])
    h2 = E.arm_trades(t, "H2")
    assert abs(h2["ret"].iloc[0] - 0.002) < 1e-9 and h2["days"].iloc[0] == 3
    day = E.arm_trades(t, "DAY")
    assert day["why"].iloc[0] == "target" and abs(day["ret"].iloc[0] - 0.06) < 1e-9
    exa = E.arm_trades(t, "EXA1")
    assert exa["why"].iloc[0] == "time" and abs(exa["ret"].iloc[0] - 0.005) < 1e-9
    exq = E.arm_trades(t, "EXQ", levels=[(0.25, 3.0)])             # stop at 99.5: day d's low after entry
    assert exq["why"].iloc[0] == "stop" and abs(exq["ret"].iloc[0] + 0.005) < 1e-9


def test_trail_exit_breakeven_then_trail_and_same_session_order():
    s = lambda rows: [(o, h, lo, c, D + pd.Timedelta(days=i)) for i, (o, h, lo, c) in enumerate(rows)]
    # long, entry 100, ATR 2, k 1.5: start stop 97; best 103 arms it -> max(entry, 103 - 3) = 100 (breakeven)
    assert E.trail_exit(100.0, 1, 2.0, 1.5, s([(np.nan, 103, 99, 102), (101, 102, 99.5, 100)])) == \
        (100.0, D + pd.Timedelta(days=1), "breakeven")
    # best 108 -> stop 105; next session opens at 104 below it: fills at the open, on the trail
    assert E.trail_exit(100.0, 1, 2.0, 1.5, s([(np.nan, 108, 100, 107), (104, 105, 103, 104)])) == \
        (104, D + pd.Timedelta(days=1), "trail")
    # one session reaching 110 and 96.5: the stop (97) is checked first, so it's a stop at 97
    assert E.trail_exit(100.0, 1, 2.0, 1.5, s([(np.nan, 110, 96.5, 105)]))[::2] == (97.0, "stop")
    # short mirrors: entry 50, ATR 1, best 47 -> stop min(50, 48.5) = 48.5; then high 49 hits it
    assert E.trail_exit(50.0, -1, 1.0, 1.5, s([(np.nan, 50.5, 47, 48), (48, 49, 47.5, 48.8)]))[::2] == (48.5, "trail")
    # never armed (best < 1 ATR up): time exit at the last close
    assert E.trail_exit(100.0, 1, 2.0, 1.5, s([(np.nan, 101, 99, 100), (100, 101.5, 99, 100.7)]))[::2] == (100.7, "time")


def test_nxt_arms_size_k_by_confidence():
    r = {"symbol": "AAA", "d": D, "entry": 100.0, "side": 1, "atr": 2.0, "price": 100.0, "role": "pick",
         "post_h": 100.5, "post_l": 96.5, "post_c": 99.0, "e": 0.03, "cut": 0.01, "_i": 0}
    for n_ in range(1, E.FRAME + 1):
        r |= {f"o{n_}": 99.0, f"h{n_}": 100.0, f"l{n_}": 98.0, f"c{n_}": 99.0, f"date{n_}": D + pd.Timedelta(days=n_)}
    t = pd.DataFrame([r])
    tr = E.arm_trades(t, "NXT-TR")                  # stop 97: day d's low 96.5 hits it
    assert tr["why"].iloc[0] == "stop" and abs(tr["ret"].iloc[0] + 0.03) < 1e-9
    tc = E.arm_trades(t, "NXT-TC")                  # c = 3, capped at 2: k 3, stop 94, never hit
    assert tc["why"].iloc[0] == "time" and abs(tc["ret"].iloc[0] + 0.01) < 1e-9
    nan_c = E.arm_trades(t.assign(e=np.nan), "NXT-TC")   # no score: c = 1, same as TR
    assert nan_c["why"].iloc[0] == "stop"
