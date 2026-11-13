import numpy as np
import pandas as pd

from tracks.main import dir_path as D

DAY = pd.Timestamp("2024-03-05")


def _path(closes, start="09:30", spread=0.001, day=DAY):
    t = pd.date_range(day + pd.Timedelta(start + ":00"), periods=len(closes), freq="min")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return t, o, np.maximum(o, c) * (1 + spread), np.minimum(o, c) * (1 - spread), c


def test_run_trade_stop_target_gap_and_ambiguous():
    o = np.array([100, 101, 103, 95.0])
    h = np.array([100.5, 102, 104, 96])
    l = np.array([99.5, 100.5, 102, 94])
    c = np.array([100, 101.5, 103.5, 95])
    assert D.run_trade(o, h, l, c, 0, 100, -1, 104, 92) == ("stop", 104.0)          # short stopped at +4%
    assert D.run_trade(o, h, l, c, 0, 100, 1, 96, 103.5) == ("target", 103.5)
    assert D.run_trade(o, h, l, c, 0, 100, 1, 96.0, 120)[0] == "stop"                # bar 3 opens below the stop
    assert D.run_trade(o, h, l, c, 0, 100, 1, 96.0, 120)[1] == 95.0                  # filled at that open (gap)
    assert D.run_trade(np.array([100.0]), np.array([105.0]), np.array([95.0]), np.array([100.0]), 0, 100, 1, 96, 104)[0] == "ambiguous"
    assert D.run_trade(o[:2], h[:2], l[:2], c[:2], 0, 100, 1, 90, 120) == ("close", 101.5)


def test_sstop_trade_entry_at_open():
    _, o, h, l, c = _path([100, 99, 97, 95, 93, 91.5])
    st, e, x = D.sstop_trade(o, h, l, c, -1, 0.04)
    assert e == 100 and st == "target" and np.isclose(x, 92)


def test_bracket_triggers_after_935_and_fills():
    closes = [100] * 5 + [100.5, 101, 102.5, 103] + [103] * 30
    t, o, h, l, c = _path(closes)
    i, side, fill, P = D.bracket(t, o, h, l, c, DAY)
    assert side == 1 and P == 100 and t[i] >= DAY + pd.Timedelta("9:35:00") and np.isclose(fill, max(102, o[i]))
    assert D.bracket(*_path([100] * 40), DAY) is None
    (sa, _xa), (sc, _xc) = D.bracket_trade(o, h, l, c, i, side, fill, P, -1)
    assert sa == "close" and sc in ("close", "stop", "target")


def test_age_call():
    d = DAY
    assert D.age_call([d + pd.Timedelta("9:00:00")], d, 0.03) == 1.0
    assert D.age_call([d + pd.Timedelta("6:00:00")], d, 0.03) == -1.0
    assert D.age_call([d + pd.Timedelta("8:00:00")], d, 0.03) == 0.0
    assert D.age_call([], d, 0.03) == 0.0


def test_sstop_report_on_falling_picks():
    rows = []
    rng = np.random.default_rng(0)
    for d in pd.bdate_range("2021-01-04", periods=80):
        for _ in range(15):
            drift = -0.0012 if rng.random() < 0.8 else 0.002
            closes = 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, 390))
            t, o, h, l, c = _path(closes, day=d)
            rows.append({"d": d, "open": 100.0, "pt": t, "po": o, "ph": h, "pl": l, "pc": c, "d6": rng.normal()})
    r = D.test_sstop(pd.DataFrame(rows))
    assert r["trades"] >= 1000 and r["mean_net"] > 0 and r["verdict"] in ("passed", "failed")
    assert set(r["exits_stop4"]) <= {"stop", "target", "close"}
    b = D.test_brkt(pd.DataFrame(rows))
    assert b["trades"] > 0 and b["per_side"]["short"]["n"] > b["per_side"]["long"]["n"]
