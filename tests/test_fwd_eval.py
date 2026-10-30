"""paper/fwd_eval.py on synthetic logs and bars (no network)."""
import json

import pandas as pd
import pytest

from paper import fwd_eval as F


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _bars(prices, days):
    return pd.DataFrame({"o": [p[0] for p in prices], "c": [p[1] for p in prices]}, index=days)


def test_long_book_pick_minus_control_after_costs(tmp_path):
    st = tmp_path / "state"
    _w(st / "longterm" / "r1" / "a.json", {"as_of": "2026-10-23", "picks": [{"symbol": "P"}], "control": ["C"],
                                          "hold_sessions": 2, "cost_round_trip": 0.002})
    _w(st / "longterm" / "ledger.jsonl", json.dumps({"kind": "R1_picks", "file": "r1/a.json", "date": "2026-10-23"}))
    cal = pd.DatetimeIndex(["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28"])
    bars = {"P": _bars([(10, 10), (10, 11), (11, 12), (12, 12)], cal), "C": _bars([(10, 10), (10, 10), (10, 10), (10, 10)], cal)}
    books = {b["book"]: b for b in F.evaluate(st, bars, cal)}
    r1 = books["R1"]
    assert r1["closed_cohorts"] == 1 and r1["closed_trades"] == 1
    assert r1["own_mean"] == pytest.approx(0.2 - 0.002)          # +20% over two sessions, minus the round trip
    assert r1["mean_pick_minus_control"] == pytest.approx(0.2)   # the control pays the same cost
    assert books["COMBO1"]["days"] == 2


def test_short_stops_at_the_first_close_35_percent_up():
    cal = pd.DatetimeIndex(["2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"])
    b = _bars([(2, 2), (2, 2.2), (2.2, 2.8), (2.8, 3.0)], cal)
    days, closed = F.trade_days(b, cal, "2026-10-30", 63, -1, 0.02, stop=0.35, borrow=0.2)
    assert closed and len(days) == 2                               # out at 2.8 (+40%), the day after entry
    assert sum(days.values()) == pytest.approx(-0.4 - 0.02 - 2 * 0.2 / 252)


def test_open_trade_is_not_closed_and_l1_books_on_exit_day(tmp_path):
    cal = pd.DatetimeIndex(["2026-10-30", "2026-11-02"])
    _, closed = F.trade_days(_bars([(5, 5), (5, 6)], cal), cal, "2026-10-30", 20, 1, 0.02)
    assert not closed
    recs = [{"events": [{"category": "offering", "group": "penny", "status": "ok", "net": 0.03, "control_net": -0.01,
                         "exit_t": "2026-10-28 10:00", "entry_day": "2026-10-27"},
                        {"category": "offering", "group": "main", "status": "ok", "net": 1, "control_net": 0,
                         "exit_t": "2026-10-28", "entry_day": "2026-10-27"}]}]
    out = F.l1_books(recs)
    assert out["L1-offering-penny"][0] == {pd.Timestamp("2026-10-28"): pytest.approx(0.04)}
    assert out["L1-late-main"] == ({}, [])


def test_a1f_reports_the_mean_without_its_best_cohort():
    cohorts = [{"closed": True, "picks": 1, "pick_net": x, "control_net": 0.0} for x in (0.01, 0.02, 0.9)]
    s = F.status("A1F", {}, cohorts)
    assert s["mean_without_best"] == pytest.approx(0.015)
