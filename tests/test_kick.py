"""Morning starter (core/kick.py): which workflows are due, on synthetic clocks (no network)."""
from datetime import date, datetime, timezone

from core import kick


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_due_starts_each_job_once_inside_its_window():
    mon = utc(2026, 10, 26, 13, 25)                      # 9:25am ET
    assert kick.due(mon, {}, set()) == ["listen.yml", "nx3-live.yml", "chr-live.yml", "day1-live.yml", "watch.yml", "scouts.yml", "dashboard.yml"]
    runs = {"listen.yml": [utc(2026, 10, 26, 8, 0)], "nx3-live.yml": [utc(2026, 10, 26, 13, 16)],
            "watch.yml": [utc(2026, 10, 23, 13, 20)], "scouts.yml": [utc(2026, 10, 26, 7, 0)]}
    assert kick.due(mon, runs, set()) == ["chr-live.yml", "day1-live.yml", "watch.yml", "scouts.yml", "dashboard.yml"]   # Friday's and pre-start runs don't count
    assert kick.due(utc(2026, 10, 26, 7, 54), {}, set()) == []          # 3:54am ET, before listen's start
    assert kick.due(utc(2026, 10, 26, 7, 56), {}, set()) == ["listen.yml"]
    assert kick.due(utc(2026, 10, 26, 20, 0), {}, set()) == ["listen.yml", "dashboard.yml"]   # 4pm ET: only listen's window left
    assert kick.due(utc(2026, 10, 27, 0, 0), {}, set()) == []           # 8pm ET


def test_no_starts_on_weekends_or_closures():
    assert kick.due(utc(2026, 10, 24, 14, 0), {}, set()) == []          # Saturday
    assert kick.due(utc(2026, 11, 26, 15, 0), {}, {date(2026, 11, 26)}) == []
    assert date(2026, 11, 26) in kick.holidays()                       # Thanksgiving is in the NYSE file


def test_dashboard_every_ten_minutes_in_market_hours_and_once_early_each_morning():
    mon = utc(2026, 10, 26, 15, 0)                       # 11am ET
    assert "dashboard.yml" not in kick.due(mon, {"dashboard.yml": [utc(2026, 10, 26, 14, 52)]}, set())
    assert "dashboard.yml" in kick.due(mon, {"dashboard.yml": [utc(2026, 10, 26, 14, 49)]}, set())
    assert "dashboard.yml" not in kick.due(utc(2026, 10, 26, 20, 46), {}, set())     # 4:46pm ET, settle is in
    assert kick.due(utc(2026, 10, 24, 9, 10), {}, set()) == ["dashboard.yml"]         # Saturday 5:10am ET
    assert kick.due(utc(2026, 10, 24, 14, 0), {}, set()) == []                        # Saturday market hours


def test_boom_short_and_n1_screen_are_started():
    from datetime import datetime as _dt
    from core import kick as K
    et = K.ET
    m = K.due(_dt(2026, 11, 10, 8, 45, tzinfo=et), {}, set())
    assert "boom-short.yml" in m
    e = K.due(_dt(2026, 11, 10, 17, 0, tzinfo=et), {"boom-short.yml": [_dt(2026, 11, 10, 8, 45, tzinfo=et)]}, set())
    assert "boom-short.yml" in e
    s = K.due(_dt(2026, 11, 14, 3, 20, tzinfo=et), {}, set())       # a Saturday
    assert "n1-screen.yml" in s and "boom-short.yml" not in s
