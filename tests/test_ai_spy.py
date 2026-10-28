"""K5 bot A (intraday momentum on SPY): signal, entry and exit times, twin and HOLD books, one run per slot.
Synthetic bars and prices only (no network)."""
import pandas as pd
import pytest

from tracks.main import ai_spy as s

DAY = "2026-10-19"


def bars(open_px=100.0, ten_px=101.0):
    """1-minute bars from 09:30 to 10:05 ET: the open is open_px, the 10:00 bar opens at ten_px."""
    out = []
    t = pd.Timestamp(f"{DAY} 09:30", tz="America/New_York")
    while t <= pd.Timestamp(f"{DAY} 10:05", tz="America/New_York"):
        o = open_px if t == pd.Timestamp(f"{DAY} 09:30", tz="America/New_York") else (
            ten_px if t >= pd.Timestamp(f"{DAY} 10:00", tz="America/New_York") else 100.5)
        out.append({"t": t.isoformat(), "o": o, "c": o})
        t += pd.Timedelta(minutes=1)
    return out


def et(hh, mm):
    return pd.Timestamp(f"{DAY} {hh:02d}:{mm:02d}", tz="America/New_York")


def T(hh, mm):
    return int(et(hh, mm).timestamp())


def test_signal_is_open_to_ten_return_and_none_when_missing():
    assert s.signal(bars(100.0, 101.0), DAY) == pytest.approx(0.01)
    assert s.signal(bars(100.0, 99.0), DAY) == pytest.approx(-0.01)
    assert s.signal([], DAY) is None
    early = [b for b in bars() if pd.Timestamp(b["t"]).strftime("%H:%M") < "10:00"]
    assert s.signal(early, DAY) is None


def test_up_morning_goes_long_at_1530_run_and_exits_at_1557():
    st = {"books": {n: s.new_book() for n in s.NAMES}, "started": None, "days": {}}
    s.step(st, DAY, et(9, 37), 0.01, 100.0, T(9, 37))                # first run: HOLD opens
    s.step(st, DAY, et(15, 27), 0.01, 100.0, T(15, 27))               # before the entry window
    assert "SPY" not in st["books"]["A"]["pos"]
    rec = s.step(st, DAY, et(15, 37), 0.01, 100.0, T(15, 37))
    assert st["books"]["A"]["pos"]["SPY"]["side"] == 1
    assert {f["book"] for f in rec["fills"]} == {"A", "A-coin"}
    s.step(st, DAY, et(15, 47), 0.01, 101.0, T(15, 47))               # still inside the hold
    assert "SPY" in st["books"]["A"]["pos"]
    s.step(st, DAY, et(15, 57), 0.01, 102.0, T(15, 57))
    assert "SPY" not in st["books"]["A"]["pos"] and st["books"]["A"]["trips"][0]["side"] == 1
    assert st["books"]["HOLD"]["pos"]["SPY"]["side"] == 1     # the benchmark never exits


def test_down_morning_goes_short_and_zero_move_trades_nothing():
    st = {"books": {n: s.new_book() for n in s.NAMES}, "started": None, "days": {}}
    s.step(st, DAY, et(15, 37), -0.002, 100.0, T(15, 37))
    assert st["books"]["A"]["pos"]["SPY"]["side"] == -1
    st2 = {"books": {n: s.new_book() for n in s.NAMES}, "started": None, "days": {}}
    s.step(st2, DAY, et(15, 37), 0.0, 100.0, T(15, 37))
    assert st2["books"]["A"]["pos"] == {} and st2["books"]["A-coin"]["pos"] == {}


def test_coin_twin_side_is_fixed_by_date_and_seed():
    assert s.coin_side(DAY) == s.coin_side(DAY)
    assert s.coin_side(DAY) in (1, -1)


def test_position_left_from_a_missed_run_is_closed_at_the_next_run():
    st = {"books": {n: s.new_book() for n in s.NAMES}, "started": None, "days": {}}
    s.step(st, "2026-10-16", et(15, 37), 0.01, 100.0, int(pd.Timestamp("2026-10-16 15:37", tz="America/New_York").timestamp()))     # yesterday's entry, never exited
    rec = s.step(st, DAY, et(9, 37), 0.0, 100.0, T(9, 37))
    assert "SPY" not in st["books"]["A"]["pos"]
    assert any(f.get("reason", f.get("why")) == "stale" for f in rec["fills"])


def test_run_logs_once_per_slot_and_skips_closed_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ai_mirror_run", None, raising=False)
    from tracks.main import ai_mirror
    monkeypatch.setattr(ai_mirror, "run", lambda *a, **k: None)
    px_time = pd.Timestamp(f"{DAY} 15:40", tz="America/New_York").tz_convert("UTC")
    kw = dict(fetch_bars=lambda day, end: bars(100.0, 101.0), latest=lambda: (100.5, px_time))
    now = pd.Timestamp(f"{DAY} 15:47", tz="America/New_York").tz_convert("UTC")
    assert s.run(tmp_path, now=now, **kw) is not None
    assert s.run(tmp_path, now=now + pd.Timedelta(minutes=2), **kw) is None     # same 10-minute slot
    assert s.run(tmp_path, now=pd.Timestamp(f"{DAY} 21:00", tz="America/New_York").tz_convert("UTC"), **kw) is None
    from core.ledger import Ledger
    assert Ledger(tmp_path / "spy").has("run", f"{now.tz_convert('America/New_York'):%Y-%m-%dT%H}:40")


def test_live_bars_waits_for_the_first_free_sip_bar_and_starts_at_the_et_open():
    calls = []
    get = lambda url, p: (calls.append(p), {"bars": {"SPY": [{"t": "x"}]}})[1]
    assert s.live_bars(DAY, et(9, 37).tz_convert("UTC") - pd.Timedelta(minutes=16), get=get) == [] and not calls
    assert s.live_bars(DAY, et(9, 47).tz_convert("UTC") - pd.Timedelta(minutes=16), get=get) == [{"t": "x"}]
    assert calls[0]["start"] == "2026-10-19T13:30:00Z" and calls[0]["end"] == "2026-10-19T13:31:00Z"
    s.live_bars("2026-11-02", pd.Timestamp("2026-11-02 15:00", tz="UTC"), get=get)    # EST: the open is 14:30 UTC
    assert calls[1]["start"] == "2026-11-02T14:30:00Z"
