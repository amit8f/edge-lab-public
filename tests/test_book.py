"""Paper book rules on synthetic prices (no network)."""
import pytest

from paper.book import apply_split, equity, load_config, new_state, process_day

CFG = load_config()
CAL = [f"2026-01-{d:02d}" for d in range(5, 31)]


def sig(symbol="AAA", side="long", track="main", hold=5, atr=1.0, close=100.0, **kw):
    return dict(track=track, idea="T1", symbol=symbol, side=side, hold_days=hold, atr20=atr, close=close, **kw)


def run(days, signals_by_day=None, state=None):
    """days: list of (day, {symbol: (open, close)})."""
    s = state or new_state(CFG)
    for day, px in days:
        bars = {(k, day): {"o": o, "c": c} for k, (o, c) in px.items()}
        s = process_day(s, CFG, day, bars, (signals_by_day or {}).get(day, []), CAL)
    return s


def test_entry_next_open_and_sizing():
    s = run([(CAL[0], {"AAA": (99, 100)}), (CAL[1], {"AAA": (102, 103)})], {CAL[0]: [sig(atr=2.0)]})
    (p,) = s["open"]
    assert p["entry_date"] == CAL[1] and p["entry_price"] == 102
    # stop distance 3 x 2 / 100 = 6%; notional = 5000 x 0.005 / 0.06 = 416.67 (below the 500 cap)
    assert p["notional"] == pytest.approx(5000 * 0.005 / 0.06)
    assert p["stop_price"] == pytest.approx(102 * 0.94)


def test_stop_checked_on_close_filled_next_open():
    days = [(CAL[0], {"AAA": (100, 100)}), (CAL[1], {"AAA": (100, 100)}),
            (CAL[2], {"AAA": (99, 90)}), (CAL[3], {"AAA": (85, 88)})]
    s = run(days, {CAL[0]: [sig(atr=2.0)]})
    (t,) = s["closed"]
    assert t["exit_reason"] == "stop" and t["exit_date"] == CAL[3] and t["exit_price"] == 85
    assert t["gross"] == pytest.approx(t["notional"] * (85 / 100 - 1))


def test_time_exit_at_close_of_last_day_with_costs():
    days = [(CAL[0], {"AAA": (100, 100)})] + [(CAL[i], {"AAA": (100, 101)}) for i in range(1, 4)]
    s = run(days, {CAL[0]: [sig(hold=3)]})
    (t,) = s["closed"]
    assert t["exit_reason"] == "time" and t["exit_date"] == CAL[3] and t["exit_price"] == 101
    assert t["costs"] == pytest.approx(t["notional"] * 0.002)
    assert s["open"] == []


def test_short_needs_borrow_and_pays_it():
    s = run([(CAL[0], {"AAA": (100, 100)})], {CAL[0]: [sig(side="short")]})
    assert s["rejected"][0]["reason"].startswith("not shortable")
    days = [(CAL[0], {"BBB": (100, 100)}), (CAL[1], {"BBB": (100, 95)}), (CAL[2], {"BBB": (95, 90)})]
    s = run(days, {CAL[0]: [sig("BBB", side="short", hold=2, shortable=True, easy_to_borrow=True)]})
    (t,) = s["closed"]
    assert t["gross"] == pytest.approx(t["notional"] * 0.10)
    assert t["costs"] == pytest.approx(t["notional"] * (0.002 + 2 * 0.03 / 252))


def test_limits_and_duplicates():
    sigs = [sig(f"S{i}") for i in range(12)] + [sig("S0")]
    s = run([(CAL[0], {})], {CAL[0]: sigs})
    assert len(s["pending_entries"]) == 10
    reasons = [r["reason"] for r in s["rejected"]]
    assert reasons.count("max positions") == 2 and "already held or pending in this track" in reasons


def test_track_pause_after_drawdown():
    # five main longs, each 25 / 0.15 = 166.67 notional, gap down 30%: the track loses 250, more than
    # 10% of the 5000 / 3 sleeve (166.67), while the account (-5%) stays above its own 15% brake
    names = [f"M{i}" for i in range(5)]
    s = run([(CAL[0], {})], {CAL[0]: [sig(n, atr=5.0) for n in names]})
    s = run([(CAL[1], {n: (100, 70) for n in names})], state=s)
    s = run([(CAL[2], {n: (70, 70) for n in names})],
            {CAL[2]: [sig("M9", atr=5.0), sig("N1", track="news", atr=1.0)]}, s)
    assert [r["reason"] for r in s["rejected"]] == ["track or account paused (drawdown)"]
    assert s["rejected"][0]["symbol"] == "M9" and s["pending_entries"][0]["symbol"] == "N1"
    assert 5000 * 0.85 < equity(s, CFG) < 5000


def test_split_rescales_prices():
    s = run([(CAL[0], {"AAA": (100, 100)}), (CAL[1], {"AAA": (100, 100)})], {CAL[0]: [sig()]})
    s = apply_split(s, "AAA", 2.0)
    s = run([(CAL[2], {"AAA": (50, 50)})], state=s)
    (p,) = s["open"]
    assert p["entry_price"] == 50 and equity(s, CFG) == pytest.approx(5000 - p["costs"])
