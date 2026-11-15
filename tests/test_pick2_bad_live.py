"""PICK2-BAD forward book: synthetic data only, no network."""
import json

import numpy as np
import pandas as pd

from tracks.main import pick2_bad_live as L
from tracks.main import pick_test as T

ET = "America/New_York"


def bars_day(day, px, seed):
    rng = np.random.default_rng(seed)
    t = pd.date_range(f"{day.date()} 09:30", f"{day.date()} 15:55", freq="5min")
    c = px * np.exp(np.cumsum(rng.normal(0, 0.004, len(t))))
    o = np.r_[px, c[:-1]]
    return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) * 1.002, "l": np.minimum(o, c) * 0.998, "c": c,
                         "v": rng.integers(20000, 60000, len(t)).astype(float)})


class Fake:
    def __init__(self, days):
        self.days = pd.DatetimeIndex(days)
        self.syms = [f"S{i}" for i in range(6)]
        self.bar_calls = []

    def calendar(self, start, end):
        d = self.days[(self.days >= pd.Timestamp(start)) & (self.days <= pd.Timestamp(end))]
        return pd.DataFrame({"date": d, "open": "09:30", "close": "16:00", "early_close": False})

    def symbols(self):
        return self.syms

    def daily(self, symbols, start, end, adjustment):
        d = self.days[(self.days >= pd.Timestamp(start)) & (self.days <= pd.Timestamp(end))]
        return {s: pd.DataFrame({"o": 50.0, "h": 51.0, "l": 49.0, "c": 50.0 + 0.1 * i, "v": 1e6},
                                index=pd.DatetimeIndex(d, name="t")) for i, s in enumerate(symbols)}

    def bars(self, symbols, day):
        self.bar_calls.append(day)
        return {s: bars_day(day, 50.0, hash((s, str(day))) % 2**32) for s in symbols}

    def news(self, start, end):
        return pd.DataFrame(columns=["id", "created_at", "headline", "symbols"])

    def finbert(self):
        raise AssertionError("not used: finbert is passed in")


def fb(texts):
    return np.tile([0.3, 0.3, 0.4], (len(texts), 1))


def test_universe_is_main_track_at_the_previous_close():
    rows = pd.DataFrame({"symbol": ["A", "B", "C"], "date": pd.Timestamp("2026-11-13"), "c": [20.0, 3.0, 8.0],
                         "dv20": [6e6, 9e6, 4e6]})
    assert list(L.universe_at(rows, pd.Timestamp("2026-11-13"))) == ["A"]


def test_bad_only_keeps_bad_picks_and_every_control():
    R = pd.DataFrame({"role": ["pick", "pick", "ctrl", "ctrl"], "bad": [True, False, False, True]})
    out = L.bad_only(R)
    assert list(out["role"]) == ["pick", "ctrl", "ctrl"]


def test_record_counts_only_trades_the_book_took():
    pk = pd.DataFrame({"symbol": ["A", "B"], "t": pd.to_datetime(["2026-11-16 10:00", "2026-11-16 10:05"]),
                       "score": [0.5, 0.4], "size": [0.4, 0.0], "net": [0.02, -0.01], "exit": ["close", "stop"],
                       "ctrl_net": [0.01, np.nan], "tier": [2, 1]})
    r = L.day_record(pd.Timestamp("2026-11-16"), pk, 100, 7)
    assert len(r["trades"]) == 1 and r["trades"][0]["usd"] == 2000.0
    assert abs(r["book_ret"] - 0.008) < 1e-12 and abs(r["ctrl_ret"] - 0.004) < 1e-12


def test_picks_are_pick_tests_bad_day_rule():
    """picks_of on bad_only rows: a normal-day row at the bad bar is not bought; sizes come from bad picks only."""
    v = {"wpt": {0: 1.5}, "normal": 0.30, "bad": 0.35}
    d = pd.Timestamp("2026-11-16")
    R = pd.DataFrame({"symbol": ["A", "B", "C"], "d": d, "t": pd.to_datetime(["2026-11-16 10:00"] * 3),
                      "track": "main", "tier": 0, "bad": [True, False, True], "role": "pick", "version": "",
                      "partner": "", "score_COMBINED": [0.40, 0.50, 0.33], "net_1.5": [0.01, 0.05, 0.02],
                      "exit_1.5": "close"})
    pk = T.picks_of(L.bad_only(R), v, "COMBINED", "main", 0.45)
    assert list(pk["symbol"]) == ["A"]


def test_run_warms_the_swing_logs_each_finished_day_once_and_chains(tmp_path):
    days = pd.bdate_range("2026-08-03", "2026-11-18")
    api = Fake(days)
    L.run(tmp_path, api=api, now=pd.Timestamp("2026-11-17 17:00", tz=ET), finbert=fb)
    root = tmp_path / "pick2bad"
    logged = sorted(p.stem for p in (root / "ledger" / "days").glob("*.json"))
    assert logged == ["2026-11-16", "2026-11-17"]
    assert len(api.bar_calls) == L.WARM + 2
    rec = json.loads((root / "ledger" / "days" / "2026-11-16.json").read_text())
    assert rec["universe"] == 6 and rec["rows"] > 0
    api2 = Fake(days)
    L.run(tmp_path, api=api2, now=pd.Timestamp("2026-11-18 16:20", tz=ET), finbert=fb)
    assert api2.bar_calls == [pd.Timestamp("2026-11-18")]           # no new warm-up, one new day
    s = json.loads((root / "summary.json").read_text())
    assert s["days"] == 3 and s["first_day"] == "2026-11-16"


def test_a_session_still_open_is_not_logged(tmp_path):
    days = pd.bdate_range("2026-08-03", "2026-11-18")
    L.run(tmp_path, api=Fake(days), now=pd.Timestamp("2026-11-16 16:10", tz=ET), finbert=fb)
    assert not list((tmp_path / "pick2bad" / "ledger").glob("days/*.json"))
