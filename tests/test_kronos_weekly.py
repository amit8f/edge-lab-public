"""K3 Kronos weekly picks: synthetic data and a fake predictor only (no network, no torch weights)."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from core.ledger import Ledger
from tracks.main import kronos_weekly as k

T = pd.Timestamp("2026-10-16")


def bars(dates, close=10.0, vol=1000, step=0.0):
    c = close * (1 + step) ** np.arange(len(dates))
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": float(vol)}, index=pd.DatetimeIndex(dates))


def sessions(end=T, n=300):
    return list(pd.bdate_range(end=end, periods=n))


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda s: None))


class FakePredictor:
    """Forecasts each series' last close times (1 + its own last-day return x 10)."""
    def __init__(self):
        self.calls = 0

    def predict_batch(self, dfs, xs, ys, pred_len, **kw):
        self.calls += 1
        assert len(dfs) <= k.CHUNK and all(len(d) == k.LOOKBACK for d in dfs) and all(len(y) == pred_len for y in ys)
        out = []
        for d, y in zip(dfs, ys):
            g = d["close"].iloc[-1] / d["close"].iloc[-2] - 1
            out.append(pd.DataFrame({"close": [d["close"].iloc[-1] * (1 + 10 * g)] * pred_len}, index=y))
        return out


def test_clean_series_cuts_at_last_long_gap_and_drops_zero_volume():
    d = list(pd.bdate_range("2026-01-01", periods=10)) + list(pd.bdate_range("2026-03-01", periods=5))
    df = bars(d)
    df.iloc[-1, df.columns.get_loc("v")] = 0
    s = k.clean_series(df)
    assert len(s) == 4 and s.index[0] == pd.Timestamp("2026-03-02")


def test_universe_ranks_by_median_dollar_volume_and_filters_price_and_today():
    d = sessions(n=30)
    raw = {"BIG": bars(d, 20, 10_000), "MID": bars(d, 20, 5_000), "CHEAP": bars(d, 4, 1_000_000),
           "STALE": bars(d[:-1], 20, 1_000_000)}
    u = k.universe(raw, T, n=1)
    assert u["symbol"].tolist() == ["BIG"]
    assert set(k.universe(raw, T)["symbol"]) == {"BIG", "MID"}


def test_model_inputs_need_256_clean_bars_ending_on_t():
    adj = {"OK": bars(sessions(n=300)), "SHORT": bars(sessions(n=100)), "OLD": bars(sessions(T - pd.Timedelta(days=3)))}
    inputs, dropped = k.model_inputs(adj, ["OK", "SHORT", "OLD", "MISSING"], T)
    assert list(inputs) == ["OK"] and len(inputs["OK"]) == k.LOOKBACK
    assert sorted(dropped) == ["MISSING", "OLD", "SHORT"]
    assert list(inputs["OK"].columns) == ["open", "high", "low", "close", "volume"]


def test_model_inputs_drop_series_with_a_data_break():
    d = sessions(n=300)
    jump = bars(d)
    jump.iloc[-10:, jump.columns.get_loc("c")] = 4.0   # -60% in one day, e.g. an unadjusted spin-off
    inputs, dropped = k.model_inputs({"OK": bars(d), "JUMP": jump}, ["OK", "JUMP"], T)
    assert list(inputs) == ["OK"] and dropped == ["JUMP"]


def test_forecast_chunks_and_pick_is_deterministic():
    d = sessions()
    inputs = {f"S{i:02d}": k.model_inputs({"x": bars(d, 10, 100, step=0.001 * (i - 25))}, ["x"], T)[0]["x"]
              for i in range(50)}
    fp = FakePredictor()
    fc = k.forecast(inputs, fp, 1)
    assert fp.calls == 2 and len(fc) == 50
    top, ctrl = k.pick(fc, T)
    assert top[0] == "S49" and top == [f"S{i:02d}" for i in range(49, 29, -1)]
    assert ctrl == k.pick(fc, T)[1] and len(set(ctrl)) == 20 and set(ctrl) <= set(fc)


def _fake_fetch(n=40):
    d = sessions()
    adj = {f"S{i:02d}": bars(d, 10 + i, 1000, step=0.0005 * (i - 20)) for i in range(n)}
    u = pd.DataFrame({"symbol": list(adj), "dollar_volume": 1.0, "close": [10.0 + i for i in range(n)]})
    return lambda t, now: (u, adj, n + 5)


def _fake_load(kronos_dir, pins):
    return FakePredictor(), {**pins, k.MODEL: pins.get(k.MODEL, "m1"), k.TOKENIZER: pins.get(k.TOKENIZER, "t1")}


def test_run_logs_once_on_week_end_and_pins_revisions(tmp_path):
    now = pd.Timestamp("2026-10-17 03:30", tz="UTC")
    kw = dict(fetch=_fake_fetch(), load=_fake_load, session=lambda n: (T, True))
    rec = k.run(tmp_path, "unused", now=now, **kw)
    assert rec["universe_size"] == 40 and len(rec["picks"]) == 20 and rec["hf_revisions"][k.MODEL] == "m1"
    assert json.loads((tmp_path / "kronos" / "pins.json").read_text())[k.MODEL] == "m1"
    led = Ledger(tmp_path / "kronos")
    assert led.has("K3_picks", "2026-10-16") and led.verify() is not False
    assert k.run(tmp_path, "unused", now=now, **kw) is None   # already logged


def test_run_skips_mid_week_and_dry_run_logs_nothing(tmp_path):
    kw = dict(fetch=_fake_fetch(), load=_fake_load, session=lambda n: (T, False))
    assert k.run(tmp_path, "unused", **kw) is None
    rec = k.run(tmp_path, "unused", dry_run=True, **kw)
    assert rec is not None and (tmp_path / "kronos" / "dryrun-2026-10-16.json").exists()
    assert not Ledger(tmp_path / "kronos").has("K3_picks", "2026-10-16")
    assert not (tmp_path / "kronos" / "pins.json").exists()


def test_score_window_entry_open_to_21st_close_and_early_end():
    s = sessions(n=60) + list(pd.bdate_range(T + pd.Timedelta(days=1), periods=30))
    rise = bars(s, 10, 100, step=0.01)
    after = [x for x in s if x > T]
    want = rise.loc[after[20], "c"] / rise.loc[after[0], "o"] - 1
    assert k._window(rise, T, s) == pytest.approx(want)
    ends = rise[rise.index <= after[5]]
    assert k._window(ends, T, s) == pytest.approx(rise.loc[after[5], "c"] / rise.loc[after[0], "o"] - 1)
    assert k._window(rise[rise.index <= T], T, s) is None


def test_score_weeks_and_summary():
    s = sessions(n=60) + list(pd.bdate_range(T + pd.Timedelta(days=1), periods=30))
    frames = {"UP": bars(s, 10, 100, step=0.01), "FLAT": bars(s, 10, 100)}
    recs = [{"as_of": "2026-10-16", "picks": [{"symbol": "UP"}], "control": ["FLAT"], "universe": ["UP", "FLAT"]},
            {"as_of": f"{s[-5]:%Y-%m-%d}", "picks": [{"symbol": "UP"}], "control": ["FLAT"], "universe": ["UP", "FLAT"]}]
    w = k.score_weeks(recs, frames, s)
    assert len(w) == 1   # the second week has no full window yet
    up = 1.01 ** 20 - 1
    assert w["excess"].iloc[0] == pytest.approx(up - up / 2 - k.COST)
    many = pd.DataFrame({"excess": np.r_[np.full(10, 0.01), np.full(10, 0.03)], "control": 0.0, "universe": 0.0})
    out = k.summary(many)
    assert out["weeks"] == 20 and out["mean"] == pytest.approx(0.02) and out["ci"][0] > 0
