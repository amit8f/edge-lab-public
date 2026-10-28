"""NX3 live scorer and shadow book on synthetic data (no network, no FinBERT; the model is a stand-in)."""
import json

import numpy as np
import pandas as pd
import requests

from core.ledger import Ledger
from tracks.news import nx3_live as L
from tracks.news import nx3_study as S

DAYS = pd.bdate_range("2025-01-01", "2026-11-30")
CAL = pd.DataFrame({"date": DAYS, "open": "09:30", "close": "16:00"})
D = pd.Timestamp("2026-10-26")                     # a Monday
OPEN = D + pd.Timedelta(hours=9, minutes=30)


def bars(prices, start=OPEN, vol=100):
    return pd.DataFrame({"t": [start + pd.Timedelta(minutes=i) for i in range(len(prices))],
                         "o": prices, "c": prices, "v": vol})


def test_intraday_matches_training_trade_on_the_same_prices():
    b = bars([10.0, 10.2, 10.4, 10.6, 10.8, 11.0] * 10)
    t = OPEN + pd.Timedelta(minutes=40)
    tr = S.trade(b, t, OPEN, 9.5, 12.0, 1e7, 9.5)
    x = L.intraday(b, t, OPEN, tr["entry"], 9.5, 1e7, 1.0)
    for k in ("mins_open", "r_open", "r_react", "rv30"):
        assert np.isclose(x[k], tr[k]), k
    assert np.isclose(L.intraday(b, t, OPEN, tr["entry"], 9.5, 1e7, 0.05)["rv30"], tr["rv30"] / 0.05)
    pre = L.intraday(b, OPEN, OPEN, 10.3, 9.5, 1e7, 0.05)        # pre-open news: price at t is the open
    assert pre["mins_open"] == 0 and np.isclose(pre["r_open"], 10 / 9.5 - 1)
    assert np.isclose(pre["r_react"], 0.03) and np.isnan(pre["rv30"])


def test_due_splits_ready_and_missed():
    ev = pd.DataFrame({"key": ["a", "b", "c", "d"], "t": [OPEN, OPEN, OPEN + pd.Timedelta(minutes=30), OPEN]})
    now = OPEN + pd.Timedelta(minutes=5)
    ready, missed = L.due(ev, {"d"}, now)
    assert ready["key"].tolist() == ["a", "b"] and missed.empty
    ready, missed = L.due(ev, set(), OPEN + pd.Timedelta(minutes=11))
    assert ready.empty and missed["key"].tolist() == ["a", "b", "d"]


def test_take_quota_and_borrow():
    assert L.take(0.5, False, "A", set(), []) == (None, "not fired")
    assert L.take(0.5, True, "A", set(), [1, 1]) == (1, None)
    assert L.take(0.5, True, "A", set(), [1, 1, 1, -1]) == (None, "side full")
    assert L.take(-0.5, True, "A", set(), []) == (None, "not easy to borrow")
    assert L.take(-0.5, True, "A", {"A"}, [-1, -1]) == (-1, None)


def test_control_matches_sides_and_is_seeded():
    dec = [{"symbol": s} for s in "ABCDEFG"]
    picks = [{"symbol": "A", "side": 1}, {"symbol": "B", "side": -1}]
    c1, c2 = L.control(dec, picks, 7), L.control(dec, picks, 7)
    assert c1 == c2 and [x["side"] for x in c1] == [1, -1] and len({x["symbol"] for x in c1}) == 2
    assert L.control(dec[:1], picks, 7) == []


class FakeApi:
    """Two stocks with pre-open news (AAA good, BBB bad), one with news at 11:00 (CCC)."""

    def __init__(self):
        self.clock = OPEN - pd.Timedelta(minutes=2)
        self.settled = []

    def calendar(self, start, end):
        return CAL.copy()

    def assets(self):
        return pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "shortable": [True, True, True],
                             "easy_to_borrow": [True, True, False]})

    @staticmethod
    def refuse(symbols):
        """Alpaca refuses a whole request that names a symbol it doesn't list (a 400)."""
        bad = sorted(set(symbols) - {"AAA", "BBB", "CCC", "SPY"})
        if bad:
            raise requests.HTTPError(f"400 Client Error: Bad Request for symbols {bad}")

    def daily(self, symbols, start, end, adjustment):
        self.refuse(symbols)
        idx = DAYS[(DAYS >= pd.Timestamp(start)) & (DAYS <= pd.Timestamp(end))]
        rng = np.random.default_rng(1)
        out = {}
        for s in symbols:
            c = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
            out[s] = pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": 1e6}, index=idx)
        return out

    def iex_bars(self, symbols, start, end, timeframe):
        self.refuse(symbols)
        if timeframe == "1Day":
            return {s: pd.DataFrame({"t": [start], "o": [1.0], "c": [1.0], "v": [2.5e5]}) for s in symbols}
        n = int((min(pd.Timestamp(end), self.clock) - OPEN).total_seconds() // 60)
        return {s: bars([20.0] * max(n, 0)) for s in symbols}

    def last_trades(self, symbols):
        self.refuse(symbols)
        return {s: (20.2, str(self.clock)) for s in symbols}

    def news(self, start, end):
        rows = [(1, "2026-10-26T12:00:00Z", "AAA raises guidance", "AAA"),
                (2, "2026-10-26T12:30:00Z", "BBB misses estimates, cuts outlook", "BBB"),
                (3, "2026-10-26T15:00:00Z", "CCC wins contract", "CCC"),
                (5, "2026-10-26T12:05:00Z", "XYZ beats estimates", "TSX:XYZ"),   # listed abroad only
                (4, "2026-10-23T15:00:00Z", "AAA older", "AAA")]
        n = pd.DataFrame(rows, columns=["id", "created_at", "headline", "symbols"])
        tt = pd.to_datetime(n["created_at"], utc=True)
        return n[(tt >= start) & (tt < end) & (tt <= pd.Timestamp(self.clock, tz="America/New_York"))]

    def finbert(self, meta_file):
        return lambda texts: np.tile([0.6, 0.2, 0.2], (len(texts), 1))

    def effect(self):
        def eff(X):
            fb_neg = X[:, S.INPUTS.index("fb_neg")]
            lm_neg = X[:, S.INPUTS.index("lm_neg")]
            return np.where(lm_neg > 0, -1.0, 1.0) * (1 + 0 * fb_neg)
        return eff

    def settle_trades(self, ev, cal):
        self.settled.append(ev)
        return S.month_trades(ev, S.sessions(cal), None)     # training's code, on the fake SIP bars below


def test_a_day_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "daily_closes", lambda syms, first, last, h: {
        (s, d): [20.0, 20.0] for s in syms for d in DAYS[(DAYS >= first) & (DAYS <= last)]})
    monkeypatch.setattr(S, "day_bars", lambda syms, d, o, c, h: {s: bars([20.0] * 390, start=o) for s in syms})
    api = FakeApi()
    clock_steps = []

    def clock():
        return api.clock

    def sleep(s):
        api.clock += pd.Timedelta(seconds=30)
        clock_steps.append(api.clock)
    rc = L.run(tmp_path, 60 * 8, api=api, clock=clock, sleep=sleep)
    assert rc == 3
    led = Ledger(tmp_path / "nx3")
    rec = json.loads((tmp_path / "nx3" / led.entry("nx3_day", "2026-10-26")["file"]).read_text())
    dec = {x["symbol"]: x for x in rec["events"]}
    assert set(dec) == {"AAA", "BBB", "CCC", "TSX:XYZ"}
    assert dec["TSX:XYZ"]["status"] == "not in universe" and dec["AAA"]["status"] == "decided"
    assert dec["AAA"]["side"] == 1 and dec["AAA"]["t"].startswith("2026-10-26T09:30")
    assert dec["BBB"]["e"] < 0 and dec["BBB"]["side"] == -1             # bad news, easy to borrow: short
    assert dec["CCC"]["t"].startswith("2026-10-26T11:00") and dec["CCC"]["inputs"]["mins_open"] == 90
    assert pd.Timestamp(dec["CCC"]["decided_at"]) >= pd.Timestamp("2026-10-26 11:01")
    assert dec["AAA"]["inputs"]["novelty"] == 1.0                      # the Friday article
    assert len(rec["picks"]) == len(rec["control"]) == 3
    assert [p["side"] for p in rec["picks"]] == [p["side"] for p in rec["control"]]
    assert "headline" not in json.dumps(rec) and "raises guidance" not in json.dumps(rec)
    assert not led.has("nx3_settle", "2026-10-26")                    # exit is Tuesday's close
    # Tuesday evening: Monday settles
    api.clock = pd.Timestamp("2026-10-27 16:30")
    assert L.run(tmp_path, 1, api=api, clock=clock, sleep=sleep) == 3
    s = json.loads((tmp_path / "nx3" / led.entry("nx3_settle", "2026-10-26")["file"]).read_text())
    assert len(s["trades"]) == 2 * len(rec["picks"]) and s["pick_minus_control"] is not None
    assert {x["days"] for x in s["trades"]} == {2} and "bug days" in s["excluded"]   # D is one of them
    assert led.verify() == []
