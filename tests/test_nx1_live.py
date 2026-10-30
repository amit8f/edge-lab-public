"""NX1 nightly scorer and shadow books on synthetic data (no network, no FinBERT, no trained model)."""
import json

import numpy as np
import pandas as pd

from tracks.news import nx1_live as L
from tracks.news import nx1_study as S

DAYS = pd.bdate_range("2025-01-01", "2026-11-30")
CAL = pd.DataFrame({"date": DAYS, "open": "09:30", "close": "16:00"})


def test_batch_day_and_window():
    now = pd.Timestamp("2026-10-22 22:30", tz="America/New_York")
    d = L.batch_day(now, DAYS)
    assert d == pd.Timestamp("2026-10-22")
    assert L.batch_day(pd.Timestamp("2026-10-22 21:59", tz="America/New_York"), DAYS) == pd.Timestamp("2026-10-21")
    start, end = L.window(pd.Timestamp("2026-10-19"), DAYS)       # a Monday: window opens Friday 10pm
    assert start == pd.Timestamp("2026-10-16 22:00", tz="America/New_York")
    assert end == pd.Timestamp("2026-10-19 22:00", tz="America/New_York")


def test_settle_fills_stops_and_exits():
    b = L.new_book()
    b["pending"] = [{"symbol": "AAA", "sign": 1, "shares": 10, "ref_px": 50, "entry_day": "2026-10-23",
                     "exit_day": "2026-10-27"},
                    {"symbol": "BBB", "sign": -1, "shares": 10, "ref_px": 50, "entry_day": "2026-10-23",
                     "exit_day": "2026-10-27"}]
    f = L.settle(b, {"AAA": (50, 51, 49, 50.5), "BBB": (50, 56, 49, 52)}, "2026-10-23")
    assert len(f) == 2 and len(b["pos"]) == 1 and b["closed"][0]["why"] == "stop"
    assert np.isclose(b["closed"][0]["exit_px"], 55.0) and b["closed"][0]["pnl"] < -50
    L.settle(b, {"AAA": (51, 52, 50, 51)}, "2026-10-26")
    assert len(b["pos"]) == 1
    L.settle(b, {"AAA": (51, 53, 50, 52)}, "2026-10-27")
    t = b["closed"][-1]
    assert t["why"] == "3 sessions" and t["exit_px"] == 52 and 0 < t["pnl"] < 20
    assert np.isclose(b["cash"], L.BOOK_CASH + sum(x["pnl"] for x in b["closed"]))


def test_gap_through_the_stop_fills_at_the_open():
    b = L.new_book()
    b["pos"] = [{"symbol": "AAA", "sign": 1, "shares": 10, "entry_px": 50.0, "entry_cost": 0.35, "stop": 45.0,
                 "entry_day": "2026-10-23", "exit_day": "2026-10-27"}]
    b["cash"] -= 500.35
    L.settle(b, {"AAA": (40, 41, 39, 40)}, "2026-10-26")
    assert b["closed"][0]["exit_px"] == 40 and b["closed"][0]["why"] == "stop (gap)"


def test_choose_respects_shortable_and_matches_control_sides():
    ev = pd.DataFrame({"symbol": list("ABCDEF"), "p_long": [0.8, 0.7, 0.1, 0.1, 0.2, 0.3],
                       "p_short": [0.1, 0.1, 0.8, 0.7, 0.1, 0.3], "cls": [2, 2, 0, 0, 1, 1],
                       "short_ok": [True, True, False, True, False, True]})
    p, c = L.choose(ev, np.random.default_rng(0))
    assert p["symbol"].tolist() == ["A", "B", "D"] and p["side"].tolist() == [1, 1, -1]
    assert c["side"].tolist() == [1, 1, -1] and c["symbol"].is_unique
    assert bool(ev.set_index("symbol").at[c["symbol"].iloc[-1], "short_ok"])


def test_plan_sizes_at_the_shadow_limits():
    b = L.new_book()
    ch = pd.DataFrame({"symbol": ["AAA", "BBB"], "side": [1, -1], "log_px": np.log([20.0, 20.0]),
                       "log_dv": np.log([5e6, 5e6])})
    out = L.plan(b, ch, {}, "2026-10-23", "2026-10-27")
    assert out[0]["shares"] == 25 and out[1]["shares"] == 12          # 10% long, 5% short of $5,000


class FakeApi:
    def __init__(self):
        self.syms = ["N1", "N2", "N3", "Q1", "Q2", "Q3", "Q4", "SPY"]

    def calendar(self, start, end):
        return CAL[(CAL["date"] >= pd.Timestamp(start)) & (CAL["date"] <= pd.Timestamp(end))].copy()

    def assets(self):
        a = pd.DataFrame({"symbol": self.syms[:-1], "excl_reason": None, "tradable": True, "shortable": True,
                          "easy_to_borrow": [True, False, True, True, True, True, True]})
        return a

    def daily(self, symbols, start, end, adjustment):
        idx = DAYS[(DAYS >= pd.Timestamp(start)) & (DAYS <= pd.Timestamp(end))]
        out = {}
        for s in symbols:
            rng = np.random.default_rng(abs(hash(s)) % 1000)
            c = 30 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
            out[s] = pd.DataFrame({"o": c * 0.99, "h": c * 1.02, "l": c * 0.98, "c": c, "v": 2e5}, index=idx)
        return out

    def news(self, start, end):
        t = lambda x: pd.Timestamp(x, tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        return pd.DataFrame({"id": [1, 2, 3, 4, 5],
                             "created_at": [t("2026-10-22 08:00"), t("2026-10-22 09:00"), t("2026-10-22 12:00"),
                                            t("2026-10-22 23:00"), t("2026-10-21 12:00")],
                             "headline": ["N1 Earnings Beat", "N2 Downgraded", "N3 Raises Guidance", "N1 later",
                                          "Q1 old news"],
                             "symbols": ["N1", "N2", "N3", "N1", "Q1"]})

    def finbert(self, meta_file):
        return lambda texts: np.array([[0.6, 0.2, 0.2]] * len(texts))

    def model(self):
        return lambda X: np.tile([0.2, 0.3, 0.5], (len(X), 1))

    def nx2(self):
        return None


class FakeNX2(FakeApi):
    def nx2(self):
        # effect follows the first news input, which differs between real and no-news rows
        return (lambda X: X[:, 0] * 10.0), {"fire_cut": 0.0, "sha256": {"news": "y"}}


def test_run_logs_a_night_and_books(tmp_path, monkeypatch):
    m = tmp_path / "m.json"
    m.write_text("{}")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"model_sha256": "x", "finbert": {"revision": "abc"}}))
    monkeypatch.setattr(L, "MODEL", m)
    monkeypatch.setattr(L, "META", meta)
    now = pd.Timestamp("2026-10-22 23:15", tz="America/New_York").tz_convert("UTC")
    out = L.run(tmp_path / "state", FakeApi(), now)
    kinds = [e["kind"] for e in out["events"]]
    assert kinds.count("real") == 3 and kinds.count("placebo") == 3
    assert {e["symbol"] for e in out["events"] if e["kind"] == "placebo"} <= {"Q2", "Q3", "Q4"}   # Q1 had news
    assert [p["symbol"] for p in out["picks"]] and all(p["side"] == 1 for p in out["picks"])
    assert len(out["control"]) == len(out["picks"])
    raw = (tmp_path / "state" / "nx1" / "nights" / "2026-10-22.json").read_text()
    assert "Earnings" not in raw and "Downgraded" not in raw
    books = json.loads((tmp_path / "state" / "nx1" / "books.json").read_text())
    assert books["picks"]["pending"][0]["entry_day"] == "2026-10-23"
    assert L.run(tmp_path / "state", FakeApi(), now) is None                 # the same night only once
    nxt = pd.Timestamp("2026-10-23 23:15", tz="America/New_York").tz_convert("UTC")
    out2 = L.run(tmp_path / "state", FakeApi(), nxt)
    assert out2["fills"]["picks"] and out2["books"]["picks"]["open"] >= 1
    assert out2["batch_headlines"] == 1                                     # the 23:00 article joins Oct 23's batch


def test_nx2_near_miss_book_is_logged_beside_nx1(tmp_path, monkeypatch):
    m = tmp_path / "m.json"
    m.write_text("{}")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"model_sha256": "x", "finbert": {"revision": "abc"}}))
    monkeypatch.setattr(L, "MODEL", m)
    monkeypatch.setattr(L, "META", meta)
    now = pd.Timestamp("2026-10-22 23:15", tz="America/New_York").tz_convert("UTC")
    out = L.run(tmp_path / "state", FakeNX2(), now)
    rec = out["nx2"]
    assert rec["as_of"] == "2026-10-22" and rec["hold_sessions"] == 3 and rec["events"] == 3
    assert len(rec["control"]) == len(rec["picks"]) and all(np.isfinite(p["cost"]) for p in rec["picks"])
    assert all(p["side"] == "long" or p["symbol"] != "N2" for p in rec["picks"])     # N2 isn't easy to borrow
    led = (tmp_path / "state" / "nx1" / "ledger.jsonl").read_text()
    assert '"kind": "NX2_picks"' in led and (tmp_path / "state" / "nx1" / "nx2" / "2026-10-22.json").exists()
