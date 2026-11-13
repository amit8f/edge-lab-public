import json

import numpy as np
import pandas as pd

from core import chart_features as C
from tracks.main import boom1_study as B
from tracks.main import boom_short as BS


def _daily(sym, n=320, seed=0, px=20.0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2025-01-02", periods=n)
    c = px * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    o = c * (1 + rng.normal(0, 0.01, n))
    h, l = np.maximum(o, c) * 1.02, np.minimum(o, c) * 0.98
    v = rng.integers(200_000, 400_000, n).astype(float)
    return pd.DataFrame({"symbol": sym, "date": d, "o": o, "h": h, "l": l, "c": c, "v": v,
                         "ao": o, "ah": h, "al": l, "ac": c, "av": v})


def test_universe_rows_prev_close_only_and_filters():
    daily = pd.concat([_daily("AAA", seed=1), _daily("BBB", seed=2, px=2.0)], ignore_index=True)
    spy = _daily("SPY", seed=3, px=400)
    prev = daily["date"].iloc[-1]
    u = BS.universe_rows(daily, spy, prev)
    assert list(u["symbol"]) == ["AAA"]                  # BBB is under $5
    assert set(C.FEATURES + C.MARKET) <= set(u.columns)
    # the row equals chart_features on bars up to prev (nothing later exists here, so drop the last day and compare)
    u2 = BS.universe_rows(daily[daily["date"] < prev], spy, daily["date"].iloc[-2])
    assert np.isclose(u2["r1"].iloc[0], daily.loc[daily["symbol"] == "AAA", "c"].pct_change().iloc[-2], rtol=1e-4)


def test_news_inputs_and_premarket():
    open_t = pd.Timestamp("2026-11-09 09:30")
    ev = pd.DataFrame({"symbol": ["AAA", "BBB"], "t": [open_t, open_t + pd.Timedelta(minutes=5)], "open_t": open_t,
                       "n_articles": [2, 1], "fb_neg": [0.7, 0.1], "fb_pos": [0.1, 0.8], "lm_net": [-1, 1],
                       "fb_max_neg": [0.9, 0.1], "fb_max_pos": [0.2, 0.9], "f_offering": [1, 0]})
    ni = BS.news_inputs(ev)
    assert list(ni["symbol"]) == ["AAA"] and ni["pre_offering"].iloc[0] == 1.0
    u = BS.with_news(pd.DataFrame({"symbol": ["AAA", "CCC"]}), ni)
    assert u.loc[1, "pre_news"] == 0 and u.loc[1, "pre_articles"] == 0 and np.isnan(u.loc[1, "pre_fb_neg"])
    day = open_t.normalize()
    b = pd.DataFrame({"t": [day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=9, minutes=14),
                            day + pd.Timedelta(hours=9, minutes=20)], "c": [10.0, 11.0, 50.0], "v": [100.0, 100.0, 1e6]})
    pm = BS.premarket(b, 10.0, 2100.0, day)
    assert np.isclose(pm["pm_gap"], 0.1) and np.isclose(pm["pm_dv"], 1.0) and pm["pm_bars"] == 2
    assert BS.premarket(None, 10.0, 1.0, day) == {"pm_gap": 0.0, "pm_dv": 0.0, "pm_bars": 0}


def test_choose_cut_borrow_cap_and_control():
    p = pd.DataFrame({"symbol": [f"S{i:02d}" for i in range(20)], "d6": np.linspace(-0.5, 0.5, 20)})
    sh, ctl, blocked = BS.choose(p, -0.39, {"S00", "S02", "S03"} | {f"S{i:02d}" for i in range(5, 20)}, 7)
    assert sh == ["S00", "S02"] and blocked == ["S01"]
    assert len(ctl) == 2 and not set(ctl) & set(sh)
    assert BS.choose(p, -0.39, set(), 7)[:2] == ([], [])
    many = BS.choose(p, 1.0, set(p["symbol"]), 7)[0]
    assert len(many) == BS.MAX_SHORTS and many[0] == "S00"


def test_settle_rows_barrier_and_costs():
    day = pd.Timestamp("2026-11-09")
    o_t, c_t = day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=16)
    t = pd.date_range(o_t, c_t - pd.Timedelta(minutes=1), freq="1min")
    px = np.full(len(t), 10.0)
    px[60:] = 9.0                                      # -10%: the -8% barrier is touched first
    b = pd.DataFrame({"t": t, "o": px, "h": px * 1.001, "l": px * 0.999, "c": px, "v": 1000.0})
    b.loc[0, "o"] = 10.0
    r = BS.settle_rows({"AAA": b}, {"AAA": (10.0, 1e6), "ZZZ": (5.0, 1e6)}, o_t, c_t)
    assert r["ZZZ"]["status"] == "no bars"
    assert r["AAA"]["lab"] == -1 and np.isclose(r["AAA"]["exit"], 9.2)
    assert 0.08 - 0.02 < r["AAA"]["net"] < 0.08


class _M1:
    def predict_proba(self, X):
        s = X[:, B.INPUTS.index("vol20")]
        return np.c_[1 - s, s]


class _M2:
    classes_ = [-1, 0, 1]

    def predict_proba(self, X):
        g = X[:, -3]                                   # pm_gap: higher gap -> more bearish here
        dn = np.clip(0.3 + g, 0, 0.9)
        return np.c_[dn, 1 - dn - 0.1, np.full(len(g), 0.1)]


class _Api:
    def __init__(self, daily):
        self.daily_rows = daily

    def daily(self, symbols, first, last):
        return self.daily_rows[self.daily_rows["date"] <= last]

    def minutes(self, symbols, start, end):
        day = pd.Timestamp(start).normalize()
        out = {}
        for i, s in enumerate(symbols):
            t = list(pd.date_range(day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=9, minutes=14), freq="1min"))
            px = [100.0 * (1 + 0.01 * i)] * len(t)
            if end > day + pd.Timedelta(hours=10):
                rt = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=15, minutes=59), freq="1min")
                t += list(rt)
                px += list(np.linspace(100, 95, len(rt)))
            px = np.array(px)
            out[s] = pd.DataFrame({"t": t, "o": px, "h": px * 1.001, "l": px * 0.999, "c": px, "v": 1000.0})
        return out

    def news(self, start, end):
        return pd.DataFrame()

    def finbert(self, meta):
        return None

    def assets(self):
        syms = self.daily_rows["symbol"].unique()
        return pd.DataFrame({"symbol": syms, "shortable": True, "easy_to_borrow": True})

    def calendar(self, start, end):
        d = pd.bdate_range("2025-01-02", "2026-12-31")
        return pd.DataFrame({"date": d, "open": "09:30", "close": "16:00", "early_close": False})


def test_run_decides_before_open_then_settles(tmp_path, monkeypatch):
    syms = [f"S{i:03d}" for i in range(120)]
    daily = pd.concat([_daily(s, n=330, seed=i) for i, s in enumerate(syms)] + [_daily("SPY", n=330, seed=999, px=400)],
                      ignore_index=True)
    days = pd.bdate_range("2025-01-02", periods=330)
    today = days[-1] + pd.offsets.BDay(1)
    man = {"cut": -0.2, "sha256": {"boom1.cbm": "x"}}
    monkeypatch.setattr(BS, "load_frozen", lambda: (man, _M1(), _M2(), syms))
    api = _Api(daily)
    clock = iter([today + pd.Timedelta(hours=8, minutes=50)] + [today + pd.Timedelta(hours=9, minutes=28)] * 50)
    assert BS.run(tmp_path, api=api, clock=lambda: next(clock), sleep=lambda s: None) == 0
    rec = json.loads((tmp_path / "boomshort" / "ledger" / "days" / f"{today:%Y-%m-%d}.json").read_text())
    assert rec["status"] == "decided" and len(rec["picks"]) == 20 and rec["universe"] >= 100
    assert rec["shorts"] and all(p["pm_bars"] == 15 for p in rec["picks"])
    assert len(rec["control"]) == len(rec["shorts"]) and not set(rec["control"]) & set(rec["shorts"])
    monkeypatch.setattr(BS, "now_et", lambda: today + pd.Timedelta(hours=17))
    BS.run(tmp_path, settle_only=True, api=api, clock=lambda: today + pd.Timedelta(hours=17))
    st = json.loads((tmp_path / "boomshort" / "ledger" / "settle" / f"{today:%Y-%m-%d}.json").read_text())
    assert st["book_mean"] is not None and set(st["trades"]) == set(rec["shorts"])
    rep = json.loads((tmp_path / "boomshort" / "report.json").read_text())
    assert rep["closed_trades"] == len(rec["shorts"])


def test_run_after_open_logs_missed(tmp_path, monkeypatch):
    man = {"cut": -0.2, "sha256": {}}
    monkeypatch.setattr(BS, "load_frozen", lambda: (man, _M1(), _M2(), ["S000"]))
    api = _Api(_daily("SPY", n=40))
    today = pd.Timestamp("2025-03-03")
    monkeypatch.setattr(BS, "now_et", lambda: today + pd.Timedelta(hours=9, minutes=45))
    BS.run(tmp_path, api=api, clock=lambda: today + pd.Timedelta(hours=9, minutes=45))
    rec = json.loads((tmp_path / "boomshort" / "ledger" / "days" / "2025-03-03.json").read_text())
    assert rec["status"] == "missed"


def test_cached_scorer_scores_each_text_once():
    calls = []

    def fb(texts):
        calls.append(list(texts))
        return np.array([[len(t), 0.0, 1.0] for t in texts])
    run = BS.cached_scorer(fb)
    a = run(["x", "yy", "x"])
    b = run(["yy", "zzz"])
    assert calls == [["x", "yy"], ["zzz"]]
    assert a[:, 0].tolist() == [1, 2, 1] and b[:, 0].tolist() == [2, 3]


def test_pre_open_events_merges_warm_news_and_fetches_only_the_tail(tmp_path, monkeypatch):
    ses = {"date": pd.Timestamp("2026-11-12"), "prev_close_t": pd.Timestamp("2026-11-11 16:00"),
           "open_t": pd.Timestamp("2026-11-12 09:30")}
    seen = {}

    class Api:
        def news(self, start, end):
            seen["start"] = start
            return pd.DataFrame({"id": [2, 3], "headline": ["b2", "c"]})
    old = pd.DataFrame({"id": [1, 2], "headline": ["a", "b"]})
    got = {}
    monkeypatch.setattr(BS.N, "headline_rows", 
                        lambda news, fb: (got.setdefault("n", news), pd.DataFrame({"t": pd.to_datetime([])}))[1])
    until = pd.Timestamp("2026-11-12 09:00")
    BS.pre_open_events(Api(), tmp_path, ses, pd.Timestamp("2026-11-12 09:28"), None, (old, until))
    assert seen["start"] == pd.Timestamp("2026-11-12 08:55", tz="America/New_York").tz_convert("UTC")
    assert sorted(got["n"]["id"]) == [1, 2, 3] and got["n"].set_index("id").loc[2, "headline"] == "b2"
