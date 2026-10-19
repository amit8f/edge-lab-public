"""K6 bot B (combined design): features, the entry test, the coin twin's counts, one run per slot. Synthetic data only."""
import random

import pandas as pd

from tracks.main import ai_combo as b

DAY = "2026-10-19"
T = int(pd.Timestamp(f"{DAY} 10:30", tz="America/New_York").timestamp())


def minute_bars(p_old, p_now, vol_now=1000, vol_old=100):
    """Bars every minute for 70 minutes: the price 60 minutes ago is p_old, the last 30 minutes trade vol_now."""
    out = []
    for m in range(70):
        t = T - 3600 - 600 + m * 60
        c = p_old if t <= T - 3600 else p_now
        v = vol_now if t > T - 1800 else vol_old
        out.append({"t": t, "c": c, "v": v})
    return out


def test_features_return_and_relative_volume():
    f = b.features(minute_bars(100.0, 102.0), 102.0, T)
    assert abs(f["ret60"] - 0.02) < 1e-9
    assert f["rv"] > 1.5                       # heavy last 30 minutes against a quiet session pace
    assert b.features([], 1.0, T) is None


def test_entry_needs_tape_news_and_peers_to_agree():
    assert b.eligible({"ret60": 0.02, "rv": 2.0}, 1, 0.01, True)[0] == 1
    assert b.eligible({"ret60": 0.02, "rv": 2.0}, 0, 0.01, True) is None          # no news
    assert b.eligible({"ret60": 0.02, "rv": 2.0}, 1, -0.01, True) is None         # peers disagree
    assert b.eligible({"ret60": 0.02, "rv": 2.0}, 1, None, True) is None          # no peers priced
    assert b.eligible({"ret60": -0.02, "rv": 2.0}, 1, -0.01, False) is None       # not shortable
    assert b.eligible({"ret60": -0.02, "rv": 2.0}, 1, -0.01, True)[0] == -1


def test_news_and_peers_from_articles():
    arts = [{"t": T - 600, "symbols": ["AAA", "BBB"]}, {"t": T - 20000, "symbols": ["AAA", "CCC"]}]
    news2, peers = b.peer_and_news(arts, T, ["AAA", "BBB", "CCC", "DDD"])
    assert news2["AAA"] == 1 and news2["BBB"] == 1 and news2["CCC"] == 0
    assert peers["AAA"] == {"BBB", "CCC"} and peers["DDD"] == set()


def test_step_opens_up_to_two_and_twin_matches_counts():
    st = {"books": {"B": b.new_book(), "B-coin": b.new_book()}, "days": {}}
    uni = [f"S{i:02d}" for i in range(12)]
    prices = {s: 50.0 for s in uni}
    feats = {s: {"ret60": 0.02, "rv": 3.0 - i * 0.1} for i, s in enumerate(uni[:6])}
    news2 = {s: 1 for s in uni}
    peers = {s: {x for x in uni[:6] if x != s} for s in uni}
    rec = b.step(st, DAY, feats, news2, peers, set(), uni, prices, T, pd.Timestamp(T, unit="s", tz="UTC").tz_convert("America/New_York"), random.Random(1))
    opened = [f for f in rec["fills"] if f["book"] == "B" and f["action"] == "open"]
    assert len(opened) == 2 and all(f["side"] == 1 for f in opened)
    assert len(st["books"]["B-coin"]["pos"]) == 2 and all(p["side"] == 1 for p in st["books"]["B-coin"]["pos"].values())


def test_run_logs_once_per_slot(tmp_path, monkeypatch):
    from tracks.main import ai_trader as k2
    now = pd.Timestamp(f"{DAY} 15:47", tz="America/New_York").tz_convert("UTC")
    monkeypatch.setattr(k2, "market_window", lambda n: (now - pd.Timedelta(minutes=0), {now.tz_convert(b.ET).date(): (now - pd.Timedelta(hours=7), now + pd.Timedelta(hours=1))}))
    uni = {"stocks": [{"symbol": "AAA"}], "shortable": []}
    kw = dict(universe_fn=lambda n: uni, minute_fn=lambda d, s, p: {"AAA": minute_bars(10.0, 10.1)},
              latest_fn=lambda s: {"AAA": (10.1, now)}, news_fn=lambda t, s: [])
    assert b.run(tmp_path, now=now, **kw) is not None
    assert b.run(tmp_path, now=now + pd.Timedelta(minutes=2), **kw) is None


def test_run_calls_news_with_time_then_symbols(tmp_path, monkeypatch):
    # articles() takes (t, syms); a swapped call crashed the first market-open run on 2026-10-19
    from tracks.main import ai_trader as k2
    seen = {}
    def news(t, syms):
        seen["args"] = (t, list(syms))
        return []
    now = pd.Timestamp("2026-10-19T14:00:00Z")
    monkeypatch.setattr(k2, "market_window", lambda n: (now, {now.tz_convert(b.ET).date(): (now - pd.Timedelta(hours=4), now + pd.Timedelta(hours=3))}))
    uni = {"stocks": [{"symbol": "AAA"}], "shortable": []}
    b.run(tmp_path, now=now, universe_fn=lambda n: uni, minute_fn=lambda d, s, p: {"AAA": minute_bars(10.0, 10.1)},
          latest_fn=lambda s: {"AAA": (10.1, now)}, news_fn=news)
    assert isinstance(seen["args"][0], int) and "AAA" in seen["args"][1]
