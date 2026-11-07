import numpy as np
import pandas as pd

from tracks.main import boom1_study as B
from tracks.main import boom2_study as B2
from tracks.news import nx3_study as X


def test_first_touch():
    lab, px = B2.first_touch([10.5, 10.9], [9.9, 9.95], [10, 10.8], 10.0, 0.08)
    assert lab == 1 and np.isclose(px, 10.8)
    lab, px = B2.first_touch([10.1, 10.0], [9.1, 9.0], [10, 9.5], 10.0, 0.08)
    assert lab == -1 and np.isclose(px, 9.2)
    assert B2.first_touch([10.9], [9.1], [10], 10.0, 0.08)[0] == 2          # both in one bar: ambiguous
    assert B2.first_touch([10.1], [9.9], [10.05], 10.0, 0.08) == (0, 10.05)


def minute_bars(day, o, drift, rng):
    t = pd.date_range(day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=15, minutes=59), freq="1min")
    c = o * np.exp(np.cumsum(rng.normal(drift, 0.002, len(t))))
    return pd.DataFrame({"t": t, "o": c, "h": c * 1.001, "l": c * 0.999, "c": c, "v": 1000.0})


def test_pick_row_sees_premarket_only_to_914():
    rng = np.random.default_rng(0)
    day = pd.Timestamp("2021-03-01")
    b = minute_bars(day, 20.0, 0.0, rng)
    r = B2.pick_row(b, 19.0, day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=16), 1e6)
    pm = b[b["t"] <= day + pd.Timedelta(hours=9, minutes=14)]["c"].iloc[-1]
    assert np.isclose(r["pm_last"], pm) and np.isclose(r["pm_gap"], pm / 19.0 - 1)
    assert r["open"] == b[b["t"] == day + pd.Timedelta(hours=9, minutes=30)]["o"].iloc[0]


def test_boom2_run_end_to_end(tmp_path, monkeypatch):
    cache = tmp_path / "nx1"
    d = B2.bdir(cache)
    (d / "minutes").mkdir(parents=True)
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2019-01-02", "2021-12-31")
    rows, mins = [], []
    for day in days:
        for i in range(20):
            drift = rng.choice([-1, 1]) * 0.0004
            news = float(rng.random() < 0.3)
            b = minute_bars(day, 20.0, drift, rng)
            rec = B2.pick_row(b, 20.0 * (1 - 3 * drift * 100), day + pd.Timedelta(hours=9, minutes=30),
                              day + pd.Timedelta(hours=16), 1e6)
            mins.append({"symbol": f"S{i}", "nxt": day} | rec)
            row = {"symbol": f"S{i}", "d": day - pd.Timedelta(days=1), "nxt": day, "c": 20.0, "dv20": 1e6,
                   "score": rng.random(), "pre_news": news, "pre_offering": float(rng.random() < 0.005),
                   "ao1": 20.0, "ah1": 21.0, "al1": 19.0, "boom": 1}
            row |= {k: rng.normal() for k in B.INPUTS if k not in row}
            rows.append(row)
    X.N.atomic_parquet(pd.DataFrame(rows), d / "picks.parquet")
    m = pd.DataFrame(mins)
    for p, g in m.groupby(m["nxt"].dt.to_period("M")):
        X.N.atomic_parquet(g, d / "minutes" / f"m_{p}.parquet")
    monkeypatch.setattr(B2, "side_d6", lambda tr, te: (np.sign(te["pm_gap"].to_numpy(float)), None, {"gap": 1.0}))
    rep = B2.run(cache, tmp_path / "out")
    assert set(rep["arms"]) == {"D1-NEWS", "D1-NONEWS", "D2", "D3", "D4", "D6"}
    assert rep["arms"]["D3"]["calls"] > 5000 and rep["verdicts"]["D2"] == "too few, not judged"
    assert 0 <= rep["d0"]["all"]["up_first"] <= 1 and (tmp_path / "out" / "BOOM2_walkforward.json").exists()
