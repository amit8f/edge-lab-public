"""Glue check on a tiny synthetic cache: W1a's events stage and W1b's signals stage (no network)."""
import numpy as np
import pandas as pd

from tracks.main import w1b_volume as B
from tracks.news import w1_data as D
from tracks.news import w1a_leaks as W

DAYS = pd.bdate_range("2016-01-04", "2016-06-30")
CAL = pd.DataFrame({"date": DAYS, "open": "09:30", "close": "16:00"})


def _frame(sym, spike):
    n = len(DAYS)
    rng = np.random.default_rng(len(sym))
    v = np.exp(rng.normal(np.log(2e5), 0.1, n))
    c = np.full(n, 10.0 if spike is not None else 10.4)       # XYZ closes near its high every day (control pool)
    if spike is not None:
        v[spike], c[spike] = 5e6, 10.4
    f = pd.DataFrame({"o": 10.0, "h": 10.5, "l": 9.5, "c": c, "v": v}, index=DAYS)
    return f, f.copy()


def _cache(tmp_path):
    rows = []
    for sym, spike in (("ABC", 80), ("XYZ", None)):
        raw, adj = _frame(sym, spike)
        rows.append(D.daily_rows(sym, raw, adj))
    u = pd.concat(rows, ignore_index=True)
    u["cik"] = np.where(u["symbol"] == "ABC", 1.0, 2.0)
    (tmp_path / "w1" / "daily").mkdir(parents=True)
    u.to_parquet(tmp_path / "w1" / "daily" / "d_0000.parquet")
    nd = tmp_path / "g1" / "news"
    nd.mkdir(parents=True)
    for per in pd.period_range(D.NEWS_FIRST, D.NEWS_LAST, freq="M"):
        n = pd.DataFrame(columns=["id", "created_at", "headline", "symbols"])
        if str(per) == "2016-03":
            n = pd.DataFrame({"id": [7, 8], "created_at": ["2016-03-15T15:00:00Z", "2016-03-16T15:00:00Z"],
                              "headline": ["Betaville Report Cites Takeover Interest In ABC", "XYZ Earnings Beat"],
                              "symbols": ["ABC", "XYZ"]})
        n.to_parquet(nd / f"news_{per}.parquet")
    (tmp_path / "w1" / "subs").mkdir(parents=True)
    pd.DataFrame({"cik": [2], "accession": ["a"], "form": ["10-Q"], "filed": ["2016-05-02"], "items": [[]]}).to_parquet(
        tmp_path / "w1" / "subs" / "s_0000.parquet")
    return tmp_path


def test_w1a_events_stage(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    monkeypatch.setattr(D, "calendar", lambda *a, **k: CAL)
    W.stage_events(cache, tmp_path / "out")
    ev = pd.read_parquet(cache / "w1" / "w1a_events.parquet")
    assert list(ev["symbol"]) == ["ABC"] and ev["group"].iloc[0] == "small" and ev["sign"].iloc[0] == 1
    assert ev["prior_day"].iloc[0] == pd.Timestamp("2016-03-14") and ev["price_group"].iloc[0] == "main"
    assert ev["cik"].iloc[0] == 1 and np.isfinite(ev["ac"].iloc[0])
    assert (tmp_path / "out" / "W1a_coverage.csv").exists()


def test_w1b_signals_stage(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    monkeypatch.setattr(D, "calendar", lambda *a, **k: CAL)
    spy = pd.DataFrame({"ao": 100.0, "ac": 100.0}, index=DAYS)
    monkeypatch.setattr(B, "spy_daily", lambda: spy)
    B.stage_signals(cache)
    s = B.load_signals(cache, "signal")
    assert list(s["symbol"]) == ["ABC"] and s["date"].iloc[0] == DAYS[80] and s["side"].iloc[0] == 1
    assert s["entry_date"].iloc[0] == DAYS[81] and s["exit_date"].iloc[0] == DAYS[80 + B.HOLD]
    pool = B.load_signals(cache, "pool", 2016)
    assert len(pool) and set(pool["symbol"]) <= {"ABC", "XYZ"} and (pool["ctrl_side"] != 0).all()
