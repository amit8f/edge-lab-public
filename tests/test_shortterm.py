"""Short-term boom models (M10, P3): synthetic data only."""
import json

import numpy as np
import pandas as pd
import pytest

from core import shortterm as st
from tracks.main.longterm import fundamentals as fu
from tracks.main.longterm.prices import align
from tests.test_longterm import _bars, _cf


def test_weekly_dates_are_last_session_of_each_week():
    cal = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-19"))   # a Friday holiday
    d = st.weekly_dates(cal, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    assert d == [pd.Timestamp(x) for x in ("2024-01-05", "2024-01-12", "2024-01-18", "2024-01-26", "2024-01-31")]


def test_features_values_labels_and_gaps():
    a, r, s = _bars(n=320, start="2020-01-01", seed=3)
    d = align(a, r, s)
    t = d.index[280]
    ends = st.horizon_ends(d.index, [t, d.index[310]], label_end=d.index[-1] + pd.Timedelta(days=1))
    f = st.features("X", d, [d.index[200], t, d.index[310]], ends)
    assert list(f["date"]) == [t, d.index[310]]                    # 200 lacks a full lookback
    row = f.iloc[0]
    c = d["c"]
    assert row["r21"] == pytest.approx(c.iloc[280] / c.iloc[259] - 1, rel=1e-5)
    assert row["mom"] == pytest.approx(c.iloc[259] / c.iloc[28] - 1, rel=1e-5)
    assert row["hi52"] == pytest.approx(c.iloc[280] / c.iloc[29:281].max(), rel=1e-5)
    assert row["ret21"] == pytest.approx(c.iloc[301] / d["o"].iloc[281] - 1, rel=1e-6) and not row["ret21_trunc"]
    assert row["ret5"] == pytest.approx(c.iloc[285] / d["o"].iloc[281] - 1, rel=1e-6)
    assert np.isnan(f.iloc[1]["ret21"])                            # window runs past the data: no label
    # a gap restarts the lookback, and a window that hits the gap is cut at the series' last close
    a, r, s = _bars(n=600, start="2020-01-01", seed=4, gap_at=290)
    d = align(a, r, s)
    t1, t2 = d.index[280], d.index[400]
    ends = st.horizon_ends(d.index, [t1], label_end=d.index[-1] + pd.Timedelta(days=1))
    f = st.features("X", d, [t1, t2, d.index[560]], ends)
    assert list(f["date"]) == [t1, d.index[560]]
    assert f.iloc[0]["ret21_trunc"] and f.iloc[0]["ret21"] == pytest.approx(d["c"].iloc[289] / d["o"].iloc[281] - 1, rel=1e-6)


def test_window_counts_label_and_ranks():
    ev = pd.DataFrame({"symbol": ["A", "A", "A", "B"],
                       "day": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-09"])})
    keys = pd.DataFrame({"symbol": ["A", "B", "C"], "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-09"])})
    c = st.window_counts(keys, ev, (7, 30))
    assert list(c[7]) == [1, 1, 0] and list(c[30]) == [2, 1, 0]   # (t - 7 days, t]: Jan 1 is outside
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-05")] * 100 + [pd.Timestamp("2024-01-12")] * 3,
                       "ret21": list(range(100)) + [np.nan, 1.0, 2.0]})
    y = st.top_label(df)
    assert y.iloc[:100].sum() == 2 and y.iloc[98] == 1 and y.iloc[99] == 1
    assert np.isnan(y.iloc[100]) and y.iloc[102] == 1
    rk = st.rank_within_date(df, ["ret21"])
    assert rk["ret21"].iloc[99] == 1.0 and np.isnan(rk["ret21"].iloc[100])


def test_walk_forward_trains_only_on_closed_windows(monkeypatch):
    dates = pd.date_range("2020-01-03", "2022-12-30", freq="W-FRI")
    end21 = {t: t + pd.Timedelta(days=30) for t in dates}
    p = pd.DataFrame({"date": np.repeat(dates, 5), "f": np.tile(np.arange(5.0), len(dates))})
    p["y"], p["ret21_rank"] = (p["f"] == 4).astype(float), p["f"] / 4
    seen = {}

    def fake(name, train, test, feats):
        seen[test["date"].dt.year.iloc[0]] = train["date"].max()
        return np.zeros(len(test)), None
    monkeypatch.setattr(st, "fit_predict", fake)
    wf = st.walk_forward(p, end21, {"xgb": ["f"]}, [2021, 2022], log=lambda s: None)
    assert set(wf["date"].dt.year) == {2021, 2022}
    for y, last in seen.items():
        first = pd.Timestamp(f"{y}-01-01")
        assert last + pd.Timedelta(days=30) < wf.loc[wf["date"].dt.year == y, "date"].min() and last < first


def _fake_world(tmp_path, monkeypatch):
    from tracks.main.shortterm import evaluate, panel as pn
    syms = [f"S{i}" for i in range(30)] + [f"P{i}" for i in range(30)]
    bars = {}
    for i, s in enumerate(syms + ["SPY"]):
        a, r, sp = _bars(n=1250, start="2019-01-01", seed=i, drift=0.0002 * (i % 5))
        k = 0.04 if s.startswith("P") else 1.0                      # P* trade at $1.5-4.5
        bars[s] = tuple(x.assign(o=x["o"] * k, c=x["c"] * k) for x in (a, r, sp))
    cal = bars["SPY"][0].index
    fake = lambda batch, start, end: {k: {s: bars[s][j] for s in batch if s in bars}
                                      for j, k in enumerate(("all", "raw", "split"))}
    monkeypatch.setattr(pn, "FIRST_SIGNAL", cal[300])
    monkeypatch.setattr(pn, "LAST_SIGNAL", cal[-1])
    monkeypatch.setattr(pn, "LABEL_END", cal[-30])
    monkeypatch.setattr(pn, "NEWS_FIRST", "2023-01")
    monkeypatch.setattr(pn, "NEWS_LAST", "2023-02")
    monkeypatch.setattr(pn, "fetch_daily_bars", lambda *a, **k: {})
    monkeypatch.setattr(pn, "to_frames", lambda raw: {"SPY": bars["SPY"][0]})
    monkeypatch.setattr(pn, "alpaca_headers", lambda: {})
    monkeypatch.setattr(pn.alpaca, "sip_end", lambda end: end)
    monkeypatch.setattr(pn.bars_batch, "assets_all", lambda: pd.DataFrame(
        {"base_symbol": syms, "symbol": syms, "name": syms, "status": "active", "excl_reason": None,
         "easy_to_borrow": [i % 2 == 0 for i in range(60)]}))
    monkeypatch.setattr(pn.sec_data, "dera_ticker_map", lambda *a: pd.DataFrame(
        {"cik": list(range(1, 61)) * 3, "symbol": syms * 3,
         "filed": [pd.Timestamp(f"{y}-06-01") for y in (2019, 2021, 2023) for _ in syms]}))
    real = pn.bar_batches
    monkeypatch.setattr(pn, "bar_batches", lambda *a, **k: real(*a, size=25, fetch=fake))
    news = tmp_path / "cache" / "news"
    news.mkdir(parents=True)
    for per, rows in (("2023-01", [("2023-01-10T15:00:00Z", "UBS Upgrades S3 to Buy", ["S3"]),
                                   ("2023-01-11T15:00:00Z", "S4 and S5 rally", ["S4", "S5"])]),
                      ("2023-02", [("2023-02-01T15:00:00Z", "Citi Downgrades P1 to Sell", ["P1"])])):
        pd.DataFrame({"id": range(len(rows)), "created_at": [r[0] for r in rows], "headline": [r[1] for r in rows],
                      "symbols": [",".join(r[2]) for r in rows]}).to_parquet(news / f"news_{per}.parquet")
    monkeypatch.setattr(evaluate, "YEARS", [2022, 2023])
    years = list(range(2018, 2024))
    sh = lambda c: [(f"{y}-07-31", 2e8, "10-Q", f"{y}-08-05", f"q{c}{y}") for y in years]
    facts = pd.concat([fu.extract(_cf(c, years, sh(c))) for c in range(1, 61)])
    return pn, facts


def test_m10_and_p3_screens_end_to_end_with_fakes(tmp_path, monkeypatch):
    pn, facts = _fake_world(tmp_path, monkeypatch)
    from tracks.main.shortterm import screen_m10
    from tracks.penny.shortterm import screen_p3
    monkeypatch.setattr(screen_m10.sec_data, "facts_for", lambda *a, **k: facts)
    monkeypatch.setattr(screen_m10.sec_data, "download_companyfacts", lambda p: p)
    res = screen_m10.run(tmp_path / "cache", tmp_path / "out_m10")
    assert res["idea"] == "M10" and res["primary"]["n_dates"] > 50
    assert (pd.read_csv(tmp_path / "out_m10" / "M10_screen_picks.csv")["price"] >= 5).all()
    assert (tmp_path / "out_m10" / "M10_xgb.json").exists() and len(res["hand_check_last_date"]) == 20
    panel = pd.read_parquet(tmp_path / "cache" / "boom" / "panel.parquet")
    s3 = panel[(panel["symbol"] == "S3") & (panel["date"] >= "2023-01-10") & (panel["date"] < "2023-04-10")]
    assert len(s3) and (s3["an_up"] == 1).all() and panel.loc[panel["symbol"] == "S4", "news30"].max() == 0
    assert panel["date"].max() < pn.LABEL_END
    res_p = screen_p3.run(tmp_path / "cache", tmp_path / "out_p3")         # reuses the cached panel
    assert "mean_without_best_pick" in res_p["primary"]
    picks = pd.read_csv(tmp_path / "out_p3" / "P3_screen_picks.csv")
    assert set(picks["symbol"].str[0]) == {"P"} and (picks["price"] < 5).all() and (picks["price"] >= 1).all()
    assert "lmcap" not in res_p["features"]
    json.dumps(res_p, default=str)


def test_panel_build_resumes_after_budget(tmp_path, monkeypatch):
    pn, _ = _fake_world(tmp_path, monkeypatch)
    assert pn.build(tmp_path / "cache", deadline=0) is None               # budget already spent: nothing done
    assert pn.build(tmp_path / "cache") is not None
