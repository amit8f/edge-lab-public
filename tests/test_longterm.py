"""M4 long-term ranking on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main.longterm import bars_batch, fundamentals as fu, prices, score, sec_data, stats
from tracks.main.longterm.screen import excess_series, month_ends


def _bars(n=400, start="2020-01-01", drift=0.0005, split_at=None, seed=0, gap_at=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    if gap_at is not None:
        idx = idx[:gap_at].append(idx[gap_at:] + pd.Timedelta(days=30))
    c = 50 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))
    raw = c.copy()
    split = c.copy()
    if split_at is not None:          # 2-for-1 split: raw halves from split_at on; adjusted series halve before
        raw[split_at:] /= 2
        split = raw.copy()
        split[:split_at] /= 2
        c = split.copy()
    f = lambda x: pd.DataFrame({"o": x * 0.999, "c": x, "v": 1e6}, index=idx)
    return f(c), f(raw), f(split)


def test_features_momentum_vol_and_forward_return():
    a, r, s = _bars()
    d = prices.align(a, r, s)
    t = d.index[300]
    ends = prices.horizon_ends(d.index, [t])
    f = prices.features("X", d, [t], ends).iloc[0]
    assert f["mom"] == pytest.approx(d["c"].iloc[279] / d["c"].iloc[48] - 1)
    # 6m window runs past the data (400 bars): horizon_ends leaves it out
    assert t not in ends["ret6"]
    assert np.isnan(f["ret6"])
    t2 = d.index[260]
    ends2 = prices.horizon_ends(d.index, [t2])
    f2 = prices.features("X", d, [t2], ends2).iloc[0]
    assert f2["ret6"] == pytest.approx(d["c"].iloc[386] / d["o"].iloc[261] - 1)
    assert not f2["ret6_trunc"]


def test_holdout_guard_blocks_windows_reaching_label_end():
    a, r, s = _bars(n=600)
    cal = a.index
    t = cal[300]
    ok = prices.horizon_ends(cal, [t], label_end=cal[426] + pd.Timedelta(days=1))
    blocked = prices.horizon_ends(cal, [t], label_end=cal[426])
    assert t in ok["ret6"] and t not in blocked["ret6"]


def test_gap_ends_series_and_truncates():
    a, r, s = _bars(n=600, gap_at=400)
    d = prices.align(a, r, s)
    cal = pd.bdate_range(d.index[0], d.index[-1])
    t = d.index[300]
    f = prices.features("X", d, [t], prices.horizon_ends(cal, [t])).iloc[0]
    assert f["ret6_trunc"] and f["ret6"] == pytest.approx(d["c"].iloc[399] / d["o"].iloc[301] - 1)
    # right after the gap there is no full lookback in the new series
    assert prices.features("X", d, [d.index[420]], {}).empty


def test_split_points_and_r_lookup():
    a, r, s = _bars(split_at=200)
    d = prices.align(a, r, s)
    rt = prices.split_points("X", d)
    assert len(rt) == 2 and rt["R"].iloc[0] == pytest.approx(2) and rt["R"].iloc[1] == pytest.approx(1)
    got = score.r_at(rt, ["X", "X", "X"], [d.index[0] - pd.Timedelta(days=50), d.index[100], d.index[300]])
    assert list(np.round(got, 6)) == [2, 2, 1]


def _cf(cik, years, shares, ni=10.0, gp=40.0, assets=100.0, equity=50.0, growth=0.1, restate=False):
    us = {"Assets": [], "NetIncomeLoss": [], "GrossProfit": [], "StockholdersEquity": []}
    for k, y in enumerate(years):
        end, filed = f"{y}-12-31", f"{y + 1}-02-20"
        a = assets * (1 + growth) ** k
        for e_, v, fl in ((end, a, filed), (f"{y - 1}-12-31", a / (1 + growth), filed)):
            us["Assets"].append({"end": e_, "val": v, "form": "10-K", "filed": fl, "accn": f"a{y}"})
        us["StockholdersEquity"].append({"end": end, "val": equity, "form": "10-K", "filed": filed, "accn": f"a{y}"})
        us["NetIncomeLoss"].append({"start": f"{y}-01-01", "end": end, "val": ni, "form": "10-K", "filed": filed, "accn": f"a{y}"})
        us["GrossProfit"].append({"start": f"{y}-01-01", "end": end, "val": gp, "form": "10-K", "filed": filed, "accn": f"a{y}"})
        if restate:   # a later 10-K restates last year's NI; the original must win
            us["NetIncomeLoss"].append({"start": f"{y - 1}-01-01", "end": f"{y - 1}-12-31", "val": 999.0,
                                        "form": "10-K", "filed": filed, "accn": f"a{y}"})
    dei = [{"end": e, "val": v, "form": fm, "filed": f, "accn": acc} for e, v, fm, f, acc in shares]
    return {"cik": cik, "facts": {"us-gaap": {k: {"units": {"USD": v}} for k, v in us.items()},
                                  "dei": {fu.SHARES: {"units": {"shares": dei}}}}}


def test_annual_records_point_in_time_and_originals():
    rows = fu.extract(_cf(7, [2019, 2020, 2021], [], restate=True))
    rec = fu.annual_records(rows)
    r20 = rec[rec["end"] == "2020-12-31"].iloc[0]
    assert r20["ni"] == 10.0                      # not the 999 restatement filed in 2022
    assert r20["avail"] == pd.Timestamp("2021-02-20")
    assert r20["assets_prev"] == pytest.approx(100 * 1.1 ** 1 / 1.1)
    keys = pd.DataFrame({"cik": [7, 7], "date": pd.to_datetime(["2021-02-19", "2021-02-22"])})
    got = fu.as_of(keys, rec, stale_days=fu.STALE_DAYS, end_col="end")
    assert list(got["end"]) == [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")]


def test_stale_annual_record_is_dropped():
    rec = fu.annual_records(fu.extract(_cf(7, [2019], [])))
    got = fu.as_of(pd.DataFrame({"cik": [7], "date": [pd.Timestamp("2022-01-03")]}), rec,
                   stale_days=fu.STALE_DAYS, end_col="end")
    assert np.isnan(got["ni"].iloc[0])


def test_shares_sum_classes_and_split_correction():
    sh = [("2020-01-31", 60, "10-K", "2020-02-20", "s1"), ("2020-01-31", 40, "10-K", "2020-02-20", "s1"),
          ("2021-01-31", 100, "10-K", "2021-02-20", "s2")]
    rows = fu.extract(_cf(7, [2019, 2020], sh))
    srec = fu.share_records(rows)
    assert sorted(srec["shares"]) == [100, 100]
    # 2-for-1 split on 2020-09-01: old 100 shares are 200 in today's terms, so issuance = log(100/200)
    rtab = pd.DataFrame({"symbol": ["X", "X"], "date": pd.to_datetime(["2015-01-01", "2020-09-01"]), "R": [2.0, 1.0]})
    panel = pd.DataFrame({"symbol": ["X"], "date": [pd.Timestamp("2021-03-31")], "cik": [7], "price": [10.0]})
    out = score.attach_fundamentals(panel, rows, rtab)
    assert out["mcap"].iloc[0] == pytest.approx(1000)
    # the split sits within 6 months of the old share date, so issuance is undefined (pre-registered)
    assert np.isnan(out["issuance"].iloc[0])
    rtab2 = pd.DataFrame({"symbol": ["X"], "date": pd.to_datetime(["2015-01-01"]), "R": [1.0]})
    out2 = score.attach_fundamentals(panel, rows, rtab2)
    assert out2["issuance"].iloc[0] == pytest.approx(0.0)
    assert out2["roa"].iloc[0] == pytest.approx(10 / 110)


def test_share_scale_errors_dropped():
    g = pd.DataFrame({"cik": [1, 1, 1, 1, 2, 2], "accn": list("abcdef"),
                      "sh_end": pd.to_datetime(["2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01",
                                                "2018-01-01", "2018-04-01"]),
                      "avail": pd.to_datetime(["2018-02-01", "2018-05-01", "2018-08-01", "2018-11-01",
                                               "2018-02-01", "2018-05-01"]),
                      # cik 1: one value in thousands; cik 2: a real 200x level shift is kept
                      "shares": [17e6, 17e3, 17.1e6, 17.2e6, 1e6, 2e8]})
    out = fu.drop_scale_errors(g)
    assert list(out["accn"]) == ["a", "c", "d", "e", "f"]


def _scored_panel(n=60, dates=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for t in pd.date_range("2020-01-31", periods=dates, freq="ME"):
        for i in range(n):
            rows.append({"symbol": f"S{i}", "cik": i, "date": t, "price": 20.0, "mcap": 1e9, "dv20": 5e6,
                         "mom": rng.normal(), "vol": abs(rng.normal()), "gpa": rng.normal(), "roa": rng.normal(),
                         "ep": rng.normal(), "bm": rng.normal(), "issuance": rng.normal(),
                         "assetgrowth": rng.normal(), "ret6": rng.normal(0, .2)})
    return pd.DataFrame(rows)


def test_eligible_filters_and_dedupes_dual_class():
    p = _scored_panel(n=3, dates=1)
    p.loc[0, "price"] = 4.99
    p.loc[1, "mcap"] = 2.99e8
    extra = p.iloc[[2]].assign(symbol="S2B", dv20=1e7)
    e = score.eligible(pd.concat([p, extra]))
    assert list(e["symbol"]) == ["S2B"]


def test_composite_ranks_and_family_minimum():
    p = _scored_panel()
    p.loc[0, ["gpa", "roa", "ep", "bm", "issuance"]] = np.nan     # 3 families left: dropped
    p.loc[1, ["issuance", "assetgrowth"]] = np.nan                # 4 left: kept
    s = score.composite(p)
    assert "S0" not in set(s[s["date"] == s["date"].min()]["symbol"])
    assert s.groupby("date")["rank"].apply(lambda r: sorted(r) == list(range(1, len(r) + 1))).all()
    best = s[s["rank"] == 1].iloc[0]
    assert best["score"] == s[s["date"] == best["date"]]["score"].max()


def test_excess_series_and_control():
    p = score.composite(_scored_panel(n=200, dates=4))
    p.loc[p["rank"] <= 20, "ret6"] += 0.10
    ex = excess_series(p, "ret6", 20, 0.002)
    assert (ex["excess"] > 0.05).all()
    ctrl = excess_series(p, "ret6", 20, 0.002, rng_seed=1)
    assert ctrl["excess"].abs().mean() < 0.08


def test_nw_and_bootstrap():
    x = np.full(50, 0.01) + np.random.default_rng(0).normal(0, 0.001, 50)
    m, se, t = stats.nw_tstat(x, 5)
    assert m == pytest.approx(0.01, abs=1e-3) and t > 10
    lo, hi = stats.block_bootstrap_ci(x, 6, 500, 1)
    assert lo < m < hi


def test_norm_symbol_and_cik_map():
    assert sec_data.norm_symbol("NYSE: BRK/B") == "BRK.B"
    assert sec_data.norm_symbol("nasdaq:opi") == "OPI"
    assert sec_data.norm_symbol("NONE ") == "NONE" and sec_data.norm_symbol("12AB") is None
    tmap = pd.DataFrame({"cik": [1, 2], "symbol": ["ABC", "ABC"],
                         "filed": pd.to_datetime(["2015-01-01", "2020-01-01"])})
    keys = pd.DataFrame({"symbol": ["ABC", "ABC", "ABC"], "date": pd.to_datetime(["2016-01-01", "2021-01-01", "2014-01-01"])})
    got = sec_data.map_ciks(keys, tmap)
    assert got[0] == 1 and got[1] == 2 and np.isnan(got[2])


def test_parse_dera_dates():
    tsv = b"ACCESSION_NUMBER\tFILING_DATE\tISSUERCIK\tISSUERTRADINGSYMBOL\nx\t31-MAR-2016\t320193\tAAPL\n"
    d = sec_data.parse_dera_submission(tsv)
    assert d.iloc[0].tolist() == [320193, "AAPL", pd.Timestamp("2016-03-31")]


def test_batch_process_with_fake_fetch():
    a, r, s = _bars(n=500)
    fake = lambda syms, start, end: {"all": {"X": a}, "raw": {"X": r}, "split": {"X": s}}
    dates = month_ends(a.index, a.index[260], a.index[-1])
    ends = prices.horizon_ends(a.index, dates)
    f, rt, n = bars_batch.process(["X", "MISSING"], "x", "y", dates, ends, fetch=fake)
    assert n == 1 and len(f) == len(dates) and f["ret6"].notna().sum() >= 1 and len(rt) == 1


def test_reasons_and_expectation():
    from tracks.main.longterm import explain
    row = {"f_mom": 0.95, "mom": 0.42, "f_lowvol": 0.2, "vol": 0.55, "f_quality": 0.8, "gpa": 0.4, "roa": 0.12,
           "f_value": 0.5, "ep": 0.05, "f_issuance": 0.9, "issuance": -0.03, "f_assetgrowth": None, "assetgrowth": None}
    good, bad = explain.reasons(row)
    assert good[0].startswith("Strong trend: up 42%") and "buybacks" in " ".join(good)
    assert bad == ["Volatile: 55% annualised, expect big swings"]
    cal = {"buckets": [{"lo": -0.001, "hi": 0.5, "median": 0.0, "p25": -.1, "p75": .1, "beat_spy": .4, "n": 9},
                       {"lo": 0.5, "hi": 1.0, "median": 0.1, "p25": 0, "p75": .2, "beat_spy": .62, "n": 9}]}
    e = explain.expectation(cal, 0.99, 100.0, 0.3)
    assert e["median"] == 0.1 and e["band_low"] < 110 < e["band_high"]
    assert explain.confidence_label(e["beat_spy"]) == "higher"
    assert explain.expectation(cal, 0.2, 10, 0.3)["beat_spy"] == .4


def test_live_run_end_to_end_with_fakes(tmp_path, monkeypatch):
    from core import ledger as ledger_mod
    from tracks.main.longterm import live
    syms = [f"S{i}" for i in range(30)]
    bars = {s: _bars(n=420, start="2024-09-02", seed=i, drift=0.0002 * (i % 7)) for i, s in enumerate(syms + ["SPY"])}
    t = bars["S0"][0].index[-1]
    fake = lambda batch, start, end: {k: {s: bars[s][j] for s in batch if s in bars}
                                      for j, k in enumerate(("all", "raw", "split"))}
    monkeypatch.setattr(live, "last_session", lambda now: (t, True))
    monkeypatch.setattr(live.bars_batch, "assets_all", lambda: pd.DataFrame(
        {"base_symbol": syms, "symbol": syms, "name": [f"Co {s}" for s in syms], "status": "active", "excl_reason": None}))
    monkeypatch.setattr(live.sec_data, "current_ticker_map", lambda: pd.DataFrame(
        {"symbol": syms, "cik": range(1, 31), "title": syms}))
    real_run_live = live.bars_batch.run_live
    monkeypatch.setattr(live.bars_batch, "run_live", lambda s, a, b, tt: real_run_live(s, a, b, tt, fetch=fake))
    monkeypatch.setattr(live.sec_data, "download_companyfacts", lambda p: p)
    facts = pd.concat([fu.extract(_cf(c, [2023, 2024], [("2025-07-31", 2e8 + c * 1e6, "10-Q", "2025-08-05", f"q{c}"),
                                                         ("2024-07-31", 2e8, "10-Q", "2024-08-05", f"p{c}")],
                                       ni=5 + c, growth=0.01 * c)) for c in range(1, 31)])
    monkeypatch.setattr(live.sec_data, "facts_for", lambda *a, **k: facts)
    monkeypatch.setattr(live, "news_for", lambda s, since: [{"time": "2026-10-01", "headline": f"{s} </script> news", "source": "x", "url": "https://e.x"}])
    monkeypatch.setattr(live, "reddit_mentions", lambda s: None)
    now = pd.Timestamp("2026-10-12T03:30:00Z")
    data = live.run(tmp_path / "state", tmp_path / "cache", now=now)
    assert len(data["picks"]) == 20 and data["ledger_ok"] and data["ledger_entries"] == 1
    html = (tmp_path / "state" / "longterm" / "dashboard.html").read_text()
    assert "</script> news" not in html and "<title>Long-Horizon Picks</title>" in html
    led = ledger_mod.Ledger(tmp_path / "state" / "longterm")
    assert led.has("M4_picks", f"{t:%Y-%m-%d}")
    again = live.run(tmp_path / "state", tmp_path / "cache", now=now)      # second run the same night: no new entry
    assert again["ledger_entries"] == 1 and again["track_record"][0]["date"] == f"{t:%Y-%m-%d}"


def _fake_screen(tmp_path, monkeypatch):
    from tracks.main.longterm import screen
    syms = [f"S{i}" for i in range(60)]
    bars = {s: _bars(n=1100, start="2021-01-04", seed=i, drift=0.0001 * (i % 9)) for i, s in enumerate(syms + ["SPY"])}
    cal = bars["SPY"][0].index
    fake = lambda batch, start, end: {k: {s: bars[s][j] for s in batch if s in bars}
                                      for j, k in enumerate(("all", "raw", "split"))}
    monkeypatch.setattr(screen, "FIRST_SIGNAL", cal[260])
    monkeypatch.setattr(screen, "LAST_SIGNAL", cal[700])
    monkeypatch.setattr(screen, "LABEL_END", cal[-1])
    monkeypatch.setattr(screen, "fetch_daily_bars", lambda *a, **k: {})
    monkeypatch.setattr(screen, "to_frames", lambda raw: {"SPY": bars["SPY"][0]})
    monkeypatch.setattr(screen, "alpaca_headers", lambda: {})
    monkeypatch.setattr(screen.bars_batch, "assets_all", lambda: pd.DataFrame(
        {"base_symbol": syms, "symbol": syms, "name": syms, "status": "active", "excl_reason": None}))
    monkeypatch.setattr(screen.sec_data, "dera_ticker_map", lambda *a: pd.DataFrame(
        {"cik": range(1, 61), "symbol": syms, "filed": pd.Timestamp("2021-06-01")}))
    real = screen.bars_batch.run_batches
    monkeypatch.setattr(screen.bars_batch, "run_batches", lambda *a: real(*a, fetch=fake))
    monkeypatch.setattr(screen.sec_data, "download_companyfacts", lambda p: p)
    years = list(range(2020, 2025))
    sh = lambda c: [(f"{y}-07-31", 2e8 * (1 + 0.002 * c) ** k, "10-Q", f"{y}-08-05", f"q{c}{y}") for k, y in enumerate(years)]
    facts = pd.concat([fu.extract(_cf(c, years, sh(c), ni=5 + c, growth=0.01 * c)) for c in range(1, 61)])
    monkeypatch.setattr(screen.sec_data, "facts_for", lambda *a, **k: facts)
    return screen


def test_screen_run_end_to_end_with_fakes(tmp_path, monkeypatch):
    screen = _fake_screen(tmp_path, monkeypatch)
    res = screen.run(tmp_path / "cache", tmp_path / "out")
    assert res["primary"]["n_dates"] > 10 and "passes" in res["primary"]
    assert (tmp_path / "out" / "M4_calibration.json").exists()
    assert res["funnel"]["with_4_families"] > 0


def test_zero_share_count_does_not_crash():
    sh = [("2020-01-31", 0, "10-K", "2020-02-20", "s1"), ("2021-01-31", 100, "10-K", "2021-02-20", "s2")]
    rows = fu.extract(_cf(7, [2019, 2020], sh)).astype({"val": object})
    rtab = pd.DataFrame({"symbol": ["X"], "date": pd.to_datetime(["2015-01-01"]), "R": [1.0]})
    panel = pd.DataFrame({"symbol": ["X"], "date": [pd.Timestamp("2021-03-31")], "cik": [7], "price": [10.0]})
    out = score.attach_fundamentals(panel, rows, rtab)
    assert np.isnan(out["issuance"].iloc[0]) and out["mcap"].iloc[0] == pytest.approx(1000)


def test_parse_dera_buys_filters_and_dedupes():
    sub = (b"ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\n"
           b"a1\t15-MAR-2020\t4\t11\na2\t16-MAR-2020\t4/A\t11\na3\t17-MAR-2020\t4\t12\n")
    own = (b"ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
           b"a1\t900\tDirector\na1\t901\tOfficer,TenPercentOwner\na2\t900\tDirector\na3\t902\tTenPercentOwner\n")
    tr = (b"ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
          b"a1\t1\tP\t100\t10\tA\na1\t2\tS\t50\t10\tD\na1\t3\tP\t0\t10\tA\na2\t4\tP\t100\t10\tA\na3\t5\tP\t100\t10\tA\n")
    b = sec_data.parse_dera_buys(sub, own, tr)
    assert len(b) == 1 and b["value"].iloc[0] == 1000 and b["owners"].iloc[0] == "900,901"


def test_trailing_buys_window_and_m5_score():
    from tracks.main.longterm import insider
    buys = pd.DataFrame({"cik": [1, 1, 1], "filed": pd.to_datetime(["2020-01-01", "2020-05-01", "2020-07-01"]),
                         "owners": ["a", "b", "a,c"], "value": [100.0, 200.0, 400.0]})
    keys = pd.DataFrame({"cik": [1, 1, 2], "date": pd.to_datetime(["2020-06-30", "2020-07-01", "2020-07-01"])})
    v, n = insider.trailing_buys(keys, buys)
    assert list(v) == [300.0, 600.0, 0.0] and list(n) == [2, 3, 0]
    uni = pd.DataFrame({"cik": [1, 2, 3], "symbol": ["A", "B", "C"], "date": pd.Timestamp("2020-07-01"),
                        "mcap": [1e6, 1e6, 1e6], "issuance": [-0.05, 0.1, np.nan]})
    s = insider.score_m5(uni, buys)
    assert list(s.sort_values("rank")["symbol"]) == ["A", "B"]


def test_screen_m5_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tracks.main.longterm import screen_m5
    _fake_screen(tmp_path, monkeypatch)
    buys = pd.DataFrame({"cik": [3, 5, 7] * 30, "filed": np.repeat(pd.date_range("2021-06-01", periods=30, freq="MS"), 3),
                         "accn": "x", "owners": "o1", "shares": 100.0, "price": 10.0, "value": 1000.0})
    monkeypatch.setattr(screen_m5.sec_data, "dera_insider_buys", lambda *a: buys)
    res = screen_m5.run(tmp_path / "cache", tmp_path / "out5")
    assert res["primary"]["n_dates"] > 10 and res["funnel"]["picks_with_any_buy"] > 0
    assert (tmp_path / "out5" / "M5_calibration.json").exists()


def test_analyst_classify_rules():
    from tracks.main.longterm.analyst import classify
    assert classify("Goldman Sachs Upgrades Apple to Buy, Raises Price Target to $250") == 1
    assert classify("Morgan Stanley Downgrades XYZ to Equal-Weight, Lowers Price Target to $40") == -1
    assert classify("Citigroup Maintains Buy on Apple, Raises Price Target to $240") == 1
    assert classify("Barclays Maintains Overweight on MSFT, Lowers Price Target to $400") == -1
    assert classify("Apple Price Target Raised to $300 at UBS") == 1
    assert classify("Citi Maintains Neutral on XYZ, Maintains $20 Price Target") == 0
    assert classify("Analyst upgrades and downgrades roundup") == 0
    assert classify("Citigroup Maintains Buy on Apple, Raises Price Target to $240", ratings_only=True) == 0


def test_analyst_events_and_score(tmp_path):
    from tracks.main.longterm import analyst, news_history
    arts = pd.DataFrame({
        "headline": ["UBS Upgrades AAA to Buy", "UBS Upgrades AAA to Buy", "Citi Downgrades BBB to Sell",
                     "Jefferies Raises Price Target on AAA", "Upgrades AAA and BBB", "Random news on CCC"],
        "created_at": ["2020-05-01T14:00:00Z", "2020-05-02T14:00:00Z", "2020-05-03T14:00:00Z",
                       "2020-06-29T23:30:00Z", "2020-05-04T14:00:00Z", "2020-05-05T14:00:00Z"],
        "symbols": ["AAA", "AAA", "BBB", "AAA", "AAA,BBB", "CCC"]})
    ev = analyst.events(arts)
    assert list(ev["symbol"]) == ["AAA", "BBB", "AAA"]          # repeat and two-symbol rows dropped
    assert ev["day"].iloc[2] == pd.Timestamp("2020-06-29")      # 23:30 UTC is still June 29 in New York
    uni = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "date": pd.Timestamp("2020-06-30"), "ret6": 0.0})
    s = analyst.score_m6(uni, ev, seed=1)
    assert list(s["symbol"]) == ["AAA", "CCC", "BBB"] and s["an_up"].iloc[0] == 2
    # resumable monthly download with a fake feed
    pages = {"2020-01": [{"news": [{"id": 1, "created_at": "2020-01-02T00:00:00Z", "headline": "h", "symbols": ["A"]}],
                          "next_page_token": "p"}, {"news": [], "next_page_token": None}]}
    calls = []

    def fake(start, end=None, page_token=None):
        calls.append(page_token)
        return pages["2020-01"][0 if page_token is None else 1]
    assert news_history.fetch_range("2020-01", "2020-01", tmp_path, fetch=fake)
    assert calls == [None, "p"] and len(news_history.load_range("2020-01", "2020-01", tmp_path)) == 1


def test_screen_m6_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tracks.main.longterm import screen_m6
    _fake_screen(tmp_path, monkeypatch)
    days = pd.date_range("2021-06-01", periods=40, freq="7D")
    rows = pd.DataFrame({"id": range(80), "headline": ["X Upgrades S3 to Buy", "Y Downgrades S5 to Sell"] * 40,
                         "created_at": np.repeat(days.strftime("%Y-%m-%dT15:00:00Z"), 2),
                         "symbols": ["S3", "S5"] * 40})
    monkeypatch.setattr(screen_m6.news_history, "fetch_range", lambda *a, **k: True)
    monkeypatch.setattr(screen_m6, "classified_rows", lambda *a, **k: (rows, 1000))
    res = screen_m6.run(tmp_path / "cache", tmp_path / "out6")
    assert res["primary"]["n_dates"] > 10 and res["funnel"]["events_after_dedupe"] > 0


def _qrows(cik, quarters, base=10.0, step=1.0, jump_at=None, q4_from_annual=False):
    """Synthetic quarterly NetIncomeLoss rows: one 10-Q per quarter end, filed 35 days later."""
    from tracks.main.longterm import earnings
    rows = []
    for k, end in enumerate(pd.date_range("2016-03-31", periods=quarters, freq="QE")):
        v = base + step * k + (50.0 if jump_at is not None and k == jump_at else 0.0)
        start, filed = end - pd.Timedelta(days=90), end + pd.Timedelta(days=35)
        if q4_from_annual and end.month == 12:
            ys = pd.Timestamp(f"{end.year}-01-01")
            rows.append((cik, "NetIncomeLoss", ys, end, 4 * v, "10-K", filed + pd.Timedelta(days=30), f"k{k}"))
            rows.append((cik, "NetIncomeLoss", ys, end - pd.Timedelta(days=92), 3 * v, "10-Q",
                         end - pd.Timedelta(days=60), f"n{k}"))
            continue
        rows.append((cik, "NetIncomeLoss", start, end, v, "10-Q", filed, f"q{k}"))
    return pd.DataFrame(rows, columns=fu.COLS)


def test_quarterly_ni_derives_q4_and_keeps_originals():
    from tracks.main.longterm import earnings
    r = _qrows(1, 8, q4_from_annual=True)
    # a later restatement of the first quarter must not replace the original
    r = pd.concat([r, pd.DataFrame([(1, "NetIncomeLoss", pd.Timestamp("2015-12-31") + pd.Timedelta(days=1),
                                     pd.Timestamp("2016-03-31"), 999.0, "10-Q", pd.Timestamp("2017-05-05"), "x")],
                                   columns=fu.COLS)])
    q = earnings.quarterly_ni(r)
    assert len(q) == 8
    first = q.iloc[0]
    assert first["ni"] == 10.0 and first["avail"] == pd.Timestamp("2016-05-05")
    q4 = q[q["end"] == pd.Timestamp("2016-12-31")].iloc[0]
    assert q4["ni"] == pytest.approx(4 * 13.0 - 3 * 13.0)   # annual minus nine months
    assert q4["avail"] == pd.Timestamp("2016-12-31") + pd.Timedelta(days=65)


def test_sue_needs_history_and_flags_the_jump():
    from tracks.main.longterm import earnings
    rng = np.random.default_rng(0)
    r = _qrows(1, 16, step=0.0, jump_at=12)
    r["val"] = r["val"] + rng.normal(0, 1, len(r))
    t = earnings.sue_table(earnings.quarterly_ni(r))
    assert t["sue"].iloc[:8].isna().all()          # 4 quarters to get a change, 4 changes of history
    assert t["sue"].iloc[12] > 5                     # the +50 quarter stands out
    assert t["sue"].iloc[13:].dropna().abs().max() < t["sue"].iloc[12]


def test_score_m7_uses_only_fresh_filed_sue():
    from tracks.main.longterm import earnings
    sue = pd.DataFrame({"cik": [1, 2, 3], "end": pd.to_datetime(["2020-03-31"] * 3),
                        "avail": pd.to_datetime(["2020-05-01", "2020-07-01", "2019-12-01"]),
                        "sue": [2.0, 9.0, 5.0]})
    uni = pd.DataFrame({"cik": [1, 2, 3], "symbol": ["A", "B", "C"], "date": pd.Timestamp("2020-06-30")})
    s = earnings.score_m7(uni, sue, 1)
    assert list(s["symbol"]) == ["A"]   # B filed after t (look-ahead), C older than 120 days


def test_screen_m7_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tracks.main.longterm import screen_m7
    screen = _fake_screen(tmp_path, monkeypatch)
    annual = screen.sec_data.facts_for()
    rng = np.random.default_rng(3)
    qs = []
    for c in range(1, 61):
        r = _qrows(c, 32, base=5 + c, step=0.1 * (c % 7))
        r["val"] = r["val"] + rng.normal(0, 1, len(r))
        qs.append(r)
    facts = pd.concat([annual] + qs, ignore_index=True)
    monkeypatch.setattr(screen.sec_data, "facts_for", lambda *a, **k: facts)
    res = screen_m7.run(tmp_path / "cache", tmp_path / "out7")
    assert res["primary"]["n_dates"] > 10 and res["funnel"]["sue_filed_after_signal_date"] == 0
    assert (tmp_path / "out7" / "M7_screen.json").exists()
