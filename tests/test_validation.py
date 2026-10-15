"""Validation studies (V1-V3): synthetic data only."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from validation import french
from validation import stats as vs


def _zip(text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("f.CSV", text)
    return b.getvalue()


def _french_daily(days, cols, data, extra_tail=True):
    lines = ["This file was created by CMPT_ME_BEME_RETS_DAILY using the 202508 CRSP database.", "",
             "," + ",".join(cols)]
    lines += [f"{d:%Y%m%d}," + ",".join(f"{100 * v:.4f}" for v in row) for d, row in zip(days, data)]
    if extra_tail:
        lines += ["", "Copyright 2025 Eugene F. Fama and Kenneth R. French"]
    return _zip("\r\n".join(lines))


def test_parse_daily_and_monthly_first_table_only():
    days = pd.bdate_range("2024-01-02", periods=3)
    raw = _french_daily(days, ["Mkt-RF", "SMB", "RF"], [[0.01, -0.002, 0.0001]] * 3)
    f = french.parse(raw, "D")
    assert list(f.columns) == ["mktrf", "smb", "rf"] and len(f) == 3
    assert f["mktrf"].iloc[0] == pytest.approx(0.01)
    monthly = "\n".join(["header text", "", ",Mkt-RF,SMB,HML,RF", "202401,   1.00,  -0.50,   0.20,   0.40",
                         "202402,   2.00,   0.10,   0.30,   0.41", "", " Annual Factors: January-December ",
                         ",Mkt-RF,SMB,HML,RF", "2024,  10.0,  1.0,  2.0,  5.0"])
    m = french.parse(_zip(monthly), "M")
    assert list(m.index) == [pd.Period("2024-01", "M"), pd.Period("2024-02", "M")]
    assert m.loc[pd.Period("2024-02", "M"), "mktrf"] == pytest.approx(0.02)


def test_ols_hac_recovers_loadings_and_holm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 0.01 + 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.1, size=500)
    beta, se, t, r2, e = vs.ols_hac(y, X, lags=3)
    assert beta == pytest.approx([0.01, 0.5, -0.2], abs=0.02) and r2 > 0.9 and t[1] > 30
    assert list(vs.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_romano_wolf_flags_only_the_real_effect():
    rng = np.random.default_rng(1)
    Z = rng.normal(scale=0.02, size=(120, 4))
    Z[:, 0] += 0.01                                       # a real effect, t about 5
    t, p = vs.romano_wolf(Z, lags=2, block=4, n_boot=2000, seed=3)
    assert p[0] < 0.01 and (p[1:] > 0.05).all() and t[0] == max(t)


def test_deflated_sharpe_falls_with_more_trials():
    sr0_few = vs.expected_max_sr(0.1, 2)
    sr0_many = vs.expected_max_sr(0.1, 100)
    assert sr0_many > sr0_few > 0
    assert vs.deflated_sr(0.3, sr0_many / 2, 100, 0, 3) < vs.deflated_sr(0.3, sr0_few / 2, 100, 0, 3)


def test_v2_window_sums_and_rejudge(tmp_path, monkeypatch):
    from validation import v2_rejudge as v2
    days = pd.bdate_range("2018-01-02", "2026-06-30")
    rng = np.random.default_rng(5)
    data = rng.normal(scale=0.01, size=(len(days), 7))
    data[:, 6] = 0.0001                                                       # rf
    ff5 = _french_daily(days, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], data[:, [0, 1, 2, 3, 4, 6]])
    mom = _french_daily(days, ["Mom   "], data[:, [5]])
    files = {"ff5_daily": ff5, "mom_daily": mom}
    monkeypatch.setattr(v2.french, "fetch", lambda name, cache: (files[name], "sha-" + name))
    f, _ = v2.daily_factors(tmp_path)
    assert list(f.columns) == v2.FACTORS + ["rf"]
    t = days[10]
    w = v2.window_sums(f, [t], 21)
    assert w["mom"].iloc[0] == pytest.approx(f["mom"].iloc[11:32].sum())     # entry day after t, 21 days
    # an idea that is pure momentum exposure: excess = 0.8 x window Mom, no alpha
    res_dir = tmp_path / "results"
    for idea, h, freq in (("M6", 126, "BME"), ("M10", 21, "W-FRI")):
        dates = [d for d in pd.date_range("2019-01-01", "2025-03-31", freq=freq) if d in days]
        ws = v2.window_sums(f, dates, h)
        ex = 0.8 * ws["mom"].to_numpy() + rng.normal(scale=0.002, size=len(dates))
        (res_dir / idea).mkdir(parents=True)
        pd.DataFrame({"date": dates, "pick_mean": ex + 0.01, "uni_mean": 0.01, "excess": ex}).to_csv(
            res_dir / idea / f"{idea}_primary_series.csv", index=False)
    monkeypatch.setattr(v2, "RW_BOOT", 500)
    res = v2.run(tmp_path, res_dir, tmp_path / "out")
    m10 = res["ideas"]["M10"]
    assert m10["loadings"]["mom"]["beta"] == pytest.approx(0.8, abs=0.05) and m10["r2"] > 0.9
    assert abs(m10["alpha_t"]) < 3 and not m10["survives"]
    assert m10["placebo_shift_52w"]["r2"] < 0.2                              # misaligned windows explain nothing
    assert set(res["ideas"]) == {"M6", "M10"} and "p_romano_wolf" in res["ideas"]["M6"]
    assert 0 <= m10["dsr"] <= 1 and (tmp_path / "out" / "V2_table.csv").exists()


def test_v1_month_rows_returns_momentum_max_and_truncation():
    from tests.test_longterm import _bars
    from tracks.main.longterm.prices import align
    from validation import v1_known_answers as v1
    a, r, s = _bars(n=600, start="2016-01-01", seed=2)
    d = align(a, r, s).iloc[:-7]                                   # the series ends a week before its last month-end
    cal = pd.bdate_range("2016-01-01", periods=600)
    me = {p: g.max() for p, g in pd.Series(cal, index=cal).groupby(cal.to_period("M"))}
    dates = [t for t in me.values() if t >= pd.Timestamp("2017-03-01")]
    rows = pd.DataFrame(v1.month_rows("X", d, dates, me)).set_index("date")
    c = d["c"]
    t = pd.Timestamp(me[pd.Period("2017-06", "M")])
    i = d.index.get_loc(t)
    nxt = d.index.get_loc(me[pd.Period("2017-07", "M")])
    assert rows.loc[t, "r_next"] == pytest.approx(c.iloc[nxt] / c.iloc[i] - 1) and not rows.loc[t, "r_next_trunc"]
    m11 = d.index.get_loc(me[pd.Period("2016-06", "M")])
    m1 = d.index.get_loc(me[pd.Period("2017-05", "M")])
    assert rows.loc[t, "mom"] == pytest.approx(c.iloc[m1] / c.iloc[m11] - 1)
    jun = d.index[(d.index.to_period("M") == pd.Period("2017-06", "M"))]
    k = [d.index.get_loc(x) for x in jun]
    assert rows.loc[t, "maxret"] == pytest.approx(max(c.iloc[j] / c.iloc[j - 1] - 1 for j in k))
    assert rows.loc[t, "r20"] == pytest.approx(c.iloc[i] / c.iloc[i - 20] - 1)
    last_t = max(rows.index)
    assert rows.loc[last_t, "r_next_trunc"] and rows.loc[last_t, "r_next"] == pytest.approx(c.iloc[-1] / c.loc[last_t] - 1)


def _uni(n_months=36, seed=7):
    """Synthetic month rows where momentum, reversal and value each carry a known premium."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in pd.date_range("2018-01-31", periods=n_months, freq="ME"):
        n = 400
        mcap = np.exp(rng.normal(21, 1.5, n))
        mom, r_cur, bm, mx = rng.normal(0, 0.3, n), rng.normal(0, 0.1, n), np.exp(rng.normal(-0.7, 0.5, n)), rng.uniform(0, 0.3, n)
        r_next = 0.01 + 0.02 * np.sign(mom) - 0.1 * r_cur + 0.01 * np.log(bm) - 0.05 * mx + rng.normal(0, 0.02, n)
        rows.append(pd.DataFrame({"date": t, "exchange": np.where(rng.uniform(size=n) < 0.4, "NYSE", "NASDAQ"),
                                  "mcap": mcap, "bm": bm, "mom": mom, "r_cur": r_cur, "maxret": mx, "r_next": r_next,
                                  "inactive": rng.uniform(size=n) < 0.1}))
    return pd.concat(rows, ignore_index=True)


def test_v1_factor_builds_and_comparison():
    from validation import v1_known_answers as v1
    f = v1.factors(_uni())
    assert (f["umd"] > 0.02).all() and (f["strev"] > 0).all() and f["hml"].mean() > 0.005 and f["max_hl_vw"].mean() < -0.005
    theirs = pd.DataFrame({"mktrf": f["mkt"] - 0.001 + 0.0005, "rf": 0.001, "smb_ff": f["smb"] * 0.9 + 0.001,
                           "hml_ff": f["hml"], "mom_ff": f["umd"], "strev_ff": -f["strev"]}, index=f.index)
    res, j = v1.compare(f, theirs)
    assert res["mkt"]["pass"] and res["umd"]["pass"] and res["mkt"]["mean_abs_diff"] == pytest.approx(0.0005)
    assert not res["strev"]["pass"] and not res["all_pass"] and res["max_effect"]["pass"]


def test_v1_reconciliation_with_fake_sources():
    from validation import v1_known_answers as v1
    days = pd.bdate_range("2020-01-01", periods=300)
    closes = pd.Series(np.linspace(10, 40, 300), index=days)
    uni = pd.DataFrame({"symbol": [f"S{i}" for i in range(60)], "date": days[100 + np.arange(60)]})
    uni["r20"] = [closes.iloc[100 + i] / closes.iloc[80 + i] - 1 for i in range(60)]
    uni.loc[3, "r20"] += 0.05                                                  # one bad row
    src = [("fake", lambda s: None if s == "S7" else closes)]
    res = v1.reconcile_prices(uni, src, n=50)
    assert res["compared"] == 50 and res["within_1pt"] >= 49 and res["pass"]
    assert any(u["symbol"] == "S7" for u in res["unavailable"]) or "S7" not in {r["symbol"] for r in res["rows"]}
    facts = pd.DataFrame({"cik": [1, 1, 2, 2], "concept": ["Assets", "NetIncomeLoss", "Assets", "Assets"],
                          "end": pd.to_datetime(["2019-12-31", "2019-12-31", "2019-12-31", "2019-12-31"]),
                          "val": [100.0, 10.0, 200.0, 999.0], "form": "10-K",
                          "filed": pd.to_datetime(["2020-02-20", "2020-02-20", "2020-03-01", "2021-03-01"]),
                          "accn": ["a1", "a1", "a2", "a3"]})
    sample = v1.sample_sec(facts, np.random.default_rng(0))
    assert len(sample) <= 3 and "a3" not in set(sample["accn"])                # a later refiling never counts
    num = pd.DataFrame({"adsh": ["a1", "a1", "a2"], "tag": ["Assets", "NetIncomeLoss", "Assets"],
                        "ddate": "20191231", "qtrs": ["0", "4", "0"], "uom": "USD", "coreg": "",
                        "value": [100.0, 10.0, 201.5]})
    fixed = pd.DataFrame({"cik": [1, 1, 2], "concept": ["Assets", "NetIncomeLoss", "Assets"],
                          "end": pd.to_datetime(["2019-12-31"] * 3), "val": [100.0, 10.0, 200.0],
                          "accn": ["a1", "a1", "a2"], "q": "2020Q1"})
    r = v1.reconcile_sec(fixed, lambda q, a: num)
    assert [x["ok"] for x in r["rows"]] == [True, True, False] and not r["pass"]


def test_v1_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tests.test_longterm import _bars, _cf
    from tracks.main.longterm import fundamentals as fu
    from validation import v1_known_answers as v1
    syms = [f"S{i}" for i in range(40)]
    bars = {s: _bars(n=2500, start="2014-06-02", seed=i, drift=0.0002 * (i % 4)) for i, s in enumerate(syms + ["SPY"])}
    fake = lambda batch, start, end: {k: {s: bars[s][j] for s in batch if s in bars}
                                      for j, k in enumerate(("all", "raw", "split"))}
    monkeypatch.setattr(v1, "fetch_daily_bars", lambda *a, **k: {})
    monkeypatch.setattr(v1, "to_frames", lambda raw: {"SPY": bars["SPY"][0]})
    monkeypatch.setattr(v1, "alpaca_headers", lambda: {})
    monkeypatch.setattr(v1.alpaca, "sip_end", lambda end: end)
    monkeypatch.setattr(v1.bars_batch, "assets_all", lambda: pd.DataFrame(
        {"base_symbol": syms, "symbol": syms, "status": ["active"] * 36 + ["inactive"] * 4, "excl_reason": None,
         "exchange": ["NYSE" if i % 2 else "NASDAQ" for i in range(40)]}))
    monkeypatch.setattr(v1.sec_data, "dera_ticker_map", lambda *a: pd.DataFrame(
        {"cik": list(range(1, 41)) * 6, "symbol": syms * 6,
         "filed": [pd.Timestamp(f"{y}-06-01") for y in range(2014, 2026, 2) for _ in syms]}))
    years = list(range(2013, 2026))
    sh = lambda c: [(f"{y}-07-31", 1e7 * (1 + c), "10-Q", f"{y}-08-05", f"q{c}{y}") for y in years]
    facts = pd.concat([fu.extract(_cf(c, years, sh(c), equity=50.0 * c)) for c in range(1, 41)])
    monkeypatch.setattr(v1.sec_data, "facts_for", lambda *a, **k: facts)
    (tmp_path / "companyfacts.zip").write_bytes(b"")
    months = pd.period_range("2016-01", "2025-09", freq="M")
    mk = lambda cols, rows: "\n".join(["hdr", "", "," + ",".join(cols)] +
                                      [f"{m.year}{m.month:02d}," + ",".join("0.50" for _ in cols) for m in rows])
    files = {"ff3_monthly": _zip(mk(["Mkt-RF", "SMB", "HML", "RF"], months)), "mom_monthly": _zip(mk(["Mom   "], months)),
             "strev_monthly": _zip(mk(["ST_Rev"], months))}
    monkeypatch.setattr(v1.french, "fetch", lambda name, cache: (files[name], "sha"))
    src = [("fake", lambda s: bars[s][0]["c"])]
    num = lambda q, a: pd.DataFrame(columns=["adsh", "tag", "ddate", "qtrs", "uom", "coreg", "value"])
    assert v1.run(tmp_path, tmp_path / "out", budget_min=0, sources=src, fs_fetch=num, fetch=fake) is None
    res = v1.run(tmp_path, tmp_path / "out", budget_min=10, sources=src, fs_fetch=num, fetch=fake)
    assert res["funnel"]["with_mcap_one_per_company"] > 1000 and res["mkt"]["months"] > 80
    assert res["prices"]["within_1pt"] == res["prices"]["compared"] == 50     # same bars on both sides
    assert not res["all_pass"]                                                  # fake French series are constants
    assert (tmp_path / "out" / "V1_monthly_factors.csv").exists()


def test_v3_part_a_power_arithmetic(tmp_path):
    from validation import v3_power as v3
    x = 0.03 + np.random.default_rng(4).normal(0, 0.1, 120)
    p = tmp_path / "s.csv"
    pd.DataFrame({"excess": x}).to_csv(p, index=False)
    a = v3.part_a(p)
    assert a["mean_21d"] == pytest.approx(a["mean_6m"] / 6) and a["ir_annual"] == pytest.approx(a["mean_21d"] / (a["sd_6m"] / np.sqrt(6)) * np.sqrt(12))
    assert a["weeks_26"]["expected_t"] < a["weeks_52"]["expected_t"] < 2
    assert a["years_for_80pct_power"] == pytest.approx(((2 + v3.Z80) / a["ir_annual"]) ** 2)


def test_v3_part_b_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tests.test_shortterm import _fake_world
    from tracks.main.longterm import screen_m6
    from validation import v3_power as v3
    pn, facts = _fake_world(tmp_path, monkeypatch)
    monkeypatch.setattr(v3.sec_data, "facts_for", lambda *a, **k: facts)
    (tmp_path / "cache" / "companyfacts.zip").write_bytes(b"")
    monkeypatch.setattr(v3.news_history, "fetch_range", lambda *a: True)
    for mod in (screen_m6, v3):
        monkeypatch.setattr(mod, "FIRST_MONTH", "2023-01")
        monkeypatch.setattr(mod, "LAST_MONTH", "2023-02")
    monkeypatch.setattr(v3.screen_m10, "MIN_PRICE", 1.0)
    monkeypatch.setattr(v3.screen_m10, "eligible", lambda df: df.drop_duplicates(["date", "cik"]))
    p = tmp_path / "m6.csv"
    pd.DataFrame({"excess": 0.03 + np.random.default_rng(4).normal(0, 0.1, 120)}).to_csv(p, index=False)
    res = v3.run(tmp_path / "cache", tmp_path / "out", m6_series=p)
    b = res["part_b"]
    assert b["coverage"]["weeks"] > 50 and set(b) >= {"top20", "random20", "ic", "net_spread"}
    w = pd.read_csv(tmp_path / "out" / "V3_weekly.csv")
    assert w["ic"].dropna().between(-1, 1).all() and w["ic"].notna().any() and "decision" in res
