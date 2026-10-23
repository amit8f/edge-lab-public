"""NX1 helpers on synthetic data (no network, no FinBERT)."""
import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import w1_data as W

DAYS = pd.bdate_range("2017-01-02", "2019-12-31")
CAL = pd.DataFrame({"date": DAYS, "open": "09:30", "close": "16:00"})


def fake_finbert(texts):
    return np.array([[0.7, 0.1, 0.2] if "beat" in t else [0.1, 0.8, 0.1] for t in texts])


def test_inputs_are_the_registered_32():
    assert len(N.INPUTS) == 32 and len(set(N.INPUTS)) == 32
    assert len(N.FLAGS) == 12


def test_flags():
    f = N.flag_row("Acme Downgrades XYZ, Lowers Price Target After FDA CRL")
    assert f["f_analyst_down"] == f["f_fda"] == 1 and f["f_analyst_up"] == 0
    assert N.flag_row("XYZ Announces $50M Registered Direct Offering")["f_offering"] == 1
    assert N.flag_row("XYZ CEO Steps Down")["f_executive"] == 1


def test_batch_day_cutoff_at_10pm():
    t = pd.to_datetime(["2017-01-03 21:59", "2017-01-03 22:00", "2017-01-03 22:01", "2017-01-07 12:00"])
    d = N.batch_days(t, CAL["date"])
    assert list(d.strftime("%m-%d")) == ["01-03", "01-03", "01-04", "01-09"]


def test_headline_rows_single_symbol_and_scores_once():
    news = pd.DataFrame({"id": [1, 2, 3, 4],
                         "created_at": ["2017-03-01T14:00:00Z", "2017-03-01T15:00:00Z", "2017-03-01T16:00:00Z",
                                        "2017-03-02T16:00:00Z"],
                         "headline": ["XYZ Earnings Beat", "Two names", "", "xyz earnings  beat"],
                         "symbols": ["XYZ", "ABC,XYZ", "XYZ", "XYZ"]})
    calls = []
    r = N.headline_rows(news, lambda t: calls.append(len(t)) or fake_finbert(t))
    assert list(r["id"]) == [1, 4] and calls == [1]
    assert r["fb_pos"].tolist() == [0.7, 0.7] and r["hkey"].nunique() == 1
    assert "headline" not in r.columns
    assert N.dedup(r)["id"].tolist() == [1]


def test_news_events_aggregate_and_bucket():
    rows = pd.DataFrame({"id": [1, 2, 3], "symbol": ["XYZ", "XYZ", "ABC"],
                         "t": pd.to_datetime(["2017-01-02 23:00", "2017-01-03 17:00", "2017-01-03 10:00"]),
                         "hkey": ["a", "b", "c"], "head_len": [10, 20, 30], "lm_neg": [1, 0, 2], "lm_pos": [0, 2, 0],
                         "fb_pos": [0.1, 0.5, 0.2], "fb_neg": [0.8, 0.1, 0.6], "fb_neu": [0.1, 0.4, 0.2],
                         **{f"f_{k}": [0, 1, 0] for k in N.FLAGS}})
    e = N.news_events(rows, CAL).set_index("symbol")
    assert e.loc["XYZ", "n_articles"] == 2 and e.loc["XYZ", "lm_net"] == 1
    assert e.loc["XYZ", "fb_max_neg"] == 0.8 and e.loc["XYZ", "time_bucket"] == 2
    assert e.loc["ABC", "time_bucket"] == 1 and e.loc["ABC", "d"] == pd.Timestamp("2017-01-03")


def test_novelty_counts_prior_30_days_only():
    nd = pd.DataFrame({"symbol": "XYZ", "date": pd.to_datetime(["2017-02-01", "2017-02-20", "2017-03-01"])})
    keys = pd.DataFrame({"symbol": ["XYZ", "ABC"], "d": pd.to_datetime(["2017-03-01", "2017-03-01"])})
    assert N.novelty(keys, nd).tolist() == [2.0, 0.0]


def _rows(sym, n=len(DAYS), price=10.0, drift=0.0, gap_at=None):
    rng = np.random.default_rng(len(sym))
    c = price * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    f = pd.DataFrame({"o": c * 0.995, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 5e5}, index=DAYS[:n])
    if gap_at is not None:
        f = f.drop(f.index[gap_at:gap_at + 15])
    u = W.daily_rows(sym, f, f.copy())
    u["cik"] = float(len(sym))
    return u


def _spy():
    c = 200 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, len(DAYS))))
    return pd.DataFrame({"date": DAYS, "ao": c * 0.999, "ac": c, "c": c})


def test_stock_panel_trade_window_and_label():
    rows, spy = _rows("XYZ"), _spy()
    p = N.stock_panel(rows, spy, CAL).set_index("d")
    d = DAYS[300]
    r = p.loc[d]
    assert r["entry"] == DAYS[301] and r["exit"] == DAYS[303]
    g = rows.set_index("date")
    assert np.isclose(r["ret"], g.at[DAYS[303], "ac"] / g.at[DAYS[301], "ao"] - 1)
    s = spy.set_index("date")
    assert np.isclose(r["mret"], r["ret"] - r["beta"] * (s.at[DAYS[303], "ac"] / s.at[DAYS[301], "ao"] - 1))
    assert np.isclose(r["price"], g.at[DAYS[301], "o"]) and r["days"] == (DAYS[303] - DAYS[301]).days + 1
    assert p.loc[DAYS[50], "beta"] == 1.0                      # under 120 returns: beta 1
    assert p["uni"].iloc[:20].sum() == 0 and p["uni"].iloc[30]


def test_stock_panel_never_crosses_a_gap():
    rows = _rows("XYZ", gap_at=400)
    p = N.stock_panel(rows, _spy(), CAL).set_index("d")
    before = DAYS[399]
    assert pd.isna(p.loc[before, "entry"])                     # next session is missing: no trade
    assert not p.loc[DAYS[415:434], "uni"].any()               # 20 fresh bars needed after the gap


def test_costs_and_classes():
    c = N.cost_frac([10.0, 2.0, 10.0], [False, False, True], [5, 5, 5])
    assert 0.004 < c[0] < 0.006 and c[1] > c[0] and c[2] > c[0]
    lo, hi = N.edges(np.arange(100.0))
    assert N.classes([0, 50, 99], lo, hi).tolist() == [0, 1, 2]
    w = N.weights(np.array([0, 1, 1, 1, 2]))
    assert np.isclose(w[0], 5 / 3) and np.isclose(w[1], 5 / 9)


def test_picks_top3_each_side_and_matched_control():
    night = pd.DataFrame({"p_long": [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.2, 0.3],
                          "p_short": [0.05, 0.1, 0.1, 0.2, 0.8, 0.7, 0.1, 0.3],
                          "cls": [2, 2, 2, 2, 0, 0, 1, 1]})
    p, c = N.picks(night, np.random.default_rng(0))
    assert p["side"].tolist() == [1, 1, 1, -1, -1] and list(p.index[:3]) == [0, 1, 2]
    assert len(c) == 5 and c["side"].tolist() == p["side"].tolist() and c.index.is_unique


def _events():
    rng = np.random.default_rng(3)
    rows = []
    for d in DAYS[60:]:
        for k in range(8):
            news = rng.normal()
            x = {i: rng.normal() for i in N.INPUTS}
            x["lm_net"] = news
            mret = 0.03 * news + rng.normal(0, 0.02)
            rows.append({"d": d, "kind": "real", **x, "mret": mret, "ret": mret, "price": 20.0, "days": 5})
        x = {i: rng.normal() for i in N.INPUTS} | N.placebo_news()
        rows.append({"d": d, "kind": "placebo", **x, "mret": rng.normal(0, 0.02), "ret": 0.0, "price": 20.0,
                     "days": 5})
    return pd.DataFrame(rows)


def test_walk_forward_finds_a_planted_news_effect():
    ev = _events()
    params = N.XGB | {"n_estimators": 40, "min_child_weight": 5}
    wf = N.walk_forward(ev, CAL, params, years=[2019])
    r = wf["2019"]
    assert r["nights"] > 200 and r["pick_minus_control"]["t_nw"] > 2
    assert r["placebo"]["n_real_fired"] > 0 and set(r["placebo"]) >= {"fire_ratio_ok", "welch_t"}
    first_2019 = DAYS[DAYS.searchsorted(pd.Timestamp("2019-01-01"))]
    assert r["train_events"] == int(((ev["kind"] == "real") & (ev["d"] <= DAYS[DAYS.get_loc(first_2019) - 6])).sum())


def test_batch_events_real_and_placebo():
    rows = pd.concat([_rows(s, price=20.0) for s in ("AAA", "BBB", "CCC", "DDD")], ignore_index=True)
    d = DAYS[300]
    ne = pd.DataFrame([{"symbol": "AAA", "d": d, **{k: 1.0 for k in N.NEWS_INPUTS}}])
    nd = pd.DataFrame({"symbol": ["AAA", "BBB"], "date": [d, d - pd.Timedelta(days=1)]})
    old = N.FIRST_DAY
    N.FIRST_DAY = DAYS[0]
    try:
        ev = N.batch_events(rows, ne, nd, _spy(), CAL, 0)
    finally:
        N.FIRST_DAY = old
    real, plac = ev[ev["kind"] == "real"], ev[ev["kind"] == "placebo"]
    assert real["symbol"].tolist() == ["AAA"] and len(plac) == 1
    assert plac["symbol"].iloc[0] in ("CCC", "DDD")             # BBB had news the day before
    assert (plac[N.NEWS_INPUTS] == 0).all().all() and real["novelty"].iloc[0] == 0
    assert ev[N.INPUTS].notna().all().all() and ev["mret"].notna().all()


def test_tone_shards_cover_every_month_once():
    allm = N.tone_months()
    parts = [m for i in range(12) for m in N.tone_months(i, 12)]
    assert sorted(parts) == allm and len(set(parts)) == len(allm) and allm[0] == pd.Period("2016-08")


def test_news_events_month_by_month_equals_all_at_once(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    t = pd.to_datetime("2017-01-25") + pd.to_timedelta(rng.uniform(0, 14 * 86400, 400), unit="s")
    rows = pd.DataFrame({"id": range(400), "symbol": rng.choice(["AAA", "BBB"], 400), "t": t,
                         "hkey": rng.choice(list("abcdef"), 400), "head_len": 10, "lm_neg": rng.integers(0, 3, 400),
                         "lm_pos": rng.integers(0, 3, 400), "fb_pos": rng.uniform(size=400),
                         "fb_neg": rng.uniform(size=400), "fb_neu": rng.uniform(size=400),
                         **{f"f_{k}": rng.integers(0, 2, 400) for k in N.FLAGS}})
    root = tmp_path / "nx1"
    (root / "tone").mkdir(parents=True)
    CAL.to_parquet(root / "cal.parquet")
    for per, g in rows.groupby(rows["t"].dt.to_period("M")):
        g.to_parquet(root / "tone" / f"tone_{per}.parquet")
    monkeypatch.setattr(N, "nx1dir", lambda cache: root)
    monkeypatch.setattr(N, "FIRST_DAY", DAYS[0])
    got = N.all_news_events("x").sort_values(["symbol", "d"]).reset_index(drop=True)
    want = N.news_events(N.dedup(rows), CAL).sort_values(["symbol", "d"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want[got.columns], check_dtype=False)
    assert got["d"].dt.month.nunique() == 2


def test_quiet_news_matches_quiet_mask():
    rng = np.random.default_rng(9)
    keys = pd.DataFrame({"symbol": rng.choice(["AAA", "BBB", "CCC"], 300), "cik": 1.0,
                         "date": rng.choice(DAYS[50:400], 300)})
    nd = pd.DataFrame({"symbol": rng.choice(["AAA", "BBB"], 200), "date": rng.choice(DAYS[40:400], 200)})
    empty = pd.DataFrame({"cik": pd.Series(dtype=float), "date": pd.Series(dtype="datetime64[ns]")})
    want = W.quiet_mask(keys, nd, empty, back=3, cal=CAL["date"])
    got = N.quiet_news(keys, nd, 3, CAL["date"])
    assert (got == want).all() and 0 < got.sum() < len(got)
