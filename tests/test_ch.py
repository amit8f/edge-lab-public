import numpy as np
import pandas as pd

from tracks.main import ch_data as C


def _rows(n=300, price=20.0, v=1e6):
    days = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"symbol": "X", "date": days, "o": price, "h": price * 1.01, "l": price * 0.99, "c": price,
                         "v": v, "ao": price, "ac": price, "dv20": price * v, "seg": 0, "cik": 1}).assign(
        h=lambda x: np.where(x.index.isin([50, 150]), price * 2, x["h"]),      # far 52-week extremes, so the
        l=lambda x: np.where(x.index.isin([50, 150]), price / 2, x["l"]))      # flat days aren't "near" them


def test_quiet_days_are_not_candidates_and_track_uses_prior_day():
    d = C.flag_days(_rows())
    assert not d["cand"].iloc[260:].any()
    assert (d["track"].iloc[1:] == "main").all() and d["track"].iloc[0] == ""


def test_volume_gap_and_breakout_flags():
    r = _rows()
    r.loc[270, "v"] = 2e6                                   # 2x volume
    r.loc[280, ["o", "ao"]] = 20.8                          # +4% gap
    r.loc[290, "h"] = 21.0                                  # new 20-day high
    d = C.flag_days(r)
    assert d["cand"].iloc[[270, 280, 290]].all()
    assert not d["cand"].iloc[[271, 281]].any()


def test_split_day_is_not_a_gap():
    r = _rows()
    r.loc[:279, ["o", "h", "l", "c"]] *= 2                  # 2:1 split on row 280: raw halves, adjusted flat
    r.loc[:279, "v"] = 0.5e6
    d = C.flag_days(r)
    assert abs(d["gap"].iloc[280]) < 1e-9 and not d["cand"].iloc[280]


def test_penny_track():
    d = C.flag_days(_rows(price=3.0, v=1e6))
    assert (d["track"].iloc[1:] == "penny").all()


from tracks.main import ch_build as B


def _session(day, closes, vols=None, o=None):
    n = len(closes)
    t = pd.date_range(day + B.OPEN, periods=n, freq="5min")
    c = np.asarray(closes, float)
    o = c if o is None else np.asarray(o, float)
    return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) * 1.001, "l": np.minimum(o, c) * 0.999, "c": c,
                         "v": 1000.0 if vols is None else vols, "vw": c})


CTX = {"prev_c": 20.0, "prev_dv20": 2e7, "vol20": 78_000.0, "hi20": 30.0, "lo20": 10.0, "hi252": 40.0, "lo252": 5.0}
CURVE = np.linspace(1 / 78, 1, 78)


def test_no_trigger_on_a_quiet_day_and_none_before_945():
    day = pd.Timestamp("2019-03-04")
    b = B.session(_session(day, [20.0] * 78), day, pd.Timedelta(hours=16))
    assert B.first_trigger(b, CTX, CURVE, "main", pd.Timedelta(hours=15, minutes=30)) == (None, None)
    v = np.full(78, 1000.0)
    v[0] = 1e6                                             # huge first bar: relative volume fires at 9:45, not 9:35
    b = B.session(_session(day, [20.0] * 78, v), day, pd.Timedelta(hours=16))
    i, trig = B.first_trigger(b, CTX, CURVE, "main", pd.Timedelta(hours=15, minutes=30))
    assert trig == "rvol" and b["end"].iloc[i] == day + B.FIRST_S


def test_gap_and_breakout_triggers():
    day = pd.Timestamp("2019-03-04")
    b = B.session(_session(day, [20.8] * 78), day, pd.Timedelta(hours=16))
    assert B.first_trigger(b, CTX, CURVE, "main", pd.Timedelta(hours=15, minutes=30))[1] == "gap"
    closes = [20.0] * 40 + [31.0] * 38
    b = B.session(_session(day, closes), day, pd.Timedelta(hours=16))
    i, trig = B.first_trigger(b, CTX, CURVE, "main", pd.Timedelta(hours=15, minutes=30))
    assert trig == "hl20" and i == 40


def test_labels_target_stop_and_costs():
    day = pd.Timestamp("2019-03-04")
    after = pd.DataFrame({"o": [100.0, 100.0], "h": [101.0, 104.5], "l": [99.5, 100.0], "c": [100.0, 104.0]})
    later = pd.DataFrame({"date": pd.bdate_range("2019-03-05", periods=4), "h": 101.0, "l": 99.0, "c": 100.0})
    lab = B.labels(after, 100.0, 2.0, later, "main", day)
    c = B.cost_frac("main", 100.0, False, 1)
    assert np.isclose(lab["long_1.0"], 0.04 - c)          # target 104 hit on day 0
    assert lab["short_1.0"] < -0.02                         # stop at 102 hit first for the short
    both = pd.DataFrame({"o": [100.0], "h": [105.0], "l": [97.0], "c": [100.0]})
    assert np.isclose(B.labels(both, 100.0, 2.0, later.iloc[:0], "main", day)["long_1.0"],
                      -0.02 - B.cost_frac("main", 100.0, False, 1))   # both in one bar: the stop counts


def test_features_count_and_no_future_bars():
    day = pd.Timestamp("2019-03-04")
    rng = np.random.default_rng(0)
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.002, 78)))
    b = B.session(_session(day, c), day, pd.Timedelta(hours=16))
    daily = pd.DataFrame({"o": 20.0, "h": 20.4, "l": 19.6, "c": 20 * np.exp(np.cumsum(rng.normal(0, 0.01, 260))),
                          "v": 1e6})
    mkt = {"dow": 0.0}
    f1 = B.features(b, 20, daily, CTX, mkt, CURVE, "rvol")
    b2 = b.copy()
    b2.loc[21:, ["o", "h", "l", "c"]] *= 3                  # changing later bars must not change the inputs
    f2 = B.features(b2, 20, daily, CTX, mkt, CURVE, "rvol")
    assert len(f1) == 68
    assert all((np.isnan(f1[k]) and np.isnan(f2[k])) or f1[k] == f2[k] for k in f1)


from tracks.main import ch_study as S


def _srows(n_days=900, per_day=6, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=n_days)
    d = pd.DataFrame({"date": np.repeat(days, per_day), "symbol": np.tile([f"S{i}" for i in range(per_day)], n_days)})
    d["track"], d["trig"], d["cik"] = "main", "rvol", 1
    d["month"] = d["date"].dt.strftime("%Y-%m")
    d["k"] = d.groupby("month").cumcount()
    d["x1"] = rng.normal(size=len(d))
    for j, lab in enumerate(B.LABELS):
        d[lab] = 0.01 * d["x1"] * (1 if lab.startswith("long") else -1) + rng.normal(0, 0.01, len(d))
    return d


def test_splits_purge_and_years():
    r = _srows()
    a, v, b, t = S.splits(r, 2019)
    assert r["date"][a].max() < pd.Timestamp("2018-01-01") and (r["date"][v].dt.year == 2018).all()
    assert r["date"][b].max() <= pd.Timestamp("2018-12-24") and (r["date"][t].dt.year == 2019).all()


def test_decide_cut_and_size():
    scale = np.linspace(0, 1, 101)                         # validation percentiles of the largest prediction
    p = np.array([[0.5, 0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0.995, 0], [0.9, 0, 0, 0, 0, 0]])
    pick, trade, size = S.decide(p, 0.8, scale)
    assert list(pick) == [0, 4, 0] and list(trade) == [False, True, True]
    assert np.isclose(size[1], S.SIZE_HI) and np.isclose(size[2], S.SIZE_LO + (S.SIZE_HI - S.SIZE_LO) * 0.1 / 0.19)


def test_walk_forward_learns_a_planted_signal_and_controls_differ():
    r = _srows()
    fit = lambda X, Y, s: [np.linalg.lstsq(np.c_[X, np.ones(len(X))], np.nan_to_num(Y[:, j]), rcond=None)[0]
                           for j in range(Y.shape[1])]
    pred = lambda m, X: np.column_stack([np.c_[X, np.ones(len(X))] @ w for w in m])
    dec, ins = S.walk_forward(r, fit, pred, "test", log=lambda *a: None)
    assert set(dec["year"]) == {2019, 2020} and dec["trade"].mean() < 0.4
    p = S.with_control(dec, r)
    assert (p["symbol"] != "").all() and p["ctrl"].notna().all()
    w = S.daily(p)
    assert w["diff"].mean() > 0                            # the planted signal beats random same-day picks
    assert ins["date"].min() < pd.Timestamp("2019-01-01")


def test_render_shapes_and_no_nan():
    rng = np.random.default_rng(1)
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.003, 156)))
    intr = np.c_[c, c * 1.002, c * 0.998, c, rng.uniform(1e3, 1e4, 156)].astype(np.float32)
    intr[120:] = np.nan                                    # today ends at s
    dc = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    dly = np.c_[dc, dc * 1.01, dc * 0.99, dc, rng.uniform(1e5, 1e6, 60)].astype(np.float32)
    img = S.render(intr, dly)
    assert img.shape == (3, 64, 192) and np.isfinite(img).all() and img[0].sum() > 0 and img[2].sum() > 0


def test_build_month_end_to_end_on_one_synthetic_candidate():
    days = pd.bdate_range("2017-01-02", periods=80)
    day, prev = days[60], days[59]
    cal = pd.DataFrame({"date": days, "close": "16:00"})
    daily = pd.DataFrame({"symbol": "X", "date": days, "o": 20.0, "h": 20.2, "l": 19.8, "c": 20.0, "v": 78_000.0,
                          "ac": 20.0, "seg": 0})
    today = _session(day, [21.0] * 78, o=[21.0] * 78).assign(symbol="X", date=day)     # +5% gap at the open
    yday = _session(prev, [20.0] * 78).assign(symbol="X", date=prev)
    spy = pd.concat([_session(d, [300.0] * 78).assign(symbol="SPY", date=d) for d in (prev, day)])
    bars = pd.concat([today, yday, spy], ignore_index=True)
    one = {"date": day, "track": "main", "ac": 20.0, "c": 20.0, **CTX, "vol20": 78_000.0, "cik": 1}
    cand = pd.DataFrame([{"symbol": "X", **one}, {"symbol": "NOBARS", **one}, {"symbol": "NODAILY", **one}, {"symbol": "X", **one, "cik": float("nan")}])
    bars = pd.concat([bars, today.assign(symbol="NODAILY")], ignore_index=True)
    etf = pd.DataFrame({"symbol": "SPY", "date": days, "c": 300.0})
    ctxd = {"etf_raw": etf, "etf_all": etf, "vix": pd.DataFrame({"date": days, "vix": 15.0}), "sic": {1: "3570"}}
    rows, intr, dimg = C.build_month(cand, bars, daily, ctxd, {"main": CURVE}, cal)
    assert len(rows) == 2 and (rows["trig"] == "gap").all() and rows["cik"].tolist() == [1, -1]
    assert intr.shape == (2, 156, 5) and dimg.shape[0] == 2
