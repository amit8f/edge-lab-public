import io
import zipfile

import numpy as np
import pandas as pd

from tracks.crypto import intra_study as I


def _hours(n, seed=0, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    qv = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq="h"), "o": c, "h": c * 1.01, "l": c * 0.99, "c": c,
                         "v": qv / c, "qv": qv, "n": 100.0, "tbqv": qv * rng.uniform(0.3, 0.7, n)})


def test_inputs_use_only_the_past_and_labels_look_forward():
    h, lead = _hours(24 * 60), _hours(24 * 60, 1)
    fund = pd.DataFrame({"t": pd.date_range("2023-01-01", periods=180, freq="8h"), "rate": np.linspace(0, 1e-3, 180)})
    met = pd.DataFrame({"t": pd.date_range("2023-01-01", periods=24 * 60 * 12, freq="5min"), "oi": 1.0, "tls": 1.0})
    a = I.coin_rows(h, lead, fund, met, 0.005)
    h2 = h.copy()
    h2.loc[h2.index[-100:], ["c", "h", "qv", "tbqv"]] *= 3     # change only the last 100 hours
    b = I.coin_rows(h2, lead, fund, met, 0.005)
    cut = h["t"].iloc[-100]                                    # close time of the first changed candle is cut + 1h
    early = a["t"] <= cut
    assert (a["slot"] % 4 == 0).all() and len(a) == len(b)
    pd.testing.assert_frame_equal(a.loc[early, I.FEATS + ["trend"]], b.loc[early, I.FEATS + ["trend"]])
    i = a.index[early][-1]
    assert a.loc[i, "net24"] != b.loc[i, "net24"] or a.loc[i, "net8"] != b.loc[i, "net8"]
    t0 = a.loc[i, "t"]
    c = h.set_index(h["t"] + pd.Timedelta(hours=1))["c"]
    assert np.isclose(a.loc[i, "net8"], c[t0 + pd.Timedelta(hours=8)] / c[t0] - 1 - 0.005)


def test_trend_gate_reads_the_last_finished_daily_close():
    n = 24 * 40
    h = _hours(n)
    h["c"] = 100.0
    h.loc[h["t"] >= pd.Timestamp("2023-02-05 23:00"), "c"] = 110.0   # the candle closing at 00:00 on Feb 6
    g = I.regular(h)
    on = pd.Series(I.trend_on(g), index=g.index)
    assert on[pd.Timestamp("2023-02-05 20:00")] == 0 and on[pd.Timestamp("2023-02-06 00:00")] == 1
    assert on[pd.Timestamp("2023-01-20 00:00")] == 0                 # no close 28 days back yet


def test_take_keeps_one_open_trade_per_coin():
    t = pd.date_range("2024-01-01", periods=6, freq="4h")
    R = pd.DataFrame({"t": list(t) * 2, "symbol": ["A"] * 6 + ["B"] * 6})
    tk = I.take(R, np.ones(12), 0.5, 8)
    assert list(tk[tk.symbol == "A"]["t"]) == [t[0], t[2], t[4]]
    assert len(I.take(R, np.zeros(12), 0.5, 8)) == 0


def test_skip_filter_and_clustered_t():
    R = pd.DataFrame({"t": pd.date_range("2023-11-12", periods=400, freq="4h"), "symbol": "A",
                      "fund": np.r_[np.linspace(0, 1, 300), [5.0] * 100], "oi24": 0.0, "trend": 1.0})
    cuts = I.skip_cuts(R, pd.Timestamp("2024-01-01"))
    ok = I.gated(R, cuts)
    assert not ok[-100:].any() and ok[:280].all()
    R.loc[:9, "fund"] = np.nan
    assert I.gated(R, cuts)[:10].all()
    day = np.repeat(np.arange(5), 4)
    net = np.r_[np.full(4, 0.02), np.full(4, -0.01), np.full(4, 0.03), np.full(4, 0.01), np.full(4, 0.0)]
    assert I.clustered_t(net, day) > 0 and np.isfinite(I.clustered_t(net, day))


def test_metrics_parse():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m.csv", "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
                   "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
                   "2024-01-01 00:05:00,BTCUSDT,100.5,1,1,1,1,0.9\n")
    m = I.parse_metrics(buf.getvalue())
    assert m["oi"].iloc[0] == 100.5 and m["tls"].iloc[0] == 0.9 and m["t"].iloc[0] == pd.Timestamp("2024-01-01 00:05")


def test_every_model_fits_and_scores_with_missing_inputs():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(size=(300, len(I.FEATS))), columns=I.FEATS)
    R["oi4"] = np.nan
    R.loc[:5, "vratio"] = np.inf
    R["net8"] = rng.normal(size=300)
    for m in I.MODELS:
        st, (sc,) = I.fit_score(m, R, 8, [R.iloc[:50]])
        assert len(st) == 300 and len(sc) == 50 and np.isfinite(sc).all()
