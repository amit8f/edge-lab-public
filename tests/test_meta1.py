import numpy as np
import pandas as pd

from tracks.main import meta1_study as M


def frame(n=300, seed=1):
    idx = pd.bdate_range("2018-01-01", periods=n)
    c = 50 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"o": c * 0.999, "c": c, "rc": c * 2, "v": 1e6}, index=idx)


def test_stock_inputs_use_only_past_bars():
    d = frame()
    t = d.index[260]
    a = M.stock_inputs(d, [t])[t]
    later = d.copy()
    later.iloc[261:, :] *= 3.0          # changing the future must not change any input
    b = M.stock_inputs(later, [t])[t]
    assert {k: v for k, v in a.items() if k != "fwd5_check"} == {k: v for k, v in b.items() if k != "fwd5_check"}
    assert np.isclose(a["ret20"], d["c"].iloc[260] / d["c"].iloc[240] - 1)
    assert np.isclose(a["fwd5_check"], d["c"].iloc[265] / d["o"].iloc[261] - 1)
    assert np.isclose(a["log_dv20"], np.log(d["c"].iloc[241:261].median() * 2 * 1e6))


def test_a_gap_starts_a_new_series():
    d = frame()
    d.index = d.index[:150].append(d.index[150:] + pd.Timedelta(days=30))
    t = d.index[160]                    # 11 bars into the new series: too short
    assert M.stock_inputs(d, [t]) == {}
    t = d.index[180]                    # 31 bars: no 60-day or 200-day inputs across the gap
    r = M.stock_inputs(d, [t])[t]
    assert np.isnan(r["ret60"]) and np.isnan(r["dist_ma200"]) and np.isfinite(r["ret20"])


def test_walk_forward_learns_a_real_signal_and_passes():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2017-01-06", "2025-09-19", freq="W-FRI")
    rows = []
    for w in weeks:
        for k in range(12):
            z = rng.normal(size=len(M.INPUTS))
            rows.append(dict(zip(M.INPUTS, z), date=w, symbol=f"S{k}", mcap=1e9 if k % 2 else 5e9,
                             fwd5=0.01 * z[0] + rng.normal(0, 0.01) + 0.01))
    x = pd.DataFrame(rows)
    x["label"] = (x["fwd5"] > x.groupby("date")["fwd5"].transform("mean")).astype(int)
    t, imp = M.walk_forward(x)
    assert t["date"].min().year == 2019 and set(imp) == {str(y) for y in range(2019, 2026)}
    r = M.evaluate(t, imp)
    assert r["PRIMARY"]["passes"] and r["PRIMARY"]["auc_oos"] > 0.6
    assert abs(r["random_control_minus_all"]["mean"]) < r["PRIMARY"]["mean"] / 3
    assert 0.3 < r["PRIMARY"]["taken_share"] < 0.7


def test_noise_does_not_pass():
    rng = np.random.default_rng(1)
    weeks = pd.date_range("2017-01-06", "2025-09-19", freq="W-FRI")
    x = pd.DataFrame([dict(zip(M.INPUTS, rng.normal(size=len(M.INPUTS))), date=w, symbol=f"S{k}", mcap=1e9,
                           fwd5=rng.normal(0, 0.03)) for w in weeks for k in range(12)])
    x["label"] = (x["fwd5"] > x.groupby("date")["fwd5"].transform("mean")).astype(int)
    r = M.evaluate(*M.walk_forward(x))
    assert abs(r["PRIMARY"]["auc_oos"] - 0.5) < 0.05
