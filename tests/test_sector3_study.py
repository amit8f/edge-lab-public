import numpy as np
import pandas as pd
import pytest

from tracks.main import sector3_study as S


def test_shapley_shares_add_up_to_the_full_r2():
    rng = np.random.default_rng(1)
    n = 2000
    m, s, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 1.0 * m + 0.5 * s + 0.2 * (m + s) + rng.normal(size=n)
    sh = S.shapley(y, {"a": m, "b": s, "c": z})
    assert sh["a"] + sh["b"] + sh["c"] == pytest.approx(sh["full"])
    assert sh["a"] > sh["b"] > abs(sh["c"]) and abs(sh["c"]) < 0.01


def test_big_stock_return_leaves_the_stock_out():
    d = pd.DataFrame({"date": ["d"] * 4, "pdv": [4.0, 3.0, 2.0, 1.0], "ret": [0.1, 0.2, 0.3, 0.4]})
    b = S.big_stock_return(d, n=2)
    assert b.tolist() == pytest.approx([0.2, 0.1, 0.15, 0.15])


def test_event_group_takes_the_first_that_applies():
    base = {k: False for k in ["earn", "news", "k8_other", "f_offer", "f_insider", "f_other"]
            + [f"t_{k}" for k in ["earnings", "guidance", "analyst", "deal", "fda", "offering", "contract"]]}
    rows = [dict(base, earn=True, news=True, t_deal=True, vr=5.0), dict(base, news=True, t_deal=True, t_offering=True, vr=1.0),
            dict(base, news=True, t_guidance=True, vr=1.0), dict(base, news=True, vr=1.0), dict(base, f_insider=True, vr=3.0),
            dict(base, vr=2.5), dict(base, vr=1.0)]
    g = S.event_group(pd.DataFrame(rows)).tolist()
    assert g == ["earn", "news_deal", "news_earnings", "news_other", "insider", "heavy_volume", "quiet"]


def test_filing_flags_mark_the_day_and_the_next_session():
    cal = pd.DatetimeIndex(pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]))
    f = pd.DataFrame({"cik": [1, 2, 3, 4], "form": ["8-K", "4", "424B4", "8-K"], "items": [["8.01"], None, None, ["2.02"]],
                      "filed": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-03"])})
    x = S.filing_flags(f, cal).set_index(["cik", "date"])
    assert x.loc[(1, pd.Timestamp("2023-01-04")), "k8_other"]
    assert x.loc[(2, pd.Timestamp("2023-01-05")), "f_insider"] and x.loc[(3, pd.Timestamp("2023-01-05")), "f_offer"]
    assert not x.loc[(4, pd.Timestamp("2023-01-03"))].any()            # an earnings 8-K is counted elsewhere


def test_chart_features_use_only_the_past():
    n = 80
    d = pd.DataFrame({"symbol": "A", "seg": 0, "date": pd.bdate_range("2023-01-02", periods=n),
                      "ac": np.linspace(10, 20, n), "h": 1.0, "l": 1.0, "c": 1.0, "vr": 1.0})
    f = S.chart_features(d)
    i = 70
    assert f["r1"].iloc[i] == pytest.approx(np.log(d["ac"].iloc[i - 1] / d["ac"].iloc[i - 2]))
    assert f["hi20"].iloc[i] == pytest.approx(0.0)         # a rising series sits at its 20-day high the day before
