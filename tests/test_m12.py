"""M12 (neutralised boom portfolio) and FF12 industries: synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from core import industry


def test_ff12_ranges_and_sic_cache(tmp_path):
    assert industry.ff12(7372) == 6 and industry.ff12(2834) == 10 and industry.ff12(6022) == 11
    assert industry.ff12(1311) == 4 and industry.ff12(None) == 12 and industry.ff12(9999) == 12
    calls = []

    class R:
        def __init__(self, sic):
            self.sic = sic

        def json(self):
            return {"sic": self.sic}

    def get(url):
        calls.append(url)
        return None if "0000000003" in url else R("7372" if "0000000001" in url else "4911")
    s = industry.industries([1, 2, 3], tmp_path, get=get)
    assert s.to_dict() == {1: 6, 2: 8, 3: 12} and len(calls) == 3
    industry.industries([1, 2, 3], tmp_path, get=get)
    assert len(calls) == 3                                         # cached


def test_capped_weights_and_industry_cap():
    from tracks.main.shortterm import screen_m12 as m12
    w = m12.capped_weights([0.1] + [1.0] * 19)
    assert w.sum() == pytest.approx(1) and w.max() <= 0.1 + 1e-9 and w[0] == pytest.approx(0.1)
    g = pd.DataFrame({"ind": [1] * 10 + list(range(2, 32)), "p": np.linspace(1, 0, 40)})
    picks = m12.select(g, np.arange(40), cap=4, n=20)
    assert len(picks) == 20 and (picks["ind"] == 1).sum() == 4


def test_neutralize_removes_volatility_and_size():
    from tracks.main.shortterm import screen_m12 as m12
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"date": pd.Timestamp("2020-01-03"), "rk_vol63": rng.uniform(size=n), "rk_lmcap": rng.uniform(size=n),
                       "ind": rng.integers(1, 13, n)})
    df["f"] = 2 * df["rk_vol63"] - df["rk_lmcap"] + 0.3 * (df["ind"] == 6) + rng.normal(0, 0.01, n)
    r = m12.neutralize(df, ["f"])["f"].astype(float)
    assert abs(np.corrcoef(r, df["rk_vol63"])[0, 1]) < 0.05 and r.std() < 0.02


def test_m12_end_to_end_with_fakes(tmp_path, monkeypatch):
    from tests.test_shortterm import _fake_world
    from tests.test_validation import _french_daily
    from tracks.main.shortterm import screen_m12 as m12
    from validation import v2_rejudge as v2
    pn, facts = _fake_world(tmp_path, monkeypatch)
    monkeypatch.setattr(m12.sec_data, "facts_for", lambda *a, **k: facts)
    (tmp_path / "cache" / "companyfacts.zip").write_bytes(b"")
    days = pd.bdate_range("2018-01-02", "2026-06-30")
    data = np.random.default_rng(5).normal(scale=0.01, size=(len(days), 7))
    ff5 = _french_daily(days, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], data[:, [0, 1, 2, 3, 4, 6]])
    mom = _french_daily(days, ["Mom"], data[:, [5]])
    files = {"ff5_daily": ff5, "mom_daily": mom}
    monkeypatch.setattr(v2.french, "fetch", lambda name, cache: (files[name], "sha"))
    monkeypatch.setattr(m12, "ROOT", tmp_path)                                  # no V2/V4 results: flat cost, no DSR
    ind = pd.Series({c: 1 + c % 5 for c in range(1, 61)})
    monkeypatch.setattr(m12, "TOP_N", 10)
    res = m12.run(tmp_path / "cache", tmp_path / "out", years=[2022, 2023], industries=ind)
    assert res["cost"] == 0.002 and res["primary"]["n_weeks"] > 20 and "passes" in res["primary"]
    ser = pd.read_csv(tmp_path / "out" / "M12_primary_series.csv")
    assert (ser["max_per_industry"] <= 4).all() and (ser["n_picks"] == 10).all()
    assert set(res["exploratory"]) >= {"not_neutralised_same_portfolio", "equal_weight", "no_industry_cap"}
