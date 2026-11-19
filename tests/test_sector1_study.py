"""SECTOR1 stage 1: synthetic daily rows only, no network."""
import numpy as np
import pandas as pd

from tracks.main import sector1_study as S


def rows(sym, n=80, px=20.0, seed=0, v=1e6):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2021-03-01", periods=n)
    c = px * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[px, c[:-1]] * (1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({"symbol": sym, "date": d, "o": o, "h": np.maximum(o, c) * 1.01, "l": np.minimum(o, c) * 0.99,
                         "c": c, "ao": o, "ac": c, "dv20": px * v, "seg": 0, "cik": 1})


def test_prepare_keeps_only_universe_days_and_computes_moves():
    d = pd.concat([rows("A"), rows("B", px=3.0, seed=1), rows("C", v=1e5, seed=2)], ignore_index=True)
    p = S.prepare(d)
    assert set(p["symbol"]) == {"A"} and len(p) == 79
    x = p.iloc[1]
    assert np.isclose(x["ret"], (1 + x["gap"]) * (1 + x["intra"]) - 1)


def test_sector_mean_leaves_the_stock_out_and_table_has_the_measures():
    parts = [rows(s, seed=i).assign(cik=i) for i, s in enumerate("ABCDE")]
    p = S.prepare(pd.concat(parts, ignore_index=True))
    p["sector"] = 6
    c = S.comove(p)
    one = c[c["date"] == c["date"].iloc[0]]
    assert np.allclose(one["sec"], (one["ret"].sum() - one["ret"]) / (len(one) - 1))
    t = S.sector_table(c)[0]
    assert t["sector"] == "Business equipment" and t["thin"] and t["stocks"] == 5
    assert 0 < t["overnight_share"] < 1 and 0 <= t["r2_market"] <= 1
