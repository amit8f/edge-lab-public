import numpy as np
import pandas as pd

from tracks.main.longterm import screen_sd1 as S


def _panel(n_dates=3, n=10):
    rows = []
    for d in pd.date_range("2018-01-31", periods=n_dates, freq="ME"):
        for i in range(n):
            rows.append({"date": d, "cik": i, "symbol": f"S{i}", "gpa": i, "roa": i, "ep": i, "bm": i,
                         "issuance": -i, "assetgrowth": -i, "ret1": 0.01 * i})
    return pd.DataFrame(rows)


def test_composite4_needs_three_families_and_ranks_within_date():
    df = _panel()
    df.loc[0, ["ep", "bm", "issuance"]] = np.nan   # two families missing: dropped
    df.loc[1, ["ep", "bm"]] = np.nan               # one missing: kept on three
    c = S.composite4(df)
    assert len(c) == len(df) - 1
    assert c.loc[1, "n_fams"] == 3
    last = c[c["date"] == c["date"].max()]
    assert last.sort_values("score")["cik"].tolist() == list(range(10))


def test_long_short_costs_and_random_control():
    df = _panel(n=10).assign(score=lambda x: x["cik"])
    ls = S.long_short(df, "score")
    assert np.allclose(ls["ls"], (0.085 - 0.005) - 2 * S.COST)
    assert (ls["n_leg"] == 2).all()
    r = S.long_short(df, "score", seed=1)
    assert len(r) == 3 and not np.allclose(r["ls"], ls["ls"])


def test_stale_ratios_use_old_filings_priced_at_t(monkeypatch):
    feat = pd.DataFrame({"date": pd.to_datetime(["2019-06-28"]), "symbol": ["A"], "cik": [1]})
    seen = {}

    def fake(f, facts, rtab):
        seen["date"] = f["date"].iloc[0]
        return pd.DataFrame({"gpa": [0.3], "roa": [0.1], "assetgrowth": [0.05], "issuance": [0.0],
                             "ep": [0.10], "bm": [0.50], "mcap": [100.0]}, index=f.index)

    monkeypatch.setattr(S, "attach_fundamentals", fake)
    fresh = pd.DataFrame({"mcap": [200.0]}, index=feat.index)
    st = S.stale_ratios(feat, None, None, fresh)
    assert seen["date"] == pd.Timestamp("2018-06-28")
    assert st.loc[0, "gpa"] == 0.3
    assert np.isclose(st.loc[0, "ep"], 0.05) and np.isclose(st.loc[0, "bm"], 0.25)


def test_verdict_rules():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-31", periods=96, freq="ME")
    months = pd.PeriodIndex([pd.Period(d, "M") + 1 for d in dates], freq="M")
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    ff5 = pd.DataFrame(rng.normal(0, 0.04, (96, 5)), index=months, columns=cols)
    umd = pd.DataFrame({"Mom": rng.normal(0, 0.04, 96)}, index=months)
    fresh = pd.DataFrame({"date": dates, "ls": 0.01 + rng.normal(0, 0.01, 96)})
    good = S.verdict(fresh, fresh.assign(ls=fresh["ls"] * 0.8), ff5, umd)
    assert good["verdict"] == "replication holds"
    bad = S.verdict(fresh, fresh.assign(ls=rng.normal(0, 0.01, 96)), ff5, umd)
    assert bad["verdict"].startswith("fails")
    none = S.verdict(fresh.assign(ls=rng.normal(0, 0.01, 96)), fresh, ff5, umd)
    assert none["verdict"].startswith("nothing to replicate")
