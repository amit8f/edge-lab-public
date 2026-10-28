"""RL1 on synthetic data (no network): month-end rows and the next month's return, the rank label, the
long-short decile with costs, the walk-forward's purge and the verdict."""
import numpy as np
import pandas as pd
import pytest

from tracks.main.longterm import screen_rl1 as R


def bars(n=700, seed=0):
    idx = pd.bdate_range("2016-01-04", periods=n)
    c = 20 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"o": c * 0.999, "c": c, "rc": c, "sc": c, "v": 1e5}, index=idx)


def test_symbol_rows_next_month_return_from_the_next_open():
    d = bars()
    me = pd.Series(d.index).groupby(d.index.to_period("M")).max().tolist()
    dates = me[13:16]
    ends1 = {dates[0]: dates[1], dates[1]: dates[2]}
    rows = pd.DataFrame(R.symbol_rows("AAA", d, dates, ends1)).set_index("date")
    i = d.index.get_loc(dates[0])
    assert rows.loc[dates[0], "ret1"] == pytest.approx(d["c"].loc[dates[1]] / d["o"].iloc[i + 1] - 1)
    assert np.isnan(rows.loc[dates[2], "ret1"]) and {"mom", "vol", "price", "dv20"} <= set(rows.columns)


def panel(signal, months=96, n=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=months, freq="ME")
    u = pd.DataFrame({"date": np.repeat(dates, n), "symbol": np.tile([f"S{i}" for i in range(n)], months)})
    for k in R.RAW_INPUTS:
        u[k] = rng.normal(size=len(u))
    u["ret1"] = signal * 0.02 * np.tanh(u["mom"]) + rng.normal(0, 0.05, len(u))
    ends1 = {t: dates[k + 1] for k, t in enumerate(dates[:-1])}
    return R.add_inputs(u), ends1


def test_rank_label_is_normal_scores_within_date():
    u, _ = panel(1.0)
    g = u[u["date"] == u["date"].iloc[0]]
    assert g["y_rank"].mean() == pytest.approx(0, abs=1e-9)
    assert g.loc[g["ret1"].idxmax(), "y_rank"] == pytest.approx(R.NormalDist().inv_cdf((60 - 0.5) / 60))
    assert g["r_mom"].between(0, 1).all()


def test_long_short_takes_deciles_and_costs():
    d = pd.DataFrame({"date": pd.Timestamp("2020-01-31"), "p": np.arange(20.0), "ret1": np.arange(20.0) / 100})
    ls = R.long_short(d, "p")
    assert ls["ls"].iloc[0] == pytest.approx(0.185 - 0.005 - 2 * R.COST - R.BORROW / 12) and ls["n_leg"].iloc[0] == 2


def test_walk_forward_trains_only_on_closed_labels_and_finds_a_signal():
    u, ends1 = panel(1.0)
    pred, ins = R.walk_forward(u, ends1, years=[2020, 2021])
    assert pred["date"].min() >= pd.Timestamp("2020-01-01") and len(ins["rank"]) == 2
    res, _, _ = R.verdict(pred, ins)
    assert res["rank"]["mean"] > 0 and res["ic"]["rank"] > 0.05 and "label_claim_passes" in res
    assert set(res["overfit_rule21"]) == {"raw", "rank"}


def test_walk_forward_purges_the_last_label(monkeypatch):
    u, ends1 = panel(0.0)
    seen = []
    monkeypatch.setattr(R, "fit", lambda X, y: seen.append(len(X)) or type("M", (), {"predict": lambda s, X: np.zeros(len(X))})())
    R.walk_forward(u, ends1, years=[2020])
    # December 2019's label closes on 2020-01-31, so training ends with November 2019
    assert seen[0] == u[(u["date"] <= "2019-11-30")]["ret1"].notna().sum()
