"""NX2 on synthetic events (no network): the effect is 0 without news, and the gate tells a real news
effect from a volatility-only one."""
import numpy as np
import pandas as pd

from tracks.news import nx1_study as N
from tracks.news import nx2_study as X

DAYS = pd.bdate_range("2017-01-02", "2019-12-31")
CAL = pd.DataFrame({"date": DAYS})


def events(news_effect, seed=0, per_night=24):
    """Real and placebo events; z = news_effect * tone + noise, scaled by a wide spread of volatilities."""
    rng = np.random.default_rng(seed)
    n = len(DAYS) * per_night
    kind = np.tile(np.r_[np.repeat("real", per_night // 2), np.repeat("placebo", per_night // 2)], len(DAYS))
    ev = pd.DataFrame({k: rng.normal(size=n) for k in N.INPUTS})
    real = kind == "real"
    nv = N.placebo_news()
    for k in N.NEWS_INPUTS:
        ev.loc[~real, k] = nv[k]
    ev.loc[real, "n_articles"] = rng.integers(1, 4, real.sum())
    ev["vol20"] = rng.uniform(0.005, 0.12, n)
    jump = rng.normal(size=n) * 1.0
    z = news_effect * np.where(real, np.tanh(ev["lm_net"]), 0) + jump
    ev["mret"] = z * ev["vol20"] * np.sqrt(3)
    ev["ret"] = ev["mret"]
    ev["kind"] = kind
    ev["d"] = np.repeat(DAYS, per_night)
    ev["exit"] = ev["d"] + pd.Timedelta(days=4)
    ev["price"] = 20.0
    ev["days"] = 4
    return ev


def test_zlabel_scales_by_volatility_and_drops_missing():
    ev = pd.DataFrame({"mret": [0.03, 0.03, 0.5, 0.01], "vol20": [0.01, 0.1, 0.01, 0.0]})
    z = X.zlabel(ev)
    assert np.isclose(z[0], 0.03 / (0.01 * np.sqrt(3))) and np.isclose(z[1], z[0] / 10)
    assert z[2] == X.Z_CLIP and np.isnan(z[3])


def test_no_news_effect_is_exactly_zero_and_news_inputs_only_are_reset():
    ev = events(1.0)
    m = X.Model(ev[ev["d"] < "2018-06-01"])
    pl = ev[ev["kind"] == "placebo"].head(500)
    assert np.all(m.effect(pl[N.INPUTS].to_numpy(float)) == 0) and not m.fires(np.zeros(3)).any()
    x = ev[N.INPUTS].head(3).to_numpy(float)
    x0 = X.no_news(x)
    keep = [N.INPUTS.index(k) for k in ["novelty"] + N.STOCK_INPUTS + N.MARKET_INPUTS]
    assert np.array_equal(x0[:, keep], x[:, keep]) and np.all(x0[:, N.INPUTS.index("lm_net")] == 0)


def test_gate_passes_a_news_effect():
    wf, pooled, gate = X.walk_forward(events(1.5, seed=1), CAL, years=[2019])
    assert gate == {"pick_minus_control": True, "pick_minus_shuffled": True}
    assert wf["2019"]["placebo_fire_rate"] == 0.0 and wf["2019"]["real_fire_rate"] > 0.05


def test_gate_fails_when_only_volatility_moves_stocks():
    wf, pooled, gate = X.walk_forward(events(0.0, seed=2), CAL, years=[2019])
    assert not any(gate.values()) and pooled["pick_minus_control"]["t_nw"] < 2
