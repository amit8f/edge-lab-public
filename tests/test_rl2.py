"""RL2 on synthetic data (no network): the event inputs count only what was public by each date, headline
inputs are missing before 2017, and the walk-forward finds a signal carried by a new input."""
import numpy as np
import pandas as pd
import pytest

from tests.test_rl1 import panel
from tracks.main.longterm import screen_rl2 as R


def test_event_inputs_use_only_the_past():
    uni = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB"], "cik": [1, 1, 2], "mcap": [1e9, 1e9, 2e9],
                        "date": pd.to_datetime(["2016-12-30", "2017-03-31", "2017-03-31"])})
    buys = pd.DataFrame({"cik": [1, 1], "filed": pd.to_datetime(["2017-03-01", "2017-04-03"]), "value": [5e6, 9e9], "owners": ["x", "y"]})
    an = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA"], "day": pd.to_datetime(["2017-03-01", "2017-03-20", "2017-04-01"]),
                       "direction": [1, 1, -1]})
    sue = pd.DataFrame({"cik": [1, 2, 2], "avail": pd.to_datetime(["2017-02-15", "2016-08-01", "2017-04-15"]),
                        "sue": [2.0, 1.0, -3.0]})
    nd = pd.DataFrame({"symbol": ["AAA"] * 3 + ["BBB"], "day": pd.to_datetime(["2017-03-01", "2017-03-15", "2017-03-31", "2017-04-01"])})
    d = R.event_inputs(uni, buys, an, sue, nd)
    assert d["insider"].tolist() == pytest.approx([0.0, 5e6 / 1e9, 0.0])
    assert np.isnan(d.loc[0, "analyst"]) and np.isnan(d.loc[0, "news_n"])          # before headlines start
    assert d.loc[1, "analyst"] == pytest.approx(2 / np.sqrt(2)) and d.loc[2, "analyst"] == 0
    assert d.loc[1, "news_n"] == 2 and d.loc[2, "news_n"] == 0                       # (date - 30d, date]
    assert d.loc[1, "sue"] == 2.0 and np.isnan(d.loc[2, "sue"])                      # BBB's only SUE is stale


def test_walk_forward_finds_a_new_input_signal():
    u, ends1 = panel(0.0)
    rng = np.random.default_rng(3)
    for k in R.EVENT_INPUTS:
        u[k] = rng.normal(size=len(u))
    u["ret1"] = 0.02 * np.tanh(u["sue"]) + rng.normal(0, 0.03, len(u))
    u = R.add_inputs(u)
    pred, ins, imp = R.walk_forward(u, ends1, years=[2020, 2021])
    base, _, _ = R.walk_forward(u, ends1, inputs=R.R1.INPUTS, years=[2020, 2021])
    res, ls = R.verdict(pred, ins, base)
    assert res["rl2"]["mean"] > 0 and res["minus_rl1_inputs"]["mean"] > 0 and not res["overfit_rule21"]
    assert max(imp["2020"], key=imp["2020"].get) == "r_sue"
