import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto import harness as H
from auto import harness2 as H2

ROOT = Path(__file__).resolve().parents[1]


def test_loop_files_pass_the_safety_check():
    for loop in H2.LOOPS:
        for f in ("start.py", "candidate.py"):
            assert H.check_source((ROOT / "auto" / "loops" / loop / f).read_text()) == [], (loop, f)


def test_candidates_start_as_their_start_recipe():
    for loop in H2.LOOPS:
        d = ROOT / "auto" / "loops" / loop
        assert (d / "start.py").read_text() == (d / "candidate.py").read_text(), loop


def test_book_takes_three_buys_and_three_shorts_by_time():
    t = pd.to_datetime(["2021-01-04 09:30", "2021-01-04 09:40", "2021-01-04 09:50", "2021-01-04 10:00",
                        "2021-01-04 10:10", "2021-01-04 10:20", "2021-01-04 10:30", "2021-01-04 10:40",
                        "2021-01-05 09:30"])
    f = pd.DataFrame({"d": t.normalize(), "t": t,
                      "net_long": [0.01, 0.02, 0.03, 0.04, -0.9, -0.9, -0.9, -0.9, -0.9],
                      "net_short": [-0.9, -0.9, -0.9, -0.9, 0.05, 0.06, 0.07, 0.08, -0.9]})
    daily, n = H2.book(f, [1, 1.5, 1, 2, -1, -3, -1, -1, 0.99])
    assert n == 6
    assert math.isclose(daily.iloc[0], (0.01 + 0.02 + 0.03 + 0.05 + 0.06 + 0.07) / 6)
    assert daily.iloc[1] == 0.0


def _events():
    d = pd.bdate_range("2019-01-02", "2020-12-31")
    n = len(d) * 3
    t = d.repeat(3) + pd.to_timedelta(np.tile([600, 610, 620], len(d)), "m")
    return pd.DataFrame({"d": d.repeat(3), "t": t, "next": d.repeat(3) + pd.Timedelta(days=1),
                         "kind": np.tile(["real", "real", "placebo"], len(d)), "mret": 0.0, "ret": 0.0,
                         "net_long": 0.001, "net_short": -0.002, "x": np.arange(n, dtype=float)})


def test_fit_sees_placebos_score_sees_no_outcomes_and_leaks_are_rejected():
    kinds = []

    class Peek:
        @staticmethod
        def fit(tr):
            kinds.append(set(tr["kind"]))
            assert "net_long" not in tr.columns and "mret" in tr.columns

        @staticmethod
        def score(m, f):
            assert not set(H2.OUTCOME) & set(f.columns)
            return np.ones(len(f))

    class DayMax:
        fit = Peek.fit

        @staticmethod
        def score(m, f):
            return f.groupby("d")["x"].transform("max").to_numpy() - f["x"].to_numpy() - 0.5

    res, _ = H2.walk_forward(_events(), Peek, (2020,))
    assert kinds[0] == {"real", "placebo"} and res["trades"] == 2 * len(pd.bdate_range("2020-01-01", "2020-12-31"))
    with pytest.raises(ValueError, match="after a row"):
        H2.walk_forward(_events(), DayMax, (2020,))
