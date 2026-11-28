import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

from auto import decide as D
from auto import harness as H

ROOT = Path(__file__).resolve().parents[1]


def test_live_code_never_imports_auto():
    for top in ("tracks", "paper", "adaptive", "core", "dashboard"):
        for f in (ROOT / top).rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                       [n.module or ""] if isinstance(n, ast.ImportFrom) else []
                assert not any(m == "auto" or m.startswith("auto.") for m in mods), f


def test_check_source_rejects_unsafe_code():
    ok = "import numpy as np\ndef fit(t):\n    return 1\ndef score(m, f):\n    return np.zeros(len(f))\n"
    assert H.check_source(ok) == []
    assert H.check_source("import os\n" + ok)
    assert H.check_source("from subprocess import run\n" + ok)
    assert H.check_source(ok + "x = open('a')\n")
    assert H.check_source(ok + "y = ().__class__\n")
    assert H.check_source("def fit(t):\n    return 1\n")          # no score()


def test_starting_candidate_passes_the_check():
    assert H.check_source((ROOT / "auto" / "candidate.py").read_text()) == []


def test_book_takes_first_three_positive_by_time_and_cash_otherwise():
    f = pd.DataFrame({"d": pd.to_datetime(["2021-01-04"] * 5 + ["2021-01-05"]),
                      "t": pd.to_datetime(["2021-01-04 10:00", "2021-01-04 09:30", "2021-01-04 09:40",
                                           "2021-01-04 09:50", "2021-01-04 11:00", "2021-01-05 09:30"]),
                      "net": [0.01, 0.02, -0.01, 0.05, 0.03, 0.04]})
    daily, n = H.book(f, [1, 1, -1, 1, 1, -1])
    assert n == 3 and math.isclose(daily.iloc[0], (0.02 + 0.05 + 0.01) / 3) and daily.iloc[1] == 0.0


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(3)
    few, many = rng.normal(0, 0.02, 5), rng.normal(0, 0.02, 500)
    assert H.deflated_sharpe(0.08, few, 250, 0, 3) > H.deflated_sharpe(0.08, many, 250, 0, 3)


def test_keep_needs_the_margin():
    assert D.keep(1.0, None) and D.keep(1.06, 1.0) and not D.keep(1.04, 1.0) and not D.keep(None, 1.0)


def test_walk_forward_trains_only_on_earlier_years():
    seen = []

    class C:
        @staticmethod
        def fit(tr):
            seen.append(tr["next"].max())

        @staticmethod
        def score(m, f):
            return np.ones(len(f))

    d = pd.bdate_range("2019-01-02", "2022-12-30")
    ev = pd.DataFrame({"d": d, "t": d + pd.Timedelta(hours=10), "next": d + pd.Timedelta(days=1), "net": 0.001})
    res, _ = H.walk_forward(ev, C, (2020, 2021, 2022))
    assert all(s < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=13) for s, y in zip(seen, (2020, 2021, 2022), strict=True))
    assert res["trades"] == len(ev[ev["d"].dt.year >= 2020])


def test_score_never_sees_outcomes_and_leaks_are_rejected():
    import pytest

    d = pd.bdate_range("2019-01-02", "2020-12-31")
    ev = pd.DataFrame({"d": d.repeat(3), "t": d.repeat(3) + pd.to_timedelta(np.tile([600, 610, 620], len(d)), "m"),
                       "next": d.repeat(3) + pd.Timedelta(days=1), "net": 0.001, "x": np.arange(len(d) * 3, dtype=float)})

    class Peek:
        @staticmethod
        def fit(tr):
            return None

        @staticmethod
        def score(m, f):
            assert "net" not in f.columns and "ret" not in f.columns
            return np.ones(len(f))

    class DayMax:     # uses later rows of the same day: a leak
        fit = Peek.fit

        @staticmethod
        def score(m, f):
            return f.groupby("d")["x"].transform("max").to_numpy() - f["x"].to_numpy() - 0.5

    H.walk_forward(ev, Peek, (2020,))
    with pytest.raises(ValueError, match="after a row"):
        H.walk_forward(ev, DayMax, (2020,))
