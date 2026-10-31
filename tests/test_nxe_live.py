"""NXE-EXL forward log on synthetic picks (no network): context, parsing and a logged day."""
import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nxe_live as X


def daily(n=20, start="2026-10-05"):
    idx = pd.bdate_range(start, periods=n)
    c = np.linspace(10, 11, n)
    return pd.DataFrame({"o": c, "h": c + 0.2, "l": c - 0.2, "c": c, "v": 1e5}, index=idx)


def test_context_uses_only_days_before_d():
    g = daily()
    d = g.index[-1]
    a, bars = X.context(g, d, 11.0)
    assert len(bars) == 5 and bars[-1][3] == round(g["c"].iloc[-2] / 11.0, 4)
    assert np.isclose(a, X.E.atr(g["h"][:-1].to_numpy(), g["l"][:-1].to_numpy(), g["c"][:-1].to_numpy()) / 11.0)
    assert X.context(g.head(10), d, 11.0) == (None, [])


def test_parse_keeps_levels_inside_the_clip():
    assert X.parse("1: 1.5, 3\n2: 0.2, 3\n3: 2 ATR, 4.5\n4: 7, 1", 5) == [(1.5, 3.0), None, (2.0, 4.5), None, None]


def test_a_day_logs_each_pick_with_the_answer(tmp_path):
    root = tmp_path / "nx3"
    root.mkdir()

    class Api:
        def daily(self, symbols, start, end, adjustment):
            assert adjustment == "all"
            return {s: daily(start="2026-10-01") for s in symbols}

    dec = [{"status": "decided", "symbol": "AAA", "t": "2026-11-02T09:31:00", "side": 1, "e": 0.02,
            "entry_iex": 11.0},
           {"status": "decided", "symbol": "BBB", "t": "2026-11-02T09:32:00", "side": None, "e": 0.0}]
    call = lambda model, user: ("1: 1.2, 2.4", {"totalTokenCount": 50})
    assert X.close_day(root, Api(), dec, pd.Timestamp("2026-10-30"), 0.01, call) is None     # before FIRST_DAY
    assert X.close_day(root, Api(), dec, pd.Timestamp("2026-11-02"), 0.01, call) is not None
    rec = json.loads((root / Ledger(root).entry("nxe_exl_day", "2026-11-02")["file"]).read_text())
    assert rec["picks"] == 1 and rec["trades"][0]["stop_atr"] == 1.2 and rec["trades"][0]["target_atr"] == 2.4
    assert rec["llm"]["model"] == X.MODELS[0] and rec["llm"]["prompt_sha256"] == X.PROMPT_SHA
