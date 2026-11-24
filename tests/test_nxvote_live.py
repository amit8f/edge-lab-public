import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3_study as S
from tracks.news import nxvote_live as VL


class Fixed:
    """Effect read from the first input column, scaled per voter."""

    def __init__(self, scale, cut=2.0):
        self.scale, self.cut = scale, cut

    def effect(self, X):
        return X[:, 0] * self.scale * self.cut


def dec(sym, x0, minute, etb=True):
    inputs = {k: 0.0 for k in S.INPUTS}
    inputs[S.INPUTS[0]] = x0
    return {"symbol": sym, "t": f"2026-11-25 10:{minute:02d}:00", "status": "decided", "etb": etb, "inputs": inputs,
            "prev_c": 20.0, "dv20": 1e7}


def test_only_unanimous_events_trade(tmp_path, monkeypatch):
    models = {"LGBM": Fixed(1.0), "MLP": Fixed(1.0), "ENS": Fixed(1.0), "RFR": Fixed(0.5)}
    monkeypatch.setattr(VL.M, "shas", lambda: {})
    decisions = [dec("AAA", 2.5, 1), dec("BBB", 1.5, 2), dec("CCC", -3.0, 3), dec("DDD", -3.0, 4, etb=False),
                 dec("EEE", 0.0, 5), dec("FFF", 0.1, 6)]
    VL.close_day(tmp_path, decisions, pd.Timestamp("2026-11-25"), 1, models=models)
    rec = json.loads((tmp_path / "nxvote/days/2026-11-25.json").read_text())
    got = {(p["symbol"], p["side"]) for p in rec["picks"]}
    assert got == {("AAA", 1), ("CCC", -1)}          # BBB: RFR at 0.75 does not fire; DDD: hard to borrow
    a = [p for p in rec["picks"] if p["symbol"] == "AAA"][0]
    assert np.isclose(a["conf"], (2.5 * 3 + 1.25) / 4)
    assert len(rec["control"]) == 2
    assert VL.close_day(tmp_path, decisions, pd.Timestamp("2026-11-25"), 1, models=models) is None


def test_before_first_day(tmp_path):
    assert VL.close_day(tmp_path, [], pd.Timestamp("2026-11-24"), 1, models={}) is None
