"""NX3-HI shadow book on synthetic decisions (no network)."""
import json

import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3hi_live as H


def dec(symbol, minute, e, fired=True, etb=True):
    return {"status": "decided", "symbol": symbol, "t": f"2026-11-04T{9 + minute // 60:02d}:{minute % 60:02d}:00",
            "e": e, "fired": fired, "etb": etb, "prev_c": 20.0, "dv20": 1e7}


def test_only_confident_events_take_a_slot_and_slots_fill_in_time_order():
    cut = 0.01
    d = [dec("LOW", 31, 0.012), dec("A", 32, 0.02), dec("B", 33, -0.02, etb=False), dec("C", 34, -0.03),
         dec("D", 35, 0.02), dec("E", 36, 0.02), dec("F", 37, 0.05), dec("NF", 38, 0.05, fired=False)]
    picks, ctrl = H.book(d, cut, 3)
    assert [p["symbol"] for p in picks] == ["A", "C", "D", "E"]      # LOW below 1.5x cut, B not borrowable, F side full
    assert [c["side"] for c in ctrl] == [p["side"] for p in picks]
    assert ctrl == H.book(d, cut, 3)[1]


def test_a_day_logs_from_the_first_day_once():
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    d = [dec("A", 32, 0.02), dec("B", 40, 0.001, fired=False)]
    assert H.close_day(root, d, pd.Timestamp("2026-11-03"), 1, 0.01) is None
    assert H.close_day(root, d, pd.Timestamp("2026-11-04"), 1, 0.01) is not None
    assert H.close_day(root, d, pd.Timestamp("2026-11-04"), 1, 0.01) is None
    rec = json.loads((root / Ledger(root).entry("nx3hi_day", "2026-11-04")["file"]).read_text())
    assert [p["symbol"] for p in rec["picks"]] == ["A"] and rec["hi"] == 1.5
