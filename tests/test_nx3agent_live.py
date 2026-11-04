"""NX3-AGENT shadow book on synthetic decisions (no network)."""
import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3agent_live as A
from tracks.news import nx3_study as S


def dec(symbol, minute, e, fired=True, etb=True):
    inp = {k: 0.0 for k in S.INPUTS} | {"vol20": 0.03, "mins_open": minute - 30, "r_open": 0.01, "f_earnings": 1}
    return {"status": "decided", "symbol": symbol, "t": f"2026-11-05T{9 + minute // 60:02d}:{minute % 60:02d}:00",
            "inputs": inp, "e": e, "fired": fired, "etb": etb, "prev_c": 20.0, "dv20": 1e7}


def test_events_are_the_trees_fired_trades_and_the_prompt_lists_them():
    d = [dec("A", 31, 0.02), dec("B", 32, -0.02, etb=False), dec("C", 33, 0.001, fired=False), dec("D", 34, -0.03)]
    _, rows = A.events(d, 0.01)
    assert [r["symbol"] for r in rows] == ["A", "D"] and [r["side"] for r in rows] == [1, -1]
    t = A.text(rows)
    assert t.startswith("1. long, confidence 2.00, news earnings") and "2. short" in t and "AAA" not in t
    assert A.parse("1: take\n2. FLIP\n3: maybe", 3) == ["TAKE", "FLIP", None]


def test_a_day_logs_once_from_the_first_day_and_settles_the_agents_choices(tmp_path):
    root = tmp_path / "nx3"
    root.mkdir()
    d = [dec("A", 31, 0.02), dec("D", 34, -0.03), dec("E", 40, 0.015), dec("F", 45, 0.0, fired=False)]
    call = lambda model, text: ("1: TAKE\n2: FLIP\n3: SKIP", {})
    assert A.close_day(root, d, pd.Timestamp("2026-11-04"), 1, 0.01, call) is None        # before the first day
    assert A.close_day(root, d, pd.Timestamp("2026-11-05"), 1, 0.01, call) is not None
    assert A.close_day(root, d, pd.Timestamp("2026-11-05"), 1, 0.01, call) is None        # once a day
    led = Ledger(root)
    rec = json.loads((root / led.entry("nx3agent_day", "2026-11-05")["file"]).read_text())
    assert [e["action"] for e in rec["events"]] == ["TAKE", "FLIP", "SKIP"] and rec["llm"]["prompt_sha256"] == A.PROMPT_SHA

    class Api:
        def settle_trades(self, ev, cal):      # every stock +10%: a long gains, a short loses
            return ev.assign(entry=20.0, exit_ac=22.0, ret=0.10, spy_ret=0.0, price=20.0, days=2)

    cal = pd.DataFrame({"date": pd.to_datetime(["2026-11-05", "2026-11-06"])})
    assert A.settle(root, Api(), cal, pd.Timestamp("2026-11-05")) == []
    A.settle(root, Api(), cal, pd.Timestamp("2026-11-06"))
    s = json.loads((root / led.entry("nx3agent_settle", "2026-11-05")["file"]).read_text())
    assert s["taken"] == 2 and s["fired"] == 3 and s["actions"] == {"0": "TAKE", "1": "FLIP", "2": "SKIP"}
    # the tree shorted D into a +10% day (loses); flipping it to a long wins, and skipping E (a winning long) forgoes it
    assert s["agent_minus_tree"] is not None and led.verify() == []
    fl = [t for t in s["trades"] if t["role"] == "flip"]
    assert len(fl) == 1 and fl[0]["side"] == 1


def test_a_missing_answer_counts_as_take():
    out = A.ask([{"side": 1, "conf": 1.2, "inputs": {}, "prev_c": 10.0}], call=lambda m, t: ("garbled", {}))
    assert out["actions"] == [None] and out["missing"] == 1
