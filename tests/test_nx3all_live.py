"""NX3-ALL shadow book and the NXS-LLM size log on synthetic decisions (no network)."""
import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx3all_live as A
from tracks.news import nx3_study as S


def dec(symbol, minute, e, fired=True, etb=True):
    inp = {k: 0.0 for k in S.INPUTS} | {"vol20": 0.03, "mins_open": minute - 30, "r_open": 0.01}
    return {"status": "decided", "symbol": symbol, "t": f"2026-11-02T{9 + minute // 60:02d}:{minute % 60:02d}:00",
            "inputs": inp, "e": e, "fired": fired, "etb": etb, "prev_c": 20.0, "dv20": 1e7}


def test_book_takes_every_fired_event_sized_by_confidence():
    d = [dec("S1", 31, 0.02), dec("S2", 32, -0.03, etb=False), dec("S3", 33, 0.001, fired=False),
         dec("S4", 34, -0.01), dec("S5", 35, 0.015)] + [dec(f"X{i}", 40 + i, 0.02) for i in range(4)]
    picks, ctrl = A.book(d, 0.01, 7)
    assert [p["symbol"] for p in picks] == ["S1", "S4", "S5", "X0", "X1", "X2", "X3"]
    assert np.allclose([p["w"] for p in picks], [0.10, 0.05, 0.075, 0.10, 0.10, 0.10, 0.10])
    assert [c["side"] for c in ctrl] == [p["side"] for p in picks] and [c["w"] for c in ctrl] == [p["w"] for p in picks]
    assert ctrl == A.book(d, 0.01, 7)[1]


def test_llm_answers_are_parsed_within_5_to_10_percent_and_fall_back():
    assert A.llm_parse("1: 7.5\n2: 12\n3: 5%\nnoise", 4) == [0.075, None, 0.05, None]
    picks, _ = A.book([dec("S1", 31, 0.02), dec("S2", 32, 0.01)], 0.01, 1)
    out = A.ask_llm(picks, call=lambda model, text: "1: 6\n2: oops")
    assert out["model"] == A.LLM_MODELS[0] and out["sizes"] == [0.06, None] and out["missing"] == 1

    def fails(model, text):
        raise RuntimeError("503")
    assert A.ask_llm(picks, call=fails)["sizes"] == [None, None]


def test_a_day_logs_from_the_first_day_and_settles(tmp_path):
    root = tmp_path / "nx3"
    root.mkdir()
    d = [dec("AAA", 31, 0.02), dec("BBB", 40, -0.01), dec("CCC", 45, 0.0, fired=False)]
    call = lambda model, text: "1: 8\n2: 9"
    assert A.close_day(root, d, pd.Timestamp("2026-10-30"), 1, 0.01, call) is None      # before FIRST_DAY
    assert A.close_day(root, d, pd.Timestamp("2026-11-02"), 1, 0.01, call) is not None
    assert A.close_day(root, d, pd.Timestamp("2026-11-02"), 1, 0.01, call) is None      # once a day
    led = Ledger(root)
    rec = json.loads((root / led.entry("nx3all_day", "2026-11-02")["file"]).read_text())
    assert [p["w_llm"] for p in rec["picks"]] == [0.08, 0.09] and rec["llm"]["prompt_sha256"] == A.PROMPT_SHA

    class Api:
        def settle_trades(self, ev, cal):
            ret = np.where(ev["role"] == "picks", 0.05, 0.0)
            return ev.assign(entry=20.0, exit_ac=21.0, ret=ret, spy_ret=0.0, price=20.0, days=2)

    cal = pd.DataFrame({"date": pd.to_datetime(["2026-11-02", "2026-11-03"])})
    assert A.settle(root, Api(), cal, pd.Timestamp("2026-11-02")) == []
    A.settle(root, Api(), cal, pd.Timestamp("2026-11-03"))
    s = json.loads((root / led.entry("nx3all_settle", "2026-11-02")["file"]).read_text())
    assert s["sized_pick_minus_control"] is not None and s["llm_sized_pick_minus_control"] is not None
    assert len(s["trades"]) == 4 and led.verify() == []
