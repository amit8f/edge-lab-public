"""NXM shadow books on synthetic decisions (no network): the committed version 1 models load and score zero
without news, picks follow NX3's rules, the control is seeded, and a day logs and settles."""
import json

import numpy as np
import pandas as pd
import pytest

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S
from tracks.news import nxm_live as M


class Lin:
    """effect = k * lm_net, cut 1."""

    def __init__(self, k):
        self.k, self.cut = k, 1.0

    def effect(self, X):
        return self.k * X[:, S.INPUTS.index("lm_net")]


def dec(symbol, minute, lm_net, etb=True):
    inp = {k: 0.0 for k in S.INPUTS} | {"lm_net": lm_net, "n_articles": 1.0, "rv30": None}
    return {"status": "decided", "symbol": symbol, "t": f"2026-10-29T{9 + minute // 60:02d}:{minute % 60:02d}:00",
            "inputs": inp, "etb": etb, "prev_c": 20.0, "dv20": 1e7}


def test_committed_models_load_and_score_zero_without_news():
    models = M.load()
    has_k = (M.RES_K / "nxk_v1_meta.json").exists()     # NXK/NXK2 version 1 lands after the nxk_v1 stage runs
    assert set(models) == set(M.ARMS if has_k else M.NXM_ARMS + M.NXR_ARMS)
    X = np.array([[0.0 if k in N.NEWS_INPUTS else 0.3 for k in S.INPUTS]])
    for a, m in models.items():
        assert np.allclose(m.effect(X), 0), a
        assert np.isfinite(m.effect(np.where(np.isnan(X), 0, X) + 1)).all() and m.cut > 0


def test_book_takes_fired_events_in_time_order_with_quotas_and_borrow():
    d = [dec("S4", 40, 3), dec("S1", 31, 2), dec("S2", 35, 0.5), dec("S3", 36, -2, etb=False), dec("S5", 50, -4),
         dec("S6", 51, 5), dec("S7", 52, 6), dec("S8", 53, 7)]
    picks, ctrl = M.book(d, Lin(1), 7)
    assert [(p["symbol"], p["side"]) for p in picks] == [("S1", 1), ("S4", 1), ("S5", -1), ("S6", 1)]
    assert [c["side"] for c in ctrl] == [1, 1, -1, 1] and ctrl == M.book(d, Lin(1), 7)[1]


def test_a_day_logs_from_the_first_day_and_settles(tmp_path, monkeypatch):
    root = tmp_path / "nx3"
    root.mkdir()
    models = {a: Lin(1 if a != "LIN" else -1) for a in M.ARMS}
    d = [dec("AAA", 31, 2), dec("BBB", 40, -2), dec("CCC", 45, 0.1)]
    assert M.close_day(root, d, pd.Timestamp("2026-10-28"), 1, models) is None      # before FIRST_DAY
    assert M.close_day(root, d, pd.Timestamp("2026-10-29"), 1, models) is not None
    led = Ledger(root)
    rec = json.loads((root / led.entry("nxm_day", "2026-10-29")["file"]).read_text())
    assert rec["arms"]["LGBM"]["fired"] == 2 and [p["symbol"] for p in rec["arms"]["LIN"]["picks"]] == ["AAA", "BBB"]
    assert [p["side"] for p in rec["arms"]["LIN"]["picks"]] == [-1, 1]

    class Api:
        def settle_trades(self, ev, cal):
            t = ev.assign(entry=20.0, exit_ac=21.0, ret=0.05, spy_ret=0.0, price=20.0, days=2)
            return t

    cal = pd.DataFrame({"date": pd.to_datetime(["2026-10-29", "2026-10-30"])})
    assert M.settle(root, Api(), cal, pd.Timestamp("2026-10-29")) == []               # exit not closed yet
    M.settle(root, Api(), cal, pd.Timestamp("2026-10-30"))
    s = json.loads((root / led.entry("nxm_settle", "2026-10-29")["file"]).read_text())
    assert s["arms"]["ENS"]["pick_minus_control"] is not None and len(s["arms"]["LIN"]["trades"]) == 4
    assert led.verify() == []


def test_a_model_file_that_changed_is_refused(tmp_path):
    for f in M.RES.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    p = tmp_path / "nxm_v1_LIN.pkl"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="sha256"):
        M.load(tmp_path)


def test_nxr_arms_start_on_their_own_first_day(tmp_path):
    root = tmp_path / "nx3"
    root.mkdir()
    models = {a: Lin(1) for a in M.ARMS}
    d = [dec("AAA", 31, 2), dec("BBB", 40, -2)]
    M.close_day(root, d, pd.Timestamp("2026-10-29"), 1, models)
    M.close_day(root, d, pd.Timestamp("2026-10-30"), 1, models)
    led = Ledger(root)
    r29 = json.loads((root / led.entry("nxm_day", "2026-10-29")["file"]).read_text())
    r30 = json.loads((root / led.entry("nxm_day", "2026-10-30")["file"]).read_text())
    assert set(r29["arms"]) == set(M.NXM_ARMS) and set(r30["arms"]) == set(M.NXM_ARMS + M.NXR_ARMS)
    assert set(r30["models_sha256"]) == set(M.NXM_ARMS + M.NXR_ARMS)
    M.close_day(root, d, pd.Timestamp("2026-11-09"), 1, models)       # NXK-KRR and NXK2-CAT start 2026-11-09
    r9 = json.loads((root / led.entry("nxm_day", "2026-11-09")["file"]).read_text())
    assert set(r9["arms"]) == set(M.ARMS)


def test_all_book_takes_every_fired_event_sized_by_confidence():
    d = [dec("S4", 40, 3), dec("S1", 31, 2), dec("S2", 35, 0.5), dec("S3", 36, -2, etb=False), dec("S5", 50, -4),
         dec("S6", 51, 5), dec("S7", 52, 6), dec("S8", 53, 7)]
    picks, ctrl = M.book_all(d, Lin(1), 7)
    assert [(p["symbol"], p["side"]) for p in picks] == [("S1", 1), ("S4", 1), ("S5", -1), ("S6", 1), ("S7", 1),
                                                         ("S8", 1)]          # no 3+3 cap; S3 not borrowable
    assert np.allclose([p["w"] for p in picks], S.conf_sizes([2, 3, 4, 5, 6, 7]))
    assert [(c["side"], c["w"]) for c in ctrl] == [(p["side"], p["w"]) for p in picks]
    assert ctrl == M.book_all(d, Lin(1), 7)[1]


def test_all_books_log_from_their_first_day_and_settle(tmp_path):
    root = tmp_path / "nx3"
    root.mkdir()
    models = {a: Lin(1) for a in M.ARMS}
    d = [dec("AAA", 31, 2), dec("BBB", 40, -2), dec("CCC", 45, 0.1), dec("DDD", 46, 1.5)]
    assert M.close_day_all(root, d, pd.Timestamp("2026-11-13"), 1, models) is None      # before ALL_FIRST
    assert M.close_day_all(root, d, pd.Timestamp("2026-11-16"), 1, models) is not None
    assert M.close_day_all(root, d, pd.Timestamp("2026-11-16"), 1, models) is None      # once a day
    led = Ledger(root)
    rec = json.loads((root / led.entry("nxmall_day", "2026-11-16")["file"]).read_text())
    assert set(rec["arms"]) == set(M.ARMS) and [p["symbol"] for p in rec["arms"]["CAT"]["picks"]] == ["AAA", "BBB", "DDD"]

    class Api:
        def settle_trades(self, ev, cal):
            return ev.assign(entry=20.0, exit_ac=21.0, ret=np.where(ev["role"] == "picks", 0.05, 0.0), spy_ret=0.0,
                             price=20.0, days=2)

    cal = pd.DataFrame({"date": pd.to_datetime(["2026-11-16", "2026-11-17"])})
    assert M.settle_all(root, Api(), cal, pd.Timestamp("2026-11-16")) == []
    M.settle_all(root, Api(), cal, pd.Timestamp("2026-11-17"))
    s = json.loads((root / led.entry("nxmall_settle", "2026-11-16")["file"]).read_text())
    a = s["arms"]["ENS"]
    assert len(a["trades"]) == 6 and a["sized_pick_minus_control"] > 0 and a["book_pnl"] is not None
    assert led.verify() == []
