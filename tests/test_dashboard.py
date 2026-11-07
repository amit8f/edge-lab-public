"""Dashboard data step and tree export (synthetic data only, no network)."""
import json
from pathlib import Path

import numpy as np
import pytest

from dashboard import build as B
from dashboard import export_trees as E


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("ALPACA_"):
            monkeypatch.delenv(k, raising=False)


def test_walk_matches_xgboost(tmp_path):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 2]) + rng.normal(scale=0.1, size=400)
    m = xgb.train({"max_depth": 3, "eta": 0.3, "objective": "reg:squarederror", "base_score": 0.1},
                  xgb.DMatrix(X, label=y), num_boost_round=20)
    m.save_model(tmp_path / "m.json")
    meta = {"sha256": {"news": "x"}, "inputs": ["a", "b", "c", "d"], "fire_cut": 0.2}
    c = E.compact(json.loads((tmp_path / "m.json").read_text()), meta)
    margin = m.predict(xgb.DMatrix(X), output_margin=True)
    for i in range(len(X)):
        row = [None if np.isnan(v) else float(v) for v in X[i]]
        assert E.score(c, row) == pytest.approx(float(margin[i]), abs=1e-5)


def test_effect_blanks_only_news_inputs():
    tree = {"l": [1, -1, -1], "r": [2, -1, -1], "f": [0, 0, 0], "c": [0.5, -1.0, 1.0], "d": [True, False, False]}
    model = {"base_score": 0.0, "inputs": ["lm_neg", "px"], "no_news": {"lm_neg": 0.0}, "trees": [tree]}
    assert E.walk(tree, [None, 0])[1] == [0, 1]          # missing follows default_left
    assert E.effect(model, [0.9, 3.0]) == 2.0             # news 0.9 -> right (+1), blanked 0 -> left (-1)


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _state(tmp_path):
    st = tmp_path / "st"
    nx = st / "state-nx3" / "nx3"
    rows = [{"status": "decided", "symbol": "AAA", "e": 0.3, "fired": True, "side": "long", "entry_iex": 10.0,
             "inputs": {"lm_neg": 0.1}, "headline": "never copied"},
            {"status": "decided", "symbol": "BBB", "e": 0.01, "fired": False, "side": None, "entry_iex": 20.0},
            {"status": "missed", "symbol": "CCC"}]
    _w(nx / "days" / "2026-10-29" / "decisions.jsonl", "\n".join(json.dumps(r) for r in rows).replace("0.01", "NaN"))
    _w(nx / "nx3" / "days" / "2026-10-29.json", {"picks": [{"symbol": "AAA"}], "control": [{"symbol": "BBB", "side": "long"}]})
    _w(nx / "nx3" / "settled" / "2026-10-29.json", {"day": "2026-10-29", "trades": [
        {"symbol": "AAA", "role": "picks", "net": 0.02, "entry": 10.0, "exit_ac": 10.2, "t": "2026-10-29T14:00", "days": 2},
        {"symbol": "BBB", "role": "control", "net": -0.01}]})
    _w(nx / "nx3" / "settled" / "2026-10-27.json", {"day": "2026-10-27", "excluded": True,
                                            "trades": [{"symbol": "Z", "role": "picks", "net": 0.5}]})
    _w(st / "state-watch" / "watch" / "books.json", {"books": {
        "PICK-MAIN": {"start": 5000, "trips": [{"net": 50}], "pos": {"X": {"qty": 2, "entry": 10, "last": 11, "side": -1}}},
        "PICK-MAIN-coin": {"start": 5000, "trips": [], "pos": {}}}})
    _w(st / "state-ai" / "ai" / "books.json", {"books": {"B": {"trips": None, "pos": None}}})
    return st


def test_books_from_synthetic_state(tmp_path):
    st = _state(tmp_path)
    nx3 = B.nx3_books(st)[0]
    assert (nx3["closed"], nx3["excluded_days"]) == (1, ["2026-10-27"])     # the excluded bug day doesn't count
    assert nx3["realized"] == pytest.approx(10.0) and nx3["pnl_pct"] == pytest.approx(0.2)
    w = {b["name"]: b for b in B.watch_books(st)}
    assert w["PICK-MAIN"]["unrealized"] == -2 and w["PICK-MAIN-coin"]["control"]
    assert B.ai_books(st)[0]["closed"] == 0                                 # None trips don't crash
    assert B.paper_books(st)[0]["closed"] == 0 and B.nx1_book(st)[0]["closed"] == 0


def test_nx3_days_keeps_inputs_and_drops_text(tmp_path, monkeypatch):
    st = _state(tmp_path)
    monkeypatch.setattr(B, "skipped_moves", lambda day, nxt, rows: {"AAA": 0.03, "BBB": -0.02})
    d = B.nx3_days(st, tmp_path / "cache")[0]
    assert d["counts"] == {"seen": 3, "decided": 2, "missed": 1, "fired": 1, "picks": 1}
    a, b = d["events"]
    assert a["inputs"] == {"lm_neg": 0.1} and "headline" not in a and a["result"]["net"] == 0.02
    assert b["e"] is None and b["next_move"] == -0.02
    assert d["control"][0]["result"]["net"] == -0.01
    assert json.loads((tmp_path / "cache" / "nx3_moves_2026-10-29.json").read_text())["AAA"] == 0.03


def test_ideas_status(tmp_path):
    _w(tmp_path / "research" / "IDEAS.md", "\n".join([
        "| ID | Track | Registered | Idea | Prereg | Results | Outcome |", "|---|---|---|---|---|---|---|",
        "| X1 | main | 2026-10-01 | a | prereg/X1.md | | FAILED: t 0.4 |",
        "| X2 | main | 2026-10-01 | b | prereg/X2.md | | Registered |",
        "| NX3 | news | 2026-10-01 | c | prereg/NX3.md | | FAILED v1; PASS v2 |"]))
    got = {i["id"]: i["status"] for i in B.ideas(tmp_path, [])}
    assert got == {"X1": "failed", "X2": "open", "NX3": "live"}


def test_accounts_without_keys_and_build(tmp_path, monkeypatch):
    assert all(r["status"] == "secrets not set" for r in B.account_rows(get=None))
    st = _state(tmp_path)
    monkeypatch.setattr(B, "skipped_moves", lambda *a: {})
    _w(tmp_path / "research" / "IDEAS.md", "| ID | Track | Registered | Idea | Prereg | Results | Outcome |\n")
    res = B.build(st, tmp_path / "out", repo_root=tmp_path)
    j = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert j["books"][0]["name"] == "NX3" and j["jobs"] == [] and res["nx3"][0]["day"] == "2026-10-29"


def test_nx3_all_book_and_its_day_button(tmp_path):
    st = _state(tmp_path)
    root = st / "state-nx3" / "nx3"
    _w(root / "nx3all" / "days" / "2026-10-29.json", {"picks": [{"symbol": "AAA", "side": 1, "w": 0.08, "w_llm": 0.06}]})
    _w(root / "nx3all" / "settled" / "2026-10-29.json", {"day": "2026-10-29", "trades": [
        {"symbol": "AAA", "role": "picks", "w": 0.08, "net": 0.05}, {"symbol": "BBB", "role": "control", "w": 0.08, "net": 0.0}]})
    b = {x["name"]: x for x in B.nx3_books(st)}["NX3-ALL"]
    assert (b["closed"], b["open"]) == (1, 0) and b["realized"] == pytest.approx(20.0)
    assert B.arm_picks(root, "2026-10-29")["ALL"]["AAA"] == {"side": 1, "net": 0.05, "w": 0.08, "w_llm": 0.06}


def test_sp1_row(tmp_path):
    from core.ledger import Ledger
    led = Ledger(tmp_path / "state" / "sp1")
    led.write_json("SP1_candidate", "111", "candidates/111.json", {"candidate": True})
    led.write_json("SP1_candidate", "333", "candidates/333.json", {"candidate": False})
    led.write_json("SP1_entry", "111", "entries/111.json", {"symbol": "SPNA"})
    sp = next(b for b in B.log_books(tmp_path) if b["name"] == "SP1")
    assert (sp["closed"], sp["open"], sp["candidates"]) == (0, 1, 1)


def test_nx3all_picks_before_the_close(monkeypatch):
    cut = json.loads(Path("research/results/NX3/nx3_v1_meta.json").read_text())["fire_cut"]
    rows = [{"symbol": "A", "t": "2026-11-02T09:31", "e": 2 * cut, "fired": True, "etb": True},
            {"symbol": "B", "t": "2026-11-02T09:32", "e": -1.2 * cut, "fired": True, "etb": False},    # hard to borrow
            {"symbol": "C", "t": "2026-11-02T09:33", "e": -1.5 * cut, "fired": True, "etb": True},
            {"symbol": "D", "t": "2026-11-02T09:34", "e": 0.5 * cut, "fired": False, "etb": True}]
    got = B._nx3all_so_far(rows)
    assert set(got) == {"A", "C"} and got["A"]["side"] == 1 and got["C"]["side"] == -1
    assert got["A"]["w"] == pytest.approx(0.10) and got["C"]["w"] == pytest.approx(0.075)


def test_open_nx3_trades_count_in_the_book_percent():
    books = [B._book("NX3", "NX3", 100, 3, 0, realized=-21.76, size=5000.0),
             B._book("NX3-ALL", "NX3-ALL", 30, 0, 0, realized=0, size=5000.0), B._book("OTHER", "X", 1, 0, 0, realized=1, size=100.0)]
    days = [{"day": "2026-11-02", "settled": False, "excluded": None, "events": [
                {"symbol": "A", "side": 1, "entry_iex": 10.0, "last_price": 11.0, "arms": {"ALL": {"side": 1, "w": 0.1}}},
                {"symbol": "B", "side": -1, "entry_iex": 20.0, "last_price": 19.0, "arms": {}},
                {"symbol": "C", "side": None, "entry_iex": 5.0, "last_price": 6.0, "arms": {"ALL": {"side": -1, "w": 0.05}}}]},
            {"day": "2026-10-29", "settled": True, "excluded": None, "events": [
                {"symbol": "Z", "side": 1, "entry_iex": 1.0, "last_price": 9.0, "arms": {}}]}]
    B.mark_open_nx3(books, days)
    nx3, allb, other = books
    assert nx3["open"] == 2 and nx3["unrealized"] == pytest.approx(500 * 0.1 + 500 * 0.05)        # settled day ignored
    assert nx3["pnl_pct"] == pytest.approx(100 * (-21.76 + 75.0) / 5000)
    assert allb["open"] == 2 and allb["unrealized"] == pytest.approx(5000 * 0.1 * 0.1 + 5000 * 0.05 * -1 * 0.2)
    assert other["unrealized"] is None


def test_nx3all_has_no_provisional_picks_before_its_first_day(tmp_path):
    rows = [{"symbol": "A", "t": "2026-10-30T09:31", "e": 9.9, "fired": True, "etb": True}]
    assert "ALL" not in B.arm_picks(tmp_path, "2026-10-30", rows)
    assert "ALL" in B.arm_picks(tmp_path, "2026-11-02", rows)


def test_hi_books_live_picks_and_open_marks():
    cut = float(json.loads(Path("research/results/NX3/nx3_v1_meta.json").read_text())["fire_cut"])
    rows = [{"symbol": "AAA", "t": "2026-11-05T09:30:00", "fired": True, "e": 2.0 * cut, "etb": True},
            {"symbol": "BBB", "t": "2026-11-05T09:31:00", "fired": True, "e": 1.2 * cut, "etb": True},   # fires but below 1.5
            {"symbol": "CCC", "t": "2026-11-05T09:32:00", "fired": True, "e": -1.8 * cut, "etb": False}]  # hard to borrow
    hi = B._hi_so_far(rows, "HI")
    assert list(hi) == ["AAA"] and hi["AAA"]["side"] == 1
    assert list(B._hi_so_far(rows, "ALLHI")) == ["AAA"]
    day = {"settled": False, "events": [{"entry_iex": 10.0, "last_price": 11.0, "arms": {"HI": {"side": 1}, "ALLHI": {"side": 1, "w": 0.1}}}]}
    books = [{"name": "NX3-HI", "size": 5000.0, "realized": 0.0}, {"name": "NX3-ALL-HI", "size": 5000.0, "realized": 0.0}]
    B.mark_open_nx3(books, [day])
    assert books[0]["open"] == 1 and abs(books[0]["pnl_pct"] - 1.0) < 1e-9      # 500 * 10% = 50 on 5,000
    assert books[1]["open"] == 1 and abs(books[1]["pnl_pct"] - 1.0) < 1e-9      # 5000 * 0.1 * 10% = 50


def test_agent_arm_picks_take_skip_flip(tmp_path):
    (tmp_path / "nx3agent" / "days").mkdir(parents=True)
    (tmp_path / "nx3agent" / "settled").mkdir(parents=True)
    ev = [{"symbol": "AAA", "side": 1, "w": 0.01, "action": "TAKE"}, {"symbol": "BBB", "side": 1, "w": 0.01, "action": "SKIP"},
          {"symbol": "CCC", "side": -1, "w": 0.01, "action": "FLIP"}]
    (tmp_path / "nx3agent" / "days" / "2026-11-05.json").write_text(json.dumps({"events": ev}))
    (tmp_path / "nx3agent" / "settled" / "2026-11-05.json").write_text(json.dumps({"trades": [
        {"symbol": "AAA", "role": "tree", "net": 0.02}, {"symbol": "CCC", "role": "flip", "net": -0.01}]}))
    out = B.arm_picks(tmp_path, "2026-11-05")["AGENT"]
    assert list(out) == ["AAA", "CCC"]
    assert out["AAA"]["side"] == 1 and out["AAA"]["net"] == 0.02
    assert out["CCC"]["side"] == 1 and out["CCC"]["net"] == -0.01          # flipped short becomes a long


def test_workflow_timeouts(tmp_path):
    from dashboard import build as B
    w = tmp_path / ".github" / "workflows"
    w.mkdir(parents=True)
    (w / "a.yml").write_text("jobs:\n  x:\n    timeout-minutes: 15\n  y:\n    timeout-minutes: 350\n")
    (w / "b.yml").write_text("jobs:\n  x:\n    runs-on: ubuntu\n")
    assert B.workflow_timeouts(tmp_path) == {".github/workflows/a.yml": 350}
