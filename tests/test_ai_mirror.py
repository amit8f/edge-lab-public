"""Alpaca mirror of the $100 bot: order planning and the no-keys path (no network)."""
import json

from tracks.main import ai_mirror as m


def book(sym=None, side=1, qty=0.0):
    return {"pos": {"sym": sym, "side": side, "qty": qty} if sym else None}


def test_targets_net_books_and_round_shorts_to_whole_shares():
    st = {"books": {"A": book("QQQ", -1, 0.1734), "B": book("NVDA", 1, 0.2302)}}
    assert m.targets(st) == {"QQQ": -4.0, "NVDA": 5.755}
    st = {"books": {"A": book("SPY", 1, 0.1), "B": book("SPY", -1, 0.04)}}
    assert m.targets(st) == {"SPY": 1.5}


def test_plan_orders_flip_closes_first_and_skips_dust():
    px = {"QQQ": 720.0, "SPY": 770.0, "NVDA": 217.0}
    o = m.plan_orders({"QQQ": -4.0, "SPY": 1.0}, {"QQQ": 3.5, "SPY": 1.0001, "NVDA": 2.0}, px)
    assert o == [("NVDA", "sell", 2.0), ("QQQ", "sell", 3.5), ("QQQ", "sell", 4.0)]


def test_run_sends_orders_logs_and_never_raises(tmp_path):
    st = {"books": {"A": book("QQQ", -1, 0.1734), "B": book()}}
    sent = []
    def post(url, body, headers):
        sent.append(body)
        return 200, {"id": "x1"}
    log = tmp_path / "mirror.jsonl"
    out = m.run(st, {"QQQ": 720.0}, log, headers={"k": "v"}, post=post, positions=lambda h: {}, wait=lambda h, i: "filled")
    assert sent == [{"symbol": "QQQ", "qty": "4.0", "side": "sell", "type": "market", "time_in_force": "day"}]
    assert out["orders"][0]["fill"] == "filled" and json.loads(log.read_text())["target"] == {"QQQ": -4.0}
    def boom(h):
        raise RuntimeError("down")
    assert "down" in m.run(st, {}, log, headers={"k": "v"}, positions=boom)["error"]


def test_no_keys_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_MIRROR_KEY_ID", raising=False)
    assert m.run({"books": {}}, {}, tmp_path / "x") is None and not (tmp_path / "x").exists()
