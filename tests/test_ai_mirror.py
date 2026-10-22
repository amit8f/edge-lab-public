"""Alpaca mirrors of the paper bots: one paper account per book; order planning and the no-keys path (no network).
The planning tests pin SCALE 25 to their old sizes; the bots now run at SCALE 1."""
import json

import pytest

from tracks.main import ai_mirror as m


@pytest.fixture(autouse=True)
def scale_25(monkeypatch):
    monkeypatch.setattr(m, "SCALE", 25)


def test_mirror_is_one_to_one_for_the_10k_books(monkeypatch):
    monkeypatch.undo()
    assert m.SCALE == 1


def book(sym=None, side=1, qty=0.0):
    return {"pos": {"sym": sym, "side": side, "qty": qty} if sym else None}


def test_targets_per_book_and_shorts_rounded_to_whole_shares():
    st = {"books": {"A": book("QQQ", -1, 0.1734), "B": book("NVDA", 1, 0.2302)}}
    assert m.targets(st, "A") == {"QQQ": -4.0} and m.targets(st, "B") == {"NVDA": 5.755}
    assert m.targets({"books": {"A": book()}}, "A") == {}


def test_plan_orders_flip_closes_first_and_skips_dust():
    px = {"QQQ": 720.0, "SPY": 770.0, "NVDA": 217.0}
    o = m.plan_orders({"QQQ": -4.0, "SPY": 1.0}, {"QQQ": 3.5, "SPY": 1.0001, "NVDA": 2.0}, px)
    assert o == [("NVDA", "sell", 2.0), ("QQQ", "sell", 3.5), ("QQQ", "sell", 4.0)]


def test_each_book_goes_to_its_own_account_and_errors_never_raise(tmp_path):
    st = {"books": {"A": book("QQQ", -1, 0.1734),
                    "C": {"pos": {"NVDA": {"side": 1, "qty": 0.2302}}}}}
    sent = []
    def post(url, body, headers):
        sent.append((headers["k"], body["symbol"], body["side"], body["qty"]))
        return 200, {"id": "x1"}
    log = tmp_path / "mirror.jsonl"
    out = m.run(st, {"QQQ": 720.0, "NVDA": 217.0}, log, headers_for=lambda b: {"k": b}, post=post,
                positions=lambda h: {}, wait=lambda h, i: "filled")
    assert sent == [("A", "QQQ", "sell", "4.0"), ("C", "NVDA", "buy", "5.755")]
    assert out["A"]["orders"][0]["fill"] == "filled" and len(log.read_text().splitlines()) == 2
    only_a = m.run(st, {}, log, headers_for=lambda b: {"k": b} if b == "A" else None,
                   positions=lambda h: (_ for _ in ()).throw(RuntimeError("down")))
    assert list(only_a) == ["A"] and "down" in only_a["A"]["error"]
    assert json.loads(log.read_text().splitlines()[-1])["book"] == "A"


def test_no_keys_does_nothing(tmp_path, monkeypatch):
    assert m.ACCOUNTS["C"][0] == "ALPACA_MIRROR_B_KEY_ID"     # C took over B's paper account
    assert m.ACCOUNTS["B"][0] == "ALPACA_BOT_B_KEY_ID"        # never the EL-DATA key pair
    for k in ("ALPACA_MIRROR_KEY_ID", "ALPACA_MIRROR_B_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    assert m.run({"books": {}}, {}, tmp_path / "x") is None and not (tmp_path / "x").exists()
