"""K4 whole-market AI bot: accounting, parsing, shortlist, twins and one logged run, on synthetic data."""
import json

import pandas as pd
import pytest

from core.ledger import Ledger
from tracks.main import ai_mirror
from tracks.main import ai_wide as w

T0 = int(pd.Timestamp("2026-10-19 14:51", tz="UTC").timestamp())


def uni(n=50):
    return {"date": "2026-10-19", "as_of": "2026-10-16", "candidates": 5000,
            "stocks": [{"symbol": f"S{i:02d}", "dv": 1e9 - i, "prev": 100.0, "c5": 100.0, "c20": 100.0} for i in range(n)],
            "shortable": [f"S{i:02d}" for i in range(0, n, 2)]}


def data(n=50, move=lambda i: 1 + i / 1000):
    d = {f"S{i:02d}": {"last": 100.0 * move(i), "last_t": "x", "1h": [[T0 - 3600, 1, 1, 1, 100.0, 1, T0]]} for i in range(n)}
    d["SPY"] = {"last": 700.0, "last_t": "x", "1h": []}
    return d


def test_rebalance_sizes_fifths_keeps_and_closes():
    b = w.new_book()
    px = {"A": 10.0, "B": 20.0, "C": 50.0}
    f = w.rebalance(b, {"A": 1, "B": -1}, px, T0)
    assert [x["action"] for x in f] == ["open", "open"] and b["pos"]["A"]["notional"] == pytest.approx(w.START_USD * w.SLOT)
    assert b["cash"] == pytest.approx(w.START_USD - 2 * w.START_USD * w.SLOT * w.FEE)
    px2 = {"A": 11.0, "B": 18.0, "C": 50.0}
    f = w.rebalance(b, {"A": 1, "C": 1}, px2, T0 + 3600)
    assert [(x["action"], x["sym"]) for x in f] == [("close", "B"), ("open", "C")]
    assert b["pos"]["A"]["entry"] == 10.0                         # kept as it was
    assert b["trips"][0]["pnl"] == pytest.approx(200.0 - 100 * 18 * w.FEE)
    assert w.equity(b, px2) == pytest.approx(b["cash"] + 200.0)   # A: 200 shares up $1; B was closed; C is flat


def test_short_pays_borrow_and_cap_of_five():
    b = w.new_book()
    px = {f"X{i}": 10.0 for i in range(7)}
    w.rebalance(b, {f"X{i}": -1 for i in range(7)}, px, T0)
    assert len(b["pos"]) == w.MAX_POS
    cash = b["cash"]
    w.charge(b, T0 + 365 * 24 * 3600)
    assert cash - b["cash"] == pytest.approx(5 * w.START_USD * w.SLOT * w.BORROW)


def test_parse_drops_bad_symbols_unshortable_shorts_and_extras():
    allowed, short = {"AA", "BB", "CC", "DD", "EE", "FF", "GG"}, {"AA"}
    txt = json.dumps({"positions": [{"symbol": "aa", "side": "short", "reason": "r"}, {"symbol": "BB", "side": "short"},
                                    {"symbol": "ZZ", "side": "long"}, {"symbol": "CC", "side": "long"},
                                    {"symbol": "DD", "side": "long"}, {"symbol": "EE", "side": "long"},
                                    {"symbol": "FF", "side": "long"}, {"symbol": "GG", "side": "long"}], "summary": "s"})
    want, dropped, summ = w.parse(txt, allowed, short)
    assert want == {"AA": -1, "CC": 1, "DD": 1, "EE": 1, "FF": 1} and "short BB" in dropped and "long ZZ" in dropped
    assert w.parse("no idea", allowed, short)[0] is None
    assert w.parse('{"positions": []}', allowed, short)[0] == {}


def test_shortlist_biggest_absolute_moves_plus_holdings():
    d = data(move=lambda i: 1 + (i - 25) / 100)
    s = w.shortlist(uni(), d, ["S25"], n=4)
    assert s[:4] == ["S00", "S01", "S49", "S02"] and s[-1] == "S25"


def test_step_twin_follows_symbols_with_coin_sides_and_hold_buys_spy():
    st = w.init_state(T0)
    d, u = data(), uni()
    syms = w.shortlist(u, d, [])
    ans = json.dumps({"positions": [{"symbol": "S49", "side": "long"}, {"symbol": "S48", "side": "short"}], "summary": "x"})
    rec = w.step(st, u, d, syms, T0, ask=lambda s, t: ans, news={"S49": ["S49 beats estimates"]})
    assert set(st["books"]["C"]["pos"]) == {"S49", "S48"} and set(st["books"]["C-coin"]["pos"]) == {"S49", "S48"}
    assert {s: p["side"] for s, p in st["books"]["C-coin"]["pos"].items()} == {s: w.coin_side(s, T0) for s in ("S48", "S49")}
    assert st["books"]["HOLD"]["pos"]["SPY"]["notional"] == pytest.approx(w.START_USD / (1 + w.FEE))
    assert rec["decision"]["want"] == {"S49": 1, "S48": -1}
    rec2 = w.step(st, u, d, syms, T0 + 3600, ask=lambda s, t: '{"positions": [{"symbol": "S49", "side": "long"}]}')
    assert set(st["books"]["C-coin"]["pos"]) == {"S49"} and rec2["fills"]["C-coin"][0]["sym"] == "S48"
    w.step(st, u, d, syms, T0 + 7200, ask=lambda s, t: "garbage")     # unreadable: nothing changes
    assert set(st["books"]["C"]["pos"]) == {"S49"}


def test_context_shows_news_and_shortable():
    st = w.init_state(T0)
    txt = w.context(st["books"]["C"], uni(), data(), ["S00", "S01"], "2026-10-19 10:35", {"S00": ["Big news"]})
    assert "S00, 100, +0.0" in txt and "news: Big news" in txt and "S01, 100.1, +0.1, +0.1, +0.1, no" in txt


def test_run_logs_universe_and_hour_once_and_mirrors_c(tmp_path, monkeypatch):
    px_time = pd.Timestamp(T0, unit="s", tz="UTC")
    seen = []
    monkeypatch.setattr(ai_mirror, "mirror_headers", lambda b: None)
    kw = dict(universe=lambda now: uni(), fetch=lambda now, u, held: (px_time, data(), {}),
              hours=lambda d, s, now, p, ses: seen.append(sorted(s)),
              ask=lambda s, t: '{"positions": [{"symbol": "S49", "side": "long"}], "summary": "up"}',
              headlines=lambda syms, p: {"S49": ["h"]})
    rec = w.run(tmp_path, now=px_time + pd.Timedelta(minutes=16), **kw)
    assert rec["equity"]["C"] < w.START_USD and rec["headline_counts"] == {"S49": 1} and "SPY" in seen[0]
    led = Ledger(tmp_path / "wide")
    assert led.has("universe", "2026-10-19") and led.has("run", "2026-10-19T15:00")
    assert w.run(tmp_path, now=px_time + pd.Timedelta(minutes=16), **kw) is None
    readme = (tmp_path / "wide" / "README.md").read_text()
    assert "long S49" in readme and "bought S49" in readme
    assert ai_mirror.targets({"books": {"C": {"pos": {"S49": {"side": 1, "qty": 0.2}, "S01": {"side": -1, "qty": 0.17}}}}}, "C") \
        == {"S49": 0.2, "S01": 0.0}      # SCALE 1: a 0.17-share short rounds to no shares


def test_fetch_hour_uses_newer_iex_trades_inside_the_session(monkeypatch):
    now = pd.Timestamp("2026-10-19 15:07", tz="UTC")
    px = now - pd.Timedelta(minutes=16)
    o, c = pd.Timestamp("2026-10-19 13:30", tz="UTC"), pd.Timestamp("2026-10-19 20:00", tz="UTC")
    monkeypatch.setattr(w.k2, "market_window", lambda n: (px, {px.tz_convert(w.ET).date(): (o, c)}))
    bar = lambda p: [{"t": (px - pd.Timedelta(minutes=1)).isoformat(), "c": p, "v": 10}]
    monkeypatch.setattr(w.k2, "_bars", lambda syms, *a: {s: bar(10.0) for s in syms})
    monkeypatch.setattr(w, "iex_latest", lambda syms: {"S00": (11.0, now + pd.Timedelta(minutes=3)),
                                                       "S01": (9.0, px - pd.Timedelta(minutes=30))})
    _, d, _ = w.fetch_hour(now, uni(3), [])
    assert d["S00"]["last"] == 11.0 and d["S00"]["src"] == "iex"     # traded after the run began: still used
    assert d["S01"]["last"] == 10.0 and d["S01"]["src"] == "sip"      # older IEX trade: SIP kept


def test_groq_book_trades_on_its_own_answer_and_old_state_gains_it():
    st = w.init_state(T0)
    for n in ("C-groq", "C-groq-coin"):
        del st["books"][n]                       # a state saved before K7
    d, u = data(), uni()
    syms = w.shortlist(u, d, [])
    gem = json.dumps({"positions": [{"symbol": "S49", "side": "long"}]})
    grq = json.dumps({"positions": [{"symbol": "S48", "side": "short"}, {"symbol": "S47", "side": "long"}]})
    rec = w.step(st, u, d, syms, T0, ask=lambda s, t: gem, ask_b=lambda s, t: grq)
    assert set(st["books"]["C"]["pos"]) == {"S49"} and set(st["books"]["C-groq"]["pos"]) == {"S48", "S47"}
    assert set(st["books"]["C-groq-coin"]["pos"]) == {"S48", "S47"}
    assert rec["decision_C-groq"]["want"] == {"S48": -1, "S47": 1}
    w.step(st, u, d, syms, T0 + 600, ask=lambda s, t: gem, ask_b=lambda s, t: (_ for _ in ()).throw(RuntimeError("429")))
    assert set(st["books"]["C-groq"]["pos"]) == {"S48", "S47"}   # a failed call changes nothing


def test_slow_books_act_once_a_day_from_1530_et_on_the_same_answers():
    st = w.init_state(T0)
    d, u = data(), uni()
    syms = w.shortlist(u, d, [])
    at = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    a1 = json.dumps({"positions": [{"symbol": "S49", "side": "long"}, {"symbol": "S48", "side": "short"}]})
    a2 = json.dumps({"positions": [{"symbol": "S47", "side": "long"}]})
    for t in ("2026-10-28 19:37", "2026-10-29 19:00"):        # before K8's first day, then before 15:30 ET
        w.step(st, u, d, syms, at(t), ask=lambda s, x: a1, ask_b=lambda s, x: a1)
        assert not st["books"]["C-slow"]["pos"] and not st["books"]["C-groq-slow"]["pos"]
    w.step(st, u, d, syms, at("2026-10-29 19:31"), ask=lambda s, x: "garbage", ask_b=lambda s, x: "garbage")
    assert not st["books"]["C-slow"]["pos"]                   # an unreadable answer doesn't use up the day
    t = at("2026-10-29 19:37")                                # 15:37 ET: the first readable answer from 15:30
    rec = w.step(st, u, d, syms, t, ask=lambda s, x: a1, ask_b=lambda s, x: a2)
    assert {s: p["side"] for s, p in st["books"]["C-slow"]["pos"].items()} == {"S49": 1, "S48": -1}
    assert {s: p["side"] for s, p in st["books"]["C-slow-coin"]["pos"].items()} == {s: w.coin_side(s, t) for s in ("S48", "S49")}
    assert set(st["books"]["C-groq-slow"]["pos"]) == set(st["books"]["C-groq-slow-coin"]["pos"]) == {"S47"}
    assert [f["sym"] for f in rec["fills"]["C-slow"]] == ["S48", "S49"] and not rec["fills"]["C"]
    w.step(st, u, d, syms, at("2026-10-29 19:47"), ask=lambda s, x: a2, ask_b=lambda s, x: a1)   # later that day
    assert set(st["books"]["C"]["pos"]) == {"S47"} and set(st["books"]["C-slow"]["pos"]) == {"S49", "S48"}
    assert set(st["books"]["C-groq-slow"]["pos"]) == {"S47"}
    w.step(st, u, d, syms, at("2026-10-30 19:37"), ask=lambda s, x: a2, ask_b=lambda s, x: a1)   # the next day
    assert set(st["books"]["C-slow"]["pos"]) == {"S47"} and set(st["books"]["C-groq-slow"]["pos"]) == {"S49", "S48"}


def test_slow_book_holdings_are_priced_but_kept_off_the_shortlist(tmp_path, monkeypatch):
    px_time = pd.Timestamp(T0, unit="s", tz="UTC")
    st = w.init_state(T0)
    w.open_pos(st["books"]["C-slow"], "S05", 1, 100.0, T0, w.START_USD)     # S05's move is too small for the top 40
    w.open_pos(st["books"]["C-groq"], "S06", 1, 100.0, T0, w.START_USD)
    (tmp_path / "wide").mkdir()
    (tmp_path / "wide" / "books.json").write_text(json.dumps(st))
    fetched, seen = [], []
    monkeypatch.setattr(ai_mirror, "mirror_headers", lambda b: None)
    w.run(tmp_path, now=px_time + pd.Timedelta(minutes=16), universe=lambda now: uni(),
          fetch=lambda now, u, held: (fetched.append(held), (px_time, data(), {}))[1],
          hours=lambda d, s, now, p, ses: seen.append(s), ask=lambda s, t: '{"positions": []}',
          headlines=lambda syms, p: {})
    assert {"S05", "S06"} <= set(fetched[0]) and "S06" in seen[0] and "S05" not in seen[0]
