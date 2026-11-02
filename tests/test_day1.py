"""DAY1 boss (tracks/day1/bot.py) on synthetic sleeve files and a fake API. No network."""
import json
import math

import pandas as pd
import pytest

from tracks.day1 import bot as D

LABELS = [f"{s}_{k}" for s in ("long", "short") for k in (1.0, 1.5, 2.0)]
DAY = pd.Timestamp("2026-11-03")
SESSIONS = [pd.Timestamp(x) for x in ("2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09",
                                      "2026-11-10", "2026-11-11")]


class FakeApi:
    def __init__(self, px, closes=None, quotes=None):
        self.px, self.cl, self.q = px, closes or {}, quotes or {}

    def quotes(self, syms):
        return {s: self.q[s] for s in syms if s in self.q}

    def last_trades(self, syms):
        return {s: (self.px[s], "2026-11-03T14:31:00Z") for s in syms if s in self.px}

    def closes(self, syms, d):
        return {s: self.cl[s] for s in syms if s in self.cl}


class FakeBroker:
    def __init__(self):
        self.sent = []

    def order(self, body):
        self.sent.append(body)
        return 200, {"status": "accepted"}


def nx3_day(tmp, decisions, cut=0.2):
    d = tmp / "sleeves" / "nx3" / "days" / "2026-11-03"
    d.mkdir(parents=True)
    (d / "ctx.json").write_text(json.dumps({"cut": cut}))
    (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decisions))


def dec(key, sym, e, vol=0.05, fired=True, t="2026-11-03 09:31:00"):
    return {"key": key, "symbol": sym, "status": "decided", "e": e, "fired": fired, "entry_iex": 10.0,
            "decided_at": t, "inputs": {"vol20": vol}}


def test_quoted_spread_and_moc_exit():
    assert D.quoted_spread((99.95, 100.05)) == pytest.approx(0.001)
    assert D.quoted_spread((100.0, 100.0)) is None and D.quoted_spread((0, 1.0)) is None
    assert D.quoted_spread((1.0, 1.5)) is None                  # wider than 10%: no usable quote
    full = D.round_trip(100.0, 200.0, 0.001)
    moc = D.round_trip(100.0, 200.0, 0.001, moc_exit=True)
    assert full == pytest.approx(0.001, abs=5e-5) and moc == pytest.approx(0.0005, abs=5e-5)


def test_round_trip_is_spread_plus_sale_fees():
    rt = D.round_trip(20.0, 100.0)
    assert 0.0041 < rt < 0.0045                                # main spread 0.41% + tiny SEC/TAF fees
    assert D.round_trip(2.0, 100.0) > 0.0065                    # penny spread
    assert D.round_trip(300.0, 50.0) < 0.005                    # fractional main shares
    assert math.isinf(D.round_trip(80.0, 50.0) if D.band(80.0) == "penny" else math.inf)


def test_size_scales_with_confidence():
    assert D.size_usd(0) == 50 and D.size_usd(1) == 200 and D.size_usd(0.5) == 125 and D.size_usd(3) == 200


def test_nx3_candidates_expected_move_and_sides(tmp_path):
    nx3_day(tmp_path, [dec("a", "AAA", 0.4), dec("b", "BBB", -0.3), dec("c", "CCC", 0.1, fired=False)])
    c, s, pool = D.nx3_cands(tmp_path / "sleeves", DAY)
    assert [x["symbol"] for x in c] == ["AAA"] and [x["symbol"] for x in s] == ["BBB"]
    assert c[0]["expected"] == pytest.approx(0.4 * 0.05 * math.sqrt(1.5))
    assert c[0]["conf"] == pytest.approx(1.0)                   # |e|/cut = 2 -> top
    assert {x["symbol"] for x in pool} == {"AAA", "BBB", "CCC"}


def run_fund(tmp_path, decisions, px, now="2026-11-03 09:32:00"):
    nx3_day(tmp_path, decisions)
    st = D.State(tmp_path / "state")
    br = FakeBroker()
    c, s, pool = D.nx3_cands(tmp_path / "sleeves", DAY)
    n = D.fund(st, FakeApi(px), br, pd.Timestamp(now), DAY, c, {"nx3": pool, "chrp": []}, SESSIONS, s)
    return st, br, n


def test_cost_check_funds_big_moves_only(tmp_path):
    # no quotes: the band average stands in. AAA: 0.4 x 5% x 1.22 = 2.4% >> 2 x 0.21% (MOC exit);
    # DDD: 0.21 x 1% x 1.22 = 0.26% < 0.42%
    st, br, n = run_fund(tmp_path, [dec("a", "AAA", 0.4), dec("d", "DDD", 0.21, vol=0.01), dec("q", "QQQ", 0.0, fired=False)],
                         {"AAA": 20.0, "DDD": 20.0, "QQQ": 30.0})
    assert n == 1
    logs = {x["symbol"]: x for x in D._jsonl(st.work / "2026-11-03" / "decisions.jsonl")}
    assert logs["AAA"]["funded"] and logs["DDD"]["reason"] == "cost check"
    pos = st.load("positions.json", [])
    bot = [p for p in pos if p["role"] == "bot"]
    assert len(bot) == 1 and bot[0]["usd"] == 200 and bot[0]["end"] == "2026-11-04"
    twin = [p for p in pos if p["role"] == "twin"]
    assert len(twin) == 1 and twin[0]["symbol"] != "AAA" and twin[0]["usd"] == 200
    assert br.sent[0]["notional"] == "200.00" and br.sent[0]["side"] == "buy"
    # a second tick sees nothing new
    c, s, pool = D.nx3_cands(tmp_path / "sleeves", DAY)
    assert D.fund(st, FakeApi({"AAA": 20.0}), br, pd.Timestamp("2026-11-03 09:33"), DAY, c,
                  {"nx3": pool, "chrp": []}, SESSIONS, s) == 0


def test_shorts_are_logged_never_sent(tmp_path):
    st, br, n = run_fund(tmp_path, [dec("b", "BBB", -0.4)], {"BBB": 20.0})
    assert n == 0 and br.sent == []
    pos = st.load("positions.json", [])
    assert [p["role"] for p in pos] == ["short"]


def test_overnight_cap_and_budget(tmp_path):
    decs = [dec(str(i), f"S{i}", 0.4, t=f"2026-11-03 09:31:{i:02d}") for i in range(4)]
    st, br, n = run_fund(tmp_path, decs, {f"S{i}": 20.0 for i in range(4)})
    reasons = [x.get("reason") for x in D._jsonl(st.work / "2026-11-03" / "decisions.jsonl")]
    # $200 each: two fit NX3's $500 budget and the $500 overnight cap, the third would make $600
    assert n == 2 and reasons.count("sleeve budget") == 2


def test_late_and_paused(tmp_path):
    st, br, n = run_fund(tmp_path, [dec("a", "AAA", 0.4)], {"AAA": 20.0}, now="2026-11-03 10:00:00")
    assert n == 0 and D._jsonl(st.work / "2026-11-03" / "decisions.jsonl")[0]["reason"] == "late"
    st2 = D.State(tmp_path / "s2")
    st2.save("account.json", {"realised": -120.0, "twin_realised": 0.0, "peak": 1000.0, "paused": False})
    c, s, pool = D.nx3_cands(tmp_path / "sleeves", DAY)
    n2 = D.fund(st2, FakeApi({"AAA": 20.0}), FakeBroker(), pd.Timestamp("2026-11-03 09:32"), DAY, c,
                {"nx3": pool, "chrp": []}, SESSIONS, s)
    assert n2 == 0 and st2.load("account.json", {})["paused"]


def chr_day(tmp, trig):
    d = tmp / "sleeves" / "chrp" / "work" / "2026-11-03"
    d.mkdir(parents=True)
    (d / "triggers.json").write_text(json.dumps(trig))


def test_chr_candidates_and_barrier_exit(tmp_path):
    pred = [0.06, 0.0, 0.0, 0.0, 0.0, 0.0]
    chr_day(tmp_path, {"PNY": {"label": "long_1.0", "pred": pred, "entry": 2.0, "atr": 0.1, "size": 0.10,
                               "decided": "2026-11-03 10:00:00", "trade": True},
                       "OTH": {"label": "long_1.0", "pred": [-0.01] + [0] * 5, "entry": 3.0, "atr": 0.2, "size": 0.0,
                               "decided": "2026-11-03 10:00:00", "reason": "below the cut"}})
    c, s, pool = D.chr_cands(tmp_path / "sleeves", DAY, LABELS)
    assert [x["symbol"] for x in c] == ["PNY"] and len(pool) == 2
    assert c[0]["expected"] > 0.06 and c[0]["exit"]["k"] == 1.0
    st = D.State(tmp_path / "state")
    br = FakeBroker()
    n = D.fund(st, FakeApi({"PNY": 2.0, "OTH": 3.0}), br, pd.Timestamp("2026-11-03 10:01"), DAY, c,
               {"nx3": [], "chrp": pool}, SESSIONS, s)
    assert n == 1 and br.sent[0]["type"] == "limit" and br.sent[0]["qty"] == "100"
    bot = [p for p in st.load("positions.json", []) if p["role"] == "bot"][0]
    assert bot["stop"] == pytest.approx(1.9) and bot["target"] == pytest.approx(2.2) and bot["end"] == "2026-11-10"
    # the stop trades through: exit at the trade below the stop
    k = D.check_barriers(st, FakeApi({"PNY": 1.85, "OTH": 3.0}), br, pd.Timestamp("2026-11-04 09:40"),
                         pd.Timestamp("2026-11-04"))
    assert k == 1
    cl = st.load("closed.json", [])
    assert cl[0]["exit_px"] == 1.85 and cl[0]["exit_why"] == "stop" and cl[0]["net"] < -0.07
    assert br.sent[-1]["side"] == "sell"


def test_after_close_settles_time_exits_once(tmp_path):
    st, br, n = run_fund(tmp_path, [dec("a", "AAA", 0.4)], {"AAA": 20.0, "TTT": 10.0})
    api = FakeApi({}, closes={"AAA": 21.0, "TTT": 10.0})
    assert D.after_close(st, api, pd.Timestamp("2026-11-03")) is True       # nothing ends today
    assert D.after_close(st, api, pd.Timestamp("2026-11-03")) is False      # logged once
    assert D.after_close(st, api, pd.Timestamp("2026-11-04")) is True
    cl = st.load("closed.json", [])
    bot = [x for x in cl if x["role"] == "bot"][0]
    assert bot["exit_px"] == 21.0 and bot["net"] == pytest.approx(0.05 - bot["round_trip"])
    acct = st.load("account.json", {})
    assert acct["realised"] == pytest.approx(200 * bot["net"])
    assert st.ledger.has("day1_day", "2026-11-04") and st.ledger.verify() == []


def test_resume_clears_pause(tmp_path):
    st = D.State(tmp_path)
    st.save("account.json", {"realised": -110.0, "twin_realised": 0.0, "peak": 1000.0, "paused": True})
    D.resume(tmp_path)
    a = st.load("account.json", {})
    assert not a["paused"] and a["peak"] == pytest.approx(890.0)


def test_positions_ending_today_free_the_budget(tmp_path):
    nx3_day(tmp_path, [dec("a", "AAA", 0.4)])
    st = D.State(tmp_path / "state")
    old = [{"id": f"o{i}", "role": "bot", "sleeve": "nx3", "symbol": f"OLD{i}", "date": "2026-11-02", "entry": 10.0,
            "usd": 250.0, "qty": 25.0, "round_trip": 0.004, "end": "2026-11-03", "exit": {"type": "close", "sessions": 1}}
           for i in range(2)]
    st.save("positions.json", old)
    c, s, pool = D.nx3_cands(tmp_path / "sleeves", DAY)
    n = D.fund(st, FakeApi({"AAA": 20.0, "OLD0": 10.0, "OLD1": 10.0}), FakeBroker(), pd.Timestamp("2026-11-03 09:32"),
               DAY, c, {"nx3": pool, "chrp": []}, SESSIONS, s)
    assert n == 1                      # $500 of NX3 closes today, so today's $200 entry fits the $500 budget


def test_tight_quote_lets_a_small_move_through(tmp_path):
    # 0.21 x 1% x 1.22 = 0.26%: fails on the 0.41% band average, passes on a 0.02% quoted spread with a MOC exit
    st, br, n = run_fund(tmp_path, [dec("d", "DDD", 0.21, vol=0.01)], {"DDD": 20.0})
    assert n == 0
    nx3 = tmp_path / "t2"
    nx3_day(nx3, [dec("d", "DDD", 0.21, vol=0.01)])
    st2 = D.State(nx3 / "state")
    c, s, pool = D.nx3_cands(nx3 / "sleeves", DAY)
    n2 = D.fund(st2, FakeApi({"DDD": 20.0}, quotes={"DDD": (19.998, 20.002)}), FakeBroker(),
                pd.Timestamp("2026-11-03 09:32"), DAY, c, {"nx3": pool, "chrp": []}, SESSIONS, s)
    assert n2 == 1


def test_penny_entry_limit_price_is_whole_cents():
    pos = {"symbol": "EVGO", "entry": 1.7845, "qty": 29, "usd": 52.0, "date": "2026-11-02"}
    o = D.entry_order(pos)
    assert o["type"] == "limit" and len(o["limit_price"].split(".")[1]) == 2


def test_non_fractionable_entry_retries_whole_shares():
    sent = []

    class B:
        def order(self, body):
            sent.append(body)
            if "notional" in body:
                return 403, {"message": 'asset "SPTX" is not fractionable'}
            return 200, {"status": "pending_new"}

    pos = {"symbol": "SPTX", "entry": 20.0, "usd": 105.0, "qty": 5.25, "date": "2026-11-02"}
    orders = []
    res = D.send(B(), D.entry_order(pos), orders)
    assert res[0] == 403
