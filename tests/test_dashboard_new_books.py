import json

from dashboard import build as B


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_live_tab_arms_for_vote_cost1_surv1(tmp_path):
    d = "2026-11-25"
    _w(tmp_path / "nxvote/days" / f"{d}.json", {"picks": [{"symbol": "AAA", "side": 1, "w": 0.06}]})
    _w(tmp_path / "nxvote/settled" / f"{d}.json", {"trades": [{"symbol": "AAA", "role": "picks", "net": 0.01}]})
    _w(tmp_path / "cost1/settled" / f"{d}.json", {"trades": [{"symbol": "BBB", "side": -1, "w": 0.05, "net_m": -0.002, "kept": True},
                                                            {"symbol": "CCC", "side": 1, "w": 0.05, "net_m": -0.03, "kept": False}]})
    _w(tmp_path / "surv1long/settled" / f"{d}.json", {"trades": [{"symbol": "DDD", "side": 1, "w": 0.07, "net": 0.04}]})
    out = B.arm_picks(tmp_path, d)
    assert out["VOTE"] == {"AAA": {"side": 1, "net": 0.01, "w": 0.06}}
    assert set(out["COST1"]) == {"BBB"} and out["SURV1"]["DDD"]["net"] == 0.04


def test_vote_and_allhi_copies_before_settlement(tmp_path):
    d = "2026-11-25"
    rows = [{"symbol": "AAA", "t": "2026-11-25 09:31", "vote_e": 1.6, "etb": True},
            {"symbol": "BBB", "t": "2026-11-25 09:32", "vote_e": -1.2, "etb": False},
            {"symbol": "CCC", "t": "2026-11-25 09:33", "vote_e": None}]
    live = B._vote_so_far(rows)
    assert set(live) == {"AAA"} and live["AAA"]["side"] == 1 and live["AAA"]["live"]
    _w(tmp_path / "nx3allhi/days" / f"{d}.json", {"picks": [{"symbol": "EEE", "side": 1, "w": 0.08}]})
    out = B.arm_picks(tmp_path, d)
    assert out["COST1"]["EEE"]["net"] is None and out["SURV1"]["EEE"]["side"] == 1
