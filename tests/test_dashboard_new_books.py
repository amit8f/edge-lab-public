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
