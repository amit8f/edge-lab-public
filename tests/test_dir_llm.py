import json

import numpy as np
import pandas as pd

from tracks.main import boom2_study as B2
from tracks.news import dir_gate as G
from tracks.news import dir_llm as L


def test_mask_hides_ticker_and_name():
    t = L.mask("Acme Robotics Inc. (NASDAQ: ACMR) wins deal; Acme's shares rise, $ACMR up", "ACMR", "Acme Robotics, Inc.")
    assert "Acme" not in t and "ACMR" not in t and t.count("Company X") >= 3
    assert L.mask("American Airlines cuts outlook", "AAL", "American Airlines Group Inc") == "Company X cuts outlook"
    assert "American Express" in L.mask("American Express beats", "AAL", "American Airlines Group Inc")
    assert L.mask("MAC upgrade at ACMRX", "ACMR", None) == "MAC upgrade at ACMRX"


def test_prompt_fixed_question_and_hash():
    p = L.prompt(["Company X wins deal"], L.Q_BOOM)
    assert p.endswith("Answer with one letter.") and "- Company X wins deal" in p
    assert L.phash(p) == L.phash(L.prompt(["Company X wins deal"], L.Q_BOOM))
    assert "2025-03-04" in L.probe_prompt("2025-03-04") and "Headlines" not in L.probe_prompt("2025-03-04")


def test_p_up_and_letter():
    c = [{"token": "A", "logProbability": np.log(0.6)}, {"token": " B", "logProbability": np.log(0.2)},
         {"token": "The", "logProbability": np.log(0.2)}]
    assert np.isclose(L.p_up_from_logprobs(c), 0.75)
    assert np.isnan(L.p_up_from_logprobs([{"token": "x", "logProbability": 0.0}]))
    assert L.letter(" b.") == "B" and L.letter("maybe") is None


def test_top_third_uses_confidence_only():
    c = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    assert L.top_third(c).sum() == 2


def test_probe_band():
    pr = pd.DataFrame({"side": [1, -1] * 50, "truth": [1] * 100})
    r = L.probe_result(pr)
    assert r["hit_rate"] == 0.5 and r["within_band"]
    assert not L.probe_result(pd.DataFrame({"side": [1] * 100, "truth": [1] * 100}))["within_band"]


def _boom(n_days, edge, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-02-03", periods=n_days):
        for _ in range(12):
            lab = int(rng.choice([1, -1]))
            right = rng.random() < 0.5 + edge
            p = 0.9 if right else 0.6
            side = lab if right else -lab
            rows.append({"d": d, "open": 20.0, "entry": 100.0, "exitp": 100 * (1 + 0.04 * lab), "lab": lab,
                         "p_up": p if side == 1 else 1 - p})
    b = pd.DataFrame(rows)
    b["conf"], b["side"] = L.confidence(b["p_up"]), np.where(b["p_up"] > 0.5, 1.0, -1.0)
    return b


def test_boom_result_rules():
    r = L.boom_result(_boom(80, 0.2), True, B2.net)
    assert r["verdict"] == "passed" and r["trades"] >= 200
    assert L.boom_result(_boom(80, 0.2), False, B2.net)["verdict"] == "failed"     # probe outside the band
    assert L.boom_result(_boom(10, 0.2), True, B2.net)["verdict"] == "too few"


def test_nx3_result_too_few_and_gain():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-02-03", periods=150)
    allong = pd.DataFrame({"d": np.repeat(days, 6), "net": rng.normal(0, 0.01, 900)})
    a = allong.assign(p_up=rng.uniform(0.5, 1.0, 900))
    a["conf"], a["side"] = L.confidence(a["p_up"]), 1.0
    a.loc[L.top_third(a["conf"]), "net"] += 0.01
    r = L.nx3_result(a, allong, True, G.boot_pooled_diff)
    assert r["verdict"] == "passed" and r["trades"] == 300
    assert L.nx3_result(a.head(200), allong, True, G.boot_pooled_diff)["verdict"] == "too few"


def test_calls_resume_and_votes_fallback(tmp_path, monkeypatch):
    it = pd.DataFrame({"id": ["probe|a", "boom|b", "nx3|c"], "kind": ["probe", "boom", "nx3"],
                       "d": pd.to_datetime(["2025-03-03"] * 3), "prompt": ["p1", "p2", "p3"], "hash": ["h1", "h2", "h3"]})
    it.to_parquet(tmp_path / "items.parquet")
    monkeypatch.setattr(L, "MIN_GAP_SEC", 0)
    monkeypatch.setattr(L, "ask_logprobs", lambda p: None)
    monkeypatch.setattr(L, "ask_votes", lambda p: {"votes": ["A", "A", "B", "A", "A"], "p_up": 0.8})
    assert not L.stage_calls(tmp_path, float("inf"), max_requests=5)
    rows = [json.loads(x) for x in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["mode"] == "votes" and rows[0]["model"] == L.MODEL
    assert L.stage_calls(tmp_path, float("inf"))
    assert len((tmp_path / "answers.jsonl").read_text().splitlines()) == 3


def test_calls_stop_on_quota(tmp_path, monkeypatch):
    pd.DataFrame({"id": ["x", "y"], "kind": ["boom"] * 2, "d": pd.to_datetime(["2025-03-03"] * 2), "prompt": ["a", "b"],
                  "hash": ["h", "i"]}).to_parquet(tmp_path / "items.parquet")
    monkeypatch.setattr(L, "MIN_GAP_SEC", 0)
    calls = iter([{"text": "A", "top": [], "p_up": 0.7}])

    def ask(p):
        try:
            return next(calls)
        except StopIteration:
            raise L.QuotaExhausted("per day") from None
    monkeypatch.setattr(L, "ask_logprobs", ask)
    assert not L.stage_calls(tmp_path, float("inf"))
    assert json.loads((tmp_path / "mode.json").read_text())["mode"] == "logprobs"
    assert len((tmp_path / "answers.jsonl").read_text().splitlines()) == 1
