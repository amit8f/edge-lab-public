"""core helpers on synthetic inputs (no network)."""
import json
import os

import pandas as pd

from core.alpaca import sip_end
from core.ledger import Ledger
from core.sec import parse_form_index


def test_sip_end_caps_at_now_minus_16_minutes():
    now = pd.Timestamp("2026-10-12T20:30:00Z")   # 16:30 ET
    assert sip_end("2026-10-12", now) == "2026-10-12T20:14:00Z"
    assert sip_end("2026-10-09", now) == "2026-10-10T03:59:59Z"


def test_ledger_chain_detects_tampering(tmp_path):
    led = Ledger(tmp_path)
    led.write_json("picks", "2026-10-12", "a.json", {"x": 1})
    led.write_json("picks", "2026-10-13", "b.json", {"x": 2})
    assert led.verify() == []
    p = tmp_path / "a.json"
    os.chmod(p, 0o644)
    p.write_text(json.dumps({"x": 99}))
    assert any("hash mismatch" in m for m in led.verify())


def test_parse_form_index():
    text = ("Description: Daily Index\n\nForm Type   Company Name   CIK   Date Filed  File Name\n"
            "-----------------------------------------------------------------\n"
            "4                ACME CORP                                 123456   20261012    edgar/data/123456/0001.txt\n"
            "NT 10-K          BETA INC                                  789      20261012    edgar/data/789/0002.txt\n")
    df = parse_form_index(text)
    assert list(df["form"]) == ["4", "NT 10-K"] and list(df["cik"]) == [123456, 789]
    assert df["filed"].iloc[0] == pd.Timestamp("2026-10-12")
