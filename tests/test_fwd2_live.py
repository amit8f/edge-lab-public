"""FWD2 forward books I1F and A1F on synthetic rows (no network)."""
import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.main.longterm import live
from tracks.penny import a1_screen, fwd2_live, i1_screen

T = pd.Timestamp("2026-10-30")


def panel():
    """12 big stocks per industry for industries 6 (up 6% in the week) and 9 (flat), pennies in both."""
    rows = []
    for ind, r5 in ((6, 0.06), (9, 0.0)):
        for k in range(12):
            rows.append({"symbol": f"B{ind}{k:02d}", "cik": ind * 100 + k, "price": 50.0, "dv20": 5e7, "nb": 300,
                         "ret5": r5, "above200": True})
        for k, r in enumerate((-0.05, 0.01, 0.09, 0.0)):
            rows.append({"symbol": f"P{ind}{k}", "cik": ind * 1000 + k, "price": 2.5, "dv20": 1e6, "nb": 100,
                         "ret5": r, "above200": False})
    rows.append({"symbol": "THIN", "cik": 1, "price": 2.0, "dv20": 1e5, "nb": 100, "ret5": -0.1, "above200": False})
    return pd.DataFrame(rows).assign(date=T)


def ind_map(p):
    return pd.Series({int(c): (6 if str(c).startswith("6") else 9) for c in p["cik"]}, dtype="int64")


def test_flags_and_i1f_picks_lagging_pennies_of_the_leading_industry():
    p = panel()
    rows = fwd2_live.flags(p, [s for s in p["symbol"] if s.startswith("B")])
    assert rows["main"].sum() == 24 and rows["penny"].sum() == 8          # THIN fails the dv20 floor
    rec = fwd2_live.i1f_record(rows, ind_map(p), 0.01, T)
    assert [x["symbol"] for x in rec["picks"]] == ["P60", "P63", "P61"]    # by gap; P62 (+9%) caught up
    assert rec["industries"][0]["ind"] == 6 and np.isclose(rec["industries"][0]["move"], 0.05)
    assert len(rec["control"]) == 3 and not set(rec["control"]) & {"P60", "P61", "P63"}
    assert rec["control_seed"] == i1_screen.SEED + T.toordinal()
    assert rec == fwd2_live.i1f_record(rows, ind_map(p), 0.01, T)        # seeded
    assert rec["hold_sessions"] == 20 and rec["universe"] == sorted(rows.loc[rows["penny"], "symbol"])
    assert fwd2_live.i1f_record(rows, ind_map(p), 0.05, T) is None        # move 1% < 2%


def scored():
    f = pd.Timestamp("2026-09-15")
    return pd.DataFrame([
        {"cik": 6000, "accn": "a0", "filed": f, "m": -1.0, "sales": 5e7, "sgi": 1.2},        # flag, shortable
        {"cik": 6001, "accn": "a1", "filed": f, "m": -1.0, "sales": 1e8, "sgi": 5.0},        # SGI above 3
        {"cik": 6002, "accn": "a2", "filed": f, "m": -1.0, "sales": 1.2e7, "sgi": 1.5},      # prior sales $8M
        {"cik": 6003, "accn": "a3", "filed": f, "m": -1.0, "sales": 5e7, "sgi": 1.0},        # not easy to borrow
        {"cik": 9000, "accn": "a4", "filed": f, "m": -2.5, "sales": 5e7, "sgi": 1.0},        # M below the flag
        {"cik": 9001, "accn": "a5", "filed": pd.Timestamp("2026-06-01"), "m": -1.0, "sales": 5e7, "sgi": 1.0},
        {"cik": 9002, "accn": "a6", "filed": f, "m": -1.0, "sales": 5e7, "sgi": 1.0}])       # logged before


def test_a1f_exclusions_borrow_and_control():
    p = panel()
    rows = fwd2_live.flags(p, [])
    ok = {"P60", "P61", "P62", "P90", "P91", "P92", "P93"}
    rec = fwd2_live.a1f_record(rows, scored(), T, ok, {"a6"})
    assert [x["symbol"] for x in rec["picks"]] == ["P60"]
    assert {x["symbol"]: x["why"] for x in rec["skipped"]} == {
        "P61": "SGI above 3", "P62": "prior-year sales under $10M", "P63": "not easy to borrow"}
    assert rec["accns"] == ["a0", "a1", "a2", "a3"]
    assert len(rec["control"]) == 1 and rec["control"][0] in {"P90", "P91", "P92", "P93"}  # a6 was used before
    assert rec["control_seed"] == a1_screen.SEED + T.toordinal()
    assert rec["hold_sessions"] == 63 and rec["stop"] == 0.35
    assert fwd2_live.a1f_record(rows, scored().iloc[4:6], T, ok, set()) is None


def test_fwd2_log_writes_both_records_once(tmp_path, monkeypatch):
    p = panel()
    root = tmp_path / "longterm"
    root.mkdir()
    led = Ledger(root)
    cache = tmp_path / "cache"
    tmap = p[["symbol", "cik"]]
    common = pd.DataFrame({"base_symbol": p["symbol"], "shortable": True,
                           "easy_to_borrow": p["symbol"] != "P63"})
    idx = pd.bdate_range(end=T, periods=10)
    closes = {"SPY": pd.DataFrame({"o": 100.0, "c": [100.0] * 9 + [101.0]}, index=idx)}
    uni = pd.DataFrame({"symbol": [s for s in p["symbol"] if s.startswith("B")]})
    sic = {6: "3674", 9: "5411"}

    class R:
        def __init__(self, c):
            self.c = c

        def json(self):
            return {"sic": sic[int(str(self.c)[0])]}

    def get_sic(url):
        return R(int(url.split("CIK")[1][:10]))

    monkeypatch.setattr(a1_screen, "scores", lambda zp, ciks, d: scored())   # no companyfacts zip here
    for _ in range(2):
        live.fwd2_log(led, root, cache, p.drop(columns="cik").to_dict("records"), uni, tmap, common, closes, T,
                      True, True, tmp_path / "cf.zip", get_sic=get_sic)
    kinds = [e["kind"] for e in led.entries()]
    assert kinds == ["I1F_picks", "A1F_picks"]
    a1 = json.loads((root / led.entry("A1F_picks", f"{T:%Y-%m-%d}")["file"]).read_text())
    assert [x["symbol"] for x in a1["picks"]] == ["P60", "P92"] and (root / "sic" / "sic.json").exists()
    assert led.verify() == []


def test_fwd2_log_before_the_first_day_only_prints(tmp_path, monkeypatch, capsys):
    t0 = pd.Timestamp("2026-10-23")
    p = panel().assign(date=t0)
    root = tmp_path / "longterm"
    root.mkdir()
    led = Ledger(root)
    idx = pd.bdate_range(end=t0, periods=10)
    closes = {"SPY": pd.DataFrame({"o": 100.0, "c": [100.0] * 9 + [101.0]}, index=idx)}
    common = pd.DataFrame({"base_symbol": p["symbol"], "shortable": True, "easy_to_borrow": True})
    monkeypatch.setattr(a1_screen, "scores", lambda zp, ciks, d: scored())
    live.fwd2_log(led, root, tmp_path, p.drop(columns="cik").to_dict("records"),
                  pd.DataFrame({"symbol": [s for s in p["symbol"] if s.startswith("B")]}), p[["symbol", "cik"]],
                  common, closes, t0, False, False, tmp_path / "cf.zip", get_sic=lambda url: None)
    out = capsys.readouterr().out
    assert led.entries() == [] and "[A1F] dry run 2026-10-23" in out and "not logged" in out
