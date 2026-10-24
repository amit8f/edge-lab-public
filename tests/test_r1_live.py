"""R1 forward shadow book (FWD1) on synthetic closes (no network)."""
import numpy as np
import pandas as pd

from tracks.main import r1_live as L
from tracks.main import r1_screen as R

IDX = pd.bdate_range("2025-01-01", periods=260)
T = IDX[-1]


def closes(n=60):
    out = {}
    for i in range(n):
        c = np.linspace(10, 20, len(IDX)) if i % 2 == 0 else np.linspace(20, 10, len(IDX))
        c = c.copy()
        c[-6:] = c[-6] * (1 + np.linspace(0, -0.10 + i / 1000, 6))   # S00 falls most over 5 days
        out[f"S{i:02d}"] = pd.DataFrame({"o": c, "c": c}, index=IDX)
    return out


def test_features_match_the_screen_definitions():
    f = L.features(closes(), ["S00", "S01", "MISSING"], T)
    assert list(f["symbol"]) == ["S00", "S01"]
    c = closes()["S00"]["c"].to_numpy()
    assert np.isclose(f["ret5"].iloc[0], c[-1] / c[-6] - 1)
    assert f["above200"].iloc[0] == (c[-1] > c[-200:].mean())


def test_record_picks_bottom_decile_uptrend_losers_with_a_control_and_top5():
    rec = L.record(closes(), sorted(closes()), T)
    f = L.features(closes(), sorted(closes()), T).sort_values(["ret5", "symbol"])
    low = f.head(len(f) // 10)
    assert [p["symbol"] for p in rec["picks"]] == low[low["above200"]]["symbol"].head(R.MAX_PICKS).tolist()
    assert rec["top5"] == [p["symbol"] for p in rec["picks"]][:5]
    assert len(rec["control"]) == len(rec["picks"]) and not set(rec["control"]) & set(rec["top5"])
    assert rec["control_seed"] == R.SEED + T.toordinal() and rec == L.record(closes(), sorted(closes()), T)


def test_record_needs_a_200_day_history_for_the_trend():
    short = {k: v.iloc[-100:] for k, v in closes().items()}
    assert L.record(short, sorted(short), T) is None


def test_meta3_keeps_the_better_half_by_the_three_ranks():
    cl = closes()
    picks = [{"symbol": s, "ret5": r} for s, r in (("S00", -0.20), ("S02", -0.15), ("S04", -0.12), ("S06", -0.10))]
    n8k = {"S00": 3, "S02": 0, "S04": 1, "S06": None}
    m = {p["symbol"]: p for p in L.meta3(picks, cl, T, n8k)}
    assert sum(p["keep"] for p in m.values()) == 2
    assert m["S06"]["n_8k"] is None and not m["S00"]["keep"]
    c = cl["S00"]["c"].to_numpy()
    assert np.isclose(m["S00"]["vol20"], np.std(np.diff(np.log(c[-21:])), ddof=1))


def test_eightk_counts_by_filing_date_and_missing_index():
    sessions = list(IDX[-10:])
    idx = pd.DataFrame({"form": ["8-K", "8-K/A", "4", "8-K"], "cik": [1, 1, 1, 2]})
    got = L.eightk_counts(["A", "B", "C"], T, sessions, lambda d: idx if d == T else idx.iloc[:0], {"A": 1, "B": 2})
    assert got == {"A": 2, "B": 1, "C": None}
    assert L.eightk_counts(["A"], T, sessions, lambda d: None, {"A": 1}) == {"A": None}
