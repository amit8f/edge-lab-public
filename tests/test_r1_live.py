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
