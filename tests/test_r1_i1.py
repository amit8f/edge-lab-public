"""R1 and I1 weekly screens on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main import r1_screen as R
from tracks.main import weekly_panel as W
from tracks.penny import i1_screen as I

CAL = list(pd.bdate_range("2016-01-04", "2017-06-30"))


def bars(close, open_=None, vol=1e6, index=None):
    idx = pd.DatetimeIndex(index if index is not None else CAL[:len(close)])
    c = np.asarray(close, dtype=float)
    o = c if open_ is None else np.asarray(open_, dtype=float)
    return pd.DataFrame({"o": o, "c": c, "rc": c, "v": vol, "sc": c}, index=idx)


def test_week_ends_and_hold_ends_stop_at_the_holdout():
    cal = list(pd.bdate_range("2025-09-15", "2025-10-10"))
    we = W.week_ends(cal)
    assert we[0] == pd.Timestamp("2025-09-19") and all(t.weekday() == 4 for t in we)
    e = W.hold_ends(cal, we, 5)
    # Fri 19 Sep: entry Mon 22, 5th day Fri 26 -> kept; Fri 26: exit Fri 3 Oct reaches the holdout
    assert e == {pd.Timestamp("2025-09-19"): pd.Timestamp("2025-09-26")}


def test_symbol_rows_ret5_200_day_and_forward():
    n = 260
    c = np.linspace(10, 20, n)
    c[-6:] = [20, 19, 18, 17, 16, 15]
    opens = c.copy()
    d = bars(np.r_[c, [15.5, 16, 16.5, 17, 18]], np.r_[opens, [15.2, 16, 16.5, 17, 18]])
    t = d.index[n - 1]
    ends = {h: {t: d.index[n - 1 + h]} if n - 1 + h < len(d) else {} for h in W.HOLDS}
    r = W.symbol_rows("AAA", d, [t], ends)[0]
    assert r["ret5"] == pytest.approx(15 / 20 - 1)
    assert r["above200"] == (15 > d["c"].iloc[n - 200:n].mean())
    assert r["fwd5"] == pytest.approx(18 / 15.2 - 1)
    assert np.isnan(r["fwd10"])


def test_symbol_rows_never_cross_a_gap_and_skip_tiny_stocks():
    idx = CAL[:100] + CAL[130:200]          # a 6-week gap starts a new series
    d = bars(np.r_[np.full(100, 3.0), np.full(70, 4.0)], index=idx)
    none = {h: {} for h in W.HOLDS}
    r = W.symbol_rows("GAP", d, [d.index[125]], none)[0]
    assert r["ret5"] == pytest.approx(0.0) and r["nb"] == 26 and not r["above200"]
    assert W.symbol_rows("GAP", d, [d.index[105]], none) == []      # 6 bars into the new series
    assert W.symbol_rows("LOW", bars(np.full(60, 0.5)), [CAL[50]], {h: {} for h in W.HOLDS}) == []


def test_forward_ends_at_the_last_close_when_a_series_stops():
    d = bars([10, 10, 11, 12], [10, 10, 11, 12])
    seg = np.zeros(4, dtype=int)
    assert W.forward(d, seg, 1, CAL[30]) == pytest.approx(12 / 11 - 1)
    assert np.isnan(W.forward(d, seg, 3, CAL[30]))


def panel(dates, n=60):
    rows = []
    for t in dates:
        for i in range(n):
            rows.append({"symbol": f"S{i:02d}", "date": t, "price": 20.0, "dv20": 3e6, "nb": 300, "cik": i,
                         "ret5": -0.30 + i / 100, "above200": i % 2 == 0, "fwd5": 0.05 if i < 6 else 0.0,
                         "fwd10": 0.0, "fwd20": 0.0, "main": True, "penny": False, "mcap": 1e9})
    return pd.DataFrame(rows)


def test_r1_picks_only_uptrend_losers_in_the_bottom_decile():
    dates = [pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-12")]
    sel = R.pick(panel(dates), "main", dates)
    p = sel[sel["role"] == "pick"]
    # bottom decile of 60 = S00..S05; above200 only for even ones
    assert set(p["symbol"]) == {"S00", "S02", "S04"}
    c = sel[sel["role"] == "control"]
    assert (c.groupby("date").size() == 3).all() and not set(c["symbol"]) & {"S00", "S02", "S04"}
    t = W.weekly(sel, R.COST)
    assert np.allclose(t["pick"], 0.05 - R.COST) and np.allclose(t["diff"], t["pick"] - t["control"])


def test_r1_control_draw_is_fixed_by_the_seed():
    dates = [pd.Timestamp("2018-01-05")]
    a = R.pick(panel(dates), "main", dates)
    b = R.pick(panel(dates), "main", dates)
    assert list(a["symbol"]) == list(b["symbol"])


def test_weekly_summary_and_alpha_shapes():
    dates = list(pd.bdate_range("2018-01-05", periods=80, freq="W-FRI"))
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"pick": rng.normal(0.01, 0.02, 80), "control": 0.0, "n_picks": 5}, index=dates)
    t["diff"] = t["pick"] - t["control"]
    s = W.summary(t, 1)
    assert s["weeks"] == 80 and s["ci95"][0] < s["mean"] < s["ci95"][1]
    cal = list(pd.bdate_range("2018-01-01", "2019-12-31"))
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(cal), 6)), index=cal,
                       columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    a = W.alpha(t, fac, cal)
    assert a["n"] == 80 and len(a["betas"]) == 6


def test_parse_form_idx_keeps_only_asked_forms():
    text = ("Form Type   Company Name   CIK   Date Filed   File Name\n" + "-" * 40 + "\n"
            "8-K         ACME CORP      123   2020-01-02   edgar/data/123/x.txt\n"
            "10-K        ACME CORP      123   2020-01-03   edgar/data/123/y.txt\n")
    d = W.parse_form_idx(text, {"8-K"})
    assert list(d["form"]) == ["8-K"] and d["cik"].iloc[0] == 123
    assert d["filed"].iloc[0] == pd.Timestamp("2020-01-02")


def i1_rows(t):
    rows = []
    for ind in (1, 2, 3):
        for i in range(12):          # big stocks: industry 1 up 6%, 2 up 3%, 3 flat
            rows.append({"symbol": f"B{ind}{i:02d}", "date": t, "main": True, "penny": False, "ind": ind,
                         "ret5": {1: 0.06, 2: 0.03, 3: 0.0}[ind], "fwd5": 0.0, "dv20": 5e6})
        for i in range(4):           # pennies: two lag, two already moved
            rows.append({"symbol": f"P{ind}{i}", "date": t, "main": False, "penny": True, "ind": ind,
                         "ret5": -0.01 if i < 2 else 0.10, "fwd5": 0.04, "dv20": 6e5})
    for i in range(10):
        rows.append({"symbol": f"Z{i}", "date": t, "main": False, "penny": True, "ind": np.nan,
                     "ret5": 0.0, "fwd5": 0.0, "dv20": 6e5})
    return pd.DataFrame(rows)


def test_i1_moves_and_picks():
    t = pd.Timestamp("2019-03-08")
    rows = i1_rows(t)
    mv = I.moves(rows, {t: 0.0})
    assert dict(zip(mv["ind"], mv["move"].round(4))) == {1: 0.06, 2: 0.03, 3: 0.0}
    sel = I.pick(rows, mv, [t])
    p = sel[sel["role"] == "pick"]
    # top 2 industries (1 and 2) both clear +2%; only their lagging pennies are picked, largest gap first
    assert list(p["symbol"]) == ["P10", "P11", "P20", "P21"]
    c = sel[sel["role"] == "control"]
    assert len(c) == 4 and not set(c["symbol"]) & set(p["symbol"])


def test_i1_needs_ten_big_stocks_and_a_2pct_move():
    t = pd.Timestamp("2019-03-08")
    rows = i1_rows(t)
    rows = rows[~((rows["ind"] == 1) & rows["main"] & (rows["symbol"] > "B103"))]   # industry 1: 4 big
    mv = I.moves(rows, {t: 0.015})
    assert set(mv["ind"]) == {2, 3}
    assert I.pick(rows, mv, [t]).empty          # industry 2's move is now +1.5%
