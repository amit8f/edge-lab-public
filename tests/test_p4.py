"""P4 insider-cluster penny screen on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main.longterm import sec_data
from tracks.penny import p4_screen as P

CAL = pd.bdate_range("2016-01-04", "2025-12-31")


def buy(cik, owner, filed, value=20_000.0, tdate=None, accn=None, sk="1"):
    return {"cik": cik, "owner": owner, "filed": pd.Timestamp(filed), "tdate": pd.Timestamp(tdate or filed),
            "value": value, "accn": accn or f"{cik}-{owner}-{filed}", "sk": sk}


# ------------------------------------------------------------------ DERA parser

def test_parse_owner_buys_roles_forms_and_codes():
    sub = (b"ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\n"
           b"a1\t05-JAN-2020\t4\t100\na2\t06-JAN-2020\t4/A\t100\na3\t07-JAN-2020\t5\t100\na4\t07-JAN-2020\t4\t100\n")
    own = (b"ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
           b"a1\t900\tDirector\na1\t901\tTenPercentOwner\na2\t902\tOfficer\na3\t903\tOfficer\na4\t904\tOther\n")
    tr = (b"ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
          b"a1\t1\t03-JAN-2020\tP\t1000\t2.5\tA\na1\t2\t03-JAN-2020\tS\t1000\t2.5\tD\n"
          b"a2\t1\t03-JAN-2020\tP\t1000\t2.5\tA\na3\t1\t02-JAN-2020\tP\t10\t3\tA\na4\t1\t02-JAN-2020\tP\t10\t3\tA\n")
    d = sec_data.parse_dera_owner_buys(sub, own, tr)
    assert sorted(d["owner"]) == [900, 901, 903]          # 10% owner kept, 4/A and "Other" dropped, S dropped
    assert set(d.loc[d["accn"] == "a1", "value"]) == {2500.0}
    assert d.loc[d["owner"] == 903, "tdate"].iloc[0] == pd.Timestamp("2020-01-02")


# ------------------------------------------------------------------ cluster and opportunistic rule

def test_cluster_two_owners_value_window():
    b = pd.DataFrame([buy(1, 10, "2020-03-01", 15_000), buy(1, 11, "2020-03-10", 15_000),
                      buy(2, 20, "2020-03-01", 15_000), buy(2, 21, "2020-03-11", 15_000),   # 11 days apart
                      buy(3, 30, "2020-03-02", 10_000), buy(3, 31, "2020-03-03", 10_000)])  # $20k only
    c = P.clusters(b).set_index(["cik", "date"])
    assert c.loc[(1, pd.Timestamp("2020-03-10")), "n_owners"] == 2
    assert c.loc[(1, pd.Timestamp("2020-03-10")), "value"] == 30_000
    assert c.loc[(2, pd.Timestamp("2020-03-11")), "n_owners"] == 1
    q = P.qualifies(c.reset_index()).to_numpy()
    flags = dict(zip(c.index, q))
    assert flags[(1, pd.Timestamp("2020-03-10"))]
    assert not flags[(2, pd.Timestamp("2020-03-11"))]
    assert not flags[(3, pd.Timestamp("2020-03-03"))]


def test_joint_filing_value_counted_once():
    b = pd.DataFrame([buy(1, 10, "2020-03-02", 30_000, accn="x", sk="1"), buy(1, 11, "2020-03-02", 30_000, accn="x", sk="1")])
    c = P.clusters(b)
    assert c["n_owners"].iloc[0] == 2 and c["value"].iloc[0] == 30_000


def test_opportunistic_rule():
    hist = [buy(1, 10, f"{y}-05-15") for y in (2017, 2018, 2019)]       # owner 10 buys every May: routine
    hist += [buy(1, 11, f"{y}-05-15") for y in (2017, 2019)]             # owner 11 missed 2018: opportunistic
    now = [buy(1, 10, "2020-05-20", 30_000), buy(1, 11, "2020-05-21", 30_000)]
    c = P.clusters(pd.DataFrame(hist + now)).set_index("date")
    assert c.at[pd.Timestamp("2020-05-21"), "n_opp"] == 1
    assert c.at[pd.Timestamp("2020-05-20"), "n_opp"] == 0
    keys = P.routine_keys(pd.DataFrame(hist).assign(filed=lambda d: pd.to_datetime(d["filed"]),
                                                    tdate=lambda d: pd.to_datetime(d["tdate"])))
    assert not P.is_opportunistic(1, 10, "2020-05-02", "2020-05-20", keys)
    assert P.is_opportunistic(1, 10, "2020-06-02", "2020-06-20", keys)          # other month
    assert P.is_opportunistic(1, 10, "2020-05-02", "2018-01-01", keys)          # history not yet filed
    # both routine -> the cluster fails rule 3
    hist2 = hist + [buy(1, 11, "2018-05-15")]
    c2 = P.clusters(pd.DataFrame(hist2 + now)).set_index("date")
    assert c2.at[pd.Timestamp("2020-05-21"), "n_opp"] == 0
    assert not P.qualifies(c2.reset_index()).iloc[-1]


# ------------------------------------------------------------------ trade path

def test_simulate_levels_gaps_and_stop_first():
    o, h, l, c = (np.array(x, float) for x in ([10, 10, 10], [10.5, 11, 11], [9.5, 9, 9], [10, 10, 10.5]))
    g, why, i = P.simulate(o, h, l, c, -0.2, 0.4)
    assert why == "end" and g == pytest.approx(0.05) and i == 2
    g, why, i = P.simulate(np.array([10, 10.]), np.array([10, 15.]), np.array([10, 7.]), np.array([10, 9.]), -0.2, 0.4)
    assert why == "stop" and g == pytest.approx(-0.2) and i == 1                  # both touched: stop first
    g, why, _ = P.simulate(np.array([10, 7.]), np.array([10, 8.]), np.array([10, 6.]), np.array([10, 7.]), -0.2, 0.4)
    assert why == "stop_gap" and g == pytest.approx(-0.3)                         # gap down fills at the open
    g, why, _ = P.simulate(np.array([10, 15.]), np.array([10, 16.]), np.array([10, 15.]), np.array([10, 15.]), -0.2, 0.4)
    assert why == "tp_gap" and g == pytest.approx(0.5)
    g, why, _ = P.simulate(np.array([10, 12.]), np.array([10, 14.5]), np.array([10, 11.]), np.array([10, 12.]), -0.2, 0.4)
    assert why == "tp" and g == pytest.approx(0.4)
    g, why, _ = P.simulate(np.array([10, 12.]), np.array([10, 14.5]), np.array([10, 7.]), np.array([10, 12.]), None, None)
    assert why == "end" and g == pytest.approx(0.2)
    g, why, _ = P.simulate(np.array([10.]), np.array([10.]), np.array([7.9]), np.array([9.]), -0.2, 0.4)
    assert why == "stop"                                                        # entry day's own low counts


def frame(n=400, px=3.0, vol=1_000_000, start="2016-06-01"):
    idx = CAL[CAL >= start][:n]
    return pd.DataFrame({"o": px, "h": px * 1.01, "l": px * 0.99, "c": px, "v": vol}, index=idx)


def test_symbol_rows_universe_entry_exit_21_days():
    raw = frame()
    adj = raw.copy()
    rows = P.symbol_rows("AAA", raw, adj, CAL).set_index("date")
    F = pd.Timestamp("2017-03-01")
    r = rows.loc[F]
    p = CAL.get_loc(F)
    assert r["entry_day"] == CAL[p + 1] and r["exit_day"] == CAL[p + 22]
    assert r["has_entry"] and not r["skip1"]
    assert r["base_why"] == "day21" and r["base_xday"] == CAL[p + 22]
    assert r["base_gross"] == pytest.approx(0.0)
    assert rows.index.min() >= P.SIGNAL_START


def test_symbol_rows_entry_limit_and_price_band():
    raw = frame()
    F = pd.Timestamp("2017-03-01")
    E = CAL[CAL.get_loc(F) + 1]
    raw.loc[E, "o"] = 3.0 * 1.011
    adj = raw.copy()
    rows = P.symbol_rows("AAA", raw, adj, CAL).set_index("date")
    assert rows.loc[F, "skip1"] and np.isnan(rows.loc[F, "base_gross"])
    raw2 = frame(px=5.0)
    assert len(P.symbol_rows("BBB", raw2, raw2.copy(), CAL)) == 0              # $5 is outside the band
    raw3 = frame(vol=100_000)
    assert len(P.symbol_rows("CCC", raw3, raw3.copy(), CAL)) == 0              # $300k dollar volume


def test_symbol_rows_delisting_exits_at_last_close():
    raw = frame(n=230)
    adj = raw.copy()
    adj.iloc[-1, adj.columns.get_loc("c")] = 3.3
    last = raw.index[-1]
    F = raw.index[-6]
    r = P.symbol_rows("AAA", raw, adj, CAL).set_index("date").loc[F]
    assert r["base_why"] == "last_close" and r["base_xday"] == last and r["base_gross"] == pytest.approx(0.1)


def test_symbol_rows_gap_ends_series():
    raw = frame(n=300)
    raw = raw.drop(raw.index[200:215])                                         # 21 calendar days missing
    rows = P.symbol_rows("AAA", raw, raw.copy(), CAL)
    after = rows[rows["date"] > raw.index[199]]
    assert after.empty or after["date"].min() >= raw.index[200 + 60 - 1 - 15] # needs 60 new bars after the gap
    before = rows[rows["date"] <= raw.index[199]].set_index("date")
    assert before.loc[raw.index[195], "base_why"] == "last_close"


def test_holdout_guard_and_rows_never_reach_it():
    raw = frame(n=2600)
    rows = P.symbol_rows("AAA", raw, raw.copy(), CAL)
    assert (pd.to_datetime(rows["exit_day"]) < P.HOLDOUT).all()
    assert pd.to_datetime(rows["exit_day"]).max() >= pd.Timestamp("2025-09-01")
    p = pd.DataFrame({"exit_day": [pd.Timestamp("2025-09-30")], "c_exit_day": [pd.Timestamp("2025-10-01")],
                      "xday": [pd.Timestamp("2025-09-30")]})
    with pytest.raises(AssertionError):
        P.holdout_guard(p)
    P.holdout_guard(p.assign(c_exit_day=pd.Timestamp("2025-09-30")))


# ------------------------------------------------------------------ exclusions

def test_reverse_split_detection_and_exclusion_order():
    raw = frame(n=400, vol=10_000_000)
    adj = raw.copy()
    d = raw.index[250]
    raw.loc[:raw.index[249], ["o", "h", "l", "c"]] /= 10                       # 1-for-10: raw $0.30 -> $3.00
    assert list(P.reverse_split_days(raw, adj)) == [d]
    rows = P.symbol_rows("AAA", raw, adj, CAL).set_index("date")
    assert rows.loc[raw.index[260], "rsplit90"] and not rows.loc[raw.index[330], "rsplit90"]
    s = pd.DataFrame({"ret5": [0.31, 0.1, 0.1, 0.1, 0.1, 0.1], "max_pct": [0.95, 0.95, 0.5, 0.5, 0.5, 0.5],
                      "rsplit90": [False, False, True, False, False, False],
                      "offer90": [False] * 3 + [True, False, False], "delist90": [False] * 4 + [True, False]})
    assert list(P.exclusion_reason(s)) == ["runup_30pct_5d", "max_top_decile", "reverse_split_90d", "offering_90d",
                                           "delisting_notice_90d", None]


def test_filing_flags_window():
    sig = pd.DataFrame({"cik": [1, 1, 2, 3], "date": pd.to_datetime(["2020-06-30", "2020-12-31", "2020-06-30", "2020-06-30"])})
    f = pd.DataFrame({"cik": [1, 2, 3, 3], "form": ["424B4", "8-K", "EFFECT", "8-K"],
                      "filed": ["2020-04-15", "2020-06-30", "2020-07-01", "2020-05-01"], "items": ["", "3.01,9.01", "", "2.02"]})
    off, dl = P.filing_flags(sig, f)
    assert list(off) == [True, False, False, False]                           # 76 days back yes; 259 no; after F no
    assert list(dl) == [False, False, True, False]                            # item 3.01 on F counts; 2.02 does not
    assert P.filing_flags(sig.iloc[[2]], f.assign(items=["", "2.02", "", ""]))[1][0] == False  # noqa: E712


def test_universe_dedupe_deciles_and_max():
    rows = pd.DataFrame({"date": pd.Timestamp("2020-01-02"), "symbol": [f"S{i}" for i in range(21)],
                         "cik": list(range(20)) + [0], "dv20": [2e6] * 20 + [9e6],
                         "vol60": np.arange(21) / 100, "max21": np.arange(21) / 100})
    u = P.universe(rows)
    assert len(u) == 20 and u.loc[u["cik"] == 0, "symbol"].iloc[0] == "S20"
    assert set(u["vol_dec"]) == set(range(1, 11))
    assert (u["max_pct"] > P.MAX_DECILE).sum() == 2


# ------------------------------------------------------------------ trades and controls

def sig_rows(n=6):
    d = pd.Timestamp("2020-01-02")
    return pd.DataFrame({"date": [d, d + pd.Timedelta(days=1), d + pd.Timedelta(days=40), d, d, d][:n],
                         "cik": [1, 1, 1, 2, 3, 4][:n], "symbol": ["A", "A", "A", "B", "C", "D"][:n],
                         "ret5": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5][:n], "max_pct": 0.5, "rsplit90": False,
                         "offer90": False, "delist90": False, "has_entry": [True, True, True, True, False, True][:n],
                         "skip1": [False, False, False, True, False, False][:n], "base_gross": 0.1,
                         "base_xday": [d + pd.Timedelta(days=30)] * n})


def test_take_trades_reentry_entry_skip_and_exclusion():
    t, f = P.take_trades(sig_rows())
    assert list(zip(t["cik"], t["date"].dt.strftime("%m-%d"))) == [(1, "01-02"), (1, "02-11")]
    assert f["skipped_open_trade"] == 1 and f["skipped_entry_above_1pct"] == 1 and f["no_entry_bar"] == 1
    assert f["excluded_runup_30pct_5d"] == 1


def control_universe():
    d = pd.Timestamp("2020-01-02")
    n = 40
    return pd.DataFrame({"date": d, "cik": range(100, 100 + n), "symbol": [f"Z{i:02d}" for i in range(n)],
                         "vol_dec": [1 + i % 2 for i in range(n)], "has_entry": True, "skip1": [i == 0 for i in range(n)],
                         "base_gross": 0.0, "exit_day": d + pd.Timedelta(days=30)})


def test_controls_same_decile_deterministic_and_exclude_signals():
    uni = control_universe()
    tr = pd.DataFrame({"date": pd.Timestamp("2020-01-02"), "cik": [1, 2, 3], "symbol": ["A", "B", "C"], "vol_dec": [1, 1, 2]})
    sigs = {pd.Timestamp("2020-01-02"): {1, 2, 3, 102, 104}}
    c1 = P.pick_controls(tr, uni, sigs)
    c2 = P.pick_controls(tr, uni, sigs)
    assert c1["symbol"].tolist() == c2["symbol"].tolist()
    assert list(c1["vol_dec"]) == [1, 1, 2]
    assert c1["cik"].nunique() == 3 and not set(c1["cik"]) & {100, 102, 104}  # skip1 row and signal names out
    # seed is SEED + date ordinal: reproduce the first draw by hand
    pool = uni[(uni["vol_dec"] == 1) & ~uni["skip1"] & ~uni["cik"].isin({1, 2, 3, 102, 104})].sort_values("symbol")
    k = int(np.random.default_rng(P.SEED + pd.Timestamp("2020-01-02").toordinal()).integers(0, len(pool)))
    assert c1["symbol"].iloc[0] == pool["symbol"].iloc[k]


def test_weekly_primary_alpha_share_and_verdict():
    rng = np.random.default_rng(0)
    n = 300
    days = pd.bdate_range("2018-01-01", periods=n * 2)[::2]
    p = pd.DataFrame({"entry_day": days, "xday": days + pd.Timedelta(days=28), "cik": np.arange(n) % 50,
                      "gross": 0.03 + rng.normal(0, 0.02, n), "c_gross": rng.normal(0, 0.02, n)})
    w = P.weekly(p)
    assert len(w) == p["entry_day"].dt.to_period("W-FRI").nunique()
    assert w["diff"].mean() > 0.02
    prim = P.primary_stats(w)
    assert prim["t_hac"] > 3 and prim["ci95"][0] > 0
    fd = pd.bdate_range("2017-12-01", "2020-12-31")
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(fd), 6)), index=fd,
                       columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    a = P.monthly_alpha(p, fac, pd.Series(0.0, index=fd))
    assert a["alpha"] > 0 and a["months"] > 12
    share, _ = P.stock_share(p)
    assert share < 0.2
    assert P.verdict(prim, a, share, 99)[0] == "INCONCLUSIVE"
    assert P.verdict(prim, {"alpha": 0.01, "t": 2.5}, share, 300) == ("PASSED", True)
    assert P.verdict(prim, {"alpha": 0.01, "t": 2.5}, 0.25, 300)[0] == "FAILED"


def test_analyse_end_to_end_on_synthetic_panel(tmp_path):
    rng = np.random.default_rng(1)
    parts = []
    for i in range(30):
        n = 900
        idx = CAL[CAL >= "2016-06-01"][:n]
        c = 3.0 * np.exp(np.cumsum(rng.normal(0, 0.03 + 0.002 * i, n)))
        c = np.clip(c, 1.2, 4.8)
        o = c * (1 + rng.normal(0, 0.003, n))
        fr = pd.DataFrame({"o": o, "h": np.maximum(o, c) * 1.02, "l": np.minimum(o, c) * 0.98, "c": c, "v": 2e6}, index=idx)
        r = P.symbol_rows(f"S{i:02d}", fr, fr.copy(), CAL)
        parts.append(r.assign(cik=i, car=np.nan, scar=np.nan, car_res=None))
    uni = P.universe(pd.concat(parts, ignore_index=True))
    days = sorted(uni["date"].unique())[::15][:40]
    b = pd.DataFrame([buy(i % 10, 1000 + i % 10, d, 20_000) for i, d in enumerate(days)]
                     + [buy(i % 10, 2000 + i % 10, d, 20_000) for i, d in enumerate(days)])
    cl = P.clusters(b)
    fd = pd.bdate_range("2016-01-01", "2020-12-31")
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(fd), 6)), index=fd, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    filings = pd.DataFrame({"cik": [0], "form": ["424B5"], "filed": [pd.Timestamp(days[0])], "items": [""]})
    res, p, w = P.analyse(uni, cl, filings, fac, pd.Series(0.0, index=fd))
    assert res["funnel"]["signals"] > 0 and res["funnel"].get("excluded_offering_90d", 0) >= 1
    assert len(p) == res["PRIMARY"]["trades"] and res["PRIMARY"]["verdict"] == "INCONCLUSIVE"
    assert (p["exit_day"] < P.HOLDOUT).all() and (p["symbol"] != p["c_symbol"]).all()
    assert "label" in res["EXPLORATORY"] and "one_insider" in res["EXPLORATORY"]
    P.write_report(tmp_path, {**res})
    assert "Exclusions" in (tmp_path / "REPORT.md").read_text()
