"""B0 burst study and short-side source parsing, on synthetic data only (no network)."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import shortdata
from tracks.penny import b0_study as b0

CAL = pd.bdate_range("2015-01-02", "2016-12-30")


def frames(close, cal=CAL, vol=500_000, factor=1.0):
    """raw and adjusted frames from a close path (open = previous close)."""
    c = pd.Series(close, index=cal[:len(close)], dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    raw = pd.DataFrame({"o": o, "h": np.maximum(o, c), "l": np.minimum(o, c), "c": c, "v": float(vol)})
    adj = raw.copy()
    adj[["o", "h", "l", "c"]] = raw[["o", "h", "l", "c"]] / factor
    return raw, adj


def jump_path(jumps):
    """$3 flat with +40% one-day jumps (that fade back the next day) on the given positions."""
    p = np.full(len(CAL), 3.0)
    for j in jumps:
        p[j] = 4.2
    return p


def test_symbol_rows_burst_flags_and_spacing():
    t0 = CAL.get_loc(pd.Timestamp("2016-02-01"))
    raw, adj = frames(jump_path([t0, t0 + 5, t0 + 30]))
    d = b0.symbol_rows("XYZ", raw, adj, CAL).set_index("date")
    assert d.index.min() >= b0.DEV[0] and set(d["group"]) == {"penny"}
    # the 10 days before each jump see it; only the first of a run counts, and again 20+ days later
    first = d.index[d["first_b30_10"]]
    assert list(first) == [CAL[t0 - 10], CAL[t0 + 20]]
    assert d.loc[CAL[t0 - 1], "maxup10"] == pytest.approx(0.4, rel=1e-5)
    assert d.loc[CAL[t0 - 1], "ret10"] == pytest.approx(0.0, abs=1e-6)   # open 3.0 the next day, 10th close 3.0
    assert d.loc[CAL[t0 - 1], "mindn10"] == pytest.approx(0.0, abs=1e-6)
    assert not d["first_b50_10"].any()
    assert list(d.index[d["first_b30_5"]])[0] == CAL[t0 - 5]


def test_symbol_rows_gap_ends_series_and_liquidity_filter():
    p = jump_path([])
    raw, adj = frames(p)
    cut = raw.index[(raw.index > "2015-12-01") & (raw.index < "2015-12-24")]
    raw, adj = raw.drop(cut), adj.drop(cut)              # a 3-week gap: a new series needs 60 bars again
    d = b0.symbol_rows("XYZ", raw, adj, CAL)
    assert d["date"].min() > pd.Timestamp("2016-03-01")
    thin, thin_adj = frames(p, vol=100_000)              # $300k a day: below the $500k floor
    assert b0.symbol_rows("THIN", thin, thin_adj, CAL) is None


def test_symbol_rows_flags_a_split():
    p = np.full(len(CAL), 3.0)
    raw, adj = frames(p)
    s = CAL.get_loc(pd.Timestamp("2016-03-01"))
    raw.iloc[s:, :4] = raw.iloc[s:, :4] * 5                # 1-for-5 reverse split: raw jumps, adjusted doesn't
    d = b0.symbol_rows("RS", raw, adj, CAL).set_index("date")
    assert d.loc[CAL[s], "split30"] and d.loc[CAL[s + 29], "split30"] and not d.loc[CAL[s + 30], "split30"]
    assert not d.loc[CAL[s - 1], "split30"] and not d["first_b30_10"].any()


def day_rows():
    syms = [f"S{i:02d}" for i in range(40)]
    d = pd.DataFrame({"date": pd.Timestamp("2016-05-02"), "symbol": syms, "group": "penny",
                      "dv20": np.arange(40) * 1e5 + 6e5, "vol60": 0.05, "maxup5": 0.0,
                      "maxup10": [0.5 if i in (3, 4) else 0.0 for i in range(40)]})
    for v in b0.VARIANTS:
        d[f"first_{v}"] = d["symbol"].isin(["S03", "S04"]) if v == "b30_10" else False
    return b0.cells(d)


def test_match_day_controls_share_the_cell_and_are_not_reused():
    d = day_rows()
    rows = b0.match_day(d, "b30_10", np.random.default_rng(1))
    again = b0.match_day(d, "b30_10", np.random.default_rng(1))
    assert rows == again
    ctrl = [r for r in rows if r[2] == "control"]
    # 40 names: dollar-volume quintiles of 8, volatility deciles of 4 (equal volatility ranks in row
    # order), so each burst's cell holds 4 names and runs short of 5 controls
    assert len(ctrl) == 3 + 3
    assert len({r[0] for r in ctrl}) == len(ctrl) and not {"S03", "S04"} & {r[0] for r in ctrl}
    cell = d.set_index("symbol")[["dvq", "vold"]].apply(tuple, axis=1)
    assert all(cell[r[0]] == cell[r[1]] for r in ctrl)


def test_regsho_parse_and_facility_fallback():
    text = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20240102|A|214302|2595|425742|B,Q,N\n" \
           "20240102|AA|543080|66|1361152|B,Q,N\n2\n"
    d = shortdata.parse_regsho(text).set_index("symbol")
    assert d.loc["A", "short"] == 214302 and d.loc["AA", "total"] == 1361152

    class R:
        def __init__(self, t):
            self.text = t

    def get(url):
        if "CNMS" in url:
            return None
        return R("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20160104|A|10|0|40|Q\n") \
            if ("FNSQ" in url or "FNYX" in url) else None
    out, how = shortdata.regsho_day("2016-01-04", get)
    assert how == "FNSQ+FNYX" and out.set_index("symbol").loc["A", "short"] == 20
    assert shortdata.regsho_day("2016-01-04", lambda u: None) == (None, None)


def ftd_zip(lines):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cnsfails202401a", "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n" + lines)
    return buf.getvalue()


def test_ftd_parse_and_mean_over_file_days():
    z = ftd_zip("20240102|C1|VMAR|3000|VISION MARINE|1.09\n20240103|C1|VMAR|1000|VISION MARINE|.\n"
                "20240103|C2|AGRI|600|AGRIFORCE|.\nTrailer record count 3\n")
    d = shortdata.parse_ftd(z)
    assert len(d) == 3
    m = shortdata.ftd_mean_fails(d)
    assert m["VMAR"] == 2000 and m["AGRI"] == 300          # two settlement days in the file
    assert shortdata.ftd_period_end("2024-01", "a") == pd.Timestamp("2024-01-15")
    assert shortdata.ftd_period_end("2024-02", "b") == pd.Timestamp("2024-02-29")


def test_short_interest_pages_until_a_short_page():
    calls = []

    class R:
        def __init__(self, rows, code=200):
            self.status_code, self._rows = code, rows

        def json(self):
            return self._rows

    def post(url, json, headers, timeout):
        calls.append(json["offset"])
        n = 2 if json["offset"] < 4 else 1
        return R([{"symbolCode": f"S{json['offset'] + i}", "settlementDate": "2020-04-15",
                   "currentShortPositionQuantity": 100 + i, "marketClassCode": "NNM"} for i in range(n)])
    d = shortdata.short_interest("2020-04-01", "2020-04-30", post=post, page=2)
    assert calls == [0, 2, 4] and len(d) == 5 and d["settlement"].iloc[0] == pd.Timestamp("2020-04-15")


def test_full_index_parse_keeps_wanted_forms():
    text = ("Form Type   Company Name                         CIK         Date Filed  File Name\n" + "-" * 120 + "\n"
            "10-K        BIG CO INC                           1591890     2016-03-11  edgar/data/1591890/a.txt\n"
            "8-K         ACME CORP                            1234567     2016-01-05  edgar/data/1234567/b.txt\n"
            "424B4       NEW CO, INC.                         7654321     2016-02-01  edgar/data/7654321/c.txt\n"
            "8-K/A       ACME CORP                            1234567     2016-01-06  edgar/data/1234567/d.txt\n")
    d = b0.parse_full_index(text)
    assert list(d["form"]) == ["8-K", "424B4"] and d["filed"].iloc[1] == pd.Timestamp("2016-02-01")


def test_cik_features():
    d = pd.DataFrame({"date": pd.to_datetime(["2016-07-01", "2016-07-01", "2016-07-01"]),
                      "symbol": ["A", "B", "C"], "cik": [1.0, 2.0, np.nan],
                      "first_bar": pd.to_datetime(["2016-01-10", "2014-12-01", "2016-01-10"])})
    ev = pd.DataFrame({"cik": [1, 1, 1, 2], "filed": pd.to_datetime(["2016-06-12", "2016-06-11", "2016-07-01", "2016-07-02"])})
    n = b0.by_cik_count(d["cik"].to_numpy(float), d["date"], ev, 20)
    assert n[0] == 2 and n[1] == 0 and np.isnan(n[2])     # 06-11 is exactly 20 days back: outside
    ipo = pd.DataFrame({"cik": [1, 2], "filed": pd.to_datetime(["2016-01-08", "2016-01-08"])})
    lk = b0.feat_lockup(d, ipo)
    assert lk[0] == 1 and lk[1] == 0 and np.isnan(lk[2])  # B's first bar is years before its 424B4
    buys = pd.DataFrame({"cik": [1, 2], "filed": pd.to_datetime(["2016-04-01", "2016-04-03"])})
    ins = b0.feat_insider(d, buys)
    assert ins[0] == 0 and ins[1] == 1                    # 91 days back is out, 89 is in


def test_days_to_cover_waits_for_publication():
    cal = pd.bdate_range("2016-01-04", "2016-03-31")
    si = pd.DataFrame({"symbol": ["A"], "settlement": [pd.Timestamp("2016-01-15")], "short_qty": [1_000_000.0]})
    avail = cal[cal.searchsorted(pd.Timestamp("2016-01-15")) + b0.SI_LAG]
    d = pd.DataFrame({"date": [avail - pd.offsets.BDay(1), avail, pd.Timestamp("2016-03-31")], "symbol": "A",
                      "avgvol20": 100_000.0, "split30": [False, False, False]})
    v = b0.feat_dtc(d, si, cal)
    assert np.isnan(v[0]) and v[1] == 10 and np.isnan(v[2])   # too early, known, too stale (>45 days)


def test_news_counts_the_five_sessions_to_t():
    caldf = pd.DataFrame({"date": pd.bdate_range("2016-01-04", "2016-01-29")})
    caldf["close"] = "16:00"
    news = pd.DataFrame({"created_at": ["2016-01-04T21:30:00Z", "2016-01-05T20:00:00Z", "2016-01-12T15:00:00Z",
                                        "2016-01-12T22:00:00Z"],
                         "symbols": ["A", "A,B", "A", "A"]})
    d = pd.DataFrame({"date": pd.to_datetime(["2016-01-11", "2016-01-12", "2016-01-12"]), "symbol": ["A", "A", "B"]})
    # 01-04 21:30Z is after the 16:00 ET close (21:00Z): it belongs to 01-05's session
    assert list(b0.feat_news(d, news, caldf)) == [2.0, 1.0, 0.0]


def test_prep_imputes_missing_as_the_development_mean():
    X = pd.DataFrame({c: [1.0, 2.0, 3.0, np.nan] for c in b0.FEATURES})
    prep = b0.fit_prep(X)
    Z = b0.transform(X, prep)
    assert Z.shape == (4, len(b0.FEATURES)) and np.all(Z[3] == 0) and np.allclose(Z[:3].mean(axis=0), 0, atol=1e-9)


def test_auc_and_mean_bootstrap_by_date():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2023-01-03", periods=50).to_numpy(), 12)
    y = np.tile([1, 0, 0, 0, 0, 0], 100)
    s = y + rng.normal(0, 1, len(y))
    auc, ci = b0.auc_ci(y, s, dates, n_boot=200)
    assert 0.6 < auc < 0.95 and ci[0] < auc < ci[1]
    m = b0.mean_ci(np.ones(len(y)), dates, n_boot=50)
    assert m["mean"] == 1 and m["ci95"] == [1, 1] and m["n"] == 600


def test_top_decile_per_date():
    u = pd.DataFrame({"date": np.repeat(pd.to_datetime(["2023-01-03", "2023-01-04"]), [20, 5])})
    top = b0.top_decile(u, np.arange(25.0))
    assert top[:20].sum() == 2 and top[18] and top[19] and top[20:].sum() == 1 and top[24]


def test_evaluate_end_to_end_on_a_planted_signal():
    rng = np.random.default_rng(3)
    rows = []
    for period, days in (("dev", pd.bdate_range("2018-01-02", periods=60)), ("test", pd.bdate_range("2023-01-03", periods=60))):
        for d in days:
            for v in b0.VARIANTS:
                for k in range(2):
                    rows.append((d, f"B{k}", f"B{k}", "burst", "penny", v))
                    rows += [(d, f"C{k}{j}", f"B{k}", "control", "penny", v) for j in range(5)]
    m = pd.DataFrame(rows, columns=["date", "symbol", "match_id", "role", "group", "variant"])
    for c in b0.FEATURES:
        m[c] = rng.normal(0, 1, len(m))
    m.loc[m["role"] == "burst", "dtc"] += 2.0             # planted: bursts had more days to cover
    m.loc[rng.random(len(m)) < 0.3, "shvr5"] = np.nan
    u = m[(m["variant"] == b0.PRIMARY) & (m["date"] >= b0.TEST[0])].drop_duplicates(["date", "symbol"]).copy()
    u["maxup10"] = np.where(u["role"] == "burst", 0.4, 0.0)
    u["mindn10"], u["ret10"] = -0.05, 0.01
    r = b0.evaluate(m, u, "penny", b0.PRIMARY, {"penny": 0.02, "main": 0.002})
    assert r["passes"] and r["test_auc"] > 0.75 and r["test_auc_ci95"][0] > 0.6
    assert r["dev_bursts"] == 120 and r["test_controls"] == 600 and r["dev_bursts_short_of_controls"] == 0
    assert r["univariate_test_auc"]["dtc"] > 0.75 and r["coefficients"]["dtc"] > 0
    assert r["top_decile"]["p_up30_10d"]["mean"] > r["base_rate"]["p_up30_10d"]["mean"]
    assert r["base_rate"]["mean_ret10_net"]["mean"] == pytest.approx(-0.01)
    assert r["coverage_by_year"][2023]["shvr5"] < 0.8
