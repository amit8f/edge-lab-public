"""A1 Beneish penny-short screen on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.penny import a1_screen as A


def year(sales, rec, ta, inc, cfo, **kw):
    return {"sales": sales, "rec": rec, "ta": ta, "inc": inc, "cfo": cfo, **kw}


def test_mscore_neutral_firm_and_required_inputs():
    same = year(10e6, 1e6, 20e6, 1e6, 1e6, cogs=6e6, ca=8e6, ppe=5e6, dep=1e6, sga=2e6, cl=3e6, ltd=2e6)
    m, ix, neutral = A.mscore(same, dict(same))
    assert all(ix[k] == pytest.approx(1.0) for k in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi"))
    assert ix["tata"] == pytest.approx(0.0) and neutral == []
    assert m == pytest.approx(-4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 - 0.327)
    # receivables doubling against flat sales and earnings far above cash flow push M over the flag
    bad = dict(same, rec=2e6, inc=4e6, cfo=-1e6)
    assert A.mscore(bad, same)[0] > A.M_FLAG
    # missing SG&A is set to 1; missing receivables leaves the filing unscored
    m2, ix2, neutral2 = A.mscore({k: v for k, v in same.items() if k != "sga"}, same)
    assert neutral2 == ["sgai"] and m2 == pytest.approx(m)
    assert np.isnan(A.mscore({k: v for k, v in same.items() if k != "rec"}, same)[0])


def fact(val, end, start=None, form="10-K", accn="0001-20-000001", filed="2020-03-01"):
    f = {"val": val, "end": end, "form": form, "accn": accn, "filed": filed}
    if start:
        f["start"] = start
    return f


def companyfacts():
    cy, py = ("2019-01-01", "2019-12-31"), ("2018-01-01", "2018-12-31")
    flows = {"Revenues": (10e6, 10e6), "NetIncomeLoss": (4e6, 1e6),
             "NetCashProvidedByUsedInOperatingActivities": (-1e6, 1e6)}
    inst = {"AccountsReceivableNetCurrent": (2e6, 1e6), "Assets": (20e6, 20e6)}
    g = {}
    for c, (a, b) in flows.items():
        g[c] = {"units": {"USD": [fact(a, cy[1], cy[0]), fact(b, py[1], py[0]),
                                  fact(a / 4, "2019-12-31", "2019-10-01", form="10-Q")]}}
    for c, (a, b) in inst.items():
        g[c] = {"units": {"USD": [fact(a, cy[1]), fact(b, py[1])]}}
    return {"cik": 42, "facts": {"us-gaap": g}}


def test_score_filings_reads_one_10k_and_its_comparatives():
    df = A.filing_facts(companyfacts())
    assert set(df["key"]) == {"sales", "inc", "cfo", "rec", "ta"}
    s = A.score_filings(df)
    assert len(s) == 1
    r = s.iloc[0]
    assert r["fy_end"] == pd.Timestamp("2019-12-31") and r["filed"] == pd.Timestamp("2020-03-01")
    assert r["dsri"] == pytest.approx(2.0) and r["tata"] == pytest.approx(5e6 / 20e6)
    assert r["m"] > A.M_FLAG and set(r["neutral"].split(",")) == {"gmi", "aqi", "depi", "sgai", "lvgi"}


def test_tiny_sales_are_not_scored():
    cf = companyfacts()
    for f in cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"]:
        f["val"] = 5e5
    s = A.score_filings(A.filing_facts(cf))
    assert np.isnan(s.iloc[0]["m"]) and "sales" in s.iloc[0]["why"]


CAL = pd.DatetimeIndex(pd.bdate_range("2017-01-02", periods=200))


def bars(c, o=None):
    c = np.asarray(c, dtype=float)
    return pd.DataFrame({"o": c if o is None else o, "c": c}, index=CAL[:len(c)])


def test_short_trade_stop_hold_and_series_end():
    seg = np.zeros(200, dtype=int)
    d = bars(np.full(200, 2.0))
    gross, held, stopped = A.short_trade(d, seg, 0, CAL, 0)
    assert gross == pytest.approx(0.0) and held == A.HOLD and not stopped
    up = np.full(200, 2.0)
    up[10:] = 2.8                              # +40% on day 10: stopped at that close
    gross, held, stopped = A.short_trade(bars(up), seg, 0, CAL, 0)
    assert stopped and gross == pytest.approx(1 - 2.8 / 2.0) and held == 10
    d = bars(np.r_[np.full(30, 2.0), [1.0]])    # stops trading on day 30 at $1
    gross, held, _ = A.short_trade(d, seg[:31], 0, CAL, 0)
    assert gross == pytest.approx(0.5) and held == 30
    assert np.isnan(A.short_trade(bars([2.0]), seg[:1], 0, CAL, 0)[0])


def test_net_charges_costs_and_daily_borrow():
    d = pd.DataFrame({"gross": [0.10], "held": [63]})
    assert A.net(d).iloc[0] == pytest.approx(0.10 - 0.02 - 0.20 * 63 / 252)


def test_pick_enters_each_10k_once_within_three_months():
    dates = [pd.Timestamp(x) for x in ("2020-02-28", "2020-03-31", "2020-04-30", "2020-06-30")]
    rows = pd.DataFrame([{"symbol": s, "date": t, "cik": c, "gross": 0.1, "held": 63, "stopped": False}
                         for t in dates for s, c in (("AAA", 1), ("BBB", 2), ("CCC", 3), ("DDD", 4))])
    flags = pd.DataFrame({"cik": [1, 2, 3], "accn": ["a", "b", "c"], "m": [-1.0, -3.0, -1.5],
                          "filed": pd.to_datetime(["2020-02-10", "2020-02-10", "2020-03-15"])})
    sel = A.pick(rows, flags, dates)
    f = sel[sel["role"] == "flag"]
    assert list(zip(f["date"].dt.strftime("%m"), f["symbol"])) == [("02", "AAA"), ("03", "CCC")]
    assert (sel[sel["role"] == "control"].groupby("date").size() == 1).all()


def test_month_ends_stop_before_the_holdout():
    cal = pd.DatetimeIndex(pd.bdate_range("2016-12-01", "2025-09-30"))
    me = A.month_ends(cal)
    assert me[0] == pd.Timestamp("2016-12-30")
    k = cal.searchsorted(me[-1])
    assert cal[k + A.HOLD] < A.LABEL_END and me[-1] < pd.Timestamp("2025-07-01")
