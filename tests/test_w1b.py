import numpy as np
import pandas as pd

from tracks.main import w1b_volume as B
from tracks.news import w1_data as D


def _daily(n=80, spike_at=65, spike_v=1e7, close_hi=True):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    v = np.exp(rng.normal(np.log(1e5), 0.1, n))
    v[spike_at] = spike_v
    c = np.full(n, 10.0)
    h, l = c + 0.5, c - 0.5
    if close_hi:
        c[spike_at] = 10.4
    else:
        c[spike_at] = 9.6
    return pd.DataFrame({"symbol": "ABC", "date": dates, "o": 10.0, "h": h, "l": l, "c": c, "v": v,
                         "ao": 10.0, "ac": c, "seg": 0})


def test_signal_frame_buy_and_sell():
    f = B.signal_frame(_daily())
    row = f.iloc[65]
    assert row["z"] > 3 and row["cl"] >= 0.7 and row["side"] == 1
    assert row["entry_date"] == f["date"].iloc[66] and row["exit_date"] == f["date"].iloc[65 + B.HOLD]
    assert (f["side"].iloc[:60] == 0).all()                         # no z before 60 prior days
    g = B.signal_frame(_daily(close_hi=False))
    assert g.iloc[65]["side"] == -1


def test_signal_frame_never_crosses_a_gap():
    d = _daily()
    d.loc[d.index >= 40, "seg"] = 1                                 # a new segment: only 30 days before the spike
    assert (B.signal_frame(d)["side"] == 0).all()


def test_sign_mid_and_retail():
    t0 = pd.Timestamp("2024-03-05 14:30", tz="UTC")
    trades = pd.DataFrame({"t": [t0 + pd.Timedelta(seconds=s) for s in (1, 2, 3, 10)],
                           "p": [10.02, 9.98, 10.0, 10.05], "s": [100, 200, 300, 20000], "x": ["V", "D", "D", "D"]})
    quotes = pd.DataFrame({"t": [t0 + pd.Timedelta(seconds=s) for s in (0.5, 1.5, 2.5)], "bp": 9.99, "ap": 10.01})
    side, t = B.sign_mid(trades, quotes)
    assert list(side[:3]) == [1, -1, 0]
    assert side.iloc[3] == 1                                         # no quote within 1s -> tick test (up)
    r = B.subpenny_retail([10.0012, 10.005, 10.00, 10.0012], ["D", "D", "D", "V"])
    assert list(r) == [True, False, False, False]


def test_footprint_day_shares():
    t0 = pd.Timestamp("2024-03-05 14:30", tz="UTC")
    trades = pd.DataFrame({"t": [t0 + pd.Timedelta(seconds=s) for s in (1, 2, 3)], "p": [10.0099, 9.9801, 10.02],
                           "s": [100, 100, 10_000], "x": ["D", "D", "V"]})
    quotes = pd.DataFrame({"t": [t0], "bp": [9.99], "ap": [10.01]})
    f = B.footprint_day(trades, quotes)
    assert abs(f["off_share"] - 200 / 10200) < 1e-12 and abs(f["block_share"] - 10000 / 10200) < 1e-12
    assert abs(f["imbalance"] - 10000 / 10200) < 1e-12               # +100 -100 +10000
    assert abs(f["retail_share"] - 200 / 10200) < 1e-12 and f["retail_imbalance"] == 0


def test_spy_returns_and_net():
    spy = pd.DataFrame({"ao": [100.0, 101.0], "ac": [100.5, 102.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    r = B.spy_returns(spy, pd.to_datetime(["2024-01-02", "2024-01-02"]), pd.to_datetime(["2024-01-03", "2024-02-01"]))
    assert abs(r[0] - 0.02) < 1e-12 and np.isnan(r[1])
    assert abs(B.net_return(-1, -0.05, 0.0, "penny", 14) - (0.05 - 0.02 - 0.2 * 14 / 365)) < 1e-12


def test_controls_same_day_same_side():
    sig = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["ABC"], "group": ["main"], "side": [1]})
    pool = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 3), "symbol": ["ABC", "X", "Y"], "group": "main",
                         "ctrl_side": [1, 1, -1], "r": [0.1, 0.05, -0.2], "r_spy": 0.0, "held": 14})
    c = B.controls(sig, pool)
    assert list(c["symbol"]) == ["X"] and abs(c["net"].iloc[0] - (0.05 - 0.002)) < 1e-12


def test_recall_test():
    days = pd.bdate_range("2018-01-01", "2022-12-30")
    ann = days[900]
    sig = pd.DataFrame({"symbol": ["ABC", "ABC"], "date": [days[895], days[100]]})
    cases = pd.DataFrame({"symbol": ["ABC"] * 30, "announce_date": [ann] * 30})
    r = B.recall_test(cases, sig, days, np.random.default_rng(0))
    assert r["cases"] == 30 and r["recall"] == 1.0 and r["control_rate_all_windows"] == 0.0
    assert r["validated"] and r["fisher_p"] < 1e-6


def test_finra_shares():
    rows = [{"weekStartDate": "2024-01-01", "totalWeeklyShareQuantity": 600, "summaryTypeCode": "ATS_W_SMBL_FIRM",
             "marketParticipantName": "SOME ATS"},
            {"weekStartDate": "2024-01-01", "totalWeeklyShareQuantity": 300, "summaryTypeCode": "OTC_W_SMBL_FIRM",
             "marketParticipantName": "Citadel Securities LLC"},
            {"weekStartDate": "2024-01-01", "totalWeeklyShareQuantity": 100, "summaryTypeCode": "OTC_W_SMBL_FIRM",
             "marketParticipantName": "Some Broker"}]
    s = B.finra_shares(rows)
    assert s["otc_shares"].iloc[0] == 1000 and s["ats_share"].iloc[0] == 0.6 and s["wholesaler_share"].iloc[0] == 0.3


def test_announcements_date_targets_by_first_8k():
    f = pd.DataFrame({"cik": [1, 1, 2], "form": ["8-K", "DEFM14A", "SC 13D"],
                      "filed": pd.to_datetime(["2020-01-10", "2020-03-01", "2020-05-05"]), "items": [["1.01"], [], []]})
    a = B.announcements(f)
    t = a[a["category"] == "merger_target"]
    assert t["date"].iloc[0] == pd.Timestamp("2020-01-10")
    assert set(a["category"]) == {"agreement_8k", "merger_target", "activist_13d"}


def test_quiet_mask():
    cal = pd.bdate_range("2024-03-04", "2024-03-29")
    keys = pd.DataFrame({"symbol": ["A", "A", "B"], "cik": [1, 1, 2],
                         "date": pd.to_datetime(["2024-03-11", "2024-03-20", "2024-03-20"])})
    news = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2024-03-09"])})    # Saturday before 03-11
    fil = pd.DataFrame({"cik": [2], "date": pd.to_datetime(["2024-03-14"])})          # 4 sessions before 03-20
    q = D.quiet_mask(keys, news, fil, 3, cal)
    assert list(q) == [False, True, True]


def test_sec_release_parsing():
    page = ('<a href="/enforcement-litigation/litigation-releases/lr-25985">x</a>'
            '<a href="https://www.sec.gov/litigation/litreleases/2016/lr23456.htm">y</a>'
            '<a href="/enforcement-litigation/litigation-releases/lr-25985">dup</a>')
    assert B.release_links(page) == [("25985", "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-25985"),
                                     ("23456", "https://www.sec.gov/litigation/litreleases/2016/lr23456.htm")]
    text = B.release_text("<p>The SEC charged Jane Doe with insider trading.</p><p>She bought shares of Target Inc. "
                          "the day before the February 16, 2023 announcement that Buyer would acquire it.</p>")
    snip = B.candidate(text)
    assert snip and "February 16, 2023 announcement" in snip
    assert B.candidate("The SEC charged a Ponzi scheme operator with fraud.") is None
