import numpy as np
import pandas as pd

from tracks.news import w1a_leaks as W

CAL = pd.DataFrame({"date": pd.bdate_range("2024-03-04", "2024-04-26"), "open": "09:30", "close": "16:00"})


def test_classify_groups_signs_and_order():
    assert W.classify("XYZ Shares Spike After Betaville Report Of Takeover Interest")[:2] == ("small", 1)
    assert W.classify("ABC Explores Sale, Bloomberg Reports")[:2] == ("big", 1)
    assert W.classify("Calcalist: Google In Talks To Acquire Wiz")[:2] == ("foreign", 1)
    assert W.classify("Muddy Waters Says It Is Short XYZ")[:2] == ("short_seller", -1)
    assert W.classify("Takeover Chatter In ABC")[:2] == ("unnamed", 1)
    # small beats big; short words flip the sign; no deal word -> nothing
    assert W.classify("Sky News, Citing Reuters: ABC Weighs Takeover Bid")[0] == "small"
    assert W.classify("Bloomberg: SEC Probes ABC Over Fraud Claims")[:2] == ("big", -1)
    assert W.classify("Bloomberg Reports ABC Earnings Beat") is None


def test_classify_ambiguous_names_need_a_cue():
    assert W.classify("The Deal Is Done: ABC Completes Merger") is None
    assert W.classify("ABC Shares Jump; The Deal Reports Takeover Interest")[0] == "small"
    assert W.classify("According To The Information, ABC Explores Sale")[0] == "small"
    assert W.classify("Golden Globes Sponsor ABC To Acquire XYZ") is None
    # a short-seller name without a short word is not a short report
    assert W.classify("Citron Founder Buys Stake In ABC") is None


def test_classified_limits_symbols_and_dates():
    news = pd.DataFrame({"id": [1, 2, 3, 4],
                         "created_at": ["2024-03-05T15:00:00Z", "2024-03-05T15:00:00Z", "2015-12-01T15:00:00Z",
                                        "2024-03-06T15:00:00Z"],
                         "headline": ["Betaville: ABC Takeover Interest"] * 3 + ["ABC Earnings"],
                         "symbols": ["ABC,XYZ", "A,B,C", "ABC", "ABC"]})
    c = W.classified(news)
    assert sorted(c["symbol"]) == ["ABC", "XYZ"] and set(c["id"]) == {1}
    assert c["t"].iloc[0] == pd.Timestamp("2024-03-05 10:00")


def test_first_per_window_and_stories():
    arts = pd.DataFrame({"id": [1, 2, 3, 4], "symbol": "ABC", "group": ["small", "small", "big", "small"],
                         "sign": 1, "t": pd.to_datetime(["2024-03-04 10:00", "2024-03-07 10:00", "2024-03-07 11:00",
                                                          "2024-03-11 17:00"])})
    k = W.first_per_window(arts, 5, CAL["date"])
    # id 2 is 3 sessions after id 1 (dropped); id 4 falls on 03-12, 6 sessions later (kept)
    assert list(k["id"]) == [1, 3, 4]
    s = W.stories(arts, CAL["date"])
    assert list(s.sort_values("id")["story_id"]) == [1, 2, 2, 4]


def test_entry_time_rules():
    day, t = W.entry_time(pd.Timestamp("2024-03-05 11:10:20"), CAL)
    assert (day, t) == (pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-05 11:41"))
    day, t = W.entry_time(pd.Timestamp("2024-03-05 15:40"), CAL)      # +30 min is after the last minute
    assert t == pd.Timestamp("2024-03-06 10:00")
    day, t = W.entry_time(pd.Timestamp("2024-03-05 07:00"), CAL)      # pre-market -> 10:00 the same day
    assert t == pd.Timestamp("2024-03-05 10:00")
    day, t = W.entry_time(pd.Timestamp("2024-03-09 12:00"), CAL)      # Saturday -> Monday 10:00
    assert t == pd.Timestamp("2024-03-11 10:00")


def _bars(day0, n_days, px):
    rows = []
    for d in pd.bdate_range(day0, periods=n_days):
        for m in pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 15:59", freq="min"):
            rows.append((m, px(m), px(m)))
    return pd.DataFrame(rows, columns=["t", "o", "c"])


def test_hold_trade_exits_five_sessions_later():
    day0 = pd.Timestamp("2024-03-05")
    stock = _bars(day0, 6, lambda m: 10.0 if m.normalize() == day0 else 11.0)
    spy = _bars(day0, 6, lambda m: 100.0)
    tr = W.hold_trade(stock, spy, CAL, day0, pd.Timestamp("2024-03-05 11:41"))
    assert tr["exit_day"] == pd.Timestamp("2024-03-12") and tr["exit_t"] == pd.Timestamp("2024-03-12 15:59")
    assert abs(tr["r"] - 0.1) < 1e-12 and tr["r_spy"] == 0 and tr["entry_px"] == 10.0


def test_resolve_uses_edgar_and_window():
    ev = pd.DataFrame({"cik": [1, 1, 2, 3], "sign": [1, -1, 1, 1],
                       "t": pd.to_datetime(["2023-01-10 10:00", "2023-02-01 10:00", "2023-03-01 10:00",
                                            "2025-01-05 10:00"])})
    f = pd.DataFrame({"cik": [1, 1, 2], "form": ["8-K", "8-K", "DEFM14A"],
                      "filed": pd.to_datetime(["2023-06-01", "2023-07-01", "2024-06-01"]),
                      "items": [["1.01"], ["4.02"], []]})
    r = W.resolve(ev, f)
    assert list(r["hit"]) == [True, True, False, False]
    assert r["resolve_date"].iloc[0] == pd.Timestamp("2023-06-01")
    assert r["resolve_date"].iloc[2] == pd.Timestamp("2024-02-29")      # miss: known at the window's end
    assert list(r["resolvable"]) == [True, True, True, False]


def test_point_in_time_rate_and_gap():
    ev = pd.DataFrame({"author": ["A"] * 6 + ["B"], "group": "small", "sign": 1, "resolvable": True,
                       "hit": [True] * 5 + [False, True],
                       "resolve_date": pd.to_datetime(["2023-01-01"] * 5 + ["2024-06-01", "2023-01-01"])})
    h, n = W.pit_rate(ev, pd.Timestamp("2024-01-01"), "A", "small")      # only A's 5 early rumours count
    assert n == 5 and abs(h - 6 / 8) < 1e-12
    h, n = W.pit_rate(ev, pd.Timestamp("2024-01-01"), "B", "small")      # B has 1 -> group's 6
    assert n == 6 and abs(h - 7 / 9) < 1e-12
    h, n = W.pit_rate(ev, pd.Timestamp("2022-01-01"), "A", "small")      # nothing known yet -> prior
    assert n == 0 and abs(h - 1 / 3) < 1e-12
    assert W.implied_prob(46.0, 40.0) == 0.5 and W.implied_prob(38.0, 40.0) == 0.0
    assert W.gap_side(0.8, 0.5) == 1 and W.gap_side(0.3, 0.5) == -1 and W.gap_side(0.5, 0.45) == 0


def test_prior_sessions():
    d = W.prior_sessions([pd.Timestamp("2024-03-05 10:00"), pd.Timestamp("2024-03-05 16:30"),
                          pd.Timestamp("2024-03-09 12:00")], CAL["date"])
    assert list(d) == list(pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-08"]))


def test_check_gate(tmp_path):
    ok, why = W.check_passed(tmp_path)
    assert not ok
    pd.DataFrame({"ok": ["y"] * 39 + ["n"] * 11, "note": ""}).to_csv(tmp_path / "W1a_check.csv", index=False)
    assert not W.check_passed(tmp_path)[0]
    pd.DataFrame({"ok": ["y"] * 40 + ["n"] * 10, "note": ""}).to_csv(tmp_path / "W1a_check.csv", index=False)
    assert W.check_passed(tmp_path)[0]


def test_beta_post_and_brier():
    b = W.beta_post(1, 3)
    assert abs(b["mean"] - 2 / 6) < 1e-12 and b["ci95"][0] < b["mean"] < b["ci95"][1]
    assert W.brier([1.0, 0.0], [1, 1]) == 0.5
    assert np.isnan(W.brier([], []))
