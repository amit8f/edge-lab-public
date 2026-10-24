import numpy as np
import pandas as pd

from tracks.main import meta2_study as M


def caldf(n=120):
    d = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": d, "close": "16:00"})


def test_extra_inputs_use_only_past_bars_and_flag_splits():
    idx = pd.bdate_range("2018-01-01", periods=100)
    d = pd.DataFrame({"o": 10.0, "c": 10.0, "rc": 10.0, "v": np.arange(100) + 1.0}, index=idx)
    d.loc[idx[60]:, "rc"] = 20.0                      # a 2-for-1 reverse split shows in raw over adjusted
    a = M.extra_inputs(d, [idx[50], idx[70], idx[95]])
    assert a[idx[50]] == {"avgvol20": float(np.arange(32, 52).mean()), "split30": False}
    assert a[idx[70]]["split30"] and not a[idx[95]]["split30"]
    later = d.copy()
    later.iloc[51:, 3] *= 9
    assert M.extra_inputs(later, [idx[50]]) == {idx[50]: a[idx[50]]}


def test_short_interest_waits_eight_sessions_and_changes():
    cal = pd.DatetimeIndex(caldf()["date"])
    si = pd.DataFrame({"symbol": "A", "settlement": [cal[10], cal[20]], "short_qty": [1000.0, 2000.0]})
    d = pd.DataFrame({"date": [cal[17], cal[18], cal[27], cal[28]], "symbol": "A", "avgvol20": 100.0, "split30": False})
    dtc, chg, sett = M.feat_si(d, si, cal)
    assert np.isnan(dtc[0]) and dtc[1] == 10.0                     # usable from 8 sessions after settlement
    assert dtc[2] == 10.0 and dtc[3] == 20.0 and np.isclose(chg[3], np.log(2)) and np.isnan(chg[1])
    assert pd.Timestamp(sett[3]) == cal[20]
    d["split30"] = [False, True, False, False]
    assert np.isnan(M.feat_si(d, si, cal)[0][1])


def test_short_volume_ratio_and_change():
    cal = pd.DatetimeIndex(caldf()["date"])
    r = np.r_[np.full(70, 0.4), np.full(5, 0.6)]
    rs = pd.DataFrame({"symbol": "A", "date": cal[:75], "short": r * 100, "total": 100.0})
    d = pd.DataFrame({"date": [cal[74], cal[30]], "symbol": "A"})
    s5, chg = M.feat_shvr(d, rs, cal)
    assert np.isclose(s5[0], 0.6) and np.isclose(chg[0], 0.2)
    assert np.isclose(s5[1], 0.4) and np.isclose(chg[1], 0.0)


def test_news_counts_use_the_five_sessions_to_t_close():
    c = caldf()
    cal = pd.DatetimeIndex(c["date"])
    utc = lambda day, hm: pd.Timestamp(f"{day:%Y-%m-%d} {hm}", tz="America/New_York").tz_convert("UTC").isoformat()
    news = pd.DataFrame({"id": range(6), "symbols": ["A", "A,B", "A", "A", "B", "A"],
                         "headline": ["Shares plunge on loss", "Record gain", "Update", "Old story", "Other", "Late"],
                         "created_at": [utc(cal[50], "10:00"), utc(cal[48], "12:00"), utc(cal[45], "16:30"),
                                        utc(cal[20], "11:00"), utc(cal[50], "11:00"), utc(cal[50], "16:01")]})
    arts = M.article_sessions(news, c, {"A"})
    d = pd.DataFrame({"date": [cal[50]], "symbol": ["A"]})
    n5, neg, pos, base = M.feat_news(d, arts, c)
    # in window: ids 0, 1, 2 (16:30 on day 45 belongs to session 46); id 5 is after t's close
    assert n5[0] == 3 and neg[0] == 1 and pos[0] == 1 and base[0] == 1
    assert set(arts["symbol"]) == {"A"}


def test_evaluate_handles_a_week_with_nothing_taken():
    from tracks.main import meta1_study as M1
    weeks = pd.date_range("2019-01-04", periods=30, freq="W-FRI")
    rng = np.random.default_rng(2)
    t = pd.DataFrame([{"date": w, "symbol": f"S{k}", "fwd5": rng.normal(0, 0.02), "mcap": 1e9, "prob": rng.random(),
                       "take": float(i > 0 and k < 5)} for i, w in enumerate(weeks) for k in range(10)])
    t["label"] = (t["fwd5"] > t.groupby("date")["fwd5"].transform("mean")).astype(int)
    r = M1.evaluate(t, {})
    assert r["PRIMARY"]["weeks"] == 30
