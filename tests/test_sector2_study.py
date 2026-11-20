import numpy as np
import pandas as pd

from tracks.main import sector2_study as S


CAL = pd.DatetimeIndex(pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]))


def test_news_after_the_close_and_on_weekends_counts_for_the_next_session():
    ts = pd.to_datetime(["2023-01-03 09:00", "2023-01-03 16:30", "2023-01-07 12:00", "2023-01-09 17:00"])
    got = S.to_session(ts, CAL)
    assert list(pd.DatetimeIndex(got[:3]).strftime("%m-%d")) == ["01-03", "01-04", "01-09"]
    assert pd.isna(got[3])


def test_earnings_window_is_the_filing_day_and_the_next_session():
    f = pd.DataFrame({"cik": [1, 2, 3], "form": ["8-K", "8-K", "10-Q"], "items": ["2.02,9.01", "8.01", "2.02"],
                      "filed": pd.to_datetime(["2023-01-07", "2023-01-04", "2023-01-04"])})
    e = S.earn_days(f, CAL)
    assert list(e["cik"]) == [1] and list(e["date"].dt.strftime("%m-%d")) == ["01-09"]
    f.loc[0, "filed"] = pd.Timestamp("2023-01-04")
    assert list(S.earn_days(f, CAL)["date"].dt.strftime("%m-%d")) == ["01-04", "01-05"]
    f["items"] = [["2.02", "9.01"], ["8.01"], ["2.02"]]  # the W1 cache stores items as lists
    assert list(S.earn_days(f, CAL)["cik"].unique()) == [1]


def test_news_types_and_groups():
    news = pd.DataFrame({"created_at": ["2023-01-03T14:00:00Z", "2023-01-03T15:00:00Z", "2023-01-04T15:00:00Z"],
                         "headline": ["XYZ Q4 EPS beats", "Analyst upgrade on XYZ", "FDA approval for ABC"],
                         "symbols": ["XYZ", "XYZ,ABC", "ABC"]})
    nf = S.news_flags(news, CAL).set_index(["symbol", "date"])
    x = nf.loc[("XYZ", pd.Timestamp("2023-01-03"))]
    assert x["n"] == 2 and x["t_earnings"] and x["t_analyst"] and not x["t_fda"]
    assert nf.loc[("ABC", pd.Timestamp("2023-01-04")), "t_fda"]
    d = pd.DataFrame({"sector": 1, "ret": [0.10, -0.09, 0.01, 0.0], "earn": [True, False, False, False],
                      "news": [True, True, False, False]})
    d["none"] = ~d["earn"] & ~d["news"]
    for k in S.TYPES:
        d[f"t_{k}"] = False
    t = {r["group"]: r for r in S.group_table(d)}
    assert t["earn"]["share_of_all_8pct_up"] == 1.0 and t["news"]["share_of_all_8pct_down"] == 1.0
    assert t["none"]["share_ge_8pct"] == 0.0 and t["news"]["up_share_of_8pct"] == 0.5


def test_macro_table_and_fred_parse():
    m = S.parse_fred("observation_date,DGS10\n2023-01-03,3.79\n2023-01-04,.\n", "DGS10")
    assert np.isnan(m.iloc[1]) and m.iloc[0] == 3.79
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=200)
    oil = pd.Series(rng.normal(0, 0.02, 200), index=idx)
    sec = pd.DataFrame({4: 0.5 * oil + rng.normal(0, 0.001, 200)}, index=idx)
    r = S.macro_table(sec, pd.DataFrame({"dgs10_chg": rng.normal(0, 0.05, 200), "oil_ret": oil}, index=idx))[0]
    assert abs(r["beta_oil_ret"] - 0.5) < 0.05 and r["mean_ret_top10_oil_ret"] > r["mean_ret_bottom10_oil_ret"]
