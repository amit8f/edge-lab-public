import numpy as np
import pandas as pd
import pytest

from tracks.main import sector5_study as S

ET = "America/New_York"


def test_after_hours_takes_the_last_bar_before_20h():
    t = [pd.Timestamp(f"2023-03-01 {h}", tz=ET).tz_convert("UTC") for h in ("15:55", "16:00", "19:55", "20:00")]
    b = pd.DataFrame({"t": t, "c": [100.0, 101.0, 102.0, 999.0]})
    p = S.after_hours(b)
    assert p.loc[pd.Timestamp("2023-03-01")] == 102.0


def test_night_session_maps_headlines_to_the_next_open():
    cal = pd.to_datetime(["2023-03-03", "2023-03-06", "2023-03-07"])     # Fri, Mon, Tue
    ts = pd.to_datetime(["2023-03-03 17:00", "2023-03-04 11:00", "2023-03-06 08:00", "2023-03-06 11:00",
                         "2023-03-06 16:30"])
    out = pd.DatetimeIndex(S.night_session(ts, cal))
    assert list(out[:3]) == [pd.Timestamp("2023-03-06")] * 3
    assert pd.isna(out[3])                                                   # inside Monday's session
    assert out[4] == pd.Timestamp("2023-03-07")


def test_tone_counts_words():
    t = S.tone(["Apple beats estimates, raises guidance", "Tesla recall widens, shares fall", "Fed meets"])
    assert list(t) == [1.0, -1.0, 0.0]


def test_calendar_flags():
    d = pd.bdate_range("2023-06-12", "2023-07-07").drop(pd.to_datetime(["2023-06-19", "2023-07-04"]))
    m = pd.DataFrame({"date": ["2023-06-14", "2023-06-13", "2023-07-07"], "event": ["FOMC", "CPI", "JOBS"]})
    f = S.calendar_flags(d, m)
    assert f.loc["2023-06-14", "c_fomc"] and f.loc["2023-06-13", "c_fomc_eve"] and f.loc["2023-06-13", "c_cpi"]
    assert f.loc["2023-06-16", "c_opex"] and f.loc["2023-06-16", "c_preholiday"]     # Juneteenth Monday
    assert f.loc["2023-07-03", "c_preholiday"] and f.loc["2023-06-30", "c_tom"] and f.loc["2023-07-05", "c_tom"]
    assert not f.loc["2023-06-22", "c_tom"] and f.loc["2023-07-07", "c_jobs"]


def test_sign_rule_finds_a_planted_reversal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = -0.5 * x * 0.01 + rng.normal(scale=0.005, size=2000)
    r = S.sign_rule(x[:1000], y[:1000], x[1000:], y[1000:], 0.0002)
    assert r["corr_test"] < -0.5 and r["hit_test"] > 0.6 and r["t_test"] > 5
    tl = S.tail_rule(x[:1000], y[:1000], x[1000:], y[1000:], 0.0002)
    assert tl["low"]["dir"] == 1 and tl["high"]["dir"] == -1 and tl["high"]["hit_test"] > 0.8


def test_flag_rule():
    rng = np.random.default_rng(1)
    f = np.zeros(1000, bool)
    f[::10] = True
    y = rng.normal(scale=0.01, size=1000) + 0.01 * f
    r = S.flag_rule(f[:500], y[:500], f[500:], y[500:], 0.0002)
    assert r["dir"] == 1 and r["diff_test"] == pytest.approx(0.01, abs=0.005) and r["t_diff_test"] > 3
