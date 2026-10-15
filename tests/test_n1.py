"""N1 pure helpers on synthetic data (no network)."""
import numpy as np
import pandas as pd

from tracks.news import llm, n1_screen as n1

CAL = pd.bdate_range("2025-03-03", "2025-03-14")


def ts(s):
    return pd.Timestamp(s, tz="America/New_York")


def test_parse_missing_and_malformed_are_neu():
    assert llm.parse("1: NEG\n2. pos\n3: maybe\n5: NEG", 4) == ["NEG", "POS", "NEU", "NEU"]
    assert llm.user_text([("AAA", "Bad  news\n here")]) == "1. [AAA] Bad news here"


def test_timing_entry_lag_and_exit():
    # Tue 08:00 -> ready 08:30 -> Tue open; exit Thu (day 3)
    assert n1.timing(ts("2025-03-04 08:00"), CAL) == (pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-04"),
                                                      pd.Timestamp("2025-03-06"))
    # Tue 09:10 -> ready 09:40, after the open -> Wed
    assert n1.timing(ts("2025-03-04 09:10"), CAL)[1] == pd.Timestamp("2025-03-05")
    # Saturday -> prev Fri, entry Mon
    p, e, _ = n1.timing(ts("2025-03-08 12:00"), CAL)
    assert p == pd.Timestamp("2025-03-07") and e == pd.Timestamp("2025-03-10")


def test_headline_rows_dedupe_and_window():
    news = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                         "created_at": ["2025-02-03T15:00:00Z", "2025-02-04T15:00:00Z", "2025-02-09T15:00:00Z",
                                        "2025-02-03T15:00:00Z", "2025-01-20T15:00:00Z"],
                         "headline": ["X falls", "x  FALLS", "X falls", "Y up", "old"],
                         "symbols": ["X", "X", "X", "X,Y", "X"]})
    r = n1.headline_rows(news, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-09-24"))
    assert list(r["id"]) == [1, 3]                    # repeat within 3 days dropped, multi-symbol and old dropped


def test_decide_and_returns():
    assert n1.decide([("NEG", "NEG"), ("NEU", "POS")]) == "short"
    assert n1.decide([("NEG", "NEG"), ("POS", "POS")]) is None
    assert n1.decide([("NEG", "NEU")]) is None
    g = pd.DataFrame({"o": [10.0, 11, 12], "c": [10.5, 11.5, 9.0], "seg": [0, 0, 1]},
                     index=pd.DatetimeIndex(["2025-03-03", "2025-03-04", "2025-03-05"]))
    px = {"X": g}
    assert np.isclose(n1.ret_between(px, "X", pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-05")), 0.15)
    assert np.isnan(n1.ret_between(px, "X", pd.Timestamp("2025-03-06"), pd.Timestamp("2025-03-07")))


def test_window_start():
    m = {"a": {"cutoff": "2025-01-31"}, "b": {"cutoff": "2024-08-31"}}
    assert n1.window_start(m) == pd.Timestamp("2025-02-01")


def test_label_all_stops_on_quota_and_resumes(tmp_path):
    items = [(str(i), "X", f"h{i}") for i in range(120)]
    calls = []

    def call(model, text):
        calls.append(text)
        if len(calls) == 2:
            raise llm.QuotaExhausted("day")
        return "\n".join(f"{i}: NEG" for i in range(1, 51))
    assert llm.label_all(items, "gemini", "m", tmp_path / "c.jsonl", 0, call=call) is False
    assert len(llm.load_cache(tmp_path / "c.jsonl")) == 50
    assert llm.label_all(items, "gemini", "m", tmp_path / "c.jsonl", 0, call=lambda m, t: "") is True
    lab = llm.load_cache(tmp_path / "c.jsonl")
    assert len(lab) == 120 and lab["0"] == "NEG" and lab["119"] == "NEU"


def test_panel_rows_series_break():
    idx = pd.bdate_range("2024-11-01", periods=40).append(pd.bdate_range("2025-01-13", periods=30))
    d = pd.DataFrame({"o": 1.0, "c": 1.0, "rc": 10.0, "v": np.arange(1, 71) * 1.0, "sc": 10.0}, index=idx)
    r = n1.panel_rows("X", d)
    assert r["date"].min() >= n1.PANEL_FIRST and r["seg"].nunique() == 1    # the gap starts series 1
    first = r.iloc[0]
    assert first["nbars"] == 1 and np.isnan(first["dv20"])
    assert r.iloc[19]["dv20"] == 10.0 * np.median(np.arange(41, 61))


def test_lm_labels_and_dictionary_baseline():
    import numpy as np
    import pandas as pd
    from tracks.news import lm
    from tracks.news.n1_screen import dictionary_baseline
    assert lm.label("Acme misses estimates, cuts guidance") == "NEG"
    assert lm.label("Acme to present at conference") == "NEU"
    days = pd.bdate_range("2025-02-03", periods=40)
    rng = np.random.default_rng(0)
    sig = pd.DataFrame({"entry": list(days) * 2, "side": ["short"] * 40 + [None] * 40,
                        "lm_side": ["short"] * 20 + [None] * 20 + ["short"] * 40,
                        "short3": np.r_[0.02 + rng.normal(0, 0.01, 40), rng.normal(0, 0.01, 40)]})
    r = dictionary_baseline(sig, "short3")
    assert r["overlap_trades"] == 20 and r["llm_only"]["n_trades"] == 20 and r["dictionary_only"]["n_trades"] == 40
    assert r["llm_minus_dictionary_per_day"]["mean"] > 0
