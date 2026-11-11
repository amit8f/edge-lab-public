"""PICK study: news join, labels, the 2024 guard and the bar / width choice (synthetic data, no network)."""
import numpy as np
import pandas as pd
import pytest

from tracks.main import pick_study as S

D3 = pd.Timestamp("2023-03-03")
SESS = np.array([pd.Timestamp(x).value for x in ["2023-03-01", "2023-03-02", "2023-03-03"]], dtype="int64")


def ts(s):
    return pd.Timestamp(s).value


def tone_row(sym, t, **kw):
    base = {"symbol": sym, "t": pd.Timestamp(t), "lm_neg": 0, "lm_pos": 0, "fb_pos": 0.5, "fb_neg": 0.5}
    base.update({f: 0 for f in S.FLAG_COLS})
    base.update(kw)
    return base


def test_news_uses_only_headlines_after_the_prior_1600_and_before_the_decision():
    rows = pd.DataFrame({"symbol": ["AAA"], "d": [D3], "t": [pd.Timestamp("2023-03-03 10:00")], "track": ["main"]})
    tone = pd.DataFrame([
        tone_row("AAA", "2023-03-02 15:59", fb_pos=0.99),                       # before the window: out
        tone_row("AAA", "2023-03-02 16:00", fb_pos=0.9, lm_pos=2),              # the window starts at 16:00: in
        tone_row("AAA", "2023-03-03 09:00", fb_pos=0.2, lm_neg=1, f_earnings=1),  # in
        tone_row("AAA", "2023-03-03 10:00", fb_pos=0.99, lm_pos=9),             # at the decision time: out
        tone_row("AAA", "2023-03-03 10:05", fb_pos=0.99, lm_pos=9),             # after it: out
        tone_row("BBB", "2023-03-03 09:30", fb_pos=0.99),                       # another symbol: out
    ])
    lo = S.prior_open(S.to_ns(rows["d"]), SESS)
    assert lo[0] == ts("2023-03-02 16:00")
    f = S.news_features(rows, tone, lo).iloc[0]
    assert f["n_news"] == 2
    assert abs(f["fb_pos_mean"] - 0.55) < 1e-12 and f["fb_pos_max"] == pytest.approx(0.9)
    assert f["lm_net"] == 1                                                    # (2 - 0) + (0 - 1)
    assert f["f_earnings"] == 1 and f["mins_last"] == 60                      # latest in-window headline 09:00


def test_no_headline_gives_the_nx1_no_news_values():
    rows = pd.DataFrame({"symbol": ["ZZZ"], "d": [D3], "t": [pd.Timestamp("2023-03-03 10:00")], "track": ["penny"]})
    tone = pd.DataFrame([tone_row("AAA", "2023-03-03 09:00")])
    f = S.news_features(rows, tone, S.prior_open(S.to_ns(rows["d"]), SESS)).iloc[0]
    assert f["n_news"] == 0 and f["mins_last"] == S.CAP and f["lm_net"] == 0 and f["fb_pos_max"] == 0


def test_label_is_1_at_three_percent_after_costs_and_nan_where_undefined():
    y = S.label([0.03, 0.0299, np.nan, 0.5, -0.02])
    assert list(y[:2]) == [1.0, 0.0] and np.isnan(y[2]) and list(y[3:]) == [1.0, 0.0]


def test_join_adds_labels_for_every_net_column_and_spy_parts_use_only_prior_data():
    days = pd.bdate_range("2023-01-02", periods=30)
    spy = pd.DataFrame({"date": days, "ao": 100.0, "ac": np.linspace(100, 110, 30), "c": 100.0})
    parts = S.spy_parts(spy)
    spy2 = spy.copy()
    spy2.loc[spy2.index[-1], ["ao", "ac"]] = [500.0, 500.0]                  # change the decision day only
    p2 = S.spy_parts(spy2)
    d = days[-1]
    assert parts.loc[d, "spy_r5"] == pytest.approx(p2.loc[d, "spy_r5"])
    assert parts.loc[d, "spy_vol20"] == pytest.approx(p2.loc[d, "spy_vol20"])
    assert parts.loc[d, "spy_r5"] == pytest.approx(spy.ac.iat[-2] / spy.ac.iat[-7] - 1)
    rows = pd.DataFrame({"symbol": ["AAA"], "d": [d], "t": [pd.Timestamp(f"{d.date()} 10:00")], "track": ["main"],
                         "net_1.5": [0.04], "net_2.0": [np.nan]})
    out = S.join_rows(rows, pd.DataFrame(columns=S.NEWS_TONE), np.unique(S.to_ns(days)), parts)
    assert out["y_1.5"].iat[0] == 1.0 and np.isnan(out["y_2.0"].iat[0])
    assert out["n_news"].iat[0] == 0


def test_fit_refuses_2024_and_later_months(tmp_path):
    with pytest.raises(ValueError):
        S.fit_year_ok("2024-01")
    with pytest.raises(ValueError):
        S.fit_year_ok("2025-09")
    (tmp_path / "joined").mkdir()
    pd.DataFrame({"track": ["main"], "d": [pd.Timestamp("2024-01-02")]}).to_parquet(tmp_path / "joined" / "2024-01.parquet")
    with pytest.raises(ValueError):
        S.read_joined(tmp_path, "2024-01", ["track", "d"])
    S.fit_year_ok("2023-12")                                                  # the last fit month is allowed


def test_empty_joined_month_reads_as_empty(tmp_path):
    (tmp_path / "joined").mkdir()
    pd.DataFrame().to_parquet(tmp_path / "joined" / "2019-01.parquet")
    assert len(S.read_joined(tmp_path, "2019-01", ["track", "d", "y_1.5"])) == 0


def test_bar_pair_takes_the_best_share_among_pairs_with_100_rows():
    s_tr = np.linspace(0, 1, 200001)
    bad_tr = np.zeros(len(s_tr), bool)
    s_va = np.linspace(0, 1, 100000)
    y_va = (s_va > 0.99).astype(float)
    best, rows, flag = S.choose_bar(s_va, y_va, np.zeros(len(s_va), bool), s_tr, bad_tr)
    assert flag is None and best["normal_q"] == 0.001 and best["n"] >= S.MIN_SEL
    assert {r["normal_q"] for r in rows} == set(S.NORMAL_Q) and len(rows) == 9


def test_bar_pair_falls_back_to_the_most_rows_when_none_reach_100():
    s_tr = np.linspace(0, 1, 200001)
    s_va = np.linspace(0, 1, 10000)
    best, rows, flag = S.choose_bar(s_va, (s_va > 0.5).astype(float), np.zeros(len(s_va), bool), s_tr,
                                    np.zeros(len(s_tr), bool))
    assert flag == "thin" and best["n"] == max(r["n"] for r in rows) and best["normal_q"] == 0.001


def test_bad_days_get_a_higher_bar_for_a_smaller_top_share():
    s_tr = np.linspace(0, 1, 200001)
    bad = np.zeros(len(s_tr), bool)
    _, rows, _ = S.choose_bar(np.linspace(0, 1, 1000), np.zeros(1000), np.ones(1000, bool), s_tr, bad)
    by = {(r["normal_q"], r["bad_divisor"]): r for r in rows}
    assert by[(0.001, 1)]["thr_bad"] == by[(0.001, 1)]["thr_normal"]
    assert by[(0.001, 4)]["thr_bad"] > by[(0.001, 2)]["thr_bad"] > by[(0.001, 1)]["thr_bad"]


def _toy_scores(n=1000):
    S_ = {w: np.linspace(0, 1, n) for w in (1.0, 1.5, 2.0)}
    thr = {w: 0.9 for w in S_}
    top = S_[1.5] >= 0.9
    return S_, thr, top


def test_width_choice_picks_the_highest_share_and_ties_go_to_the_middle():
    S_, thr, top = _toy_scores()
    tier = np.zeros(len(top), int)
    normal = np.ones(len(top), bool)
    Y = {1.0: np.where(top, (np.arange(len(top)) % 2).astype(float), 0.0),
         1.5: np.where(top, 1.0, 0.0), 2.0: np.where(top, 1.0, 0.0)}
    wpt, table = S.choose_width(S_, Y, tier, normal, thr, (1.0, 1.5, 2.0), 1.5, [0])
    assert wpt[0] == 1.5                                                     # 1.5 and 2.0 tie at 100%: middle
    Y[2.0] = np.where(top, 0.0, 0.0)
    wpt, _ = S.choose_width(S_, Y, tier, normal, thr, (1.0, 1.5, 2.0), 1.5, [0])
    assert wpt[0] == 1.5
    Y[1.5] = np.where(top, (np.arange(len(top)) % 2).astype(float), 0.0)     # now 1.0 and 1.5 are both 50%
    wpt, _ = S.choose_width(S_, Y, tier, normal, thr, (1.0, 1.5, 2.0), 1.5, [0])
    assert wpt[0] == 1.5
    Y[1.0] = np.where(top, 1.0, 0.0)                                         # 1.0 alone is best
    wpt, table = S.choose_width(S_, Y, tier, normal, thr, (1.0, 1.5, 2.0), 1.5, [0])
    assert wpt[0] == 1.0 and table["0"]["1.0"]["share"] == 1.0


def test_width_choice_falls_back_to_the_middle_for_an_empty_tier():
    S_, thr, _ = _toy_scores()
    n = len(S_[1.5])
    wpt, _ = S.choose_width(S_, {w: np.ones(n) for w in S_}, np.ones(n, int), np.ones(n, bool), thr,
                            (1.0, 1.5, 2.0), 1.5, [0, 1])
    assert wpt[0] == 1.5 and wpt[1] == 1.5


def test_tiers_are_cut_by_the_frozen_terciles():
    x = np.array([0.5, 1.5, 2.5, 1.0])
    assert list(S.tier_of(x, [1.0, 2.0])) == [0, 1, 2, 0]
    assert list(S.tier_of(x, [])) == [0, 0, 0, 0]


def test_row_scores_take_the_width_of_their_tier():
    S_ = {1.0: np.array([1.0, 1.0, 1.0]), 1.5: np.array([2.0, 2.0, 2.0])}
    out = S.row_scores(S_, np.array([0, 1, 1]), {0: 1.5, 1: 1.0})
    assert list(out) == [2.0, 1.0, 1.0]


def test_market_score_is_the_mean_of_the_four_z_scores_with_volatility_flipped():
    D = pd.DataFrame({"spy_gap": [1.0, 3.0], "spy_open": [0.0, 0.0], "spy_r5": [0.0, 0.0], "spy_vol20": [1.0, 1.0]})
    p = {"spy_gap": (2.0, 1.0), "spy_open": (0.0, 1.0), "spy_r5": (0.0, 1.0), "spy_vol20": (0.0, 1.0)}
    score = S.market_score(D, p)
    assert score[0] == pytest.approx((-1 + 0 + 0 - 1) / 4) and score[1] == pytest.approx((1 + 0 + 0 - 1) / 4)
