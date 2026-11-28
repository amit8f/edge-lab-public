import numpy as np
import pandas as pd

from tracks.news import jump1_study as J
from tracks.news import veto1_study as V

IDX = """Form Type   Company Name                 CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------
NT 10-K          ACME CORP                     1234567     2023-03-31  edgar/data/1234567/0001.txt
NT 10-Q          BETA INC                      7654321     2023-05-16  edgar/data/7654321/0002.txt
NT 11-K          GAMMA PLAN                    1111111     2023-06-29  edgar/data/1111111/0003.txt
10-K             ACME CORP                     1234567     2023-04-14  edgar/data/1234567/0004.txt
"""


def test_nt_rows_keep_only_late_10k_10q_notices():
    df = V.nt_rows(IDX)
    assert list(df["form"]) == ["NT 10-K", "NT 10-Q"] and df["cik"].tolist() == [1234567, 7654321]
    assert df["filed"].iloc[0] == pd.Timestamp("2023-03-31")


def test_veto_window_is_60_days_and_excludes_the_trade_day():
    nt = pd.DataFrame({"symbol": ["ACME"], "filed": pd.to_datetime(["2023-03-31"])})
    tr = pd.DataFrame({"symbol": ["ACME", "ACME", "ACME", "ACME", "BETA"],
                       "d": pd.to_datetime(["2023-03-31", "2023-04-01", "2023-05-30", "2023-05-31", "2023-04-03"])})
    assert V.vetoed(tr, nt).tolist() == [False, True, True, False, False]


def test_jump_and_verdict():
    assert np.isclose(J.jump_of({"r_open": 0.10, "r_react": 0.05}), 0.155)
    assert np.isnan(J.jump_of({"r_open": None, "r_react": 0.0}))
    b = pd.DataFrame({"day": [f"2026-06-{i % 28 + 1:02d}" for i in range(80)], "net": [0.02] * 40 + [-0.01] * 40,
                      "mins_open": [0.0] * 40 + [30.0] * 40, "jump": [0.2] * 40 + [0.0] * 40})
    v = J.verdict(b, 0.1)
    assert v["jump1"] == 40 and v["net_per_trade"] == 0.02 and v["gap"] > 0
    assert J.verdict(b.head(10), 0.1)["passes"] is None


def test_wire_leads_match_by_title_and_skip_first_reads():
    from core import listener as LI
    from tracks.news import wire_speed as W
    w = pd.DataFrame({"source": ["PR Newswire", "PR Newswire", "Business Wire"],
                      "sha": [LI.title_sha("Acme Wins Contract"), LI.title_sha("Old item"), LI.title_sha("Beta Q3")],
                      "first_seen": pd.to_datetime(["2026-11-25T13:00:00Z", "2026-11-25T09:00:00Z",
                                                    "2026-11-25T14:00:00Z"], utc=True),
                      "window_s": [20.0, None, 20.0], "tickers": ["ACME", "", "BETA"]})
    bz = pd.DataFrame({"headline": ["ACME wins contract!", "Old item", "Beta Q3"],
                       "created_at": ["2026-11-25T13:01:30Z", "2026-11-25T09:05:00Z", "2026-11-25T13:59:50Z"],
                       "symbols": ["ACME", "", "BETA"]})
    m = W.leads(w, bz).set_index("source")
    assert len(m) == 2 and m.loc["PR Newswire", "lead_s"] == 90 and m.loc["Business Wire", "lead_s"] == -10
    s = W.summary(W.leads(w, bz))
    assert s["all"]["matched"] == 2 and s["all"]["share_earlier_60s"] == 0.5
