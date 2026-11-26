import numpy as np
import pandas as pd
import pytest

from tracks.main import lit1_study as S

C0 = S.COST


def path(rows):
    a = np.array(rows, float)            # one trade: rows of (o, h, l, c) per day
    return a[:, 0][None], a[:, 1][None], a[:, 2][None], a[:, 3][None]


def test_wilson():
    assert S.wilson_lo(60, 100) == pytest.approx(0.502, abs=0.002)
    assert S.wilson_lo(0, 0) == 0.0


def test_bracket_take_profit_and_stop():
    O, H, L, C = path([(100, 106, 99, 104), (104, 105, 103, 104)])
    assert S.bracket(O, H, L, C, np.array([0.05]), 1)[0] == pytest.approx(0.05 - C0)
    O, H, L, C = path([(100, 106, 94, 104), (104, 105, 103, 104)])   # both touched: stop first
    assert S.bracket(O, H, L, C, np.array([0.05]), 1)[0] == pytest.approx(-0.05 - C0)
    O, H, L, C = path([(100, 101, 99, 100), (90, 91, 89, 90)])        # gap through the stop fills at the open
    assert S.bracket(O, H, L, C, np.array([0.05]), 1)[0] == pytest.approx(-0.10 - C0)
    O, H, L, C = path([(100, 101, 94, 96), (96, 97, 95, 96)])         # short: down 5% is the take-profit
    assert S.bracket(O, H, L, C, np.array([0.05]), -1)[0] == pytest.approx(0.05 - C0)
    O, H, L, C = path([(100, 101, 99, 100.5), (100.5, 101, 100, 101)])
    assert S.bracket(O, H, L, C, np.array([0.05]), 1)[0] == pytest.approx(0.01 - C0)   # out at the last close


def test_ratchet_daily_moves_up_while_on_and_exits_when_off():
    O, H, L, C = path([(100, 106, 99, 106), (106, 112, 105, 112), (112, 113, 111, 112)])
    net, days = S.ratchet_daily(O, H, L, C, np.array([[True, False, False]]), np.array([0.05]), 1)
    assert net[0] == pytest.approx(0.12 - C0) and days[0] == 2      # ratcheted at 106, exited at 112 when off
    net, days = S.ratchet_daily(O, H, L, C, np.array([[False, False, False]]), np.array([0.05]), 1)
    assert net[0] == pytest.approx(0.06 - C0) and days[0] == 1


def bars(day_prices):
    """5-minute bars ending at the given minutes with o=h=l=c=price."""
    return pd.DataFrame([(m, p, p, p, p) for m, p in day_prices], columns=["hm", "o", "h", "l", "c"])


def test_ratchet_intraday_checks_three_times():
    d1 = bars([(575, 100), (690, 103), (840, 106), (955, 106), (960, 106)])
    d2 = bars([(575, 106), (690, 107), (840, 112), (955, 112)])
    net, j = S.ratchet_intraday([d1, d2], [True, True], [True, False], 0.05, 1)
    # 14:00 day 1: 106 >= 105 and on -> ratchet (T 111.3, S 100.7); 14:00 day 2: 112 >= 111.3, on_prev True ->
    # ratchet again; 15:55 day 2 (last day) exits at 112
    assert net == pytest.approx(0.12 - C0) and j == 1
    net, j = S.ratchet_intraday([d1, d2], [False, True], [False, False], 0.05, 1)
    assert net == pytest.approx(0.06 - C0) and j == 0               # setup off at 14:00 day 1
    d3 = bars([(575, 100), (600, 94), (690, 96)])
    assert S.ratchet_intraday([d3], [True], [True], 0.05, 1)[0] == pytest.approx(-0.06 - C0)  # bar opens below stop
    assert np.isnan(S.ratchet_intraday([None], [True], [True], 0.05, 1)[0])


def test_streak_prev_light():
    df = pd.DataFrame({"symbol": ["A"] * 5 + ["B"] * 2, "seg": [0] * 7,
                       "ret": [0.01, 0.02, -0.01, -0.02, -0.03, 0.01, 0.01]})
    st = S.streak(df["ret"], df["symbol"])
    assert list(st) == [1, 2, -1, -2, -3, 1, 2]
    p = S.prev_index(df)
    assert list(p) == [-1, 0, 1, 2, 3, -1, 5]
    on = np.array([True, True, False, True, True, True, True])
    assert list(S.light(on, p)) == [True, False, False, True, False, True, False]


def test_parse_trades_keeps_officer_buys_and_sales():
    sub = "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\na1\t05-MAR-2024\t4\t123\na2\t06-MAR-2024\t4/A\t123\n"
    own = "ACCESSION_NUMBER\tRPTOWNER_RELATIONSHIP\na1\tOfficer\na2\tDirector\n"
    tr = ("ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
          "a1\t1\tP\t100\t10\tA\na1\t2\tS\t50\t10\tD\na1\t3\tM\t10\t1\tA\na2\t4\tP\t1\t1\tA\n")
    x = S.parse_trades(sub.encode(), own.encode(), tr.encode())
    assert sorted(x["code"]) == ["P", "S"] and x["value"].sum() == pytest.approx(1500)
    assert (x["cik"] == 123).all() and x["filed"].iloc[0] == pd.Timestamp("2024-03-05")


def test_cut_conditions():
    df = pd.DataFrame({f: np.arange(10.0) for f in S.CONT} | {f: [True] * 10 for f in S.FLAGS})
    c = S.cut_conditions(df, {"r1": [1, 2, 7, 8]})
    assert c["r1<=p10"].sum() == 2 and c["r1>=p90"].sum() == 2 and c["earn"].all()
