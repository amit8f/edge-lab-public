"""NX3-ALL-SDS: shorts bought back at 15:55 the same day, longs held to the next close (synthetic, no network)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S
from tracks.news import nx3sds_live as D

CLOSE = pd.Timestamp("2026-11-10 16:00")


def minute_bars(closes_by_minute):
    t = [CLOSE - pd.Timedelta(minutes=m) for m in closes_by_minute]
    return pd.DataFrame({"t": t, "o": list(closes_by_minute.values()), "c": list(closes_by_minute.values()),
                         "v": 100}).sort_values("t").reset_index(drop=True)


def test_exit_price_is_the_bar_ending_at_1555():
    b = minute_bars({10: 9.0, 6: 9.5, 5: 99.0, 1: 99.0})      # bars starting 15:50, 15:54, 15:55, 15:59
    assert D.exit_price(b, CLOSE) == 9.5 and S.sds_exit(b, CLOSE) == 9.5
    assert np.isnan(D.exit_price(pd.DataFrame(columns=["t", "c"]), CLOSE))


def test_only_shorts_take_the_same_day_exit():
    t = pd.DataFrame({"symbol": ["L", "S", "X"], "side": [1, -1, -1], "entry": [10.0, 10.0, 10.0],
                      "exit_ac": [11.0, 12.0, 12.0], "ret": [0.1, 0.2, 0.2], "days": [2, 2, 2]})
    out = D.same_day_shorts(t, {"S": 9.0, "X": np.nan})
    assert list(out["symbol"]) == ["L", "S"]                    # X has no same-day price: dropped
    assert out.set_index("symbol").loc["L", "ret"] == 0.1      # the long keeps the next close
    assert abs(out.set_index("symbol").loc["S", "ret"] + 0.1) < 1e-12 and out.set_index("symbol").loc["S", "days"] == 1


class FakeApi:
    def settle_trades(self, ev, cal):
        t = ev.copy()
        t["entry"], t["entry_t"] = 10.0, t["t"]
        t["exit_ac"] = np.where(t["side"] > 0, 11.0, 12.0)      # next close: +10% long, short loses 20%
        t["ret"] = t["exit_ac"] / t["entry"] - 1
        t["price"], t["days"] = 10.0, 2
        return t

    def day_bars(self, symbols, d, open_t, close_t):
        return {s: minute_bars({6: 9.0}) for s in symbols}


def test_settle_uses_nx3all_picks_and_the_same_day_short_price():
    root = Path(tempfile.mkdtemp())
    led = Ledger(root)
    picks = [{"symbol": "L", "t": "2026-11-10T09:30:00", "side": 1, "i": 0, "w": 0.05},
             {"symbol": "S", "t": "2026-11-10T09:30:00", "side": -1, "i": 1, "w": 0.05}]
    ctrl = [{"symbol": "C1", "t": "2026-11-10T09:30:00", "side": 1, "i": 0, "w": 0.05},
            {"symbol": "C2", "t": "2026-11-10T09:30:00", "side": -1, "i": 1, "w": 0.05}]
    for day in ("2026-11-09", "2026-11-10"):
        led.write_json("nx3all_day", day, f"nx3all/days/{day}.json", {"day": day, "picks": picks, "control": ctrl})
    cal = pd.DataFrame({"date": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]),
                        "open": "09:30", "close": "16:00"})
    assert D.settle(root, FakeApi(), cal, pd.Timestamp("2026-11-10")) == []        # 11-09 is before the first day
    out = D.settle(root, FakeApi(), cal, pd.Timestamp("2026-11-11"))
    assert len(out) == 1 and D.settle(root, FakeApi(), cal, pd.Timestamp("2026-11-11")) == []
    rec = json.loads((root / Ledger(root).entry("nx3sds_settle", "2026-11-10")["file"]).read_text())
    s = {t["symbol"]: t for t in rec["trades"]}
    assert s["S"]["exit"] == "same_day" and abs(s["S"]["ret"] + 0.1) < 1e-9       # bought back at 9.0
    assert s["L"]["exit"] == "next_close" and abs(s["L"]["ret"] - 0.1) < 1e-9
    assert rec["book_pnl"] > 0


def test_history_day_changes_only_the_shorts():
    p = pd.DataFrame({"symbol": ["L", "S"], "side": [1, -1], "entry": [10.0, 10.0], "ret": [0.1, 0.2],
                      "price": [10.0, 10.0], "days": [2, 2]})
    c = pd.DataFrame({"symbol": ["CL", "CS"], "side": [1, -1], "entry": [10.0, 10.0], "ret": [0.0, 0.0],
                      "price": [10.0, 10.0], "days": [2, 2]})
    w = np.array([0.05, 0.05])
    r = S.sds_day(p, c, w, {"S": 9.0, "CS": 10.0})
    assert r["kept"] == 2
    long_net = float(N.net(p.iloc[[0]]).iloc[0])
    assert abs(r["SDS"]["sized_net"] - r["ALL"]["sized_net"]
               - 0.05 * (float(N.net(p.assign(ret=[0.1, -0.1], days=[2, 1]).iloc[[1]]).iloc[0])
                         - float(N.net(p.iloc[[1]]).iloc[0]))) < 1e-12
    assert r["SDS"]["sized_net"] > r["ALL"]["sized_net"] and long_net > 0
    r2 = S.sds_day(p, c, w, {"S": 9.0})                  # the control short has no price: the pair drops in both arms
    assert r2["kept"] == 1 and r2["ALL"]["short_net"] == 0.0
