import numpy as np
import pandas as pd

from tracks.news import surv1long_live as SL


class Fake:
    meta = {"pstar": 0.3, "inputs": ["j"]}

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.c_[np.full(len(X), 1 - self.p), np.full(len(X), self.p)]


def longs(n):
    return pd.DataFrame({"conf": [1.6] * n, "side": [1] * n, "price": [20.0] * n, "atr": [0.03] * n,
                         "d": [pd.Timestamp("2026-11-25")] * n})


def test_exits_at_target_or_when_the_model_gives_up():
    paths = np.array([[0.01, 0.035, 0.04, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 0.02]])
    sp = np.zeros_like(paths)
    hold = SL.long_exits(longs(2), paths, sp, Fake(0.05), 0.3)     # low hazard: chance of +3% by close 5 under 0.3
    assert hold == [(0, 0.01), (0, 0.01)]
    keep = SL.long_exits(longs(2), paths, sp, Fake(0.5), 0.3)      # high hazard: hold to +3% or close 5
    assert keep == [(1, 0.035), (4, 0.02)]


def test_settle_end_to_end(tmp_path):
    from core.ledger import Ledger
    dates = pd.bdate_range("2026-08-01", "2026-12-15")
    cal = pd.DataFrame({"date": dates})
    led = Ledger(tmp_path)
    d = "2026-11-25"
    led.write_json("nx3allhi_day", d, f"nx3allhi/days/{d}.json",
                   {"day": d, "picks": [{"symbol": "AAA", "i": 0, "w": 0.06, "conf": 1.7, "prev_c": 20.0},
                                        {"symbol": "BBB", "i": 1, "w": 0.05, "conf": 1.6, "prev_c": 30.0}]})
    led.write_json("nx3allhi_settle", d, f"nx3allhi/settled/{d}.json",
                   {"day": d, "trades": [{"symbol": "AAA", "role": "picks", "i": 0, "w": 0.06, "side": 1, "ret": 0.01, "net": 0.004},
                                         {"symbol": "BBB", "role": "picks", "i": 1, "w": 0.05, "side": -1, "ret": -0.02, "net": 0.015}]})

    def daily(syms, first, last):
        rows = []
        for s in syms:
            for j, x in enumerate(dates[(dates >= first) & (dates <= last)]):
                c = 20.0 * (1 + 0.01 * j) if s == "AAA" else 400.0
                rows.append({"symbol": s, "date": x, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 1e6,
                             "ao": c, "ah": c * 1.01, "al": c * 0.99, "ac": c, "av": 1e6})
        return pd.DataFrame(rows)

    out = SL.settle(tmp_path, daily, cal, pd.Timestamp("2026-12-10"), model=Fake(0.5))
    assert len(out) == 1
    import json
    rec = json.loads((tmp_path / "surv1long/settled/2026-11-25.json").read_text())
    a = [t for t in rec["trades"] if t["symbol"] == "AAA"][0]
    b = [t for t in rec["trades"] if t["symbol"] == "BBB"][0]
    assert a["closes_held"] > 1 and a["net"] > a["net_plain"]       # rising stock: held to +3%
    assert b["net"] == 0.015 and b["closes_held"] == 1             # short: unchanged
    assert SL.settle(tmp_path, daily, cal, pd.Timestamp("2026-12-10"), model=Fake(0.5)) == []
