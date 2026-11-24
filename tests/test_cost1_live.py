import json

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import cost1_live as CL


class Api:
    h = None

    def settle_trades(self, ev, cal):
        t = ev.copy()
        t["entry_t"] = t["t"] + pd.Timedelta(minutes=1)
        t["ret"], t["price"], t["days"] = [0.02, -0.01], [20.0, 30.0], [1, 1]
        t["side"] = [1, -1]
        return t


def test_skips_wide_entry_and_keeps_missing_quote(tmp_path):
    dates = pd.bdate_range("2026-11-01", "2026-12-15")
    cal = pd.DataFrame({"date": dates, "open": "09:30", "close": "16:00"})
    led = Ledger(tmp_path)
    d = "2026-11-25"
    led.write_json("nx3allhi_day", d, f"nx3allhi/days/{d}.json",
                   {"day": d, "picks": [{"symbol": "AAA", "i": 0, "w": 0.06, "t": "2026-11-25 10:00:00"},
                                        {"symbol": "BBB", "i": 1, "w": 0.05, "t": "2026-11-25 10:05:00"}]})
    led.write_json("nx3allhi_settle", d, f"nx3allhi/settled/{d}.json", {"day": d, "trades": []})
    seen = []

    def quotes(syms, t):
        seen.append(t)
        if t.hour == 15:
            return {s: 0.001 for s in syms}
        return {"AAA": 0.01} if "AAA" in syms else {}          # AAA wide at entry, BBB no entry quote

    out = CL.settle(tmp_path, Api(), cal, pd.Timestamp("2026-12-01"), quotes=quotes)
    assert len(out) == 1
    rec = json.loads((tmp_path / "cost1/settled/2026-11-25.json").read_text())
    a, b = rec["trades"]
    assert not a["kept"] and b["kept"] and b["hs_in"] is None
    from adaptive import costs as AC
    assert np.isclose(a["net_m"], a["net_flat"] - (0.011 - AC.SPREAD_ROUND_TRIP["main"]))
    assert np.isclose(rec["gain"], -0.06 * a["net_m"])
    assert pd.Timestamp("2026-11-26 15:59") in seen
    assert CL.settle(tmp_path, Api(), cal, pd.Timestamp("2026-12-01"), quotes=quotes) == []


def test_before_first_day_is_ignored(tmp_path):
    cal = pd.DataFrame({"date": pd.bdate_range("2026-11-01", "2026-12-15"), "open": "09:30", "close": "16:00"})
    led = Ledger(tmp_path)
    led.write_json("nx3allhi_settle", "2026-11-20", "nx3allhi/settled/2026-11-20.json", {"day": "2026-11-20"})
    assert CL.settle(tmp_path, Api(), cal, pd.Timestamp("2026-12-01"), quotes=lambda s, t: {}) == []
