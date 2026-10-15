import numpy as np
import pandas as pd

from core import costs


def test_bucket_and_sample():
    assert costs.bucket_of(3.0, 1e6) == "penny" and costs.bucket_of(3.0, 1e5) is None
    assert costs.bucket_of(20.0, 3e6) == "small" and costs.bucket_of(20.0, 2e7) == "mid"
    assert costs.bucket_of(200.0, 1e9) == "large" and costs.bucket_of(20.0, 1e6) is None
    day = pd.Timestamp("2025-03-03")
    prev = pd.DataFrame({"day": day, "symbol": [f"S{i}" for i in range(300)], "price": 20.0, "dv20": 3e6})
    s1, s2 = costs.sample(prev, [day]), costs.sample(prev, [day])
    assert len(s1) == costs.PER_BUCKET and list(s1["symbol"]) == list(s2["symbol"])


def test_as_of_skips_bad_and_later_quotes():
    t = pd.Timestamp("2025-03-03 14:35", tz="UTC")
    q = [{"t": "2025-03-03T14:34:58Z", "bp": 10.0, "ap": 10.02},
         {"t": "2025-03-03T14:34:59Z", "bp": 10.05, "ap": 10.0},     # crossed
         {"t": "2025-03-03T14:35:00Z", "bp": 10.0, "ap": 10.01}]     # not before t
    got = costs.as_of(q, t)
    assert got["ap"] == 10.02 and np.isclose(costs.half_spread(got), 0.01 / 10.01)
    assert costs.as_of([], t) is None


def test_measure_day_falls_back_to_wider_window_and_summarizes():
    day = pd.Timestamp("2025-03-03")

    def fetch(syms, start, end, headers):
        out = {}
        for s in syms:
            if s == "LIQ" or (end - start).total_seconds() > 10:
                out[s] = [{"t": (end - pd.Timedelta(seconds=1)).isoformat(), "bp": 10.0, "ap": 10.02 if s == "LIQ" else 10.2}]
        return out

    rows = pd.DataFrame(costs.measure_day(day, ["LIQ", "ILL"], None, fetch))
    assert len(rows) == 2 * len(costs.INSTANTS)
    assert set(rows.loc[rows["symbol"] == "ILL", "window_s"]) == {60}
    m = rows.assign(bucket=np.where(rows["symbol"] == "LIQ", "large", "penny"))
    cs = pd.DataFrame({"bucket": ["large", "penny"], "cs": [0.002, 0.02]})
    s = costs.summarize(m, cs)
    assert np.isclose(s["large"]["round_trip_0935_1559"]["p50"], 2 * 0.01 / 10.01)
    assert s["penny"]["instants"]["09:35"]["coverage"] == 1.0
