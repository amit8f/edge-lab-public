import numpy as np
import pandas as pd

from tracks.main import pick_data as P
from tracks.main import pick_test as T


def _bars(n=78, seed=0, day="2024-03-05"):
    rng = np.random.default_rng(seed)
    t = pd.date_range(f"{day} 09:30", periods=n, freq="5min")
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[20.0, c[:-1]]
    return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) * 1.003, "l": np.minimum(o, c) * 0.997, "c": c,
                         "v": rng.integers(1000, 5000, n).astype(float)})


def test_inputs_all_matches_pick_data_inputs():
    b, spy = _bars(seed=1), _bars(seed=2)
    ks = np.arange(1, 55)
    got = T.inputs_all(b, ks, 19.5, 2e7, 0.01, spy)
    for j, k in enumerate(ks):
        want = P.inputs(b, int(k), 19.5, 2e7, 0.01, spy)
        for c, v in want.items():
            g = got[c].iat[j]
            assert (np.isnan(v) and np.isnan(g)) or np.isclose(g, v, rtol=1e-9), (k, c, g, v)


def test_control_pick_is_seeded_and_never_the_same_stock():
    t = pd.Timestamp("2024-03-05 10:00")
    a = T.control_pick({"A", "B", "C", "D"}, t.normalize(), t, "A", "COMBINED")
    assert a in {"B", "C", "D"} and a == T.control_pick({"D", "C", "B", "A"}, t.normalize(), t, "A", "COMBINED")
    assert T.control_pick({"A"}, t.normalize(), t, "A", "NEWS") is None


def test_size_and_book_room():
    assert T.size_of(0.5, 0.5, 0.9) == 0.10 and T.size_of(0.9, 0.5, 0.9) == 0.40 and T.size_of(2, 0.5, 0.9) == 0.40
    assert np.isclose(T.size_of(0.7, 0.5, 0.9), 0.25)
    t0, t1 = pd.Timestamp("2024-03-05 10:00"), pd.Timestamp("2024-03-05 11:00")
    day = pd.DataFrame({"t": [t1, t0, t0, t0, t1], "score": [0.9, 0.6, 0.8, 0.7, 0.95],
                        "want": [0.4, 0.4, 0.4, 0.4, 0.4]})
    got = T.book_day(day)
    # 10:00: 0.8 then 0.7 take 40% each, 0.6 takes the 20% left; 11:00: nothing left
    assert got.tolist() == [0.0, 0.2, 0.4, 0.4, 0.0] or np.allclose(got.tolist(), [0.0, 0.2, 0.4, 0.4, 0.0])


def _frozen_v():
    return {"wpt": {0: 1.5}, "normal": 0.5, "bad": 0.7}


def test_picks_of_bar_first_clear_and_control():
    d = pd.Timestamp("2024-03-05")
    t0, t1 = d + pd.Timedelta(hours=10), d + pd.Timedelta(hours=11)
    base = {"track": "main", "tier": 0, "version": "", "partner": "", "exit_1.5": "close"}
    R = pd.DataFrame([
        base | {"symbol": "A", "d": d, "t": t0, "bad": False, "role": "pick", "score_COMBINED": 0.6, "net_1.5": 0.05},
        base | {"symbol": "A", "d": d, "t": t1, "bad": False, "role": "pick", "score_COMBINED": 0.9, "net_1.5": -0.9},
        base | {"symbol": "B", "d": d, "t": t0, "bad": True, "role": "pick", "score_COMBINED": 0.6, "net_1.5": 0.05},
        base | {"symbol": "Z", "d": d, "t": t0, "bad": False, "role": "ctrl", "version": "COMBINED", "partner": "A",
                "score_COMBINED": 0.1, "net_1.5": -0.01}])
    pk = T.picks_of(R, _frozen_v(), "COMBINED", "main", top=0.9)
    assert pk["symbol"].tolist() == ["A"]                   # B is a bad row under the 0.7 bar; A's first clear
    assert pk["net"].iat[0] == 0.05 and pk["ctrl_net"].iat[0] == -0.01 and np.isclose(pk["size"].iat[0], 0.175)
    ev = T.evaluate(pk, pd.bdate_range("2024-03-04", "2024-03-08"))
    assert ev["trades"] == 1 and ev["verdict"] == "too few" and ev["share3"] == 1.0 and ev["share3_control"] == 0.0


class _F:
    def __init__(self):
        self.v = {V: {"normal": 0.5} for V in T.VERSIONS}


def test_keep_rows_keeps_clears_and_draws_one_control_per_version():
    d = pd.Timestamp("2024-03-05")
    t0 = d + pd.Timedelta(hours=10)
    D = pd.DataFrame({"symbol": ["A", "B", "C"], "d": d, "t": t0, "score_NEWS": [0.6, 0.1, 0.1],
                      "score_CHART": [0.6, 0.1, 0.1], "score_COMBINED": [0.1, 0.1, 0.1]})
    K = T.keep_rows(D, _F())
    assert K[K["role"] == "pick"]["symbol"].tolist() == ["A"]
    c = K[K["role"] == "ctrl"]
    assert sorted(c["version"]) == ["CHART", "NEWS"] and set(c["partner"]) == {"A"} and not ({"A"} & set(c["symbol"]))


def test_frozen_models_load_and_score():
    import json
    from pathlib import Path
    fit = json.loads(Path("research/results/PICK/PICK_fit.json").read_text())
    for tr in ("main", "penny"):
        F = T.Frozen(fit, tr)
        cols = sorted({c for V in T.VERSIONS for c in F.v[V]["feats"]})
        D = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (6, len(cols))), columns=cols)
        D["swing30"] = [0.004, 0.006, 0.01, 0.004, 0.006, 0.01]
        tier = F.tiers(D["swing30"])
        for V in T.VERSIONS:
            s = F.score(V, D, tier)
            assert s.shape == (6,) and np.all((s >= 0) & (s <= 1))


def test_score_day_end_to_end_with_frozen_models():
    import json
    from pathlib import Path

    from tracks.main import pick_study as S
    fit = json.loads(Path("research/results/PICK/PICK_fit.json").read_text())
    F = {tr: T.Frozen(fit, tr) for tr in ("main", "penny")}
    F["main"].v["CHART"]["normal"] = 0.0                     # every main row clears CHART: exercises the trades
    day = pd.Timestamp("2024-03-05")
    bars = {s: _bars(seed=i) for i, s in enumerate(["A", "B", "C", "SPY"])}
    uni = {"A": ("main", 20.0, 2e7), "B": ("main", 20.0, 2e7), "C": ("penny", 3.0, 2e6)}
    swings = {s: [0.01] * 12 for s in uni}
    D = T.day_inputs(day, uni, bars, swings)
    assert len(D) == 3 * 54 and (D["k"] >= 1).all()
    tone = pd.DataFrame(columns=S.NEWS_TONE)
    parts = pd.DataFrame({"spy_gap": [0.0], "spy_r5": [0.0], "spy_vol20": [0.01]}, index=pd.DatetimeIndex([day]))
    sess = np.array([(day - pd.Timedelta(days=1)).value, day.value])
    R = T.score_day(D, tone, sess, parts, F, bars)
    main_p = R[(R["role"] == "pick") & (R["track"] == "main")]
    assert len(main_p) == 2 * 54 and np.isfinite(main_p["net_1.5"]).all()
    c = R[(R["role"] == "ctrl") & (R["version"] == "CHART")]
    assert len(c) == len(main_p) and (c["symbol"] != c["partner"]).all()
