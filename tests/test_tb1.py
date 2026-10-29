"""TB1 on synthetic picks (no network, no TabICL download): the walk-forward with any classifier matches
META1's for LightGBM, rule 21 fails a model that memorises its training rows, and the tails are reported."""
import numpy as np
import pandas as pd

from tracks.main import meta1_study as M
from tracks.main import tb1_study as T


def picks(signal, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2017-01-06", "2025-09-19", freq="W-FRI")
    rows = []
    for w in weeks:
        for k in range(12):
            z = rng.normal(size=len(M.INPUTS))
            rows.append(dict(zip(M.INPUTS, z), date=w, symbol=f"S{k}", mcap=1e9 if k % 2 else 5e9,
                             fwd5=signal * 0.01 * z[0] + rng.normal(0, 0.01) + 0.01))
    x = pd.DataFrame(rows)
    x["label"] = (x["fwd5"] > x.groupby("date")["fwd5"].transform("mean")).astype(int)
    return x


def test_lightgbm_arm_equals_meta1():
    x = picks(3.0)
    a, _ = T.walk_forward(x, T.lgbm)
    b, _ = M.walk_forward(x)
    assert np.allclose(a["prob"], b["prob"]) and (a["take"] == b["take"]).all()
    res, _ = T.arm(x, T.lgbm)
    assert res["PRIMARY"]["passes"] and not res["PRIMARY"]["rule21"]["fails"]
    tl = res["alongside"]["rule22_tails"]
    assert tl["taken_share_top5pct"] > tl["taken_share"] > tl["taken_share_bottom5pct"]


class Memoriser:
    """Perfect on rows it has seen, a coin flip on new ones."""

    def fit(self, X, y):
        self.seen = dict(zip(map(tuple, np.round(X.to_numpy(), 9)), y))
        return self

    def predict_proba(self, X):
        rng = np.random.default_rng(len(X))
        p = np.array([self.seen.get(k, rng.uniform()) for k in map(tuple, np.round(X.to_numpy(), 9))], float)
        return np.c_[1 - p, p]


def test_rule21_fails_a_memoriser():
    res, _ = T.arm(picks(0.0), Memoriser)
    r = res["PRIMARY"]["rule21"]
    assert r["auc_in"] > 0.99 and abs(r["auc_oos"] - 0.5) < 0.05 and r["fails"]
    assert not res["PRIMARY"]["passes"]
