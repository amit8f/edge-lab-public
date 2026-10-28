"""NXM on synthetic data (no network): the XGBoost arm reproduces NX3, every arm runs NX3's design, the
ensemble, Holm, rule 21 and resuming from finished years."""
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_nx3 import CAL, events
from tracks.news import nx3_study as X3
from tracks.news import nxm_study as M


@pytest.fixture(autouse=True)
def small(monkeypatch):
    monkeypatch.setitem(M.RF, "n_estimators", 20)
    monkeypatch.setitem(M.RF, "min_samples_leaf", 20)
    monkeypatch.setitem(M.MLP, "max_iter", 20)


def ev(effect=1.5, seed=1):
    e = events(effect, seed=seed)
    e["symbol"] = "S" + pd.Series(range(len(e))).astype(str)
    return e


def test_xgb_arm_is_nx3_and_every_arm_reports(tmp_path):
    e = ev()
    rep = M.run(e, CAL, tmp_path / "s", tmp_path / "o", years=[2019], models=False)
    _, pooled, _ = X3.walk_forward(e, CAL, years=[2019])
    for k in pooled:
        assert rep["arms"]["XGB"]["pooled"][k] == pooled[k]
    assert set(rep["arms"]) == set(M.ARMS) and set(rep["react"]["spearman_with_xgb"]) == set(M.ARMS)
    assert rep["react"]["spearman_with_xgb"]["XGB"] == pytest.approx(1.0)
    assert rep["react"]["pick_overlap_with_xgb"]["XGB"] == 1.0
    for a in M.NEW_ARMS:
        assert "holm_p" in rep["arms"][a]["vs_xgb"] and rep["arms"][a]["verdict"] in ("passed", "failed", "overfit (rule 21)")
    assert len(rep["react"]["by_tone_decile"]["LIN"]["mean_e"]) == 10
    saved = json.loads((tmp_path / "o" / "NXM_walkforward.json").read_text())
    assert saved["passing_arms"] == rep["passing_arms"]


def test_every_arm_scores_zero_without_news():
    e = ev()
    models = M.fit_arms(e[e["d"] < "2018-06-01"])
    pl = e[e["kind"] == "placebo"].head(300)[X3.INPUTS].to_numpy(float)
    for a, m in models.items():
        assert np.allclose(m.effect(pl), 0), a


def test_ensemble_is_the_mean_of_scaled_effects():
    class F:
        def __init__(self, k, cut):
            self.k, self.cut = k, cut

        def effect(self, X):
            return self.k * X[:, 0]
    X = np.arange(10.0).reshape(-1, 1)
    ens = M.Ensemble([F(1, 2.0), F(4, 4.0)], X)
    assert np.allclose(ens.effect(X), (X[:, 0] / 2 + X[:, 0]) / 2)
    assert ens.cut == pytest.approx(np.quantile(np.abs(ens.effect(X)), 0.9))


def test_holm():
    assert np.allclose(M.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


def test_rule21_marks_an_overfit_arm(monkeypatch):
    days = pd.DataFrame({"d": pd.bdate_range("2019-01-01", periods=50), "pick_minus_control": 0.01,
                         "pick_minus_shuffled": 0.01, "pick_net": 0.0})
    rng = np.random.default_rng(0)
    for k in ("pick_minus_control", "pick_minus_shuffled"):
        days[k] += rng.normal(0, 0.001, 50)
    days = pd.concat([days.assign(arm=a) for a in M.ARMS], ignore_index=True)
    s = {a: {"insample_mean": 0.01, "insample_days": 100} for a in M.ARMS} | {"RF": {"insample_mean": 0.03, "insample_days": 100}}
    monkeypatch.setattr(M, "react", lambda eff, picks: {})
    rep = M.report({"2019": s}, days, None, None)
    assert rep["arms"]["RF"]["overfit_rule21"] and rep["arms"]["RF"]["verdict"] == "overfit (rule 21)"
    assert rep["arms"]["LGBM"]["verdict"] == "passed"


def test_resumes_from_finished_years(tmp_path, monkeypatch):
    e = ev()
    M.run(e, CAL, tmp_path / "s", tmp_path / "o", years=[2019], models=False)
    monkeypatch.setattr(M, "year", lambda *a: pytest.fail("refit a finished year"))
    rep = M.run(e, CAL, tmp_path / "s", tmp_path / "o", years=[2019], models=False)
    assert rep["arms"]["XGB"]["pooled"]["pick_minus_control"]["days"] > 0


def test_saves_v1_models_for_passing_arms(tmp_path):
    e = ev()
    M.save_v1(e, ["LGBM", "ENS"], tmp_path)
    meta = json.loads((tmp_path / "nxm_v1_meta.json").read_text())
    assert {"LGBM", "ENS", "member_XGB", "member_MLP"} <= set(meta) and (tmp_path / "nxm_v1_LGBM.pkl").exists()
