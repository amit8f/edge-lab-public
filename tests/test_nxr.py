"""NXR on synthetic data (no network): the XGBoost arm reproduces NX3, the tamer arms run NX3's design and
score zero without news, and the version 1 files are written for a passing arm."""
import json

import numpy as np
import pytest

from tests.test_nxm import ev
from tests.test_nx3 import CAL
from tracks.news import nx3_study as X3
from tracks.news import nxr_study as R


@pytest.fixture(autouse=True)
def small(monkeypatch):
    monkeypatch.setitem(R.RFR, "n_estimators", 20)
    monkeypatch.setitem(R.RFR, "min_samples_leaf", 50)
    monkeypatch.setitem(R.XGBR, "min_child_weight", 20)


def test_xgb_arm_is_nx3_and_tame_arms_report(tmp_path):
    e = ev()
    rep = R.run(e, CAL, tmp_path / "s", tmp_path / "o", years=[2019])
    _, pooled, _ = X3.walk_forward(e, CAL, years=[2019])
    for k in pooled:
        assert rep["arms"]["XGB"]["pooled"][k] == pooled[k]
    assert rep["arms"]["XGB"]["pick_overlap_with_xgb"] == 1.0
    for a in R.NEW_ARMS:
        assert "holm_p" in rep["arms"][a]["vs_xgb"] and rep["arms"][a]["verdict"] in ("passed", "failed", "overfit (rule 21)")
    meta = tmp_path / "o" / "nxr_v1_meta.json"
    assert (json.loads(meta.read_text()).keys() >= set(rep["passing_arms"])) if rep["passing_arms"] else not meta.exists()
    again = R.run(e, CAL, tmp_path / "s", tmp_path / "o", years=[2019], models=False)    # resumes from the year file
    assert again["arms"]["XGBR"]["pooled"] == rep["arms"]["XGBR"]["pooled"]


def test_tame_arms_score_zero_without_news():
    e = ev()
    tr = e[e["d"] < "2018-06-01"]
    pl = e[e["kind"] == "placebo"].head(300)[X3.INPUTS].to_numpy(float)
    for a in R.NEW_ARMS:
        assert np.allclose(X3.Model(tr, R.learner(a)).effect(pl), 0), a
