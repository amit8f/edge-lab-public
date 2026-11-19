"""BOOMX helpers: synthetic data only."""
import numpy as np
import pandas as pd

from tracks.main import boomx_study as X


def test_auc_and_one_input_find_a_planted_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    y = (x + rng.normal(size=4000) > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y, "year": np.repeat([2021, 2022], 2000)})
    r = X.one_input(df, "x", "y")
    assert r["auc"] > 0.7 and r["rate_high10"] > r["rate_low10"] and set(r["by_year"]) == {"2021", "2022"}
    assert np.isnan(X.auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_groups_grid_and_confident():
    g = X.group_share({"atr14": 3.0, "pm_gap": 1.0})
    assert abs(g["volatility and range"] - 0.75) < 1e-12 and abs(g["gap and pre-market"] - 0.25) < 1e-12
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    df["y"] = (df["a"] > 0).astype(int)
    gr = X.grid(df, "a", "b", "y")
    assert gr["rate"][0][0] == 0 and gr["rate"][4][4] == 1 and sum(map(sum, gr["n"])) == 1000
    c = X.confident(df["y"], df["a"])
    assert c["top_rate"] == 1 and c["bottom_other_rate"] == 1


def test_study_runs_every_model_on_a_small_frame():
    rng = np.random.default_rng(2)
    n = 6000
    df = pd.DataFrame({"atr14": rng.normal(size=n), "pm_gap": rng.normal(size=n), "r1": rng.normal(size=n),
                       "year": rng.choice([2019, 2020, 2021, 2022, 2023, 2024, 2025], n)})
    df["d"] = pd.to_datetime(df["year"].astype(str) + "-03-02")
    df["y"] = (df["atr14"] + rng.normal(size=n) > 0).astype(int)
    r = X.study(df, ["atr14", "pm_gap", "r1"], "y", X.conf("y"))
    assert set(r["models"]) >= {"CatBoost", "LightGBM", "XGBoost", "ExtraTrees", "Logistic", "Tree depth 3", "CatBoost top 10"}
    assert r["models"]["CatBoost"]["2023"]["auc"] > 0.6 and next(iter(r["shap"])) == "atr14"
    assert "atr14" in r["tree_depth3"] and X.md({"p": {"k": r | {"label": "x"}}}).startswith("# BOOMX")
