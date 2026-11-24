import numpy as np
import pandas as pd

from tracks.numerai import signals as S


def test_rank01_open_interval_and_ties():
    r = S.rank01([3.0, 1.0, 2.0, 2.0])
    assert r.min() > 0 and r.max() < 1 and r[2] == r[3] and r[0] == r.max()


def test_features_skip_country():
    assert S.features(["numerai_ticker", "feature_country", "feature_a", "target"]) == ["feature_a"]


def test_date_corr_perfect_and_summary():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": np.repeat(["2020-01-03", "2021-01-08"], 100), "target": rng.random(200)})
    df["p"] = df["target"]
    s = S.date_corr(df, "p")
    assert np.allclose(s.values, 1.0)
    by = S.summary(s)["by_year"]
    assert set(by) == {"2020", "2021"} and all(abs(v - 1) < 1e-9 for v in by.values())


def test_read_keeps_features_as_float32_and_thins_dates(tmp_path):
    df = pd.DataFrame({"numerai_ticker": ["A US"] * 8, "date": np.repeat(["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"], 1),
                       "feature_a": np.arange(8, dtype=float), "feature_country": ["US"] * 8, "target": 0.5,
                       "target_x_60": 0.1})
    f = tmp_path / "t.parquet"
    df.to_parquet(f)
    got, feats = S.read(f, keep=("date", "target"))
    assert feats == ["feature_a"] and list(got.columns) == ["date", "feature_a", "target"]
    assert got["feature_a"].dtype == np.float32
    assert S.thin_dates(got, every=4)["date"].tolist() == ["d1", "d5"]
    thin, _ = S.read(f, keep=("date", "target"), every=4)
    assert thin["date"].tolist() == ["d1", "d5"] and thin["feature_a"].tolist() == [0.0, 4.0]


def test_export_and_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MODEL_DIR", tmp_path / "models" / "num1")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.txt").write_text("tree")
    (cache / "model.json").write_text("{}")
    assert S.model_dir(cache) == cache
    S.stage_export(cache, None)
    assert (S.MODEL_DIR / "model.txt").read_text() == "tree" and S.model_dir(cache) == S.MODEL_DIR


def test_read_filters_each_batch(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = tmp_path / "v.parquet"
    pq.write_table(pa.table({"date": ["a", "a", "b"], "feature_x": [1.0, 2.0, 3.0], "target": [0.5, None, 0.25],
                             "data_type": ["validation", "validation", "test"]}), f)
    df, feats = S.read(f, keep=("date", "target", "data_type"),
                       rows=lambda d: (d["data_type"] == "validation").to_numpy() & d["target"].notna().to_numpy())
    assert list(df["feature_x"]) == [1.0] and df["feature_x"].dtype == "float32"


def test_pick_features_is_a_seeded_subset_in_file_order():
    feats = [f"feature_{i}" for i in range(50)]
    a, b = S.pick_features(feats, limit=10), S.pick_features(feats, limit=10)
    assert a == b and len(a) == 10 and a == [c for c in feats if c in set(a)]
    assert S.pick_features(feats[:5], limit=10) == feats[:5]


def test_read_uses_only_the_given_features(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = tmp_path / "t.parquet"
    pq.write_table(pa.table({"date": ["a", "b"], "feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0], "target": [0.1, 0.2]}), f)
    df, feats = S.read(f, keep=("date", "target"), use=["feature_b"])
    assert feats == ["feature_b"] and "feature_a" not in df.columns


def test_stage_data_downloads_each_file_once(tmp_path, monkeypatch, capsys):
    got = []

    class A:
        def download_dataset(self, name, dest):
            got.append(name)
            open(dest, "wb").write(b"x" * 10)

    monkeypatch.setattr(S, "api", lambda: A())
    S.stage_data(str(tmp_path))
    S.stage_data(str(tmp_path))                      # a second call finds the files and downloads nothing
    assert len(got) == 3 and all(n.startswith(f"signals/{S.VERSION}/") for n in got)
