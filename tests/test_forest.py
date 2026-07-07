"""Tests for RandomForest."""

import os
import tempfile

import pytest

from cartlet import TASK_REGRESSION, RandomForest


class TestBasicTraining:
    """Test basic forest training and prediction."""

    def test_simple_classification(self):
        # Separating these four classes needs BOTH features, so use
        # max_features=None (otherwise sqrt(2)=1 feature per split leaves each
        # tree unable to learn the mapping) and a fixed seed for determinism --
        # without them this asserted exact per-point predictions from an
        # under-powered, unseeded forest and was flaky across RNG/sklearn
        # versions.
        rf = RandomForest(
            n_estimators=10, max_features=None, feature_names=["color", "size"]
        )
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        rf.load_data(X, y)
        rf.train(random_state=42)

        assert rf.predict(["red", "small"]) == "apple"
        assert rf.predict(["blue", "large"]) == "box"

    def test_regression(self):
        rf = RandomForest(
            n_estimators=10,
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[float(i)] for i in range(100)]
        y = [float(i) for i in range(100)]
        rf.load_data(X, y)
        rf.train()

        pred = rf.predict([50.0])
        assert isinstance(pred, float)
        assert 30 <= pred <= 70  # Should be in reasonable range

    def test_n_estimators(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        X = [["a"], ["b"]] * 10
        y = ["1", "2"] * 10
        rf.load_data(X, y)
        rf.train()

        assert len(rf.trees) == 5

    def test_max_features_sqrt(self):
        rf = RandomForest(
            n_estimators=5, max_features="sqrt", feature_names=["a", "b", "c", "d"]
        )
        X = [["1", "2", "3", "4"], ["5", "6", "7", "8"]] * 10
        y = ["x", "y"] * 10
        rf.load_data(X, y)
        rf.train()

        assert len(rf.trees) == 5

    @staticmethod
    def _multi_feature_data():
        X = [["a", "p"], ["b", "q"], ["c", "r"], ["a", "q"], ["b", "r"]] * 8
        y = ["1", "2", "3", "1", "2"] * 8
        return X, y

    def test_no_bootstrap_no_sampling_gives_identical_trees(self):
        """bootstrap=False + max_features=None is fully deterministic, so every
        tree in the forest must be identical (guards that the flags do
        something rather than being silently ignored)."""
        X, y = self._multi_feature_data()
        rf = RandomForest(
            n_estimators=4,
            bootstrap=False,
            max_features=None,
            feature_names=["x", "y"],
        )
        rf.load_data(X, y)
        rf.train(random_state=42)
        models = [t.model for t in rf.trees]
        assert all(m == models[0] for m in models)

    def test_bootstrap_true_varies_trees(self):
        """With bootstrap on, resampling must make at least some trees differ."""
        X, y = self._multi_feature_data()
        rf = RandomForest(
            n_estimators=6,
            bootstrap=True,
            max_features=None,
            feature_names=["x", "y"],
        )
        rf.load_data(X, y)
        rf.train(random_state=42)
        models = [t.model for t in rf.trees]
        assert any(m != models[0] for m in models)

    def test_max_features_restricts_split_search(self):
        """max_features=1 draws a single feature per split, so trees differ
        across estimators even without bootstrap."""
        X, y = self._multi_feature_data()
        rf = RandomForest(
            n_estimators=6,
            bootstrap=False,
            max_features=1,
            feature_names=["x", "y"],
        )
        rf.load_data(X, y)
        rf.train(random_state=42)
        models = [t.model for t in rf.trees]
        assert any(m != models[0] for m in models)


class TestPrediction:
    """Test prediction methods."""

    @pytest.fixture
    def trained_forest(self):
        rf = RandomForest(n_estimators=10, feature_names=["x"])
        X = [["a"], ["a"], ["a"], ["b"], ["b"]] * 10
        y = ["yes", "yes", "no", "maybe", "maybe"] * 10
        rf.load_data(X, y)
        rf.train()
        return rf

    def test_predict_returns_majority(self, trained_forest):
        result = trained_forest.predict(["a"])
        assert result == "yes"  # 2 yes vs 1 no

    def test_predict_proba(self, trained_forest):
        proba = trained_forest.predict_proba(["a"])
        assert isinstance(proba, dict)
        assert sum(proba.values()) == pytest.approx(1.0)
        assert "yes" in proba or "no" in proba

    def test_predict_proba_regression_raises(self):
        rf = RandomForest(
            n_estimators=5,
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[1.0], [2.0], [3.0]] * 10
        y = [1.0, 2.0, 3.0] * 10
        rf.load_data(X, y)
        rf.train()

        with pytest.raises(ValueError, match="not available for regression"):
            rf.predict_proba([1.5])


class TestFeatureImportances:
    """Test feature importance computation."""

    def test_feature_importances(self):
        rf = RandomForest(n_estimators=10, feature_names=["important", "noise"])
        # Feature "important" determines the class
        X = [["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]] * 10
        y = ["A", "A", "B", "B"] * 10
        rf.load_data(X, y)
        rf.train(random_state=42)

        importances = rf.feature_importances_
        assert "important" in importances
        assert "noise" in importances
        assert sum(importances.values()) == pytest.approx(1.0)
        # "important" should have higher importance
        assert importances["important"] > importances["noise"]

    def test_feature_importances_untrained_raises(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        with pytest.raises(ValueError, match="not trained"):
            _ = rf.feature_importances_


class TestExportImport:
    """Test JSONL export and import."""

    @pytest.fixture
    def trained_forest(self):
        rf = RandomForest(n_estimators=5, feature_names=["color", "size"])
        X = [["red", "small"], ["red", "large"], ["blue", "small"]] * 10
        y = ["apple", "apple", "ball"] * 10
        rf.load_data(X, y)
        rf.train()
        return rf

    def test_export_creates_file(self, trained_forest):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.cart")
            trained_forest.export(path)
            assert os.path.exists(path)

    def test_export_cart_format(self, trained_forest):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.cart")
            trained_forest.export(path)

            # Verify binary format with CART magic
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == b"CART"

    def test_roundtrip_predictions_match(self, trained_forest):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.cart")
            trained_forest.export(path)

            rf2 = RandomForest()
            rf2.load_model(path)

            test_vecs = [["red", "small"], ["red", "large"], ["blue", "small"]]
            for vec in test_vecs:
                assert trained_forest.predict(vec) == rf2.predict(vec)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        rf.load_data([], [])
        with pytest.raises(ValueError, match="No training data"):
            rf.train()

    def test_mismatched_lengths_raises(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        with pytest.raises(ValueError):
            rf.load_data([["a"], ["b"]], ["1"])

    def test_predict_without_training_raises(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        with pytest.raises(ValueError, match="not trained"):
            rf.predict(["a"])

    def test_export_without_training_raises(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        with (
            pytest.raises(ValueError, match="No forest"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            rf.export(os.path.join(tmpdir, "forest.cart"))

    def test_get_params(self):
        rf = RandomForest(
            n_estimators=50,
            max_features="log2",
            bootstrap=False,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        params = rf.get_params()
        assert params["n_estimators"] == 50
        assert params["max_features"] == "log2"
        assert params["bootstrap"] is False
        assert params["max_depth"] == 10
        assert params["min_samples_split"] == 5
        assert params["min_samples_leaf"] == 2

    def test_repr_untrained(self):
        rf = RandomForest(n_estimators=10)
        assert "untrained" in repr(rf)

    def test_repr_trained(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        rf.load_data([["a"], ["b"]] * 10, ["1", "2"] * 10)
        rf.train()
        assert "5 trees" in repr(rf)
