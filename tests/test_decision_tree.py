"""Tests for DecisionTree."""

import os
import tempfile

import pytest

from cartlet import TASK_CLASSIFICATION, TASK_REGRESSION, DecisionTree


class TestBasicTraining:
    """Test basic tree training and prediction."""

    def test_simple_classification(self):
        dt = DecisionTree(feature_names=["color", "size"])
        X = [["red", "small"], ["red", "large"], ["blue", "small"], ["blue", "large"]]
        y = ["apple", "apple", "ball", "box"]
        dt.load_data(X, y)
        dt.train()

        assert dt.predict(["red", "small"]) == "apple"
        assert dt.predict(["red", "large"]) == "apple"
        assert dt.predict(["blue", "small"]) == "ball"
        assert dt.predict(["blue", "large"]) == "box"

    def test_pure_node_returns_string(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["a"], ["a"]]
        y = ["same", "same", "same"]
        dt.load_data(X, y)
        dt.train()

        # Pure node should return string, not dict
        result = dt.predict(["a"])
        assert result == "same"
        assert isinstance(result, str)

    def test_auto_feature_names(self):
        dt = DecisionTree()
        X = [["a", "1"], ["b", "2"]]
        y = ["x", "y"]
        dt.load_data(X, y)
        dt.train()

        assert dt.feature_names == ["0", "1"]
        assert dt.predict(["a", "1"]) == "x"

    def test_weighted_instances(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["a"]]
        y = ["yes", "no"]
        counts = [100, 1]  # "yes" has much higher weight
        dt.load_data(X, y, counts)
        dt.train()

        assert dt.predict(["a"]) == "yes"

    def test_min_samples_split(self):
        dt = DecisionTree(feature_names=["x"], min_samples_split=10)
        X = [["a"], ["b"], ["c"]]
        y = ["1", "2", "3"]
        dt.load_data(X, y)
        dt.train()

        # With min_samples_split=10 and only 3 samples, should be a leaf
        assert isinstance(dt.model, (str, dict))


class TestPrediction:
    """Test prediction methods."""

    @pytest.fixture
    def trained_tree(self):
        dt = DecisionTree(feature_names=["x"], store_distributions=True)
        X = [["a"], ["a"], ["a"], ["b"], ["b"]]
        y = ["yes", "yes", "no", "maybe", "maybe"]
        dt.load_data(X, y)
        dt.train()
        return dt

    def test_predict_returns_best(self, trained_tree):
        result = trained_tree.predict(["a"])
        assert result == "yes"  # 2 yes vs 1 no

    def test_predict_with_distribution(self, trained_tree):
        result = trained_tree.predict(["a"], return_dist=True)
        assert isinstance(result, dict)
        assert "yes" in result
        assert "no" in result

    def test_predict_with_confidence(self, trained_tree):
        pred, conf = trained_tree.predict_with_confidence(["b"])
        assert pred == "maybe"
        assert conf == 1.0  # pure node

    def test_predict_nbest(self, trained_tree):
        nbest = trained_tree.predict_nbest(["a"], n=2)
        assert len(nbest) <= 2
        assert nbest[0][0] == "yes"  # best prediction
        assert all(isinstance(p, float) for _, p in nbest)


class TestExportImport:
    """Test JSONL export and import."""

    @pytest.fixture
    def trained_tree(self):
        dt = DecisionTree(feature_names=["color", "size"])
        X = [["red", "small"], ["red", "large"], ["blue", "small"]]
        y = ["apple", "apple", "ball"]
        dt.load_data(X, y)
        dt.train()
        return dt

    def test_export_creates_file(self, trained_tree):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            trained_tree.export(path)
            assert os.path.exists(path)

    def test_export_cart_format(self, trained_tree):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            trained_tree.export(path)

            # Verify binary format with CART magic
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == b"CART"

    def test_roundtrip_predictions_match(self, trained_tree):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            trained_tree.export(path)

            dt2 = DecisionTree()
            dt2.load_model(path)

            test_vecs = [["red", "small"], ["red", "large"], ["blue", "small"]]
            for vec in test_vecs:
                assert trained_tree.predict(vec) == dt2.predict(vec)


class TestRegression:
    """Test regression functionality."""

    def test_simple_regression(self):
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        # Linear-ish data
        X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        dt.load_data(X, y)
        dt.train()

        # Should predict something reasonable
        pred = dt.predict([5.5])
        assert isinstance(pred, float)
        assert 4.0 <= pred <= 7.0  # Should be in reasonable range

    def test_regression_auto_detect(self):
        dt = DecisionTree(feature_names=["x"])
        X = [[1], [2], [3], [4]]
        y = [1.0, 2.0, 3.0, 4.0]  # Numerical targets
        dt.load_data(X, y)

        assert dt._effective_task() == TASK_REGRESSION

    def test_classification_auto_detect(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = ["x", "y", "z"]  # String targets
        dt.load_data(X, y)

        assert dt._effective_task() == TASK_CLASSIFICATION

    def test_numerical_feature_splits(self):
        dt = DecisionTree(
            features=[{"name": "age", "dtype": "int", "type": "num"}],
            task=TASK_CLASSIFICATION,
        )
        # Young vs old classification
        X = [[10], [15], [20], [25], [50], [55], [60], [65]]
        y = ["young", "young", "young", "young", "old", "old", "old", "old"]
        dt.load_data(X, y)
        dt.train()

        assert dt.predict([12]) == "young"
        assert dt.predict([58]) == "old"

    def test_mixed_feature_types(self):
        dt = DecisionTree(
            features=[
                {"name": "color", "dtype": "str", "type": "cat"},
                {"name": "size", "dtype": "float", "type": "num"},
            ],
            task=TASK_CLASSIFICATION,
        )
        X = [
            ["red", 1.0],
            ["red", 2.0],
            ["blue", 1.0],
            ["blue", 2.0],
        ]
        y = ["small_red", "big_red", "small_blue", "big_blue"]
        dt.load_data(X, y)
        dt.train()

        # Should be able to predict
        pred = dt.predict(["red", 1.5])
        assert pred in ["small_red", "big_red"]

    def test_regression_export_import(self):
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)

            # Load and verify
            dt2 = DecisionTree()
            dt2.load_model(path)

            assert dt2._effective_task() == TASK_REGRESSION

            # Predictions should match
            for x in X:
                assert dt.predict(x) == dt2.predict(x)


class TestPruning:
    """Test reduced error pruning."""

    def _make_noisy_data(self):
        """Dataset large enough for a meaningful train/validation split."""
        import random

        rng = random.Random(42)
        X = []
        y = []
        for _ in range(200):
            color = rng.choice(["red", "blue", "green"])
            size = rng.choice(["small", "medium", "large"])
            X.append([color, size])
            if color == "red":
                label = "apple" if rng.random() < 0.8 else "cherry"
            elif color == "blue":
                label = "ball"
            else:
                label = rng.choice(["apple", "ball", "cherry"])
            y.append(label)
        return X, y

    def test_pruning_runs_without_error(self):
        X, y = self._make_noisy_data()
        dt = DecisionTree(feature_names=["color", "size"])
        dt.load_data(X, y)
        dt.train(prune=True, validation_split=0.2)
        assert dt.model is not None
        assert dt.predict(["red", "small"]) is not None

    def test_pruning_reduces_tree_size(self):
        from cartlet import count_nodes

        X, y = self._make_noisy_data()

        dt_unpruned = DecisionTree(feature_names=["color", "size"])
        dt_unpruned.load_data(X, y)
        dt_unpruned.train(prune=False, random_state=42)

        dt_pruned = DecisionTree(feature_names=["color", "size"])
        dt_pruned.load_data(X, y)
        dt_pruned.train(prune=True, validation_split=0.2, random_state=42)

        # Pruned tree should be no larger (usually smaller)
        assert count_nodes(dt_pruned.model) <= count_nodes(dt_unpruned.model)

    def test_pruning_preserves_predictions(self):
        X, y = self._make_noisy_data()
        dt = DecisionTree(feature_names=["color", "size"])
        dt.load_data(X, y)
        dt.train(prune=True, validation_split=0.2, random_state=42)

        for vec in [["red", "small"], ["blue", "large"], ["green", "medium"]]:
            pred = dt.predict(vec)
            assert isinstance(pred, str)
            assert pred in {"apple", "ball", "cherry"}

    def test_pruning_with_auto_validation_split(self):
        """Pruning with no explicit validation_split uses default."""
        X, y = self._make_noisy_data()
        dt = DecisionTree(feature_names=["color", "size"])
        dt.load_data(X, y)
        dt.train(prune=True)
        assert dt.model is not None

    def test_pruning_export_import_roundtrip(self):
        X, y = self._make_noisy_data()
        dt = DecisionTree(feature_names=["color", "size"])
        dt.load_data(X, y)
        dt.train(prune=True, validation_split=0.2, random_state=42)
        preds_before = [dt.predict(row) for row in X[:10]]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pruned.cart")
            dt.export(path)

            dt2 = DecisionTree()
            dt2.load_model(path)
            preds_after = [dt2.predict(row) for row in X[:10]]

        assert preds_before == preds_after


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises(self):
        dt = DecisionTree(feature_names=["x"])
        dt.load_data([], [])
        with pytest.raises(ValueError, match="No training data"):
            dt.train()

    def test_mismatched_lengths_raises(self):
        dt = DecisionTree(feature_names=["x"])
        with pytest.raises(ValueError):
            dt.load_data([["a"], ["b"]], ["1"])

    def test_predict_without_training_raises(self):
        dt = DecisionTree(feature_names=["x"])
        with pytest.raises(ValueError, match="not trained"):
            dt.predict(["a"])

    def test_export_without_training_raises(self):
        dt = DecisionTree(feature_names=["x"])
        with (
            pytest.raises(ValueError, match="No model"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            dt.export(os.path.join(tmpdir, "model.cart"))

    def test_load_nonexistent_file_raises(self):
        dt = DecisionTree()
        with pytest.raises(FileNotFoundError):
            dt.load_model("/nonexistent/path/model.cart")

    def test_single_class(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = ["same", "same", "same"]
        dt.load_data(X, y)
        dt.train()

        assert dt.predict(["a"]) == "same"
        assert dt.predict(["z"]) == "same"  # unseen value
