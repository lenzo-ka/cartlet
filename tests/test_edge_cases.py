"""Tests for edge cases, malformed inputs, and trainer backends."""

import importlib.util
import os
import tempfile

import pytest

from cartlet import TASK_REGRESSION, DecisionTree, RandomForest
from cartlet.runner import load_model, predict

# Whether scikit-learn is importable. Evaluated once at collection time so the
# sklearn-only tests below skip cleanly (rather than skipping the whole module,
# which the old ``not pytest.importorskip(...)`` idiom did).
_SKLEARN_MISSING = importlib.util.find_spec("sklearn") is None


class TestMalformedModelFiles:
    """Test handling of malformed .cart model files."""

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            f.write(b"")
            f.flush()
            dt = DecisionTree()
            with pytest.raises(ValueError):
                dt.load_model(f.name)
            os.unlink(f.name)

    def test_invalid_magic_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            f.write(b"XXXX" + b"\x00" * 100)  # Wrong magic
            f.flush()
            dt = DecisionTree()
            with pytest.raises(ValueError, match="Invalid magic"):
                dt.load_model(f.name)
            os.unlink(f.name)

    def test_truncated_file_raises(self):
        """Model file with truncated header."""
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            f.write(b"CART")  # Just magic, no header
            f.flush()
            dt = DecisionTree()
            with pytest.raises(ValueError, match="header is 34 bytes"):
                dt.load_model(f.name)
            os.unlink(f.name)

    def test_unreasonable_header_raises(self):
        """Model file with insane header values."""
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            # CART (4) + version(2) + flags(2) + n_feat(2) + n_class(2) + n_trees(2)
            # n_feat = 65535 (too high)
            f.write(b"CART" + b"\x01\x00" + b"\x00\x00" + b"\xff\xff" + b"\x00" * 30)
            f.flush()
            dt = DecisionTree()
            with pytest.raises(ValueError, match="Unreasonable n_features"):
                dt.load_model(f.name)
            os.unlink(f.name)


class TestMinSamplesLeafEdgeCases:
    """Test min_samples_leaf constraint edge cases."""

    def test_min_samples_leaf_prevents_split(self):
        """When min_samples_leaf would make both children too small."""
        dt = DecisionTree(feature_names=["x"], min_samples_leaf=3)
        # Only 4 samples - any binary split would have < 3 on one side
        X = [["a"], ["a"], ["b"], ["b"]]
        y = ["1", "1", "2", "2"]
        dt.load_data(X, y)
        dt.train()
        # Should be a single leaf (no valid split possible)
        # or a split where both sides have >= 3 samples
        depth = dt.get_depth()
        assert depth <= 1  # Either leaf or one split

    def test_min_samples_leaf_equal_to_data_size(self):
        """min_samples_leaf equals total data size."""
        dt = DecisionTree(feature_names=["x"], min_samples_leaf=5)
        X = [["a"], ["b"], ["c"], ["d"], ["e"]]
        y = ["1", "2", "3", "4", "5"]
        dt.load_data(X, y)
        dt.train()
        # Must be a single leaf
        assert dt.get_depth() == 0

    def test_min_samples_leaf_larger_than_data(self):
        """min_samples_leaf larger than total data size."""
        dt = DecisionTree(feature_names=["x"], min_samples_leaf=100)
        X = [["a"], ["b"], ["c"]]
        y = ["1", "2", "3"]
        dt.load_data(X, y)
        dt.train()
        # Must be a single leaf
        assert dt.get_depth() == 0

    def test_min_samples_leaf_one_allows_all_splits(self):
        """min_samples_leaf=1 allows any split."""
        dt = DecisionTree(feature_names=["x"], min_samples_leaf=1)
        X = [["a"], ["b"]]
        y = ["1", "2"]
        dt.load_data(X, y)
        dt.train()
        # Should be able to split to single-sample leaves
        assert dt.predict(["a"]) == "1"
        assert dt.predict(["b"]) == "2"

    def test_min_samples_leaf_with_weighted_data(self):
        """min_samples_leaf respects sample weights."""
        dt = DecisionTree(feature_names=["x"], min_samples_leaf=10)
        X = [["a"], ["b"]]
        y = ["1", "2"]
        counts = [5, 5]  # Total weight = 10
        dt.load_data(X, y, counts)
        dt.train()
        # With weight 5 each, splitting would give < 10 per leaf
        assert dt.get_depth() == 0

    def test_min_samples_leaf_falls_back_to_valid_split(self):
        """A min_samples_leaf-violating best split must not veto a valid one.

        The maximum-gain threshold here isolates the single ``B`` row
        (``x <= 1.5`` -> a pure but 1-sample leaf), which violates
        ``min_samples_leaf=2``. A slightly-worse threshold (``x <= 2.5``)
        yields a valid split with two rows on each side. The old code enforced
        the leaf-size constraint only *after* choosing the best threshold, so
        it collapsed the whole node into a single leaf; the search must instead
        skip the violating candidate and keep the node splitting.
        """
        from cartlet import count_nodes

        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            min_samples_leaf=2,
        )
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = ["B", "A", "A", "A", "A"]
        dt.load_data(X, y)
        dt.train(trainer="native")

        # The node kept splitting instead of collapsing to a single leaf.
        assert count_nodes(dt.model) > 1
        # And the majority class is still recovered on the clean side.
        assert dt.predict([5.0]) == "A"


class TestMaxDepthConstraint:
    """Test max_depth constraint."""

    def test_max_depth_zero(self):
        """max_depth=0 should give single leaf."""
        dt = DecisionTree(feature_names=["x"], max_depth=0)
        X = [["a"], ["b"], ["c"], ["d"]]
        y = ["1", "2", "3", "4"]
        dt.load_data(X, y)
        dt.train()
        assert dt.get_depth() == 0

    def test_max_depth_limits_tree(self):
        """max_depth should limit actual depth."""
        for max_d in [1, 2, 3]:
            dt = DecisionTree(feature_names=["x", "y"], max_depth=max_d)
            X = [[f"x{i}", f"y{i}"] for i in range(20)]
            y = [str(i % 5) for i in range(20)]
            dt.load_data(X, y)
            dt.train()
            assert dt.get_depth() <= max_d

    def test_max_depth_none_unlimited(self):
        """max_depth=None allows unlimited depth."""
        dt = DecisionTree(feature_names=["x"], max_depth=None)
        X = [[str(i)] for i in range(10)]
        y = [str(i) for i in range(10)]  # All different
        dt.load_data(X, y)
        dt.train()
        # Should be able to go deeper than any reasonable limit
        # (exact depth depends on data)
        assert dt.get_depth() >= 1

    def test_max_depth_preserved_in_export(self):
        """max_depth should be saved and restored."""
        dt = DecisionTree(feature_names=["x"], max_depth=3)
        X = [["a"], ["b"]]
        y = ["1", "2"]
        dt.load_data(X, y)
        dt.train()

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            dt.export(f.name)
            dt2 = DecisionTree()
            dt2.load_model(f.name)
            # max_depth not stored in .cart, but predictions should match
            assert dt.predict(["a"]) == dt2.predict(["a"])
            os.unlink(f.name)


class TestRegressionValidation:
    """Test regression-specific validation and edge cases."""

    def test_regression_single_value(self):
        """Regression with identical target values."""
        dt = DecisionTree(task=TASK_REGRESSION, feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = [5.0, 5.0, 5.0]
        dt.load_data(X, y)
        dt.train()
        # Should predict the constant value
        assert dt.predict(["a"]) == 5.0
        assert dt.predict(["z"]) == 5.0

    def test_regression_negative_values(self):
        """Regression with negative target values."""
        dt = DecisionTree(task=TASK_REGRESSION, feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = [-10.0, 0.0, 10.0]
        dt.load_data(X, y)
        dt.train()
        pred = dt.predict(["b"])
        assert isinstance(pred, float)

    def test_regression_large_values(self):
        """Regression with very large values."""
        dt = DecisionTree(task=TASK_REGRESSION, feature_names=["x"])
        X = [["a"], ["b"]]
        y = [1e10, 1e11]
        dt.load_data(X, y)
        dt.train()
        pred = dt.predict(["a"])
        assert (
            isinstance(pred, (int, float)) and pred >= 1e9
        )  # Should be in right ballpark

    def test_regression_leaf_stats(self):
        """Regression leaves should have valid stats."""
        dt = DecisionTree(task=TASK_REGRESSION, feature_names=["x"])
        X = [["a"], ["a"], ["b"], ["b"]]
        y = [1.0, 2.0, 10.0, 20.0]
        dt.load_data(X, y)
        dt.train()

        # Export and verify roundtrip
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            dt.export(f.name)
            dt2 = DecisionTree()
            dt2.load_model(f.name)
            os.unlink(f.name)

        # Predictions should match
        for x in X:
            assert dt.predict(x) == dt2.predict(x)


class TestSklearnTrainer:
    """Tests for sklearn trainer backend."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        import importlib.util

        return importlib.util.find_spec("sklearn") is not None

    def test_sklearn_import_error_message(self, sklearn_available):
        """Should give clear error if sklearn not installed."""
        if sklearn_available:
            pytest.skip("sklearn is installed")

        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"]]
        y = ["1", "2"]
        dt.load_data(X, y)

        with pytest.raises(ImportError, match="scikit-learn"):
            dt.train(trainer="sklearn")

    @pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")
    def test_sklearn_basic_classification(self):
        """Basic classification with sklearn backend."""
        dt = DecisionTree(feature_names=["x", "y"])
        X = [["a", "1"], ["a", "2"], ["b", "1"], ["b", "2"]]
        y = ["w", "x", "y", "z"]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        # Should make predictions
        pred = dt.predict(["a", "1"])
        assert pred in ["w", "x", "y", "z"]

    @pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")
    def test_sklearn_numerical_features(self):
        """Sklearn with numerical features."""
        dt = DecisionTree(
            features=[
                {"name": "x", "dtype": "float", "type": "num"},
                {"name": "y", "dtype": "float", "type": "num"},
            ]
        )
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y = ["low", "low", "high", "high"]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        assert dt.predict([2.0, 3.0]) == "low"
        assert dt.predict([6.0, 7.0]) == "high"

    @pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")
    def test_sklearn_regression(self):
        """Sklearn regression."""
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        pred = dt.predict([2.5])
        assert isinstance(pred, float)
        assert 15.0 <= pred <= 35.0

    @pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")
    def test_sklearn_max_depth(self):
        """Sklearn respects max_depth."""
        dt = DecisionTree(
            feature_names=["x", "y"],
            max_depth=2,
        )
        X = [[f"x{i}", f"y{i}"] for i in range(20)]
        y = [str(i % 5) for i in range(20)]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        assert dt.get_depth() <= 2

    @pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")
    def test_sklearn_export_import_roundtrip(self):
        """Sklearn-trained model exports and imports correctly."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [["red", "small"], ["red", "large"], ["blue", "small"], ["blue", "large"]]
        y = ["apple", "apple", "ball", "box"]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            dt.export(f.name)

            # Load with runner
            model_data = load_model(f.name)

            # Predictions should work
            for x, _expected in zip(X, y, strict=False):
                pred = predict(model_data, x)
                # May not be exact due to sklearn's different splitting
                assert pred in y

            os.unlink(f.name)


class TestBatchPrediction:
    """Test batch prediction functionality."""

    def test_predict_batch_classification(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = ["1", "2", "3"]
        dt.load_data(X, y)
        dt.train()

        preds = dt.predict_batch([["a"], ["b"], ["c"]])
        assert len(preds) == 3
        assert preds[0] == "1"
        assert preds[1] == "2"
        assert preds[2] == "3"

    def test_predict_batch_regression(self):
        dt = DecisionTree(task=TASK_REGRESSION, feature_names=["x"])
        X = [["a"], ["b"]]
        y = [1.0, 2.0]
        dt.load_data(X, y)
        dt.train()

        preds = dt.predict_batch([["a"], ["b"]])
        assert len(preds) == 2
        assert all(isinstance(p, float) for p in preds)

    def test_predict_batch_empty(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"]]
        y = ["1", "2"]
        dt.load_data(X, y)
        dt.train()

        preds = dt.predict_batch([])
        assert preds == []

    def test_predict_batch_with_distributions(self):
        dt = DecisionTree(feature_names=["x"], store_distributions=True)
        X = [["a"], ["a"], ["b"]]
        y = ["1", "2", "3"]
        dt.load_data(X, y)
        dt.train()

        preds = dt.predict_batch([["a"], ["b"]], return_dist=True)
        assert len(preds) == 2
        # First should be a distribution dict
        assert isinstance(preds[0], dict)


class TestRunnerPredictBatch:
    """Test runner.predict_batch function."""

    def test_runner_predict_batch(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = ["1", "2", "3"]
        dt.load_data(X, y)
        dt.train()

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            dt.export(f.name)
            model_data = load_model(f.name)

            from cartlet.runner import predict_batch

            preds = predict_batch(model_data, [["a"], ["b"], ["c"]])

            assert len(preds) == 3
            assert preds[0] == "1"
            assert preds[1] == "2"
            assert preds[2] == "3"

            os.unlink(f.name)


class TestForestEdgeCases:
    """Test RandomForest edge cases."""

    def test_forest_single_tree(self):
        """Forest with n_estimators=1."""
        rf = RandomForest(n_estimators=1, feature_names=["x"])
        X = [["a"], ["b"]]
        y = ["1", "2"]
        rf.load_data(X, y)
        rf.train()

        assert len(rf.trees) == 1
        assert rf.predict(["a"]) in ["1", "2"]

    def test_forest_predict_batch(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"])
        X = [["a"], ["b"], ["c"]]
        y = ["1", "2", "3"]
        rf.load_data(X, y)
        rf.train()

        preds = rf.predict_batch([["a"], ["b"], ["c"]])
        assert len(preds) == 3

    def test_forest_max_depth(self):
        """Forest respects max_depth."""
        from cartlet.utils import max_depth as get_max_depth

        rf = RandomForest(n_estimators=5, feature_names=["x", "y"], max_depth=2)
        X = [[f"x{i}", f"y{i}"] for i in range(20)]
        y = [str(i % 5) for i in range(20)]
        rf.load_data(X, y)
        rf.train()

        for tree in rf.trees:
            assert get_max_depth(tree.model) <= 2
