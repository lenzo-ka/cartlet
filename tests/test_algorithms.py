"""Tests for Gini impurity, Extra-Trees, and Isolation Forest."""

import os
import tempfile

import pytest

from cartlet import TASK_REGRESSION, DecisionTree, IsolationForest, RandomForest

# =============================================================================
# Gini impurity
# =============================================================================


class TestGiniCriterion:
    """Test Gini impurity as split criterion."""

    def test_gini_classification(self):
        dt = DecisionTree(feature_names=["color", "size"], criterion="gini")
        X = [["red", "small"], ["red", "large"], ["blue", "small"], ["blue", "large"]]
        y = ["apple", "apple", "ball", "box"]
        dt.load_data(X, y)
        dt.train()

        assert dt.predict(["red", "small"]) == "apple"
        assert dt.predict(["blue", "large"]) == "box"

    def test_gini_produces_valid_tree(self):
        dt = DecisionTree(feature_names=["x", "y"], criterion="gini")
        X = [["a", "1"], ["a", "2"], ["b", "1"], ["b", "2"]] * 5
        y = ["A", "A", "B", "B"] * 5
        dt.load_data(X, y)
        dt.train()

        assert dt.model is not None
        assert dt.get_depth() > 0

    def test_gini_vs_entropy_both_work(self):
        X = [["a"], ["a"], ["b"], ["b"]] * 10
        y = ["yes", "no", "yes", "no"] * 10

        dt_gini = DecisionTree(feature_names=["x"], criterion="gini")
        dt_gini.load_data(X, y)
        dt_gini.train()

        dt_entropy = DecisionTree(feature_names=["x"], criterion="entropy")
        dt_entropy.load_data(X, y)
        dt_entropy.train()

        # Both should produce valid predictions
        assert dt_gini.predict(["a"]) in ("yes", "no")
        assert dt_entropy.predict(["a"]) in ("yes", "no")

    def test_gini_regression_unaffected(self):
        """Gini should have no effect on regression (variance is always used)."""
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
            criterion="gini",
        )
        X = [[float(i)] for i in range(50)]
        y = [float(i) for i in range(50)]
        dt.load_data(X, y)
        dt.train()

        pred = dt.predict([25.0])
        assert isinstance(pred, (int, float, list))

    def test_gini_with_forest(self):
        rf = RandomForest(n_estimators=5, feature_names=["x"], criterion="gini")
        X = [["a"], ["b"]] * 20
        y = ["1", "2"] * 20
        rf.load_data(X, y)
        rf.train()

        assert len(rf.trees) == 5
        assert rf.predict(["a"]) in ("1", "2")

    def test_gini_export_roundtrip(self):
        dt = DecisionTree(feature_names=["x", "y"], criterion="gini")
        X = [["a", "1"], ["b", "2"]] * 10
        y = ["A", "B"] * 10
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gini.cart")
            dt.export(path)

            dt2 = DecisionTree()
            dt2.load_model(path)

            assert dt.predict(["a", "1"]) == dt2.predict(["a", "1"])
            assert dt.predict(["b", "2"]) == dt2.predict(["b", "2"])

    def test_gini_numerical_features(self):
        dt = DecisionTree(
            features=[
                {"name": "x", "dtype": "float", "type": "num"},
                {"name": "y", "dtype": "float", "type": "num"},
            ],
            criterion="gini",
        )
        X = [[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]] * 5
        y = ["low", "low", "high", "high"] * 5
        dt.load_data(X, y)
        dt.train()

        assert dt.predict([1.0, 1.5]) == "low"
        assert dt.predict([5.0, 5.5]) == "high"


# =============================================================================
# Extra-Trees
# =============================================================================


class TestExtraTrees:
    """Test Extra-Trees random split variant."""

    def test_basic_classification(self):
        rf = RandomForest(
            n_estimators=20, extra_trees=True, feature_names=["color", "size"]
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

    def test_regression(self):
        rf = RandomForest(
            n_estimators=20,
            extra_trees=True,
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[float(i)] for i in range(100)]
        y = [float(i) for i in range(100)]
        rf.load_data(X, y)
        rf.train(random_state=42)

        pred = rf.predict([50.0])
        assert isinstance(pred, float)
        assert 20 <= pred <= 80

    def test_repr_shows_extra_trees(self):
        rf = RandomForest(n_estimators=5, extra_trees=True, feature_names=["x"])
        assert "ExtraTrees" in repr(rf)

    def test_repr_trained(self):
        rf = RandomForest(n_estimators=5, extra_trees=True, feature_names=["x"])
        rf.load_data([["a"], ["b"]] * 10, ["1", "2"] * 10)
        rf.train()
        assert "ExtraTrees" in repr(rf)
        assert "5 trees" in repr(rf)

    def test_get_params(self):
        rf = RandomForest(n_estimators=10, extra_trees=True)
        params = rf.get_params()
        assert params["extra_trees"] is True

    def test_export_cart_roundtrip(self):
        rf = RandomForest(
            n_estimators=5, extra_trees=True, feature_names=["color", "size"]
        )
        X = [["red", "small"], ["blue", "large"]] * 20
        y = ["apple", "ball"] * 20
        rf.load_data(X, y)
        rf.train(random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "et.cart")
            rf.export(path)

            rf2 = RandomForest()
            rf2.load_model(path)

            test_vecs = [["red", "small"], ["blue", "large"]]
            for vec in test_vecs:
                assert rf.predict(vec) == rf2.predict(vec)

    def test_predict_proba(self):
        rf = RandomForest(n_estimators=20, extra_trees=True, feature_names=["x"])
        X = [["a"], ["a"], ["a"], ["b"], ["b"]] * 10
        y = ["yes", "yes", "no", "maybe", "maybe"] * 10
        rf.load_data(X, y)
        rf.train(random_state=42)

        proba = rf.predict_proba(["a"])
        assert isinstance(proba, dict)
        assert sum(proba.values()) == pytest.approx(1.0)

    def test_feature_importances(self):
        rf = RandomForest(
            n_estimators=20, extra_trees=True, feature_names=["signal", "noise"]
        )
        X = [["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]] * 20
        y = ["A", "A", "B", "B"] * 20
        rf.load_data(X, y)
        rf.train(random_state=42)

        importances = rf.feature_importances_
        assert sum(importances.values()) == pytest.approx(1.0)

    def test_numerical_features(self):
        rf = RandomForest(
            n_estimators=20,
            extra_trees=True,
            features=[
                {"name": "x", "dtype": "float", "type": "num"},
                {"name": "y", "dtype": "float", "type": "num"},
            ],
        )
        X = [[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]] * 20
        y = ["low", "low", "high", "high"] * 20
        rf.load_data(X, y)
        rf.train(random_state=42)

        assert rf.predict([1.5, 0.0]) == "low"
        assert rf.predict([10.5, 0.0]) == "high"

    def test_with_gini_criterion(self):
        rf = RandomForest(
            n_estimators=10,
            extra_trees=True,
            criterion="gini",
            feature_names=["x"],
        )
        X = [["a"], ["b"]] * 20
        y = ["1", "2"] * 20
        rf.load_data(X, y)
        rf.train(random_state=42)

        assert rf.predict(["a"]) in ("1", "2")


# =============================================================================
# Isolation Forest
# =============================================================================


class TestIsolationForest:
    """Test Isolation Forest anomaly detection."""

    @pytest.fixture
    def normal_data(self):
        """Generate normal cluster data with clear outliers."""
        import random as stdlib_random

        rng = stdlib_random.Random(42)
        X_normal = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
        X_anomaly = [[10.0, 10.0], [-10.0, -10.0]]
        return X_normal, X_anomaly

    def test_anomalies_score_higher(self, normal_data):
        X_normal, X_anomaly = normal_data
        X = X_normal + X_anomaly

        ifo = IsolationForest(n_estimators=100, random_state=42)
        ifo.load_data(X)
        ifo.train()

        normal_scores = [ifo.predict(x) for x in X_normal[:10]]
        anomaly_scores = [ifo.predict(x) for x in X_anomaly]

        assert max(normal_scores) < 0.5
        assert min(anomaly_scores) > 0.5

    def test_score_range(self, normal_data):
        X_normal, X_anomaly = normal_data
        ifo = IsolationForest(n_estimators=50, random_state=42)
        ifo.load_data(X_normal + X_anomaly)
        ifo.train()

        all_scores = ifo.predict_batch(X_normal + X_anomaly)
        for score in all_scores:
            assert 0.0 <= score <= 1.0

    def test_decision_function(self, normal_data):
        X_normal, X_anomaly = normal_data
        ifo = IsolationForest(n_estimators=50, random_state=42)
        ifo.load_data(X_normal + X_anomaly)
        ifo.train()

        # Anomalies should have negative decision_function (sklearn convention)
        for x in X_anomaly:
            assert ifo.decision_function(x) < 0

        # Normal points should tend positive
        normal_decisions = [ifo.decision_function(x) for x in X_normal[:20]]
        assert sum(1 for d in normal_decisions if d > 0) > len(normal_decisions) // 2

    def test_json_roundtrip(self, normal_data):
        X_normal, X_anomaly = normal_data
        ifo = IsolationForest(n_estimators=20, random_state=42)
        ifo.load_data(X_normal + X_anomaly)
        ifo.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ifo.json")
            ifo.export(path)

            ifo2 = IsolationForest()
            ifo2.load_model(path)

            for x in X_anomaly:
                assert ifo.predict(x) == pytest.approx(ifo2.predict(x))

    def test_pickle_roundtrip(self, normal_data):
        X_normal, _ = normal_data
        ifo = IsolationForest(n_estimators=20, random_state=42)
        ifo.load_data(X_normal)
        ifo.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ifo.pkl")
            ifo.export(path)

            ifo2 = IsolationForest()
            ifo2.load_model(path)

            score1 = ifo.predict(X_normal[0])
            score2 = ifo2.predict(X_normal[0])
            assert score1 == pytest.approx(score2)

    def test_auto_feature_names(self):
        ifo = IsolationForest(n_estimators=5, random_state=42)
        ifo.load_data([[1.0, 2.0], [3.0, 4.0]])
        ifo.train()

        assert ifo.feature_names == ["f0", "f1"]

    def test_explicit_feature_names(self):
        ifo = IsolationForest(n_estimators=5, feature_names=["x", "y"], random_state=42)
        ifo.load_data([[1.0, 2.0], [3.0, 4.0]])
        ifo.train()

        assert ifo.feature_names == ["x", "y"]

    def test_max_samples(self):
        X = [[float(i), float(i)] for i in range(1000)]
        ifo = IsolationForest(n_estimators=5, max_samples=64, random_state=42)
        ifo.load_data(X)
        result = ifo.train()

        assert result["max_samples"] == 64

    def test_max_depth(self):
        X = [[float(i), float(i)] for i in range(100)]
        ifo = IsolationForest(n_estimators=5, max_depth=3, random_state=42)
        ifo.load_data(X)
        ifo.train()

        assert len(ifo.trees) == 5
        # Tree depth should be limited
        score = ifo.predict([50.0, 50.0])
        assert 0.0 <= score <= 1.0

    def test_predict_untrained_raises(self):
        ifo = IsolationForest()
        with pytest.raises(ValueError, match="not trained"):
            ifo.predict([1.0, 2.0])

    def test_train_no_data_raises(self):
        ifo = IsolationForest()
        with pytest.raises(ValueError, match="No training data"):
            ifo.train()

    def test_repr(self):
        ifo = IsolationForest(n_estimators=50)
        assert "untrained" in repr(ifo)

        ifo.load_data([[1.0, 2.0], [3.0, 4.0]] * 10)
        ifo.train()
        assert "50 trees" in repr(ifo)

    def test_unsupported_export_format(self):
        ifo = IsolationForest(n_estimators=5, random_state=42)
        ifo.load_data([[1.0], [2.0]])
        ifo.train()

        with pytest.raises(ValueError, match="supports .json and .pkl"):
            ifo.export("model.cart")

    def test_single_feature(self):
        """Isolation forest should work with single feature."""
        import random as stdlib_random

        rng = stdlib_random.Random(42)
        X = [[rng.gauss(0, 1)] for _ in range(100)] + [[100.0]]
        ifo = IsolationForest(n_estimators=50, random_state=42)
        ifo.load_data(X)
        ifo.train()

        normal_score = ifo.predict([0.0])
        anomaly_score = ifo.predict([100.0])
        assert anomaly_score > normal_score
