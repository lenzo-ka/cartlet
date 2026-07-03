"""Tests for evaluation functions."""

import pytest

from cartlet import (
    DecisionTree,
    confusion_matrix,
    cross_validate,
    evaluate_predictions,
    evaluate_tree,
    per_class_metrics,
)


class TestEvaluatePredictions:
    """Test evaluate_predictions function."""

    def test_perfect_accuracy(self):
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        result = evaluate_predictions(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["correct"] == 3
        assert result["total"] == 3

    def test_zero_accuracy(self):
        y_true = ["a", "b", "c"]
        y_pred = ["x", "y", "z"]
        result = evaluate_predictions(y_true, y_pred)
        assert result["accuracy"] == 0.0

    def test_partial_accuracy(self):
        y_true = ["a", "b", "c", "d"]
        y_pred = ["a", "b", "x", "y"]
        result = evaluate_predictions(y_true, y_pred)
        assert result["accuracy"] == 0.5

    def test_empty_raises(self):
        result = evaluate_predictions([], [])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            evaluate_predictions(["a", "b"], ["a"])


class TestConfusionMatrix:
    """Test confusion_matrix function."""

    def test_basic_matrix(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        cm = confusion_matrix(y_true, y_pred)

        assert cm[("a", "a")] == 1  # true positive for a
        assert cm[("a", "b")] == 1  # a predicted as b
        assert cm[("b", "b")] == 2  # true positive for b
        assert ("b", "a") not in cm  # no b predicted as a


class TestPerClassMetrics:
    """Test per_class_metrics function."""

    def test_perfect_classification(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b"]
        metrics = per_class_metrics(y_true, y_pred)

        assert metrics["a"]["precision"] == 1.0
        assert metrics["a"]["recall"] == 1.0
        assert metrics["a"]["f1"] == 1.0
        assert metrics["b"]["precision"] == 1.0
        assert metrics["b"]["recall"] == 1.0
        assert metrics["b"]["f1"] == 1.0

    def test_support_counts(self):
        y_true = ["a", "a", "a", "b"]
        y_pred = ["a", "a", "a", "b"]
        metrics = per_class_metrics(y_true, y_pred)

        assert metrics["a"]["support"] == 3
        assert metrics["b"]["support"] == 1


class TestCrossValidate:
    """Test cross_validate function."""

    def test_basic_cv(self):
        X = [["a"], ["a"], ["b"], ["b"], ["c"], ["c"]] * 5  # 30 samples
        y = ["1", "1", "2", "2", "3", "3"] * 5
        result = cross_validate(
            DecisionTree,
            X,
            y,
            n_folds=3,
            feature_names=["x"],
        )

        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert result["metric"] == "accuracy"
        assert len(result["scores"]) == 3
        assert 0 <= result["mean"] <= 1

    def test_cv_with_shuffle(self):
        X = [["a"]] * 10 + [["b"]] * 10
        y = ["1"] * 10 + ["2"] * 10
        result = cross_validate(
            DecisionTree,
            X,
            y,
            n_folds=2,
            shuffle=True,
            random_state=42,
            feature_names=["x"],
        )

        assert result["mean"] > 0  # With shuffle, should do better than chance

    def test_cv_reproducible_with_seed(self):
        X = [["a"], ["b"], ["c"]] * 10
        y = ["1", "2", "3"] * 10
        result1 = cross_validate(
            DecisionTree, X, y, n_folds=3, shuffle=True, random_state=42
        )
        result2 = cross_validate(
            DecisionTree, X, y, n_folds=3, shuffle=True, random_state=42
        )

        assert result1["scores"] == result2["scores"]

    def test_cv_too_few_folds_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            cross_validate(DecisionTree, [["a"]], ["1"], n_folds=1)

    def test_cv_not_enough_data_raises(self):
        with pytest.raises(ValueError, match="Not enough data"):
            cross_validate(DecisionTree, [["a"], ["b"]], ["1", "2"], n_folds=5)


class TestEvaluateTree:
    """Test evaluate_tree function."""

    def test_evaluate_trained_tree(self):
        dt = DecisionTree(feature_names=["x"])
        X = [["a"], ["b"]]
        y = ["1", "2"]
        dt.load_data(X, y)
        dt.train()

        result = evaluate_tree(dt, X, y)
        assert result["task"] == "classification"
        assert result["accuracy"] == 1.0

    def test_empty_test_set_raises(self):
        dt = DecisionTree(feature_names=["x"])
        dt.load_data([["a"]], ["1"])
        dt.train()

        with pytest.raises(ValueError, match="empty test set"):
            evaluate_tree(dt, [], [])

    def test_mismatched_lengths_raises(self):
        dt = DecisionTree(feature_names=["x"])
        dt.load_data([["a"], ["b"]], ["1", "2"])
        dt.train()

        with pytest.raises(ValueError, match="length mismatch"):
            evaluate_tree(dt, [["a"], ["b"]], ["1"])
