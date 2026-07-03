"""
Evaluation metrics for decision tree models.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from .types import TASK_CLASSIFICATION, TASK_REGRESSION


def evaluate_predictions(
    y_true: list[Any],
    y_pred: list[Any],
) -> dict[str, float]:
    """
    Evaluate prediction accuracy.

    Compares labels via exact equality, so this is intended for classification.
    For regression error metrics use `evaluate_tree`.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (same length as `y_true`).

    Returns:
        Dict with `accuracy`, `correct`, and `total` keys. Returns
        `{"accuracy": 0.0, "total": 0}` if both lists are empty.

    Raises:
        ValueError: If `len(y_true) != len(y_pred)`.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if not y_true:
        return {"accuracy": 0.0, "total": 0}

    correct = sum(1 for true, pred in zip(y_true, y_pred, strict=False) if true == pred)
    total = len(y_true)

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


def regression_metrics(
    y_true: list[Any],
    y_pred: list[Any],
    *,
    include_r2: bool = False,
) -> dict[str, float]:
    """
    Compute regression error metrics from parallel target/prediction lists.

    Args:
        y_true: True target values (any numeric-castable type).
        y_pred: Predicted target values (same length as `y_true`).
        include_r2: If True, also compute coefficient of determination (R^2).
            Returns 0.0 when the total sum of squares is zero (i.e. all
            targets equal), matching the convention used by the CLI.

    Returns:
        Dict with `mse`, `mae`, `rmse`, `total`, and optionally `r2`.

    Raises:
        ValueError: If `len(y_true) != len(y_pred)` or both are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute regression metrics on empty input")

    errors = [float(t) - float(p) for t, p in zip(y_true, y_pred, strict=False)]
    ss_res = sum(e * e for e in errors)
    mse = ss_res / n
    mae = sum(abs(e) for e in errors) / n
    result: dict[str, float] = {"mse": mse, "mae": mae, "rmse": mse**0.5, "total": n}

    if include_r2:
        y_floats = [float(t) for t in y_true]
        y_mean = sum(y_floats) / n
        ss_tot = sum((t - y_mean) ** 2 for t in y_floats)
        result["r2"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return result


def evaluate_tree(
    tree_model: Any,
    X_test: list[list[Any]],
    y_test: list[Any],
) -> dict[str, Any]:
    """
    Evaluate a model on test data.

    Dispatches on task: classification returns accuracy via exact equality,
    regression returns MSE/MAE/RMSE. The task is taken from the model's
    `_is_regression()` method when available, otherwise inferred from
    `y_test` (all-numeric, non-bool values are treated as regression).

    Args:
        tree_model: Any model exposing predict(vector). DecisionTree,
            RandomForest, XGBoostTree, etc.
        X_test: Test feature vectors
        y_test: Test targets

    Returns:
        A dict whose ``task`` key is either ``"classification"`` or
        ``"regression"``. For classification: also has ``accuracy``,
        ``correct``, ``total``. For regression: also has ``mse``, ``mae``,
        ``rmse``, ``total``. Callers can dispatch on ``result["task"]``
        without having to probe which metric keys are present.

    Raises:
        ValueError: If `X_test` is empty (no predictions to score against).
    """
    if not X_test:
        raise ValueError("Cannot evaluate on empty test set")
    if len(y_test) != len(X_test):
        raise ValueError(
            f"X_test/y_test length mismatch: {len(X_test)} vs {len(y_test)}"
        )

    y_pred = [tree_model.predict(x) for x in X_test]

    is_regression_fn = getattr(tree_model, "_is_regression", None)
    if callable(is_regression_fn):
        is_regression = is_regression_fn()
    else:
        is_regression = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in y_test
        )

    if is_regression:
        metrics: dict[str, Any] = dict(regression_metrics(y_test, y_pred))
        metrics["task"] = TASK_REGRESSION
        return metrics
    metrics = dict(evaluate_predictions(y_test, y_pred))
    metrics["task"] = TASK_CLASSIFICATION
    return metrics


def confusion_matrix(
    y_true: list[Any],
    y_pred: list[Any],
) -> dict[tuple[Any, Any], int]:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict mapping (true_label, predicted_label) -> count
    """
    matrix: dict[tuple[Any, Any], int] = {}

    for true, pred in zip(y_true, y_pred, strict=False):
        key = (true, pred)
        matrix[key] = matrix.get(key, 0) + 1

    return matrix


def per_class_metrics(
    y_true: list[Any],
    y_pred: list[Any],
) -> dict[Any, dict[str, float]]:
    """
    Compute per-class precision, recall, F1.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict mapping class -> metrics dict
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = set(y_true) | set(y_pred)

    results = {}
    for cls in classes:
        # True positives, false positives, false negatives
        tp = cm.get((cls, cls), 0)
        fp = sum(cm.get((other, cls), 0) for other in classes if other != cls)
        fn = sum(cm.get((cls, other), 0) for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[cls] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": sum(1 for y in y_true if y == cls),
        }

    return results


def cross_validate(
    tree_class: Any,
    X: list[list[Any]],
    y: list[Any],
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    **tree_kwargs,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation.

    For classification, the per-fold score is accuracy (higher is better).
    For regression, the per-fold score is MSE (lower is better). The task
    is detected by `evaluate_tree`.

    Args:
        tree_class: Model class (DecisionTree, RandomForest, ...)
        X: Feature vectors
        y: Targets
        n_folds: Number of folds
        shuffle: Whether to shuffle data before splitting (default: True)
        random_state: Random seed for reproducibility (default: None)
        **tree_kwargs: Arguments to pass to the model constructor

    Returns:
        Dict with keys:
          - "scores": per-fold score (accuracy for classification, MSE for regression)
          - "metric": "accuracy" or "mse"
          - "mean", "std": summary statistics across folds
          - "n_folds": number of folds
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    if len(X) < n_folds:
        raise ValueError("Not enough data for cross-validation")

    indices = list(range(len(X)))
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(indices)

    fold_size = len(X) // n_folds
    scores: list[float] = []
    metric: str | None = None

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(X)

        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]

        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]

        tree = tree_class(**tree_kwargs)
        tree.load_data(X_train, y_train)
        tree.train()

        fold_metrics = evaluate_tree(tree, X_test, y_test)
        if fold_metrics["task"] == TASK_CLASSIFICATION:
            metric = "accuracy"
            scores.append(fold_metrics["accuracy"])
        else:
            metric = "mse"
            scores.append(fold_metrics["mse"])

    return {
        "scores": scores,
        "metric": metric,
        "mean": statistics.mean(scores),
        "n_folds": n_folds,
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }
