"""
Evaluation metrics for decision tree models.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .types import TASK_CLASSIFICATION, TASK_REGRESSION, ModelData
from .validation import validate_dataset


def _inspection_model_data(model: Any) -> ModelData | None:
    """Return runner model data when *model* is a Predictor or ModelData."""
    from .runner import Predictor

    if isinstance(model, Predictor):
        return model.model
    if isinstance(model, dict) and {
        "meta",
        "decisions",
        "leaves",
        "tree_offsets",
    }.issubset(model):
        return cast(ModelData, model)
    return None


def _model_task_and_features(model: Any) -> tuple[str, list[str]]:
    """Return the supervised task and ordered feature names for inspection."""
    model_data = _inspection_model_data(model)
    if model_data is not None:
        if model_data.get("is_xgboost"):
            raise ValueError(
                "permutation_importance supports DecisionTree and RandomForest models, "
                "not XGBoost-derived models"
            )
        meta = model_data["meta"]
        return meta["task"], [feature["name"] for feature in meta["features"]]

    from .forest import RandomForest
    from .tree import DecisionTree

    if not isinstance(model, (DecisionTree, RandomForest)):
        raise TypeError(
            "model must be a DecisionTree, RandomForest, Predictor, or ModelData"
        )
    return model._effective_task(), list(model.feature_names)


def _predict_for_importance(model: Any, row: list[Any], missing: str) -> Any:
    """Predict through either the nested-model or runner API."""
    model_data = _inspection_model_data(model)
    if model_data is not None:
        from .runner import predict

        return predict(model_data, row, missing=missing)
    return model.predict(row, missing=missing)


def _validate_importance_parameters(
    n_repeats: int, random_state: int | None, missing: str
) -> None:
    """Validate options shared by held-out and OOB permutation importance."""
    if missing not in ("error", "right"):
        raise ValueError("missing must be 'error' or 'right'")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, int)
    ):
        raise TypeError("random_state must be an int or None")


def _importance_units(
    feature_names: list[str],
    feature_groups: Mapping[str, Sequence[str | int]] | None,
) -> list[tuple[str, list[int]]]:
    """Resolve the ordered features or feature groups to permute."""
    if not feature_names:
        raise ValueError("model has no feature schema")
    if feature_groups is None:
        return [(name, [index]) for index, name in enumerate(feature_names)]
    if not isinstance(feature_groups, Mapping) or not feature_groups:
        raise ValueError("feature_groups must be a nonempty mapping")
    units = []
    for group_name, members in feature_groups.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("feature group names must be nonempty strings")
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise ValueError(
                f"feature group {group_name!r} must contain feature names or indexes"
            )
        indexes: list[int] = []
        for member in members:
            if isinstance(member, bool):
                raise ValueError(
                    f"invalid feature reference {member!r} in group {group_name!r}"
                )
            if isinstance(member, int):
                index = member
            elif isinstance(member, str) and member in feature_names:
                index = feature_names.index(member)
            else:
                raise ValueError(f"unknown feature {member!r} in group {group_name!r}")
            if index < 0 or index >= len(feature_names):
                raise ValueError(
                    f"feature index {index} in group {group_name!r} is out of range"
                )
            if index in indexes:
                raise ValueError(
                    f"feature {member!r} appears more than once in group {group_name!r}"
                )
            indexes.append(index)
        if not indexes:
            raise ValueError(f"feature group {group_name!r} must not be empty")
        units.append((group_name, indexes))
    return units


def _importance_loss(task: str, targets: list[Any], predictions: list[Any]) -> float:
    """Return the task-appropriate loss used by permutation importance."""
    if task == TASK_CLASSIFICATION:
        return 1.0 - evaluate_predictions(targets, predictions)["accuracy"]
    return regression_metrics(targets, predictions)["mse"]


def permutation_importance(
    model: Any,
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    *,
    feature_groups: Mapping[str, Sequence[str | int]] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
    missing: str = "error",
) -> dict[str, Any]:
    """Measure held-out permutation importance for a tree or forest.

    Classification uses the increase in error rate and regression uses the
    increase in mean squared error. Group members share one row permutation,
    preserving their within-row relationship. The caller's rows are never
    mutated.

    Args:
        model: Trained or reloaded DecisionTree/RandomForest, or a `.cart`
            Predictor/ModelData for one of those model types.
        X: Nonempty held-out feature rows.
        y: Held-out targets aligned with ``X``. Classification targets are
            canonicalized to the model's string label representation.
        feature_groups: Optional insertion-ordered group-name mapping. Members
            are feature names or zero-based indexes. When omitted, each model
            feature is measured separately.
        n_repeats: Positive number of permutations per feature or group.
        random_state: Integer seed or None.
        missing: Missing-feature policy passed to every prediction.

    Returns:
        A JSON-compatible report ordered by decreasing mean loss increase.

    Raises:
        TypeError: If ``random_state`` is not an int or None.
        ValueError: If rows, targets, groups, repeats, or missing policy are
            invalid, or the model is unsupported.
    """
    _validate_importance_parameters(n_repeats, random_state, missing)
    if not isinstance(X, Sequence) or isinstance(X, (str, bytes)) or not X:
        raise ValueError("X must contain at least one held-out row")
    if not isinstance(y, Sequence) or isinstance(y, (str, bytes)):
        raise ValueError("y must be a sequence of held-out targets")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in X):
        raise ValueError("held-out rows must be sequences of feature values")

    task, feature_names = _model_task_and_features(model)
    rows = [list(row) for row in X]
    targets = list(y)
    if task == TASK_CLASSIFICATION:
        targets = [str(value) for value in targets]

    units = _importance_units(feature_names, feature_groups)

    def loss(sample_rows: list[list[Any]]) -> float:
        predictions = [
            _predict_for_importance(model, row, missing) for row in sample_rows
        ]
        return _importance_loss(task, targets, predictions)

    baseline = loss(rows)
    rng = random.Random(random_state)
    importances: list[dict[str, Any]] = []
    for name, indexes in units:
        values = []
        for _ in range(n_repeats):
            donors = list(range(len(rows)))
            rng.shuffle(donors)
            permuted = [row.copy() for row in rows]
            for destination, donor in enumerate(donors):
                for index in indexes:
                    donor_value = (
                        rows[donor][index] if index < len(rows[donor]) else None
                    )
                    while index >= len(permuted[destination]):
                        permuted[destination].append(None)
                    permuted[destination][index] = donor_value
            values.append(loss(permuted) - baseline)
        importances.append(
            {
                "name": name,
                "features": [feature_names[index] for index in indexes],
                "values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    importances.sort(key=lambda record: float(record["mean"]), reverse=True)
    return {
        "task": task,
        "metric": "error_rate" if task == TASK_CLASSIFICATION else "mse",
        "baseline": baseline,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "importances": importances,
    }


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
        Dict with `accuracy`, `correct`, and `total` keys (always all three,
        so callers can index any of them). Returns
        `{"accuracy": 0.0, "correct": 0, "total": 0}` if both lists are empty.

    Raises:
        ValueError: If `len(y_true) != len(y_pred)`.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if not y_true:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

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
    X_test: Sequence[Sequence[Any]],
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
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
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

    # Row sums (support = # true==cls) and column sums (# predicted==cls) in a
    # single pass over the matrix cells, so per-class fp/fn are O(1) derivations
    # rather than nested O(classes) sums, and support isn't re-counted over
    # y_true per class.
    row_sum: dict[Any, int] = {}
    col_sum: dict[Any, int] = {}
    for (true, pred), count in cm.items():
        row_sum[true] = row_sum.get(true, 0) + count
        col_sum[pred] = col_sum.get(pred, 0) + count

    results = {}
    for cls in classes:
        tp = cm.get((cls, cls), 0)
        fp = col_sum.get(cls, 0) - tp  # predicted cls but true != cls
        fn = row_sum.get(cls, 0) - tp  # true cls but predicted != cls

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
            "support": row_sum.get(cls, 0),
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
    X, targets, _ = validate_dataset(X, y)
    assert targets is not None
    y = targets
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    if len(X) < n_folds:
        raise ValueError("Not enough data for cross-validation")

    indices = list(range(len(X)))
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(indices)

    fold_size, extra = divmod(len(X), n_folds)
    test_start = 0
    scores: list[float] = []
    metric: str | None = None

    for fold_idx in range(n_folds):
        test_end = test_start + fold_size + (fold_idx < extra)

        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]

        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]

        test_start = test_end
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
