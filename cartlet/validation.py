"""Shared input contracts for models and operational workflows."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any

MODEL_SCHEMA_VERSION = 2


def validate_training_parameters(
    *,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    n_estimators: int = 1,
    n_jobs: int | None = None,
    random_state: int | None = None,
    trainer: str | None = None,
    criterion: str = "entropy",
    categorical_split: str = "exact",
    store_distributions: bool = True,
    prune: bool = False,
    extra_trees: bool = False,
    bootstrap: bool = True,
) -> None:
    """Shared tree/workflow parameter bounds, including backend restrictions.

    Native depth zero means one leaf. Sklearn requires positive depth and at
    least two samples for a split. None selects the native trainer; valid
    integer n_jobs is ignored by native training.
    """
    for name, value in (
        ("min_samples_split", min_samples_split),
        ("min_samples_leaf", min_samples_leaf),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators <= 0
    ):
        raise ValueError("n_estimators must be a positive integer")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be a nonnegative integer or None")
    for name, value in (("random_state", random_state), ("n_jobs", n_jobs)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name} must be an integer or None")
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero")
    for name, value in (
        ("store_distributions", store_distributions),
        ("prune", prune),
        ("extra_trees", extra_trees),
        ("bootstrap", bootstrap),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    if trainer not in (None, "native", "sklearn"):
        raise ValueError("trainer must be native or sklearn")
    if criterion not in ("entropy", "gini") or categorical_split not in (
        "exact",
        "fast",
    ):
        raise ValueError("invalid criterion or categorical_split")
    if trainer == "sklearn":
        if max_depth == 0 or min_samples_split < 2:
            raise ValueError(
                "sklearn requires positive max_depth and min_samples_split >= 2"
            )
        if random_state is not None and not 0 <= random_state < 2**32:
            raise ValueError("sklearn random_state must be in [0, 2**32)")


def effective_validation_split(
    prune: bool, validation_split: float, supported: bool
) -> float:
    """Reserve validation only for supported pruning, with no hidden fallback."""
    if not prune or not supported:
        return 0.0
    if validation_split == 0:
        raise ValueError("supported pruning requires positive validation_split")
    return validation_split


def validate_dataset(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any] | None = None,
    counts: Sequence[float] | None = None,
) -> tuple[list[list[Any]], list[Any] | None, list[Any]]:
    """Copy rectangular finite scalar data, omitting zero-weight observations.

    Weights must be finite, nonnegative and have positive total mass. Missing
    values and nested/unhashable cells are not supported training inputs.
    Inputs are Python sequences; convert NumPy arrays with .tolist() first.
    """
    if not isinstance(X, Sequence) or isinstance(X, (str, bytes)):
        raise ValueError(
            "training data must be Python sequences; convert arrays with .tolist()"
        )
    if not X:
        raise ValueError("training data must contain observations")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have same length")
    if counts is not None and len(counts) != len(X):
        raise ValueError("counts and X must have same length")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in X):
        raise ValueError("training rows must be sequences of feature values")
    rows = [list(row) for row in X]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("training rows must be rectangular with at least one feature")
    targets = list(y) if y is not None else None
    for value in chain(chain.from_iterable(rows), targets or []):
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, (int, float)) and not math.isfinite(value)
        ):
            raise ValueError("training values must be finite scalar strings or numbers")
    weights = list(counts) if counts is not None else [1] * len(rows)
    if any(
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights
    ) or not any(weight > 0 for weight in weights):
        raise ValueError(
            "weights must be finite nonnegative numbers with positive total"
        )
    try:
        total = math.fsum(weights)
    except OverflowError as exc:
        raise ValueError("total training weight must be finite") from exc
    if not math.isfinite(total):
        raise ValueError("total training weight must be finite")
    active = [i for i, weight in enumerate(weights) if weight > 0]
    return (
        [rows[i] for i in active],
        [targets[i] for i in active] if targets is not None else None,
        [weights[i] for i in active],
    )


def validate_splits(validation_split: float = 0.0, test_split: float = 0.0) -> None:
    """Require finite fractions leaving a positive training fraction."""
    if (
        any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value < 1
            for value in (validation_split, test_split)
        )
        or validation_split + test_split >= 1
    ):
        raise ValueError(
            "split fractions must be finite, nonnegative and sum to less than 1"
        )


def align_features(
    rows: Sequence[Sequence[Any]],
    input_names: Sequence[str],
    model_names: Sequence[str],
) -> list[list[Any]]:
    """Align named rows to model columns; no-header callers use positional rows."""
    if len(set(input_names)) != len(input_names) or len(set(model_names)) != len(
        model_names
    ):
        raise ValueError("feature names must be unique")
    missing = set(model_names) - set(input_names)
    if missing:
        raise ValueError(f"missing model features: {', '.join(sorted(missing))}")
    indices = [input_names.index(name) for name in model_names]
    if any(len(row) != len(input_names) for row in rows):
        raise ValueError("input rows do not match feature columns")
    return [[row[i] for i in indices] for row in rows]


def require_distinct_paths(
    inputs: Sequence[str | Path], outputs: Sequence[str | Path]
) -> None:
    """Reject output aliases of input artifacts or of another output."""

    def same(a: str | Path, b: str | Path) -> bool:
        if Path(a).resolve() == Path(b).resolve():
            return True
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)

    for i, output in enumerate(outputs):
        if any(same(output, source) for source in inputs) or any(
            same(output, previous) for previous in outputs[:i]
        ):
            raise ValueError("output paths must differ from inputs and other outputs")


def validate_model_data(data: Any, *, forest: bool = False) -> None:
    """Validate decoded supervised model fields before applying instance state.

    This checks the in-memory tree grammar; codecs remain owned by the IO layer.
    Positional decision references do not require feature names.
    """
    from .types import VALID_DTYPES, VALID_TASKS, VALID_TYPES
    from .utils import is_decision_node

    if not isinstance(data, dict) or data.get("isolation_forest"):
        raise ValueError("expected a supervised model object")
    if data.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "unsupported model schema; retrain or re-export with this release"
        )
    names = data.get("feature_names", [])
    if (
        not isinstance(names, list)
        or any(not isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("model feature_names must be unique strings")
    specs = data.get("feature_specs", [])
    if not isinstance(specs, list) or any(
        not isinstance(s, dict) or not isinstance(s.get("name"), str) for s in specs
    ):
        raise ValueError("model feature_specs must contain named objects")
    if specs and [spec["name"] for spec in specs] != names:
        raise ValueError("model feature_specs must match feature_names in order")
    for spec in specs:
        dtype, kind = spec.get("dtype", "str"), spec.get("type")
        if (
            not isinstance(dtype, str)
            or dtype not in VALID_DTYPES
            or (
                kind is not None
                and (not isinstance(kind, str) or kind not in VALID_TYPES)
            )
        ):
            raise ValueError("invalid model feature dtype or type")
        values = spec.get("values")
        if values is not None and (
            not isinstance(values, (list, tuple, set))
            or any(not isinstance(v, (str, int, float, bool)) for v in values)
        ):
            raise ValueError("model categorical values must be scalar sequences")
        if values is not None and any(
            isinstance(v, (int, float)) and not math.isfinite(v) for v in values
        ):
            raise ValueError("model categorical values must be finite")
    if (
        not isinstance(data.get("metadata", {}), dict)
        or not isinstance(data.get("task", "auto"), str)
        or data.get("task", "auto") not in VALID_TASKS
    ):
        raise ValueError("invalid model metadata or task")
    if forest:
        roots = data.get("trees")
        if not isinstance(roots, list) or not roots:
            raise ValueError("forest must contain trees")
    else:
        if "model" not in data:
            raise ValueError("tree model is missing model field")
        roots = [data["model"]]
    pending = [(root, False) for root in roots]
    active: set[int] = set()
    while pending:
        node, exiting = pending.pop()
        if exiting:
            active.remove(id(node))
            continue
        if isinstance(node, str):
            continue
        if (
            isinstance(node, dict)
            and node
            and all(
                isinstance(label, str)
                and isinstance(p, (int, float))
                and not isinstance(p, bool)
                and math.isfinite(p)
                and p >= 0
                for label, p in node.items()
            )
            and any(node.values())
        ):
            continue
        if (
            isinstance(node, list)
            and len(node) == 3
            and all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in node
            )
            and node[1] >= 0
            and node[2] > 0
        ):
            continue
        if is_decision_node(node):
            feature, op, value, left, right = node
            if not (
                (isinstance(feature, str) and feature in names)
                or (
                    isinstance(feature, int)
                    and not isinstance(feature, bool)
                    and feature >= 0
                    and (not names or feature < len(names))
                )
            ) or op not in ("=", "<=", "<"):
                raise ValueError("invalid model decision reference or operator")
            if op in ("<=", "<"):
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("invalid model numerical threshold") from exc
                if not finite:
                    raise ValueError("invalid model numerical threshold")
            if id(node) in active:
                raise ValueError("model tree contains a cycle")
            active.add(id(node))
            pending.extend([(node, True), (left, False), (right, False)])
            continue
        raise ValueError("invalid model tree node")
