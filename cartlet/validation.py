"""Shared input contracts for models and operational workflows."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any


def validate_dataset(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any] | None = None,
    counts: Sequence[float] | None = None,
) -> tuple[list[list[Any]], list[Any] | None, list[Any]]:
    """Copy rectangular finite scalar data, omitting zero-weight observations.

    Weights must be finite, nonnegative and have positive total mass. Missing
    values and nested/unhashable cells are not supported training inputs.
    """
    if not X:
        raise ValueError("training data must contain observations")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have same length")
    if counts is not None and len(counts) != len(X):
        raise ValueError("counts and X must have same length")
    if any(not isinstance(row, (list, tuple)) for row in X):
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
