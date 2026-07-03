"""
Shared helpers for cartlet example scripts.

The helpers here keep boilerplate (loading public datasets, formatting
reports, argparse) in one place so each example module can focus on the
cartlet calls it is demonstrating.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split

from cartlet import TASK_CLASSIFICATION, TASK_REGRESSION

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_FRACTION = 0.3
_REPORT_WIDTH = 60


@dataclass
class Dataset:
    """A train/test split of a public sklearn dataset, ready for cartlet."""

    name: str
    task: str
    X_train: list[list[float]]
    X_test: list[list[float]]
    y_train: list[Any]
    y_test: list[Any]
    feature_names: list[str]
    class_names: list[str] | None  # None for regression

    @property
    def feature_specs(self) -> list[dict[str, Any]]:
        """Numerical feature specs for use with DecisionTree/RandomForest."""
        return [
            {"name": n, "dtype": "float", "type": "num"} for n in self.feature_names
        ]


_LOADERS: dict[str, tuple[Callable[[], Any], str]] = {
    "iris": (load_iris, TASK_CLASSIFICATION),
    "wine": (load_wine, TASK_CLASSIFICATION),
    "breast_cancer": (load_breast_cancer, TASK_CLASSIFICATION),
    "diabetes": (load_diabetes, TASK_REGRESSION),
}


def available_datasets() -> list[str]:
    """Names of datasets supported by ``load_dataset``."""
    return sorted(_LOADERS)


def load_dataset(
    name: str,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dataset:
    """
    Load a public sklearn dataset and split it deterministically.

    Args:
        name: One of ``available_datasets()``.
        test_fraction: Fraction held out for evaluation.
        random_state: Seed for the train/test split.

    Returns:
        A populated ``Dataset``. Targets are class-name strings for
        classification tasks and Python floats for regression.
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {available_datasets()}"
        )
    loader, task = _LOADERS[name]
    data = loader()

    if task == TASK_CLASSIFICATION:
        class_names = [str(n) for n in data.target_names]
        y_all: list[Any] = [class_names[int(c)] for c in data.target.tolist()]
    else:
        class_names = None
        y_all = [float(v) for v in data.target.tolist()]

    X_train, X_test, y_train, y_test = train_test_split(
        data.data.tolist(),
        y_all,
        test_size=test_fraction,
        random_state=random_state,
    )
    feature_names = [str(n).replace(" ", "_") for n in data.feature_names]
    return Dataset(
        name=name,
        task=task,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
    )


def print_header(title: str, *, file: Any = None) -> None:
    """Print a banner. Default destination is stdout."""
    if file is None:
        file = sys.stdout
    bar = "=" * _REPORT_WIDTH
    print(f"\n{bar}\n{title}\n{bar}", file=file)


def print_dataset_summary(dataset: Dataset, *, file: Any = None) -> None:
    """Print a few lines describing the loaded dataset."""
    if file is None:
        file = sys.stdout
    print(f"  Dataset:      {dataset.name} ({dataset.task})", file=file)
    print(
        f"  Train / test: {len(dataset.X_train)} / {len(dataset.X_test)}",
        file=file,
    )
    print(f"  Features:     {len(dataset.feature_names)}", file=file)
    if dataset.class_names is not None:
        print(
            f"  Classes:      {len(dataset.class_names)} "
            f"({', '.join(dataset.class_names)})",
            file=file,
        )


def print_classification_report(
    metrics: dict[str, Any], *, title: str = "Test set", file: Any = None
) -> None:
    """Format and print accuracy/total from ``evaluate_predictions``."""
    if file is None:
        file = sys.stdout
    print(
        f"\n{title}:\n"
        f"  Accuracy: {metrics['accuracy']:.2%} "
        f"({metrics['correct']}/{metrics['total']})",
        file=file,
    )


def print_regression_report(
    metrics: dict[str, Any], *, title: str = "Test set", file: Any = None
) -> None:
    """Format and print MSE/MAE/RMSE (and R^2 if present)."""
    if file is None:
        file = sys.stdout
    print(f"\n{title}:", file=file)
    print(f"  MSE:  {metrics['mse']:.4f}", file=file)
    print(f"  MAE:  {metrics['mae']:.4f}", file=file)
    print(f"  RMSE: {metrics['rmse']:.4f}", file=file)
    if "r2" in metrics:
        print(f"  R^2:  {metrics['r2']:.4f}", file=file)


def print_feature_importances(
    importances: dict[str, float],
    *,
    top: int = 10,
    title: str = "Top feature importances",
    file: Any = None,
) -> None:
    """Print the highest-importance features in descending order."""
    if file is None:
        file = sys.stdout
    print(f"\n{title}:", file=file)
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top]
    width = max((len(name) for name, _ in ranked), default=0)
    for name, score in ranked:
        print(f"  {name:<{width}}  {score:.4f}", file=file)


def build_example_parser(description: str) -> argparse.ArgumentParser:
    """
    Build an argparse parser with the flags every example shares.

    Flags:
        ``--random-state``: seed for splits and training (default 42).
        ``--test-fraction``: held-out fraction for evaluation (default 0.3).
        ``-o/--output``: optional path to export the trained model.
        ``-q/--quiet``: suppress narrative output; results still print.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed for the train/test split and trainer (default: %(default)s)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction held out for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Export the trained model to PATH (format inferred from suffix)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress headers and dataset summary; results still print",
    )
    return parser
