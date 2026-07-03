"""
Regression on the diabetes dataset with a task-aware metric report.

This walks through the regression pipeline:

* ``DecisionTree(task="regression")`` trains an MSE-based tree.
* ``evaluate_tree`` autodetects regression and returns MSE/MAE/RMSE
  (rather than accuracy), so the same call works for both tasks.
* ``regression_metrics`` is shown with ``include_r2=True`` for callers
  who want R^2 alongside the error norms.

Run::

    python -m examples.diabetes_regression
    python -m examples.diabetes_regression --max-depth 8 --output /tmp/d.cart
"""

from __future__ import annotations

import sys

from cartlet import DecisionTree, evaluate_tree
from cartlet.evaluation import regression_metrics

from .common import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_FRACTION,
    build_example_parser,
    load_dataset,
    print_dataset_summary,
    print_header,
    print_regression_report,
)

DEFAULT_MAX_DEPTH = 5


def run(
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    output: str | None = None,
    quiet: bool = True,
) -> dict:
    """
    Train a regression tree on diabetes and return its metrics.

    The returned dict is the output of ``evaluate_tree`` (``task='regression'``,
    ``mse``, ``mae``, ``rmse``, ``total``) plus an ``r2`` key.
    """
    dataset = load_dataset(
        "diabetes", test_fraction=test_fraction, random_state=random_state
    )

    if not quiet:
        print_header(f"DecisionTree regression on diabetes (max_depth={max_depth})")
        print_dataset_summary(dataset)

    model = DecisionTree(
        features=dataset.feature_specs,
        task=dataset.task,
        max_depth=max_depth,
    )
    model.load_data(dataset.X_train, dataset.y_train)
    model.train(random_state=random_state)

    metrics = evaluate_tree(model, dataset.X_test, dataset.y_test)
    r2 = regression_metrics(
        dataset.y_test, model.predict_batch(dataset.X_test), include_r2=True
    )["r2"]
    metrics["r2"] = r2

    if not quiet:
        print_regression_report(metrics)
    if output:
        model.export(output)
        if not quiet:
            print(f"\nModel exported to {output}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_example_parser(__doc__ or "")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum tree depth (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    run(
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_fraction=args.test_fraction,
        output=args.output,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
