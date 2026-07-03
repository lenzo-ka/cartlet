"""
Iris classification with a native CART decision tree.

This is the simplest possible cartlet workflow: build a ``DecisionTree``,
load training data, call ``train()``, then predict and score on a held-out
test split.

Run::

    python -m examples.iris_decision_tree
    python -m examples.iris_decision_tree --random-state 7 -o /tmp/iris.cart
"""

from __future__ import annotations

import sys

from cartlet import DecisionTree, evaluate_predictions

from .common import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_FRACTION,
    build_example_parser,
    load_dataset,
    print_classification_report,
    print_dataset_summary,
    print_header,
)


def run(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    output: str | None = None,
    quiet: bool = True,
) -> dict:
    """
    Train and evaluate a DecisionTree on iris.

    Returns the metrics dict from ``evaluate_predictions`` (``accuracy``,
    ``correct``, ``total``).
    """
    dataset = load_dataset(
        "iris", test_fraction=test_fraction, random_state=random_state
    )

    if not quiet:
        print_header("DecisionTree on iris")
        print_dataset_summary(dataset)

    model = DecisionTree(features=dataset.feature_specs, task=dataset.task)
    model.load_data(dataset.X_train, dataset.y_train)
    model.train(random_state=random_state)

    metrics = evaluate_predictions(dataset.y_test, model.predict_batch(dataset.X_test))

    if not quiet:
        print_classification_report(metrics)
    if output:
        model.export(output)
        if not quiet:
            print(f"\nModel exported to {output}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_example_parser(__doc__ or "").parse_args(argv)
    run(
        random_state=args.random_state,
        test_fraction=args.test_fraction,
        output=args.output,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
