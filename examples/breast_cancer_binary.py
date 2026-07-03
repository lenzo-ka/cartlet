"""
Binary classification on the breast cancer dataset.

Beyond plain accuracy, this example exercises three calibrated-output
helpers cartlet exposes on classifiers:

* ``predict_with_confidence`` returns ``(label, confidence)`` for one row.
* ``predict_nbest`` returns the top-N classes by probability.
* ``predict_proba`` returns the full class -> probability dict.

We keep ``store_distributions=True`` and ``min_confidence=1.0`` on the
``DecisionTree`` so leaf distributions are retained, which is what the
three calls above read from.

Run::

    python -m examples.breast_cancer_binary
    python -m examples.breast_cancer_binary --random-state 7
"""

from __future__ import annotations

import sys

from cartlet import DecisionTree, RandomForest, evaluate_predictions

from .common import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_FRACTION,
    build_example_parser,
    load_dataset,
    print_classification_report,
    print_dataset_summary,
    print_header,
)

DEFAULT_N_ESTIMATORS = 10
SAMPLE_ROWS = 3


def run(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    output: str | None = None,
    quiet: bool = True,
) -> dict:
    """
    Train, score, and exercise the per-sample probability helpers.

    Returns a dict combining the ``evaluate_predictions`` metrics with two
    extra keys for tests to inspect:

    * ``forest_proba_sum`` -- sum of ``RandomForest.predict_proba`` for one
      row; must be ~1.0.
    * ``nbest_first_label`` -- top label from ``predict_nbest`` for the
      same row; used to sanity-check the helper returns valid classes.
    """
    dataset = load_dataset(
        "breast_cancer", test_fraction=test_fraction, random_state=random_state
    )

    if not quiet:
        print_header("Binary classification on breast_cancer")
        print_dataset_summary(dataset)

    tree = DecisionTree(
        features=dataset.feature_specs,
        task=dataset.task,
        store_distributions=True,
        min_confidence=1.0,
    )
    tree.load_data(dataset.X_train, dataset.y_train)
    tree.train(random_state=random_state)

    metrics = evaluate_predictions(dataset.y_test, tree.predict_batch(dataset.X_test))

    if not quiet:
        print_classification_report(metrics, title="DecisionTree test set")
        print("\nPer-row confidence and nbest (first few test rows):")
        for row in dataset.X_test[:SAMPLE_ROWS]:
            label, confidence = tree.predict_with_confidence(row)
            nbest = tree.predict_nbest(row, n=2)
            ranked = ", ".join(f"{lbl}={prob:.2%}" for lbl, prob in nbest)
            print(f"  {label:<10}  confidence={confidence:.2%}  nbest=[{ranked}]")

    forest = RandomForest(
        n_estimators=DEFAULT_N_ESTIMATORS,
        features=dataset.feature_specs,
        task=dataset.task,
    )
    forest.load_data(dataset.X_train, dataset.y_train)
    forest.train(random_state=random_state)

    proba = forest.predict_proba(dataset.X_test[0])
    proba_sum = sum(proba.values())
    forest_metrics = evaluate_predictions(
        dataset.y_test, forest.predict_batch(dataset.X_test)
    )

    if not quiet:
        print_classification_report(
            forest_metrics,
            title=f"RandomForest ({DEFAULT_N_ESTIMATORS} trees) test set",
        )
        ranked = ", ".join(f"{lbl}={prob:.2%}" for lbl, prob in proba.items())
        print(f"\nForest predict_proba on first row: [{ranked}] (sum={proba_sum:.4f})")

    if output:
        tree.export(output)
        if not quiet:
            print(f"\nDecisionTree exported to {output}")

    return {
        **metrics,
        "forest_accuracy": forest_metrics["accuracy"],
        "forest_proba_sum": proba_sum,
        "nbest_first_label": tree.predict_nbest(dataset.X_test[0], n=1)[0][0],
    }


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
