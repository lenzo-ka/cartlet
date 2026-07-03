"""
Wine classification with a RandomForest, including feature importances.

The wine dataset has 13 numerical features describing chemical analyses of
178 wines from three cultivars, so a small forest comfortably crosses 90%
accuracy and the per-feature importances are interpretable.

Run::

    python -m examples.wine_random_forest
    python -m examples.wine_random_forest --n-estimators 50 --random-state 7
"""

from __future__ import annotations

import sys

from cartlet import RandomForest, evaluate_predictions

from .common import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_FRACTION,
    build_example_parser,
    load_dataset,
    print_classification_report,
    print_dataset_summary,
    print_feature_importances,
    print_header,
)

DEFAULT_N_ESTIMATORS = 20


def run(
    *,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    output: str | None = None,
    quiet: bool = True,
) -> dict:
    """
    Train and evaluate a RandomForest on wine.

    Returns a dict combining the ``evaluate_predictions`` metrics with a
    ``feature_importances`` key (name -> importance, summing to 1.0).
    """
    dataset = load_dataset(
        "wine", test_fraction=test_fraction, random_state=random_state
    )

    if not quiet:
        print_header(f"RandomForest ({n_estimators} trees) on wine")
        print_dataset_summary(dataset)

    model = RandomForest(
        n_estimators=n_estimators,
        features=dataset.feature_specs,
        task=dataset.task,
    )
    model.load_data(dataset.X_train, dataset.y_train)
    model.train(random_state=random_state)

    metrics = evaluate_predictions(dataset.y_test, model.predict_batch(dataset.X_test))
    importances = model.feature_importances_

    if not quiet:
        print_classification_report(metrics)
        print_feature_importances(importances, top=5)
    if output:
        model.export(output)
        if not quiet:
            print(f"\nModel exported to {output}")

    return {**metrics, "feature_importances": importances}


def main(argv: list[str] | None = None) -> int:
    parser = build_example_parser(__doc__ or "")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees in the forest (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    run(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        test_fraction=args.test_fraction,
        output=args.output,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
