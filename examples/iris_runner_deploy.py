"""
Train, export to ``.cart``, then reload via the zero-dependency Predictor.

This is the deployment pattern: training uses the full cartlet package
(which depends on sklearn/numpy), but the resulting ``.cart`` file can be
served from a process that imports only ``cartlet.runner.Predictor`` --
which has no dependencies outside the standard library.

The example also embeds a JSON ``metadata`` blob into the .cart trailer
and reads it back via ``Predictor.metadata`` so users can see how to ship
config (e.g. locale, model version) alongside the model itself.

Run::

    python -m examples.iris_runner_deploy
    python -m examples.iris_runner_deploy -o /tmp/iris.cart
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from cartlet import DecisionTree, Predictor, evaluate_predictions

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
    Train, export, reload via Predictor, and verify predictions match.

    Returns a dict with:

    * ``accuracy`` -- in-memory test accuracy.
    * ``agreement`` -- fraction of test rows where the in-memory model
      and the reloaded ``Predictor`` produce identical labels. Should be
      ~1.0 (a small number of boundary rows may flip because ``.cart``
      stores thresholds as float32).
    * ``metadata`` -- the trailer dict read back from the exported file.
    """
    dataset = load_dataset(
        "iris", test_fraction=test_fraction, random_state=random_state
    )

    if not quiet:
        print_header("Iris: train -> export -> Predictor")
        print_dataset_summary(dataset)

    model = DecisionTree(features=dataset.feature_specs, task=dataset.task)
    model.load_data(dataset.X_train, dataset.y_train)
    model.train(random_state=random_state)

    in_memory_preds = model.predict_batch(dataset.X_test)
    metrics = evaluate_predictions(dataset.y_test, in_memory_preds)

    metadata = {
        "model": "iris-decision-tree",
        "version": "1.0.0",
        "random_state": random_state,
    }

    if output is None:
        tmpdir = tempfile.mkdtemp(prefix="cartlet_example_")
        cart_path = str(Path(tmpdir) / "iris.cart")
    else:
        cart_path = output

    model.export(cart_path, metadata=metadata)

    predictor = Predictor(cart_path)
    from_disk_preds = [predictor.predict(row) for row in dataset.X_test]

    n = len(in_memory_preds)
    agreements = sum(a == b for a, b in zip(in_memory_preds, from_disk_preds))
    agreement = agreements / n if n else 1.0

    if not quiet:
        print_classification_report(metrics, title="In-memory test set")
        print(f"\nPredictor reload agreement: {agreements}/{n} ({agreement:.2%})")
        print(f"Predictor.metadata:        {predictor.metadata}")
        print(f"Exported model:            {cart_path}")

    return {
        "accuracy": metrics["accuracy"],
        "agreement": agreement,
        "metadata": predictor.metadata,
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
