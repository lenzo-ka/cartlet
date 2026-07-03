"""
Runnable cartlet examples.

Each module in this package trains a model on a public sklearn dataset,
evaluates it on a held-out split, and prints a short report. Examples are
self-contained scripts intended both as user-facing tutorials and as
integration tests (see ``tests/test_examples.py``).

Run an example directly::

    python -m examples.iris_decision_tree
    python -m examples.wine_random_forest --random-state 7
    python -m examples.diabetes_regression --output /tmp/diabetes.cart

All examples share a small set of helpers in ``examples.common`` for dataset
loading, reporting, and argparse so the example bodies stay focused on the
cartlet calls they demonstrate.

Not part of the published ``cartlet`` package.
"""
