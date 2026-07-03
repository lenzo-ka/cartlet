"""
Runner throughput smoke test.

Runs the bundled Python runner on a tiny iris model and asserts that it
returns within a generous absolute throughput floor. The threshold is
intentionally permissive: a healthy runner is ~1000x above the floor, so
this only fires on an order-of-magnitude regression (or an outright hang).
"""

from __future__ import annotations

import pytest

# The throughput harness pulls in sklearn (public datasets); skip the whole
# module when it is unavailable so the no-optional-deps CI job can collect.
pytest.importorskip("sklearn")

from benchmarks.runner_throughput import (  # noqa: E402
    _SMOKE_BATCH,
    _bench_one,
    _public_datasets,
    _train_model,
)

# Pred/s floor below which we treat the result as a regression. Sized to
# clear typical CI variance (cold caches, contended runners) by ~5x while
# still catching a 10x slowdown in the actual hot path.
_PYTHON_FLOOR = 200


def test_python_runner_meets_throughput_floor(tmp_path):
    """Catches order-of-magnitude regressions in bundled Python runner throughput."""
    X_train, X_test, y_train, _, names, task = _public_datasets()["iris"]
    dt = _train_model(X_train, y_train, names, task, max_depth=4)

    model_path = str(tmp_path / "iris.cart")
    dt.export(model_path)

    results = _bench_one(
        dataset_name="iris",
        X_test=X_test,
        feature_names=names,
        max_depth=4,
        model_path=model_path,
        bundle_dir=str(tmp_path),
        batch=_SMOKE_BATCH,
    )

    assert results, "Python runner produced no result"
    rate = results[0].predictions_per_second
    assert rate >= _PYTHON_FLOOR, (
        f"Python runner ran at {rate:.0f} pred/s, below smoke floor {_PYTHON_FLOOR}"
    )
