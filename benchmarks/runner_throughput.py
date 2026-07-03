"""
Steady-state throughput benchmark for the bundled cartlet Python runner.

Trains one model per public dataset (and a depth sweep on breast cancer),
exports to .cart, bundles for the Python runner, and times how long the
runner takes to process a large batch fed over stdin.

This measures steady-state cost only: interpreter / process startup is
amortised over thousands of predictions, so the dominant work is loading
the model once and running tree traversal in a tight loop.

Run with ``make bench`` or directly::

    python -m benchmarks.runner_throughput
    python -m benchmarks.runner_throughput --json results.json
    python -m benchmarks.runner_throughput --batch 50000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split

from cartlet import DecisionTree, bundle

_DEFAULT_BATCH = 10_000
_DEFAULT_DEPTH = 8
_DEPTH_SWEEP = (4, 8, 16)
_SMOKE_BATCH = 200


@dataclass
class Result:
    """One (dataset, depth) throughput measurement for the Python runner."""

    dataset: str
    n_features: int
    max_depth: int
    n_predictions: int
    wall_seconds: float
    predictions_per_second: float


# =============================================================================
# Dataset + model preparation
# =============================================================================


def _numeric_specs(names: list[str]) -> list[dict]:
    return [{"name": n, "dtype": "float", "type": "num"} for n in names]


def _public_datasets() -> dict[str, tuple]:
    """
    Return ``{name: (X_train, X_test, y_train, y_test, feature_names, task)}``
    for every public dataset we benchmark on.
    """
    datasets = {}
    for name, loader, task in (
        ("iris", load_iris, "classification"),
        ("wine", load_wine, "classification"),
        ("breast_cancer", load_breast_cancer, "classification"),
        ("diabetes", load_diabetes, "regression"),
    ):
        data = loader()
        X = data.data.tolist()
        y: list
        if task == "classification":
            target_names = list(data.target_names)
            y = [str(target_names[i]) for i in data.target]
        else:
            y = [float(v) for v in data.target]
        names = list(data.feature_names)
        splits = train_test_split(X, y, test_size=0.3, random_state=42)
        datasets[name] = (*splits, names, task)
    return datasets


def _train_model(
    X_train: list[list[float]],
    y_train: list,
    feature_names: list[str],
    task: str,
    max_depth: int,
) -> DecisionTree:
    dt = DecisionTree(
        features=_numeric_specs(feature_names),
        task=task,
        max_depth=max_depth,
    )
    dt.load_data(X_train, y_train)
    dt.train(random_state=42)
    return dt


# =============================================================================
# Runner orchestration
# =============================================================================


def _format_batch(X_batch: list[list[float]]) -> bytes:
    """Render a batch of feature vectors as the stdin format the runner expects."""
    return (
        "\n".join(" ".join(repr(v) for v in row) for row in X_batch) + "\n"
    ).encode()


def _time_runner(runner_path: str, stdin_bytes: bytes) -> float:
    """
    Wall-clock seconds for a single end-to-end invocation that consumes
    ``stdin_bytes`` and writes one prediction per line to stdout.
    """
    cmd = [sys.executable, runner_path, "-f", "-"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, input=stdin_bytes, capture_output=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"python runner failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    return elapsed


def _bench_one(
    dataset_name: str,
    X_test: list[list[float]],
    feature_names: list[str],
    max_depth: int,
    model_path: str,
    bundle_dir: str,
    batch: int,
) -> list[Result]:
    """Bundle the Python runner once and time it on ``batch`` predictions."""
    # Cycle X_test up to ``batch`` rows so the input is dataset-sized but
    # the timing batch is large enough to swamp interpreter startup.
    pool = X_test * ((batch // len(X_test)) + 1)
    stdin_bytes = _format_batch(pool[:batch])

    runner_path = os.path.join(bundle_dir, f"{dataset_name}_d{max_depth}.py")
    bundle(model_path, runner_path)
    wall = _time_runner(runner_path, stdin_bytes)
    return [
        Result(
            dataset=dataset_name,
            n_features=len(feature_names),
            max_depth=max_depth,
            n_predictions=batch,
            wall_seconds=wall,
            predictions_per_second=batch / wall if wall > 0 else float("inf"),
        )
    ]


# =============================================================================
# Public entry points
# =============================================================================


def run_benchmarks(
    batch: int = _DEFAULT_BATCH,
    include_depth_sweep: bool = True,
) -> list[Result]:
    """
    Train one model per public dataset at the default depth, plus a depth
    sweep on breast cancer when ``include_depth_sweep`` is True. Returns
    one Result per (dataset, depth).
    """
    datasets = _public_datasets()

    plans: list[tuple[str, int]] = [(name, _DEFAULT_DEPTH) for name in datasets]
    if include_depth_sweep:
        for depth in _DEPTH_SWEEP:
            if depth != _DEFAULT_DEPTH:
                plans.append(("breast_cancer", depth))

    results: list[Result] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, depth in plans:
            X_train, X_test, y_train, _, names, task = datasets[name]
            dt = _train_model(X_train, y_train, names, task, depth)
            model_path = os.path.join(tmpdir, f"{name}_d{depth}.cart")
            dt.export(model_path)
            results.extend(
                _bench_one(
                    dataset_name=name,
                    X_test=X_test,
                    feature_names=names,
                    max_depth=depth,
                    model_path=model_path,
                    bundle_dir=tmpdir,
                    batch=batch,
                )
            )
    return results


def render_markdown(results: list[Result]) -> str:
    """Pretty-print a results list as a markdown table."""
    if not results:
        return "_no results_\n"

    header = "| dataset | n_features | depth | n_predictions | pred/s |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r.dataset} | {r.n_features} | {r.max_depth} "
            f"| {r.n_predictions:,} | {r.predictions_per_second:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=int,
        default=_DEFAULT_BATCH,
        help=f"Predictions per runner invocation (default: {_DEFAULT_BATCH:,})",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Also write raw results as JSON to PATH",
    )
    parser.add_argument(
        "--no-depth-sweep",
        action="store_true",
        help="Skip the breast_cancer depth sweep",
    )
    args = parser.parse_args(argv)

    results = run_benchmarks(
        batch=args.batch,
        include_depth_sweep=not args.no_depth_sweep,
    )

    print(render_markdown(results))

    if results:
        rates = [r.predictions_per_second for r in results]
        print(
            f"# python: mean {statistics.mean(rates):,.0f} pred/s "
            f"across {len(rates)} configs"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"# raw results -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
