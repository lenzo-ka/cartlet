"""Generated-data scalability measurements; run with ``python -m benchmarks.scalability``."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cartlet
from cartlet import Predictor, TrainingSettings, train_model
from cartlet.io import load_training_data, read_vectors

CASES = ("numeric", "categorical", "forest")
STAGES = ("read_vectors", "load_training_data", "train", "export", "load", "predict")


@dataclass(frozen=True)
class BenchmarkSettings:
    """Workload controls shared by the Python API and developer CLI."""

    rows: int = 500
    features: int = 6
    categories: int = 128
    trees: int = 16
    depth: int = 6
    predictions: int = 2000
    repeats: int = 3
    seed: int = 42
    backend: str = "native"
    data_format: str = "tsv"

    def __post_init__(self) -> None:
        for name in (
            "rows",
            "features",
            "categories",
            "trees",
            "depth",
            "predictions",
            "repeats",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.backend not in ("native", "sklearn"):
            raise ValueError("backend must be native or sklearn")
        if self.data_format not in ("tsv", "jsonl"):
            raise ValueError("data_format must be tsv or jsonl")


def generate_dataset(case: str, settings: BenchmarkSettings, path: Path) -> str:
    """Write deterministic labeled TSV or JSONL, returning its content SHA-256.

    Numeric/forest labels combine two numeric signals. Categorical values are
    sampled from a bounded vocabulary; labels depend on the first category.
    These are scalability workloads, not predictive-quality benchmarks.
    """
    if case not in CASES:
        raise ValueError(f"unknown case: {case}")
    rng = random.Random(settings.seed)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        names = [f"x{i}" for i in range(settings.features)] + ["label"]
        if settings.data_format == "tsv":
            writer.writerow(names)
        for _ in range(settings.rows):
            if case == "categorical":
                values = [
                    rng.randrange(settings.categories) for _ in range(settings.features)
                ]
                row = [f"category_{value:06d}" for value in values]
                label = str(values[0] % 3)
            else:
                numeric_values = [rng.random() for _ in range(settings.features)]
                row = [format(value, ".12g") for value in numeric_values]
                label = str(int(numeric_values[0] + numeric_values[-1] > 1))
            if settings.data_format == "tsv":
                writer.writerow(row + [label])
            else:
                stream.write(
                    json.dumps(dict(zip(names, row + [label], strict=True))) + "\n"
                )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(operation: Callable[[], Any], repeats: int) -> dict[str, Any]:
    """Time warmed calls, then separately trace one call's Python allocations.

    Results stay alive until the end of each measurement. Memory is additional
    traced allocation, not process RSS; preexisting inputs and native backend
    buffers are excluded. Timing runs never have tracing enabled.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if tracemalloc.is_tracing():
        raise RuntimeError("stop existing tracemalloc tracing before benchmarking")
    result = operation()  # Warm caches and imports outside every measurement.
    del result
    seconds = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = operation()
        seconds.append(time.perf_counter() - start)
        del result
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
    return {
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "peak_python_bytes": peak,
    }


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if Path(cartlet.__file__).resolve().parent != root / "cartlet":
        raise RuntimeError(
            "benchmark must import Cartlet from the same source checkout"
        )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    versions = {"cartlet": cartlet.__version__, "python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "joblib"):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    source_digest = hashlib.sha256()
    sources = sorted((root / "cartlet").rglob("*.py")) + [Path(__file__).resolve()]
    for path in sources:
        source_digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        source_digest.update(path.read_bytes() + b"\0")
    return {
        "source_sha256": source_digest.hexdigest(),
        "revision": revision,
        "dirty": dirty,
        "versions": versions,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def run_benchmarks(
    settings: BenchmarkSettings | None = None,
    *,
    cases: Sequence[str] = CASES,
    stages: Sequence[str] = STAGES,
) -> dict[str, Any]:
    """Return a JSON-compatible report without printing or retaining artifacts.

    Select only loader stages for large datasets without paying training cost.
    Stage prerequisites run outside the stage measurement.
    """
    settings = settings or BenchmarkSettings()
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(CASES):
        raise ValueError("cases must be nonempty, unique supported case names")
    if not stages or len(set(stages)) != len(stages) or set(stages) - set(STAGES):
        raise ValueError("stages must be nonempty, unique supported stage names")
    report: dict[str, Any] = {
        "schema_version": 1,
        "generator_version": 1,
        "measurement": {
            "timing": "perf_counter, warmed, untraced, GC collected before each call",
            "memory": "one separate tracemalloc run, additional Python allocation peak, bytes; excludes existing inputs and native buffers",
            "model_format": "cart",
            "warmups": 1,
        },
        "provenance": _provenance(),
        "settings": asdict(settings),
        "results": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        for case in cases:
            data_path = Path(directory) / f"data.{settings.data_format}"
            model_path = Path(directory) / "model.cart"
            digest = generate_dataset(case, settings, data_path)
            parity_rows = 0
            operations: dict[str, Callable[[], Any]] = {
                "read_vectors": partial(read_vectors, str(data_path)),
                "load_training_data": partial(load_training_data, str(data_path)),
            }
            if set(stages) - set(operations):
                X, y, names, _ = load_training_data(str(data_path))
                specs = [
                    {
                        "name": name,
                        "dtype": "str" if case == "categorical" else "float",
                        "type": "cat" if case == "categorical" else "num",
                    }
                    for name in names
                ]
                config = TrainingSettings(
                    model_type="forest" if case == "forest" else "tree",
                    task="classification",
                    trainer=settings.backend,
                    n_estimators=settings.trees,
                    max_depth=settings.depth,
                    random_state=settings.seed,
                    validation_split=0,
                    test_split=0,
                    n_jobs=1
                    if settings.backend == "sklearn" and case == "forest"
                    else None,
                )
                operations["train"] = partial(
                    train_model, X, y, settings=config, feature_specs=specs
                )
                if set(stages) & {"export", "load", "predict"}:
                    model = operations["train"]().model
                    operations["export"] = partial(model.export, str(model_path))
                    operations["export"]()
                    operations["load"] = partial(Predictor, str(model_path))
                    predictor = operations["load"]()
                    parity_rows = min(100, len(X))
                    expected = [model.predict(row) for row in X[:parity_rows]]
                    if predictor.predict_batch(X[:parity_rows]) != expected:
                        raise RuntimeError(
                            f"{case}: exported predictions differ from training model"
                        )
                    batch = [X[i % len(X)] for i in range(settings.predictions)]
                    operations["predict"] = partial(predictor.predict_batch, batch)
            for stage in stages:
                metrics = measure(operations[stage], settings.repeats)
                report["results"].append(
                    {
                        "case": case,
                        "stage": stage,
                        "dataset_sha256": digest,
                        "dataset_bytes": data_path.stat().st_size,
                        "parity_rows": parity_rows,
                        "model_bytes": model_path.stat().st_size
                        if stage in {"export", "load", "predict"}
                        else None,
                        **metrics,
                        "predictions_per_second": settings.predictions
                        / metrics["median_seconds"]
                        if stage == "predict"
                        else None,
                    }
                )
            model_path.unlink(missing_ok=True)
    return report


def render_tsv(report: dict[str, Any]) -> str:
    """Render one sortable row per stage, with plain numeric values and units."""
    stream = io.StringIO(newline="")
    columns = [
        "case",
        "stage",
        "median_seconds",
        "peak_python_bytes",
        "predictions_per_second",
        "dataset_bytes",
        "model_bytes",
        "parity_rows",
        "rows",
        "features",
        "categories",
        "trees",
        "depth",
        "predictions",
        "repeats",
        "seed",
        "backend",
        "dataset_sha256",
        "data_format",
        "revision",
        "dirty",
        "source_sha256",
    ]
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for result in report["results"]:
        writer.writerow({**report["settings"], **report["provenance"], **result})
    return stream.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    """Thin CLI: settings in, report out; TSV is the default."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BenchmarkSettings()
    for name in (
        "rows",
        "features",
        "categories",
        "trees",
        "depth",
        "predictions",
        "repeats",
        "seed",
    ):
        parser.add_argument(
            f"--{name}",
            type=int,
            default=getattr(defaults, name),
            help=f"{name.capitalize()} (default: %(default)s)",
        )
    parser.add_argument(
        "--data-format",
        choices=("tsv", "jsonl"),
        default="tsv",
        help="Generated fixture format (default: tsv)",
    )
    parser.add_argument(
        "--backend",
        choices=("native", "sklearn"),
        default="native",
        help="Training backend; sklearn requires its optional extra",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=CASES,
        help="Generated workload cases",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=STAGES,
        help="Stages to measure; loader-only runs skip training",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "json"),
        default="tsv",
        help="Output format (default: tsv); JSON includes provenance and timing samples",
    )
    args = vars(parser.parse_args(argv))
    cases, stages, output_format = (
        args.pop("cases"),
        args.pop("stages"),
        args.pop("format"),
    )
    try:
        report = run_benchmarks(BenchmarkSettings(**args), cases=cases, stages=stages)
    except (ValueError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(
        json.dumps(report, indent=2, allow_nan=False)
        if output_format == "json"
        else render_tsv(report),
        end="\n" if output_format == "json" else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
