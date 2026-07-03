#!/usr/bin/env python
"""
Command-line interface for cartlet.

Usage:
    cartlet train data.csv -o model.cart
    cartlet train data.tsv --target class --no-header
    cartlet train data.jsonl --target label
    cartlet predict model.cart data.csv
    cartlet evaluate model.cart test.csv
    cartlet stats model.cart
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, TextIO

# Optional YAML support
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

from . import __version__, _detect_is_forest, convert
from .evaluation import evaluate_predictions, per_class_metrics, regression_metrics
from .forest import RandomForest
from .io import detect_delimiter, detect_format, load_training_data, resolve_format
from .io.bytes import bundle
from .isolation import (
    ANOMALY_SCORE_MIDPOINT,
    DEFAULT_MAX_SAMPLES,
    IsolationForest,
)
from .runner import load_model, predict_batch
from .tree import DecisionTree
from .types import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_SPLIT,
    TASK_AUTO,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    infer_feature_specs,
    is_likely_regression,
    normalize_feature_spec,
)
from .utils import max_depth, tree_stats

_MAX_CAT_VALUES_DISPLAY = 10
_MAX_TARGET_VALUES_DISPLAY = 20
_ISO8601_PREFIX_LEN = 19  # "YYYY-MM-DDTHH:MM:SS"


def _require_yaml() -> None:
    """Raise a uniform ImportError when PyYAML is needed but missing."""
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML required for .yaml config files: pip install pyyaml")


def _parse_column_names(args: argparse.Namespace) -> list[str] | None:
    """Parse comma-separated column names from CLI args, or None if absent."""
    raw = getattr(args, "column_names", None)
    if raw:
        return [n.strip() for n in raw.split(",")]
    return None


def _add_tabular_input_args(
    parser: argparse.ArgumentParser,
    *,
    target_help: str = "Target column name or index (default: last)",
    delimiter_help: str = "Column delimiter (auto-detect if not specified)",
    column_names: bool = True,
) -> None:
    """
    Add the common `--target / --delimiter / --no-header / --column-names`
    options to a subparser that consumes tabular data.

    Centralizes both flag wiring and help text so all subcommands stay in
    lockstep when shortcuts or wording change.
    """
    parser.add_argument("-t", "--target", metavar="COL", help=target_help)
    parser.add_argument("-d", "--delimiter", metavar="CHAR", help=delimiter_help)
    parser.add_argument(
        "-H", "--no-header", action="store_true", help="Data has no header row"
    )
    if column_names:
        parser.add_argument(
            "-N",
            "--column-names",
            metavar="NAMES",
            help="Column names when no header (comma-separated, e.g., 'age,color,size,label')",
        )


@contextmanager
def _output_file(path: str | None) -> Iterator[TextIO]:
    """Context manager for output that defaults to stdout."""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            yield f
    else:
        yield sys.stdout


# Args to exclude from config (positional or meta)
_CONFIG_EXCLUDE = {
    "func",
    "command",
    "config",
    "save_config",
    "config_file_used",
    "config_preset_used",
}

# Built-in config presets
_BUILTIN_CONFIGS: dict[str, dict[str, Any]] = {
    "defaults": {
        # All default values - use as a template
        "task": TASK_AUTO,
        "criterion": "entropy",
        "forest": False,
        "extra_trees": False,
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "max_depth": None,  # unlimited
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "trainer": "native",
        "n_jobs": None,
        "random_seed": None,
        "prune": False,
        "validation_split": DEFAULT_VALIDATION_SPLIT,
        "test_split": DEFAULT_TEST_SPLIT,
    },
    "fast": {
        # Quick training for testing/iteration
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    },
    "accurate": {
        # More thorough training for production
        "forest": True,
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "small": {
        # Smaller model size
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    },
    "forest": {
        # Default forest settings
        "forest": True,
        "n_estimators": 50,
    },
    "forest-large": {
        # Large forest for best accuracy
        "forest": True,
        "n_estimators": 200,
        "trainer": "sklearn",
        "n_jobs": -1,
    },
    "sklearn": {
        # Use sklearn backend
        "trainer": "sklearn",
    },
    "extra-trees": {
        # Extra-Trees: random splits, no bootstrap
        "forest": True,
        "extra_trees": True,
        "n_estimators": 100,
    },
    "g2p": {
        # Settings tuned for grapheme-to-phoneme: many small classes with rare
        # but important alignments. Entropy splits + fine-grained leaves +
        # distributions kept (the default) so `predict_nbest` is usable.
        "trainer": "sklearn",
        "criterion": "entropy",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
}


def load_config(path_or_name: str) -> dict[str, Any]:
    """
    Load config from built-in preset, YAML, or JSON file.

    Args:
        path_or_name: Built-in preset name or path to config file

    Returns:
        Dict of config values
    """
    # Check for built-in preset first
    if path_or_name in _BUILTIN_CONFIGS:
        return _BUILTIN_CONFIGS[path_or_name].copy()

    # Not a preset: it must be a config file. If the name doesn't look like a
    # config file and doesn't exist, it is almost certainly a mistyped preset
    # -- say so instead of leaking a bare FileNotFoundError on the name.
    if not os.path.exists(path_or_name) and not path_or_name.endswith(
        (".yaml", ".yml", ".json")
    ):
        presets = ", ".join(sorted(_BUILTIN_CONFIGS))
        raise ValueError(
            f"Unknown config preset {path_or_name!r}. Valid presets: {presets}. "
            "(Or pass a path to a .yaml/.yml/.json config file.)"
        )

    # Load from file
    with open(path_or_name, encoding="utf-8") as f:
        content = f.read()

    if path_or_name.endswith((".yaml", ".yml")):
        _require_yaml()
        return yaml.safe_load(content) or {}
    return json.loads(content)


def save_config(args: argparse.Namespace, path: str) -> None:
    """
    Save args to config file (YAML or JSON).

    Args:
        args: Parsed arguments
        path: Output path (.yaml, .yml, or .json)
    """
    # Convert args to dict, excluding meta keys
    config = {
        k: v
        for k, v in vars(args).items()
        if k not in _CONFIG_EXCLUDE and v is not None
    }

    # Handle special cases
    if "no_header" in config and not config["no_header"]:
        del config["no_header"]  # Only save if True

    # Check the optional YAML dependency before opening the file, so a missing
    # PyYAML doesn't leave a truncated 0-byte config behind.
    is_yaml = path.endswith((".yaml", ".yml"))
    if is_yaml:
        _require_yaml()

    with open(path, "w", encoding="utf-8") as f:
        if is_yaml:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, indent=2)
            f.write("\n")

    print(f"Config saved to {path}", file=sys.stderr)


def merge_config_with_argv(config: dict[str, Any], argv: list[str]) -> list[str]:
    """
    Merge config values into argv. CLI args take precedence.

    Args:
        config: Config dict
        argv: Original command line args

    Returns:
        New argv with config values inserted as defaults
    """
    # Build set of args already in argv (to avoid overriding)
    provided = set()
    for arg in argv:
        if arg.startswith("-"):
            # Strip leading dashes and get the dest name
            clean = arg.lstrip("-").replace("-", "_")
            provided.add(clean)

    # Build new argv with config values prepended (CLI will override)
    new_args = []
    for key, value in config.items():
        if key in provided or key in _CONFIG_EXCLUDE:
            continue
        if key == "data":
            continue  # Positional, handled separately

        # Convert to CLI arg format
        cli_key = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                new_args.append(cli_key)
        elif value is not None:
            new_args.append(cli_key)
            new_args.append(str(value))

    return new_args + argv


def load_feature_specs(spec_input: str, feature_names: list[str]) -> list[dict]:
    """
    Load feature specifications from JSON file or inline JSON.

    Args:
        spec_input: Path to JSON file or inline JSON string
        feature_names: Feature names from data (for validation)

    Returns:
        List of feature spec dicts

    JSON format (array or object):
        Array: [{"name": "age", "dtype": "int", "type": "num"}, ...]
        Object: {"age": {"dtype": "int", "type": "num"}, ...}
        Simple: {"age": "num", "color": "cat"}  # Just type
    """
    # Try to load from file first
    if os.path.exists(spec_input):
        with open(spec_input, encoding="utf-8") as f:
            specs = json.load(f)
    else:
        # Try parsing as inline JSON
        try:
            specs = json.loads(spec_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid feature specs JSON: {e}") from e

    # Normalize different formats
    if isinstance(specs, list):
        # Already in list format
        return specs

    if isinstance(specs, dict):
        result = []
        for name in feature_names:
            if name not in specs:
                # Default to categorical string when caller didn't specify
                result.append({"name": name, "dtype": "str", "type": "cat"})
                continue
            fs = normalize_feature_spec(specs[name], name=name)
            result.append(
                {
                    "name": fs.name,
                    "dtype": fs.dtype,
                    "type": fs.type or "cat",
                }
            )
        return result

    raise ValueError(f"Feature specs must be a list or object, got: {type(specs)}")


def _cmd_train_isolation_forest(
    args: argparse.Namespace,
    X: list[list[Any]],
    feature_names: list[str],
) -> int:
    """Train an IsolationForest (called from cmd_train)."""
    max_samples = min(DEFAULT_MAX_SAMPLES, len(X))
    print(
        f"Training IsolationForest with {args.n_estimators} trees, "
        f"max_samples={max_samples}...",
        file=sys.stderr,
    )

    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=max_samples,
        max_depth=args.max_depth,
        feature_names=feature_names,
        random_state=args.random_seed,
        verbose=args.verbose,
    )
    model.load_data(X)
    model.train()
    print(f"  Trees: {len(model.trees)}", file=sys.stderr)

    scores = model.predict_batch(X)
    n_anomalies = sum(1 for s in scores if s > ANOMALY_SCORE_MIDPOINT)
    print(
        f"  Anomaly scores: min={min(scores):.3f}, max={max(scores):.3f}, "
        f"mean={sum(scores) / len(scores):.3f}, "
        f"anomalies(>{ANOMALY_SCORE_MIDPOINT})={n_anomalies}/{len(X)}",
        file=sys.stderr,
    )

    if args.output:
        model.export(args.output)
        print(f"Model saved to {args.output}", file=sys.stderr)

    return 0


def _create_model(
    args: argparse.Namespace,
    feature_specs: list[dict],
    target_spec: dict,
    task: str,
) -> RandomForest | DecisionTree:
    """Create and return the appropriate model from CLI args."""
    extra_trees = getattr(args, "extra_trees", False)
    if extra_trees:
        args.forest = True

    common_kwargs: dict[str, Any] = {
        "features": feature_specs,
        "target": target_spec,
        "task": task,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "criterion": getattr(args, "criterion", "entropy"),
        "verbose": args.verbose,
    }

    if args.forest:
        kind = "ExtraTrees" if extra_trees else "RandomForest"
        print(f"Training {kind} with {args.n_estimators} trees...")
        return RandomForest(
            n_estimators=args.n_estimators,
            extra_trees=extra_trees,
            **common_kwargs,
        )

    print("Training DecisionTree...")
    return DecisionTree(**common_kwargs)


def _split_train_test(
    args: argparse.Namespace,
    X: list[list[Any]],
    y: list[Any],
    task: str,
    column_names: list[str] | None,
) -> tuple[list[list[Any]], list[Any], list[list[Any]], list[Any]]:
    """Split data into train and test sets based on CLI args."""
    if args.test_file:
        X_test, y_test, _, _ = load_training_data(
            args.test_file,
            delimiter=args.delimiter,
            has_header=not args.no_header,
            target_col=args.target,
            column_names=column_names,
        )
        if task == TASK_REGRESSION:
            y_test = [float(v) for v in y_test]
        print(f"  Test file: {args.test_file} ({len(X_test)} samples)")
        return X, y, X_test, y_test
    if args.test_split > 0:
        # Shuffle before splitting so a class-sorted file (e.g. iris) does not
        # yield a single-class test set -- and drop that class from training
        # entirely. Seeded by --random-seed for reproducibility.
        indices = list(range(len(X)))
        random.Random(args.random_seed).shuffle(indices)
        split_idx = int(len(X) * (1 - args.test_split))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        print(f"  Train/test split: {split_idx}/{len(X) - split_idx}")
        return (
            [X[i] for i in train_idx],
            [y[i] for i in train_idx],
            [X[i] for i in test_idx],
            [y[i] for i in test_idx],
        )
    return X, y, [], []


def _build_training_meta(
    args: argparse.Namespace,
    task: str,
    n_features: int,
    n_train: int,
) -> dict[str, Any]:
    """Build the training metadata dict for export."""
    config = {
        "forest": args.forest,
        "extra_trees": getattr(args, "extra_trees", False) or None,
        "n_estimators": args.n_estimators if args.forest else None,
        "criterion": getattr(args, "criterion", "entropy"),
        "task": task,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "trainer": args.trainer,
        "n_jobs": getattr(args, "n_jobs", None),
        "random_seed": args.random_seed,
        "prune": args.prune if not args.forest else None,
        "validation_split": args.validation_split
        if args.validation_split > 0
        else None,
        "test_split": args.test_split if args.test_split > 0 else None,
        "test_file": args.test_file,
        "config_file": getattr(args, "config_file_used", None),
        "config_preset": getattr(args, "config_preset_used", None),
    }
    return {
        "trained_at": datetime.now().isoformat(),
        "data_source": os.path.basename(args.data),
        "samples": n_train,
        "features": n_features,
        "config": {k: v for k, v in config.items() if v is not None},
    }


def _evaluate_test(
    model: RandomForest | DecisionTree,
    X_test: list[list[Any]],
    y_test: list[Any],
    task: str,
) -> dict[str, Any] | None:
    """Evaluate model on test data and print results. Returns metrics dict or None."""
    if not X_test:
        return None
    y_pred = model.predict_batch(X_test)
    if task == TASK_CLASSIFICATION:
        metrics = evaluate_predictions(y_test, y_pred)
        print(
            f"  Test accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})"
        )
        return {"accuracy": metrics["accuracy"], "samples": len(X_test)}
    metrics = regression_metrics(y_test, y_pred)
    print(f"  Test MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "samples": len(X_test),
    }


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model."""
    if args.save_config:
        save_config(args, args.save_config)

    print(f"Loading data from {args.data}...", file=sys.stderr)

    column_names = _parse_column_names(args)

    X, y, feature_names, target_name = load_training_data(
        args.data,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        target_col=args.target,
        column_names=column_names,
    )
    print(f"  Loaded {len(X)} samples, {len(feature_names)} features", file=sys.stderr)

    if getattr(args, "isolation_forest", False):
        return _cmd_train_isolation_forest(args, X, feature_names)

    if getattr(args, "validation_file", None):
        print(
            "Error: --validation-file is not yet implemented; "
            "use --validation-split to set a fraction instead.",
            file=sys.stderr,
        )
        return 2

    # Load or infer feature specs
    if args.features:
        feature_specs = load_feature_specs(args.features, feature_names)
        print("  Using provided feature specs", file=sys.stderr)
    else:
        feature_specs = infer_feature_specs(X, feature_names)

    # Determine task and target spec
    task = args.task
    if task == TASK_AUTO:
        task = TASK_REGRESSION if is_likely_regression(y) else TASK_CLASSIFICATION

    if task == TASK_REGRESSION:
        target_spec = {"name": target_name, "dtype": "float", "type": "num"}
    else:
        all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in y)
        target_dtype = "int" if all_int else "str"
        target_spec = {"name": target_name, "dtype": target_dtype, "type": "cat"}

    print(f"  Task: {task}", file=sys.stderr)

    model = _create_model(args, feature_specs, target_spec, task)

    # Split data
    X_train, y_train, X_test, y_test = _split_train_test(args, X, y, task, column_names)

    # Validation split logic. When test rows come from --test-split, the
    # training set is only (1 - test_split) of the input, so scale up the
    # validation fraction so it stays the intended fraction of the original.
    # When test rows come from --test-file, X_train is the full input, so no
    # adjustment is needed.
    validation_split = args.validation_split
    if args.prune and validation_split <= 0 and not args.forest:
        validation_split = DEFAULT_VALIDATION_SPLIT

    if args.test_split > 0 and validation_split > 0 and not args.test_file:
        adjusted_validation_split = validation_split / (1 - args.test_split)
    else:
        adjusted_validation_split = validation_split

    # Train
    model.load_data(X_train, y_train)
    if args.forest:
        if not isinstance(model, RandomForest):
            raise TypeError("Expected RandomForest model for --forest flag")
        if args.n_jobs and args.trainer != "sklearn":
            print(
                "  Note: --n-jobs ignored for native trainer (use -B sklearn for parallelism)",
                file=sys.stderr,
            )
        model.train(
            trainer=args.trainer,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    else:
        if not isinstance(model, DecisionTree):
            raise TypeError("Expected DecisionTree model")
        model.train(
            validation_split=adjusted_validation_split,
            prune=args.prune,
            random_state=args.random_seed,
            trainer=args.trainer,
        )
        if args.prune:
            actual_train_frac = 1.0 - args.test_split - validation_split
            print(
                f"  Splits: train={actual_train_frac:.0%}, "
                f"val={validation_split:.0%}, test={args.test_split:.0%}"
            )

    # Metadata
    training_meta = _build_training_meta(args, task, len(feature_names), len(X_train))

    # Stats
    if args.forest:
        if not isinstance(model, RandomForest):
            raise TypeError("Expected RandomForest model for --forest flag")
        trees = model.trees
        avg_depth = sum(max_depth(t.model) for t in trees) / len(trees)
        print(f"  Trees: {len(trees)}, Avg depth: {avg_depth:.1f}")
        training_meta["stats"] = {
            "n_trees": len(trees),
            "avg_depth": round(avg_depth, 1),
        }
    else:
        if not isinstance(model, DecisionTree):
            raise TypeError("Expected DecisionTree model")
        stats = tree_stats(model.model)
        print(
            f"  Nodes: {stats['total_nodes']} ({stats['leaf_nodes']} leaves), Depth: {stats['max_depth']}"
        )
        training_meta["stats"] = stats

    # Evaluate
    test_metrics = _evaluate_test(model, X_test, y_test, task)
    if test_metrics:
        training_meta["test"] = test_metrics

    # Export
    if args.output:
        store_dist = not getattr(args, "no_distributions", False)
        model.export(
            args.output,
            metadata={"training": training_meta},
            store_distributions=store_dist,
        )
        print(f"Model saved to {args.output}")

    return 0


def load_raw_data(
    path: str,
    delimiter: str | None = None,
    has_header: bool = True,
) -> tuple[list[Any], list[str] | None, str]:
    """
    Load raw data from CSV/TSV/JSONL file without processing.

    Returns:
        Tuple of (rows, header, format) where:
        - For CSV/TSV: rows is list[list[str]], header is column names
        - For JSONL: rows is list[Dict], header is list of keys from first record
        - format is "csv", "tsv", or "jsonl"
    """
    file_format = detect_format(path)

    if file_format == "jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"Empty file: {path}")
        header = list(records[0].keys())
        return records, header, "jsonl"

    if delimiter is None:
        delimiter = detect_delimiter(path)

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {path}")

    if has_header:
        return rows[1:], rows[0], file_format
    return rows, None, file_format


def _build_feature_vectors(
    raw_rows: list[Any],
    header: list[str] | None,
    model_feature_names: list[str | None],
    model_features: list[Any],
    is_jsonl: bool,
) -> list[list[Any]]:
    """Build feature vectors from raw data rows for prediction."""
    if is_jsonl:
        return [[row.get(name) for name in model_feature_names] for row in raw_rows]

    if header:
        feature_indices: list[int | None] = []
        for name in model_feature_names:
            if name in header:
                feature_indices.append(header.index(name))
            else:
                try:
                    feature_indices.append(int(name))
                except ValueError:
                    feature_indices.append(None)
    else:
        feature_indices = list(range(len(model_features)))

    vectors = []
    for row in raw_rows:
        features = []
        for idx in feature_indices:
            if idx is not None and idx < len(row):
                val = row[idx]
                if isinstance(val, str):
                    with contextlib.suppress(ValueError):
                        val = float(val) if "." in val else int(val)
                features.append(val)
            else:
                features.append(None)
        vectors.append(features)
    return vectors


def _write_predictions(
    output_data: list[dict[str, Any]] | list[str],
    output_format: str,
    output_path: str | None,
) -> None:
    """Write prediction output to file or stdout."""

    def _write(f, data, fmt):
        if fmt == "json":
            json.dump(data, f, indent=2)
            f.write("\n")
        elif fmt == "jsonl":
            for record in data:
                f.write(json.dumps(record) + "\n")
        else:
            f.write("\n".join(data) + "\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            _write(f, output_data, output_format)
        print(f"Output saved to {output_path}", file=sys.stderr)
    elif output_format == "json":
        print(json.dumps(output_data, indent=2))
    elif output_format == "jsonl":
        for record in output_data:
            print(json.dumps(record))
    else:
        for line in output_data:
            print(line)


def cmd_predict(args: argparse.Namespace) -> int:
    """
    Make predictions using a trained model.

    Output is shaped by `--mode` (default `values`):

      - `values`: emit just the predicted values, one per input row. With
        --output-format json/jsonl, each prediction is wrapped in a record
        keyed by `--prediction-column` (default "prediction").
      - `append`: re-emit every input column followed by the prediction in a
        new column named `--prediction-column`. Header row is preserved for
        CSV/TSV inputs; JSON/JSONL outputs gain the prediction as a new key.
      - `inplace`: replace the target column with the prediction. The target
        column is taken from `--target` (name or index), falling back to the
        last column / last JSON key.

    Format detection: `--output-format` overrides everything; otherwise the
    output file extension is used; otherwise the input format is reused.
    `--output-delimiter` overrides the format's default delimiter (CSV=`,`,
    TSV=`\\t`, SSV=` `).
    """
    print(f"Loading model from {args.model}...", file=sys.stderr)
    model_data = load_model(args.model)
    meta = model_data["meta"]
    model_features = meta.get("features", [])
    model_feature_names = [f.get("name") for f in model_features]

    delimiter = args.delimiter or detect_delimiter(args.data)
    has_header = not args.no_header

    raw_rows, header, input_format = load_raw_data(args.data, delimiter, has_header)
    print(f"  Loaded {len(raw_rows)} samples ({input_format} format)", file=sys.stderr)

    if not raw_rows:
        print("No data to predict", file=sys.stderr)
        return 1

    is_jsonl = input_format == "jsonl"

    # Determine target column for inplace mode
    target_key = None
    target_idx = None
    if args.mode == "inplace":
        if args.target:
            target_key = args.target
            if not is_jsonl and header and args.target in header:
                target_idx = header.index(args.target)
            elif not is_jsonl and args.target.isdigit():
                target_idx = int(args.target)
        elif is_jsonl:
            target_key = header[-1] if header else None
        else:
            target_idx = len(raw_rows[0]) - 1 if raw_rows else 0

    feature_vectors = _build_feature_vectors(
        raw_rows, header, model_feature_names, model_features, is_jsonl
    )

    predictions = predict_batch(model_data, feature_vectors)

    # Determine output format and delimiter
    output_format = args.output_format
    if output_format is None:
        output_format = detect_format(args.output) if args.output else input_format

    if args.output_delimiter is not None:
        output_delimiter = args.output_delimiter
    elif output_format == "tsv":
        output_delimiter = "\t"
    elif output_format == "csv":
        output_delimiter = ","
    elif output_format == "ssv":
        output_delimiter = " "
    else:
        output_delimiter = delimiter

    def row_to_dict(row):
        if is_jsonl:
            return dict(row)
        return (
            dict(zip(header, row))
            if header
            else {str(i + 1): v for i, v in enumerate(row)}
        )

    def row_to_list(row):
        if is_jsonl and header:
            return [str(row.get(k, "")) for k in header]
        return list(row)

    is_structured = output_format in ("json", "jsonl")
    output_data: list[dict[str, Any]] | list[str]

    if args.mode == "values":
        if is_structured:
            output_data = [{args.prediction_column: pred} for pred in predictions]
        else:
            output_data = [str(pred) for pred in predictions]

    elif args.mode == "append":
        if is_structured:
            output_data = [
                dict(row_to_dict(row), **{args.prediction_column: pred})
                for row, pred in zip(raw_rows, predictions)
            ]
        else:
            lines = []
            if header:
                lines.append(output_delimiter.join(header + [args.prediction_column]))
            for row, pred in zip(raw_rows, predictions):
                lines.append(output_delimiter.join(row_to_list(row) + [str(pred)]))
            output_data = lines

    else:  # args.mode == "inplace"
        if is_structured:
            records: list[dict[str, Any]] = []
            for row, pred in zip(raw_rows, predictions):
                record = row_to_dict(row)
                if target_key:
                    record[target_key] = pred
                records.append(record)
            output_data = records
        else:
            lines = []
            if header:
                lines.append(output_delimiter.join(header))
            for row, pred in zip(raw_rows, predictions):
                new_row = row_to_list(row)
                idx = (
                    header.index(target_key)
                    if target_key and header and target_key in header
                    else target_idx
                )
                if idx is not None and idx < len(new_row):
                    new_row[idx] = str(pred)
                lines.append(output_delimiter.join(new_row))
            output_data = lines

    _write_predictions(output_data, output_format, args.output)

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """
    Evaluate a model on test data.

    The model's task (loaded from its metadata) selects the metric set:

      - classification: accuracy, plus per-class precision/recall/F1/support
        when `--verbose` or `--json` is set.
      - regression: MSE, RMSE, MAE, R^2.

    Output:
      - Default: human-readable text to stdout (or `--output FILE`).
      - `--json`: machine-readable JSON; status lines go to stderr so the
        stdout/output file stays parseable.
    """

    # Status to stderr when outputting data
    def log(msg: str) -> None:
        if args.json or args.output:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    log(f"Loading model from {args.model}...")
    model_data = load_model(args.model)

    column_names = _parse_column_names(args)

    log(f"Loading test data from {args.data}...")
    X, y, feature_names, _ = load_training_data(
        args.data,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        target_col=args.target,
        column_names=column_names,
    )
    log(f"  Loaded {len(X)} samples")

    # Make predictions
    predictions = predict_batch(model_data, X)

    # Evaluate
    task = model_data["meta"].get("task", TASK_CLASSIFICATION)

    results: dict[str, Any] = {"task": task, "samples": len(y)}

    if task == TASK_CLASSIFICATION:
        metrics = evaluate_predictions(y, predictions)
        class_metrics = per_class_metrics(y, predictions)
        results["accuracy"] = metrics["accuracy"]
        results["correct"] = metrics["correct"]
        results["total"] = metrics["total"]
        if args.verbose or args.json:
            results["classes"] = {
                str(cls): {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in m.items()
                }
                for cls, m in class_metrics.items()
            }
    else:
        reg = regression_metrics(y, predictions, include_r2=True)
        results["mse"] = round(reg["mse"], 6)
        results["rmse"] = round(reg["rmse"], 6)
        results["mae"] = round(reg["mae"], 6)
        results["r2"] = round(reg["r2"], 6)

    # Output
    if args.json:
        with _output_file(args.output) as out:
            print(json.dumps(results, indent=2), file=out)
        if args.output:
            log(f"Results saved to {args.output}")
        return 0

    with _output_file(args.output) as out:
        if task == TASK_CLASSIFICATION:
            print(
                f"\nAccuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})",
                file=out,
            )
            if args.verbose:
                print("\nPer-class metrics:", file=out)
                print(
                    f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
                    file=out,
                )
                print("-" * 62, file=out)
                for cls, m in sorted(class_metrics.items(), key=lambda x: str(x[0])):
                    print(
                        f"{str(cls):<20} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%} {m['support']:>10}",
                        file=out,
                    )
        else:
            print(f"\nMSE:  {results['mse']:.4f}", file=out)
            print(f"RMSE: {results['rmse']:.4f}", file=out)
            print(f"MAE:  {results['mae']:.4f}", file=out)
            print(f"R^2:   {results['r2']:.4f}", file=out)

    if args.output:
        log(f"Results saved to {args.output}")

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect data and output inferred feature specs."""
    column_names = _parse_column_names(args)

    print(f"Inspecting {args.data}...", file=sys.stderr)
    X, y, feature_names, target_name = load_training_data(
        args.data,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        target_col=args.target,
        column_names=column_names,
    )
    print(f"  {len(X)} samples, {len(feature_names)} features", file=sys.stderr)

    # Infer specs
    specs = infer_feature_specs(X, feature_names)

    # target_name already loaded from data

    # Infer target type
    if is_likely_regression(y):
        target_type = "num"
        target_dtype = "float"
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in y):
        target_type = "cat"
        target_dtype = "int" if all(isinstance(v, int) for v in y) else "float"
    else:
        target_type = "cat"
        target_dtype = "str"

    target_spec = {"name": target_name, "dtype": target_dtype, "type": target_type}

    # Format output
    output: dict[str, Any]
    if args.format == "simple":
        # Simple format: {"name": "type"}
        output = {s["name"]: s["type"] for s in specs}
        output["_target"] = f"{target_type} ({target_name})"
    elif args.format == "full":
        # Object format with full specs
        output = {s["name"]: {"dtype": s["dtype"], "type": s["type"]} for s in specs}
        output["_target"] = target_spec
    else:  # array
        # Array format
        output = {"features": specs, "target": target_spec}

    # Output
    json_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"Spec saved to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


def _print_stats_human(
    out: TextIO,
    args: argparse.Namespace,
    stats_dict: dict[str, Any],
    meta: Any,
    is_forest: bool,
    n_trees: int,
    total_nodes: int,
) -> None:
    """Print human-readable model statistics to output stream."""

    def p(msg: str) -> None:
        print(msg, file=out)

    p(f"Loading model from {args.model}...")
    p("\n" + "=" * 50)
    p("MODEL STATISTICS")
    p("=" * 50)

    p(f"\nModel Type:    {stats_dict['model_type']}")
    p(f"Task:          {stats_dict['task']}")
    p(f"Format:        {stats_dict['format_version']}")
    p(f"Cartlet:       {stats_dict['cartlet_version']}")

    # Features
    features_list = meta.get("features", [])
    p(f"\nFeatures ({len(features_list)}):")
    for f in features_list:
        ftype = f.get("type", "?")
        dtype = "str"
        values = f.get("values", [])
        if values and len(values) <= _MAX_CAT_VALUES_DISPLAY:
            p(f"  {f.get('name'):<20} {ftype:<5} {dtype:<6} values: {values}")
        elif values:
            p(
                f"  {f.get('name'):<20} {ftype:<5} {dtype:<6} ({len(values)} unique values)"
            )
        else:
            p(f"  {f.get('name'):<20} {ftype:<5} {dtype:<6}")

    # Target info
    target_info = meta.get("target")
    if target_info:
        p(f"\nTarget: {target_info.get('name')} ({target_info.get('type', '?')})")
        target_values = target_info.get("values", [])
        if target_values:
            if len(target_values) <= _MAX_TARGET_VALUES_DISPLAY:
                p(f"  Classes: {target_values}")
            else:
                p(f"  Classes: {len(target_values)} unique values")

    # Tree/Forest stats
    if is_forest:
        p("\nForest:")
        p(f"  Trees:         {n_trees}")
        p(f"  Total nodes:   {total_nodes}")
        if args.verbose:
            p(f"  Avg nodes/tree: {total_nodes / n_trees:.1f}")
    else:
        p("\nTree:")
        p(f"  Total nodes:   {total_nodes}")

    # Training info
    training = meta.get("training")
    if training:
        p("\nTraining:")
        trained_at = str(training.get("trained_at", "?"))
        p(f"  Date:          {trained_at[:_ISO8601_PREFIX_LEN].replace('T', ' ')}")
        p(f"  Data source:   {training.get('data_source', '?')}")
        p(f"  Samples:       {training.get('samples', '?')}")
        config_meta = training.get("config", {})
        if config_meta and isinstance(config_meta, dict) and config_meta.get("prune"):
            p(
                f"  Pruned:        Yes ({config_meta.get('validation_split', 0):.0%} validation)"
            )
        test_meta = training.get("test")
        if test_meta and isinstance(test_meta, dict):
            if "accuracy" in test_meta:
                p(
                    f"  Test accuracy: {test_meta['accuracy']:.2%} ({test_meta['samples']} samples)"
                )
            elif "mse" in test_meta:
                p(
                    f"  Test MSE:      {test_meta['mse']:.4f} ({test_meta['samples']} samples)"
                )

    p("\nParameters:")
    p(f"  min_samples_split: {meta.get('min_samples_split', '?')}")
    p(f"  min_samples_leaf:  {meta.get('min_samples_leaf', '?')}")
    if not is_forest:
        p(f"  store_distributions: {meta.get('store_distributions', '?')}")

    fmt = meta.get("format")
    if fmt and args.verbose:
        p("\nNode Format:")
        for k, desc in fmt.items():
            p(f"  {k}: {desc}")


def cmd_stats(args: argparse.Namespace) -> int:
    """Show model statistics."""
    model_data = load_model(args.model)
    meta = model_data["meta"]
    is_forest = model_data["is_forest"]
    n_trees = model_data["n_trees"]
    n_decisions = len(model_data["decisions"])
    n_leaves = len(model_data["leaves"])
    total_nodes = n_decisions + n_leaves

    stats_dict: dict[str, Any] = {
        "model_type": "RandomForest" if is_forest else "DecisionTree",
        "task": meta.get("task", "unknown"),
        "format_version": "cart-1.0",
        "cartlet_version": __version__,
        "features": meta.get("features", []),
        "target": meta.get("target"),
    }

    if is_forest:
        stats_dict["forest"] = {"n_trees": n_trees, "total_nodes": total_nodes}
    else:
        stats_dict["tree"] = {"total_nodes": total_nodes}

    if "training" in meta:
        stats_dict["training"] = meta["training"]

    stats_dict["params"] = {
        "min_samples_split": meta.get("min_samples_split"),
        "min_samples_leaf": meta.get("min_samples_leaf"),
    }

    if args.json:
        with _output_file(args.output) as out:
            print(json.dumps(stats_dict, indent=2), file=out)
        if args.output:
            print(f"Stats saved to {args.output}", file=sys.stderr)
        return 0

    with _output_file(args.output) as out:
        _print_stats_human(out, args, stats_dict, meta, is_forest, n_trees, total_nodes)

    if args.output:
        print(f"Stats saved to {args.output}", file=sys.stderr)

    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert model between formats."""
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    input_format = getattr(args, "input_format", None)
    output_format = getattr(args, "output_format", None)

    ext_in, gz_in = resolve_format(input_path, input_format)
    ext_out, gz_out = resolve_format(output_path, output_format)

    print(f"Converting {input_path} -> {output_path}", file=sys.stderr)

    try:
        is_forest = _detect_is_forest(input_path, format=input_format)
        convert(
            input_path,
            output_path,
            input_format=input_format,
            output_format=output_format,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    model_type = "RandomForest" if is_forest else "DecisionTree"

    label_in = f"{ext_in}{'.gz' if gz_in else ''}"
    label_out = f"{ext_out}{'.gz' if gz_out else ''}"
    print(f"  Model type: {model_type}", file=sys.stderr)
    print(f"  Input:  {in_size:,} bytes ({label_in})", file=sys.stderr)
    print(f"  Output: {out_size:,} bytes ({label_out})", file=sys.stderr)

    if ext_in in (".json", ".jsonl", ".pkl", ".pickle") and ext_out == ".cart":
        print(
            "  Note: Distributions lost in .cart format (nbest will return 1 result)",
            file=sys.stderr,
        )

    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """Bundle a model with the Python runner into a standalone file."""
    model_path = args.model
    output_path = args.output
    library_only = args.library_only
    embed_model = not getattr(args, "no_model", False)

    if embed_model and model_path is None:
        print("Error: Model path required (or use --no-model)", file=sys.stderr)
        return 1

    if model_path and not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    model_format = getattr(args, "model_format", None)

    try:
        bundle(
            model_path,
            output_path,
            library_only=library_only,
            embed_model=embed_model,
            model_format=model_format,
        )
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_size = os.path.getsize(output_path)
    mode = "library" if library_only else "executable"
    if embed_model:
        in_size = os.path.getsize(model_path)
        print(f"Bundled {model_path} -> {output_path}", file=sys.stderr)
        print(f"  Model size: {in_size:,} bytes", file=sys.stderr)
    else:
        print(f"Created runner library -> {output_path}", file=sys.stderr)
        print("  Model: none (load at runtime)", file=sys.stderr)
    print(f"  Mode: {mode}", file=sys.stderr)
    print(f"  Output size: {out_size:,} bytes", file=sys.stderr)

    if not library_only:
        print(
            f"\nTo run:      chmod +x {output_path} && ./{output_path} feature1 ...",
            file=sys.stderr,
        )
    elif not embed_model:
        print(
            f"\nUsage: from {os.path.splitext(os.path.basename(output_path))[0]} "
            "import Predictor",
            file=sys.stderr,
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="cartlet",
        description="Decision tree training and inference CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cartlet {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet train data.csv -o model.cart
  cartlet train data.tsv --target label -F -n 100
  cartlet train data.jsonl --config fast -o fast_model.cart
  cartlet train data.csv -X '{"age": "num", "color": "cat"}' -P
""",
    )
    train_parser.add_argument("data", help="Training data file (CSV/TSV/JSONL)")
    train_parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output model file (.cart, .json, .jsonl, .pkl, .skl/.joblib; "
        "add .gz for gzip)",
    )
    train_parser.add_argument(
        "-c",
        "--config",
        metavar="NAME|FILE",
        help=(
            "Config preset or file. Presets: "
            + ", ".join(_BUILTIN_CONFIGS)
            + ". Or path to .yaml/.json file"
        ),
    )
    train_parser.add_argument(
        "--save-config",
        metavar="FILE",
        help="Save current args to config file (.yaml or .json)",
    )
    _add_tabular_input_args(train_parser)
    train_parser.add_argument(
        "-X",
        "--features",
        metavar="JSON",
        help='Feature specs as JSON file or inline (e.g., \'{"age": "num", "color": "cat"}\')',
    )
    train_parser.add_argument(
        "-T",
        "--task",
        choices=[TASK_AUTO, TASK_CLASSIFICATION, TASK_REGRESSION],
        default=TASK_AUTO,
        help="Task type (default: auto-detect)",
    )
    train_parser.add_argument(
        "-F",
        "--forest",
        action="store_true",
        help="Train RandomForest instead of DecisionTree",
    )
    train_parser.add_argument(
        "--extra-trees",
        action="store_true",
        help="Train ExtraTrees forest (random splits, implies --forest)",
    )
    train_parser.add_argument(
        "--isolation-forest",
        action="store_true",
        help="Train IsolationForest for anomaly detection (unsupervised)",
    )
    train_parser.add_argument(
        "-n",
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        metavar="N",
        help=f"Number of trees for forest (default: {DEFAULT_N_ESTIMATORS})",
    )
    train_parser.add_argument(
        "-D", "--max-depth", type=int, metavar="N", help="Maximum tree depth"
    )
    train_parser.add_argument(
        "-s",
        "--min-samples-split",
        type=int,
        default=2,
        metavar="N",
        help="Minimum samples to split (default: 2)",
    )
    train_parser.add_argument(
        "-l",
        "--min-samples-leaf",
        type=int,
        default=1,
        metavar="N",
        help="Minimum samples in leaf (default: 1)",
    )
    train_parser.add_argument(
        "-S",
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        metavar="FRAC",
        help=f"Fraction for test evaluation (default: {DEFAULT_TEST_SPLIT})",
    )
    train_parser.add_argument(
        "-e",
        "--test-file",
        metavar="FILE",
        help="Separate test data file (alternative to --test-split)",
    )
    train_parser.add_argument(
        "-V",
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        metavar="FRAC",
        help=f"Fraction for validation/pruning (default: {DEFAULT_VALIDATION_SPLIT})",
    )
    train_parser.add_argument(
        "-E",
        "--validation-file",
        metavar="FILE",
        help="(not yet implemented) Separate validation data file",
    )
    train_parser.add_argument(
        "-P",
        "--prune",
        action="store_true",
        help="Enable reduced error pruning (uses validation split)",
    )
    train_parser.add_argument(
        "-R",
        "--random-seed",
        type=int,
        metavar="SEED",
        help="Random seed for reproducibility",
    )
    train_parser.add_argument(
        "-C",
        "--criterion",
        choices=["entropy", "gini"],
        default="entropy",
        help="Split criterion for classification (default: entropy)",
    )
    train_parser.add_argument(
        "-B",
        "--trainer",
        choices=["native", "sklearn"],
        default="native",
        help="Training backend (default: native)",
    )
    train_parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        metavar="N",
        help="Parallel jobs for sklearn trainer only (-1=all cores)",
    )
    train_parser.add_argument(
        "--no-distributions",
        action="store_true",
        help="Omit distributions in .cart output (smaller file, no nbest)",
    )
    train_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet predict model.cart input.csv
  cartlet predict model.cart data.tsv -m append -o results.tsv
  cartlet predict model.cart input.jsonl --prediction-column score
""",
    )
    predict_parser.add_argument("model", help="Model file (.cart)")
    predict_parser.add_argument("data", help="Input data file (CSV/TSV/JSONL)")
    predict_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Output file (default: stdout)"
    )
    _add_tabular_input_args(
        predict_parser,
        target_help="Target column for inplace mode (default: last)",
        delimiter_help="Input column delimiter (auto-detect)",
        column_names=False,
    )
    predict_parser.add_argument(
        "-D",
        "--output-delimiter",
        metavar="CHAR",
        help="Output delimiter (default: same as input)",
    )
    predict_parser.add_argument(
        "-m",
        "--mode",
        choices=["values", "append", "inplace"],
        default="values",
        help="Output mode: 'values', 'append', or 'inplace' (default: values)",
    )
    predict_parser.add_argument(
        "-c",
        "--prediction-column",
        default="prediction",
        metavar="NAME",
        help="Column name for predictions in append mode (default: 'prediction')",
    )
    predict_parser.add_argument(
        "-f",
        "--output-format",
        choices=["csv", "tsv", "ssv", "json", "jsonl"],
        help="Output format: csv, tsv, ssv, json, jsonl (default: same as input)",
    )
    predict_parser.set_defaults(func=cmd_predict)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        aliases=["eval"],
        help="Evaluate model on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet eval model.cart test.csv
  cartlet eval model.cart test.tsv -v
  cartlet eval model.cart test.jsonl --json
""",
    )
    eval_parser.add_argument("model", help="Model file (.cart)")
    eval_parser.add_argument("data", help="Test data file (CSV/TSV/JSONL)")
    eval_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Output to file (default: stdout)"
    )
    eval_parser.add_argument(
        "-J", "--json", action="store_true", help="Output as JSON (machine-readable)"
    )
    _add_tabular_input_args(eval_parser, delimiter_help="Column delimiter")
    eval_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-class metrics"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        aliases=["info"],
        help="Show model statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet stats model.cart
  cartlet info model.cart -v
  cartlet stats model.cart --json
""",
    )
    stats_parser.add_argument("model", help="Model file (.cart)")
    stats_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Output to file (default: stdout)"
    )
    stats_parser.add_argument(
        "-J", "--json", action="store_true", help="Output as JSON (machine-readable)"
    )
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed statistics"
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Inspect command - infer and output feature specs
    inspect_parser = subparsers.add_parser(
        "inspect",
        aliases=["schema"],
        help="Infer feature types and output spec JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet inspect data.csv
  cartlet inspect data.csv -o specs.json
  cartlet schema data.tsv --format full
""",
    )
    inspect_parser.add_argument("data", help="Data file to inspect (CSV/TSV/JSONL)")
    inspect_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Output spec to file (default: stdout)"
    )
    _add_tabular_input_args(
        inspect_parser,
        target_help="Target column (default: last)",
        delimiter_help="Column delimiter",
    )
    inspect_parser.add_argument(
        "-f",
        "--format",
        choices=["simple", "full", "array"],
        default="simple",
        help="Output format (default: simple)",
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # Convert command - convert between model formats
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert model between formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet convert model.cart model.json
  cartlet convert model.json model.pkl
  cartlet convert model.cart model.cart.gz
""",
    )
    convert_parser.add_argument("input", help="Input model file")
    convert_parser.add_argument("output", help="Output model file")
    convert_parser.add_argument(
        "--input-format",
        dest="input_format",
        default=None,
        help="Override input format detection (e.g. jsonl, json, cart)",
    )
    convert_parser.add_argument(
        "--output-format",
        dest="output_format",
        default=None,
        help="Override output format selection (e.g. jsonl, json, cart)",
    )
    convert_parser.set_defaults(func=cmd_convert)

    # Bundle command - bundle a model with the Python runner into a standalone file
    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Bundle a model with the Python runner into a standalone file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cartlet bundle model.cart predict.py
  cartlet bundle model.json predict.py            # auto-converted to .cart
  cartlet bundle --no-model --library-only cart.py
""",
    )
    bundle_parser.add_argument(
        "model",
        nargs="?",
        help="Model file (any supported format, auto-converted to .cart)",
    )
    bundle_parser.add_argument("output", help="Output file path")
    bundle_parser.add_argument(
        "--library-only",
        action="store_true",
        help="Omit CLI code, produce library-only output for import use",
    )
    bundle_parser.add_argument(
        "--no-model",
        action="store_true",
        help="Output runner code only without embedded model (for dynamic loading)",
    )
    bundle_parser.add_argument(
        "--model-format",
        dest="model_format",
        default=None,
        help="Override input model format (e.g. jsonl) when the extension is custom",
    )
    bundle_parser.set_defaults(func=cmd_bundle)

    return parser


def _extract_config_option(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pull the ``--config``/``-c`` value out of argv, returning
    ``(config_name_or_None, argv_without_the_option)``.

    Handles every argparse spelling -- ``--config X``, ``--config=X``,
    ``-c X``, ``-cX``, ``-c=X`` -- so a config passed with ``=`` is no longer
    silently ignored.
    """
    config_name: str | None = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-c", "--config"):
            if i + 1 < len(argv):
                config_name = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("-c") and not arg.startswith("--") and len(arg) > 2:
            val = arg[2:]
            config_name = val[1:] if val.startswith("=") else val
            i += 1
            continue
        out.append(arg)
        i += 1
    return config_name, out


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = _build_parser()

    if argv is None:
        argv = sys.argv[1:]

    config_file_used = None
    config_preset_used = None

    try:
        if len(argv) >= 1 and argv[0] == "train":
            config_name, rest = _extract_config_option(argv[1:])
            if config_name:
                config = load_config(config_name)
                if config_name in _BUILTIN_CONFIGS:
                    config_preset_used = config_name
                    print(f"Using preset config: {config_name}", file=sys.stderr)
                else:
                    config_file_used = config_name
                    print(f"Loaded config from {config_name}", file=sys.stderr)
                argv = [argv[0]] + merge_config_with_argv(config, rest)

        args = parser.parse_args(argv)

        # Attach config info to args for metadata
        if config_file_used:
            args.config_file_used = config_file_used
        elif config_preset_used:
            args.config_preset_used = config_preset_used

        if args.command is None:
            parser.print_help()
            return 1

        return args.func(args)
    except (OSError, ValueError, ImportError, json.JSONDecodeError) as e:
        # Turn expected user/runtime errors into a clean one-line message and a
        # nonzero exit code instead of a raw traceback.
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
