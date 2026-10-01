"""CLI handlers for held-out importance and structural path inspection."""

from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from typing import Any, TextIO

from .evaluation import permutation_importance
from .inspection import decisive_leaves, leaf_paths
from .io import detect_format, load_training_data, read_vectors
from .runner import load_model
from .types import ModelData
from .validation import align_features


@contextmanager
def _output_file(path: str | None):
    if path:
        with open(path, "w", encoding="utf-8") as stream:
            yield stream
    else:
        yield sys.stdout


def _column_names(args: Any) -> list[str] | None:
    raw = getattr(args, "column_names", None)
    return [name.strip() for name in raw.split(",")] if raw else None


def _load_data(
    args: Any, model_data: ModelData, *, target_optional: bool
) -> tuple[list[list[Any]], list[Any] | None]:
    column_names = _column_names(args)
    labeled = not target_optional or args.target is not None
    if labeled:
        X, y, feature_names, _ = load_training_data(
            args.data,
            delimiter=args.delimiter,
            has_header=not args.no_header,
            target_col=args.target,
            column_names=column_names,
        )
    else:
        X, y, feature_names, _ = read_vectors(
            args.data,
            delimiter=args.delimiter,
            has_header=not args.no_header,
            labeled=False,
        )
        if column_names is not None:
            if len(column_names) != len(feature_names):
                raise ValueError(
                    f"Column names count ({len(column_names)}) doesn't match "
                    f"data columns ({len(feature_names)})"
                )
            feature_names = column_names
    if (
        column_names is not None
        or not args.no_header
        or detect_format(args.data) == "jsonl"
    ):
        model_names = [
            feature["name"] for feature in model_data["meta"].get("features", [])
        ]
        X = align_features(X, feature_names, model_names)
    return X, y


def _load_groups(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if os.path.isfile(value):
            with open(value, encoding="utf-8") as source:
                groups = json.load(source)
        else:
            groups = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid feature groups JSON: {exc}") from exc
    if not isinstance(groups, dict):
        raise ValueError("feature groups must be a JSON object")
    return groups


def cmd_importance(args: Any) -> int:
    """Measure held-out permutation importance from a `.cart` model."""
    print(f"Loading model from {args.model}...", file=sys.stderr)
    model_data = load_model(args.model)
    print(f"Loading held-out data from {args.data}...", file=sys.stderr)
    X, y = _load_data(args, model_data, target_optional=False)
    assert y is not None
    report = permutation_importance(
        model_data,
        X,
        y,
        feature_groups=_load_groups(args.groups),
        n_repeats=args.repeats,
        random_state=args.random_seed,
        missing=args.missing,
    )
    with _output_file(args.output) as out:
        if args.format == "json":
            print(json.dumps(report, indent=2, allow_nan=False), file=out)
        else:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(["name", "features", "baseline", "mean", "std", "values"])
            for record in report["importances"]:
                writer.writerow(
                    [
                        record["name"],
                        json.dumps(record["features"], separators=(",", ":")),
                        report["baseline"],
                        record["mean"],
                        record["std"],
                        json.dumps(record["values"], separators=(",", ":")),
                    ]
                )
    if args.output:
        print(f"Results saved to {args.output}", file=sys.stderr)
    return 0


def _write_leaf_tsv(export: dict[str, Any], out: TextIO) -> None:
    writer = csv.writer(out, delimiter="\t")
    writer.writerow(
        [
            "tree",
            "leaf",
            "prediction",
            "predicted_class",
            "support",
            "purity",
            "data_support",
            "data_purity",
            "path",
            "class_distribution",
            "class_counts",
            "data_class_counts",
        ]
    )
    for tree in export["trees"]:
        for leaf in tree["leaves"]:
            writer.writerow(
                [
                    leaf["tree"],
                    leaf["leaf"],
                    leaf["prediction"],
                    leaf["predicted_class"],
                    leaf["support"],
                    leaf["purity"],
                    leaf.get("data_support"),
                    leaf.get("data_purity"),
                    json.dumps(leaf["path"], separators=(",", ":")),
                    json.dumps(leaf["class_distribution"], separators=(",", ":")),
                    json.dumps(leaf["class_counts"], separators=(",", ":")),
                    json.dumps(leaf.get("data_class_counts"), separators=(",", ":")),
                ]
            )


def cmd_leaves(args: Any) -> int:
    """Export root-to-leaf paths and optional empirical support."""
    print(f"Loading model from {args.model}...", file=sys.stderr)
    model_data = load_model(args.model)
    X = y = None
    if args.data:
        print(f"Loading inspection data from {args.data}...", file=sys.stderr)
        X, y = _load_data(args, model_data, target_optional=True)
    elif args.target is not None:
        raise ValueError("--target requires --data")
    export = leaf_paths(model_data, X, y, missing=args.missing)
    if args.class_label is not None:
        selected = decisive_leaves(
            model_data,
            args.class_label,
            X,
            y,
            min_support=args.min_support,
            min_purity=args.min_purity,
            missing=args.missing,
        )
        selected_paths = {
            (record["tree"], record["leaf"], json.dumps(record["path"], sort_keys=True))
            for record in selected
        }
        for tree in export["trees"]:
            tree["leaves"] = [
                record
                for record in tree["leaves"]
                if (
                    record["tree"],
                    record["leaf"],
                    json.dumps(record["path"], sort_keys=True),
                )
                in selected_paths
            ]
    elif args.min_support is not None or args.min_purity is not None:
        raise ValueError("--min-support and --min-purity require --class")
    with _output_file(args.output) as out:
        if args.format == "json":
            print(json.dumps(export, indent=2, allow_nan=False), file=out)
        else:
            _write_leaf_tsv(export, out)
    if args.output:
        print(f"Results saved to {args.output}", file=sys.stderr)
    return 0


__all__ = ["cmd_importance", "cmd_leaves"]
