"""Quiet operational training workflows shared by Python callers and the CLI."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation import evaluate_tree
from .forest import RandomForest
from .io import load_training_data, read_vectors
from .isolation import IsolationForest
from .tree import DecisionTree
from .types import (
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
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
from .utils import tree_stats
from .validation import (
    align_features,
    effective_validation_split,
    require_distinct_paths,
    validate_dataset,
    validate_splits,
    validate_training_parameters,
)


@dataclass(frozen=True)
class TrainingSettings:
    """Serializable model and split settings; no parser or implicit output state."""

    model_type: str = "tree"
    task: str = TASK_AUTO
    trainer: str | None = "native"
    n_estimators: int = DEFAULT_N_ESTIMATORS
    extra_trees: bool = False
    max_depth: int | None = None
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF
    criterion: str = "entropy"
    categorical_split: str = "exact"
    prune: bool = False
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    test_split: float = DEFAULT_TEST_SPLIT
    random_state: int | None = None
    n_jobs: int | None = None
    store_distributions: bool = True

    def validate(self) -> None:
        """Reject unsupported settings before model construction or writing."""
        validate_splits(self.validation_split, 0)
        validate_splits(0, self.test_split)
        if self.model_type not in ("tree", "forest", "isolation") or self.task not in (
            TASK_AUTO,
            TASK_CLASSIFICATION,
            TASK_REGRESSION,
        ):
            raise ValueError("invalid model_type or task")
        validate_training_parameters(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            trainer=self.trainer,
            criterion=self.criterion,
            categorical_split=self.categorical_split,
            store_distributions=self.store_distributions,
            prune=self.prune,
            extra_trees=self.extra_trees,
        )
        if self.extra_trees and self.model_type != "forest":
            raise ValueError("extra_trees requires forest model_type")
        if self.model_type == "isolation" and self.trainer not in (None, "native"):
            raise ValueError("isolation workflow uses the native trainer")

    def to_dict(self) -> dict[str, Any]:
        """Return native JSON-compatible settings."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    """Trained model, actual populations, metrics and optional export provenance."""

    model: DecisionTree | RandomForest | IsolationForest
    settings: TrainingSettings
    task: str
    source_samples: int
    training_samples: int
    test_samples: int
    validation_samples: int
    metrics: dict[str, Any]
    stats: dict[str, Any]
    output_path: Path | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return serializable report data; the live model remains a Python field."""
        return {
            "model_type": self.settings.model_type,
            "task": self.task,
            "settings": self.settings.to_dict(),
            "source_samples": self.source_samples,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "validation_samples": self.validation_samples,
            "metrics": self.metrics,
            "stats": self.stats,
            "output_path": str(self.output_path)
            if self.output_path is not None
            else None,
            "warnings": list(self.warnings),
        }


def resolve_feature_specs(
    specs: str | Mapping[str, Any] | Sequence[Any], names: list[str]
) -> list[dict[str, Any]]:
    """Resolve explicit Python specs or a JSON path/string into named specs."""
    if isinstance(specs, str):
        if not specs.lstrip().startswith(("{", "[")) and Path(specs).is_file():
            specs = json.loads(Path(specs).read_text(encoding="utf-8"))
        else:
            specs = json.loads(specs)
    if isinstance(specs, Mapping):
        unknown = set(specs) - set(names)
        if unknown:
            raise ValueError(
                f"unknown feature specifications: {', '.join(sorted(unknown))}"
            )
        resolved = [
            normalize_feature_spec(specs.get(name, "cat"), name=name) for name in names
        ]
    elif isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)):
        resolved = [normalize_feature_spec(spec) for spec in specs]
        if [spec.name for spec in resolved] != names:
            raise ValueError(
                "feature specifications must match input column names and order"
            )
    else:
        raise ValueError("feature specifications must be an object or sequence")
    return [{"name": s.name, "dtype": s.dtype, "type": s.type} for s in resolved]


def train_model(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any] | None = None,
    *,
    settings: TrainingSettings | None = None,
    feature_names: Sequence[str] | None = None,
    feature_specs: Sequence[dict[str, Any]] | None = None,
    counts: Sequence[float] | None = None,
    test_data: tuple[Sequence[Sequence[Any]], Sequence[Any]] | None = None,
    output: str | Path | None = None,
) -> TrainingResult:
    """Compose validation, splitting, training, evaluation and optional export.

    Quiet by default. Split evaluation uses the same model feature order; file
    callers should prefer train_file for named-column alignment. Zero weights are
    omitted before splitting; source_samples counts the remaining active rows. Unsupported pruning yields a report warning.
    """
    config = settings or TrainingSettings()
    config.validate()
    rows, targets, weights = validate_dataset(X, y, counts)
    source_samples = len(rows)
    names = (
        list(feature_names)
        if feature_names is not None
        else [normalize_feature_spec(spec).name for spec in feature_specs]
        if feature_specs is not None
        else [str(i) for i in range(len(rows[0]))]
    )
    if (
        any(not isinstance(name, str) for name in names)
        or len(names) != len(rows[0])
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature names must be unique and match data width")
    warnings: list[str] = []
    validation_samples = 0
    model: DecisionTree | RandomForest | IsolationForest
    test_X: list[list[Any]] = []
    test_y: list[Any] = []
    if config.model_type == "isolation":
        if y is not None:
            raise ValueError("isolation training requires unlabeled data")
        if counts is not None:
            raise ValueError("isolation training does not support instance weights")
        if feature_specs is not None:
            raise ValueError(
                "isolation workflow accepts numeric inputs without feature specifications"
            )
        if test_data is not None:
            raise ValueError("isolation training does not use labeled test data")
        if config.prune or config.task != TASK_AUTO:
            raise ValueError(
                "isolation training does not support pruning or supervised task settings"
            )
        isolation_model = IsolationForest(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            feature_names=names,
            random_state=config.random_state,
        )
        isolation_model.load_data(rows)
        isolation_model.train()
        model = isolation_model
        task = "anomaly"
        stats = {"n_trees": len(isolation_model.trees)}
    else:
        if targets is None:
            raise ValueError("supervised training requires targets")
        task = (
            config.task
            if config.task != TASK_AUTO
            else (
                TASK_REGRESSION
                if is_likely_regression(targets)
                else TASK_CLASSIFICATION
            )
        )
        if task == TASK_CLASSIFICATION:
            targets = [str(target) for target in targets]
        specs = (
            resolve_feature_specs(feature_specs, names)
            if feature_specs is not None
            else infer_feature_specs(rows, names)
        )
        kwargs: dict[str, Any] = {
            "features": specs,
            "task": task,
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
            "min_samples_leaf": config.min_samples_leaf,
            "criterion": config.criterion,
            "categorical_split": config.categorical_split,
        }
        if test_data is not None:
            test_X, explicit_y, _ = validate_dataset(*test_data)
            assert explicit_y is not None
            test_y = explicit_y
            if len(test_X[0]) != len(names):
                raise ValueError("test data width must match model features")
        elif config.test_split:
            indices = list(range(len(rows)))
            random.Random(config.random_state).shuffle(indices)
            size = int(len(rows) * config.test_split)
            test_ids, train_ids = indices[:size], indices[size:]
            test_X, test_y = [rows[i] for i in test_ids], [targets[i] for i in test_ids]
            rows, targets, weights = (
                [rows[i] for i in train_ids],
                [targets[i] for i in train_ids],
                [weights[i] for i in train_ids],
            )
        if task == TASK_CLASSIFICATION:
            test_y = [str(target) for target in test_y]
        effective_prune = (
            config.prune
            and config.model_type == "tree"
            and config.trainer == "native"
            and task == TASK_CLASSIFICATION
        )
        active_val = effective_validation_split(
            config.prune, config.validation_split, effective_prune
        )
        validate_splits(active_val, config.test_split if test_data is None else 0)
        if config.prune and not effective_prune:
            warnings.append(
                "pruning is unsupported for the selected model/task/trainer; no validation rows held out"
            )
        if config.model_type == "forest":
            model = RandomForest(
                n_estimators=config.n_estimators,
                extra_trees=config.extra_trees,
                **kwargs,
            )
            model.load_data(rows, targets, weights)
            model.train(
                trainer=config.trainer,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
            stats = {"n_trees": len(model.trees)}
        else:
            model = DecisionTree(
                store_distributions=config.store_distributions, **kwargs
            )
            model.load_data(rows, targets, weights)
            val = active_val
            model._train(
                trainer=config.trainer,
                random_state=config.random_state,
                prune=effective_prune,
                validation_split=val if effective_prune else 0,
                validation_count=int(source_samples * val) if effective_prune else None,
            )
            stats = tree_stats(model.model)
            validation_samples = model.training_summary["validation_samples"]
    metrics = evaluate_tree(model, test_X, test_y) if test_X else {}
    output_path = Path(output) if output is not None else None
    result = TrainingResult(
        model,
        config,
        task,
        source_samples,
        len(rows) - validation_samples,
        len(test_X),
        validation_samples,
        metrics,
        stats,
        output_path,
        tuple(warnings),
    )
    if output_path is not None:
        if isinstance(model, IsolationForest):
            model.export(str(output_path))
        else:
            model.export(
                str(output_path),
                metadata={
                    "training": {
                        "trained_at": datetime.now().isoformat(),
                        **result.to_dict(),
                    }
                },
                store_distributions=config.store_distributions,
            )
    return result


def train_file(
    data: str | Path,
    *,
    settings: TrainingSettings | None = None,
    output: str | Path | None = None,
    test_file: str | Path | None = None,
    delimiter: str | None = None,
    has_header: bool = True,
    target: str | int | None = None,
    column_names: list[str] | None = None,
    features: str | Mapping[str, Any] | Sequence[Any] | None = None,
) -> TrainingResult:
    """Read tabular input, align named test columns, and delegate to train_model.

    Isolation input uses every column unless target explicitly identifies a
    label to exclude. Outputs must differ from every file input before writing.
    """
    config = settings or TrainingSettings()
    config.validate()
    inputs: list[str | Path] = [data]
    if test_file is not None:
        inputs.append(test_file)
    if (
        isinstance(features, str)
        and not features.lstrip().startswith(("{", "["))
        and Path(features).is_file()
    ):
        inputs.append(features)
    require_distinct_paths(inputs, [output] if output is not None else [])
    if config.model_type == "isolation" and target is None:
        X, y, names, _ = read_vectors(
            str(data), delimiter=delimiter, has_header=has_header, labeled=False
        )
        if column_names is not None:
            if X and len(column_names) != len(X[0]):
                raise ValueError("column names must match input width")
            names = column_names
    else:
        X, y, names, _ = load_training_data(
            str(data),
            delimiter=delimiter,
            has_header=has_header,
            target_col=target,
            column_names=column_names,
        )
    if config.model_type == "isolation":
        y = None  # An explicitly selected label column is excluded, not trained.
    test_data = None
    if test_file is not None:
        test_X, test_y, test_names, _ = load_training_data(
            str(test_file),
            delimiter=delimiter,
            has_header=has_header,
            target_col=target,
            column_names=column_names,
        )
        if has_header or str(test_file).endswith(".jsonl"):
            test_X = align_features(test_X, test_names, names)
        test_data = test_X, test_y
    specs = resolve_feature_specs(features, names) if features is not None else None
    return train_model(
        X,
        y,
        settings=config,
        feature_names=names,
        feature_specs=specs,
        test_data=test_data,
        output=output,
    )
