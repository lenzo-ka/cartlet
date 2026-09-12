"""
Cartlet - Decision Trees for Classification and Regression

A clean implementation of CART-style decision trees supporting:
- Categorical features (equality splits)
- Numerical features (threshold splits)
- Classification (entropy or Gini)
- Regression (variance reduction)

Training backends:
- Native: Pure Python, zero dependencies (default)
- Sklearn: Uses scikit-learn (optional, faster)
"""

from __future__ import annotations

import json
import pickle
import struct
from dataclasses import dataclass

from .base import _read_model_artifact
from .evaluation import (
    confusion_matrix,
    cross_validate,
    evaluate_predictions,
    evaluate_tree,
    per_class_metrics,
    regression_metrics,
)
from .forest import RandomForest
from .io import open_file, open_file_binary
from .io.bytes import bundle

# Aliased to underscores: these are binary-format internals, not public API.
from .io.cart_format import FLAG_IS_FOREST as _FLAG_IS_FOREST
from .io.cart_format import HEADER_SIZE as _HEADER_SIZE
from .io.cart_format import MAGIC as _MAGIC
from .io.cart_format import OFF_FLAGS as _OFF_FLAGS
from .io.utils import require_joblib, resolve_format
from .isolation import IsolationForest
from .runner import (
    Predictor,
    get_vocabulary,
    is_oov,
    load_model,
    predict,
    predict_batch,
    read_cart_metadata,
)
from .trainer import Native, Trainer
from .training import TrainingResult, TrainingSettings, train_file, train_model
from .tree import DecisionTree
from .types import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    DEFAULT_MIN_DIST_ENTROPY,
    DTYPE_BOOL,
    DTYPE_FLOAT,
    DTYPE_INT,
    DTYPE_STR,
    PROB_HIGH_CONFIDENCE,
    TASK_AUTO,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TYPE_CAT,
    TYPE_NUM,
    ClassificationLeaf,
    DecisionNode,
    FeatureSpec,
    LeafNode,
    RegressionLeaf,
    TreeNode,
)
from .utils import count_leaves, count_nodes, max_depth, tree_stats
from .validation import require_distinct_paths
from .xgboost import XGBoostTree


@dataclass(frozen=True)
class ConversionResult:
    """Artifact conversion provenance for library callers and CLI presentation."""

    model_type: str
    input_path: str
    output_path: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible conversion provenance."""
        return {
            "model_type": self.model_type,
            "input_path": self.input_path,
            "output_path": self.output_path,
        }


def convert(
    input_path: str,
    output_path: str,
    *,
    input_format: str | None = None,
    output_format: str | None = None,
) -> ConversionResult:
    """
    Convert a model between formats.

    Supported formats (by extension):
      - .cart: Compact binary (cross-language)
      - .json: Full model as JSON (preserves distributions)
      - .jsonl: Full model as JSON Lines
      - .pkl/.pickle: Python pickle
      - .skl/.joblib: sklearn model (requires sklearn; export requires sklearn-trained model)

    Note: A `.cart` file exported with `store_distributions=False` cannot
        recover full distributions when converted back to JSON/pickle —
        `predict_nbest` will return only the top class.
    Note: Converting to .skl requires the model to have been trained with sklearn.

    Args:
        input_path: Path to source model file.
        output_path: Path to output model file.
        input_format: Explicit format for the input file (e.g. ``"jsonl"``);
            bypasses extension detection. Use this when the file lives under
            a custom suffix like ``model.g2p.gz``.
        output_format: Explicit format for the output file; same semantics.

    Raises:
        ValueError: If the input is an IsolationForest export (use
            `IsolationForest` directly), if a `.cart` file has bad magic,
            if exporting to `.skl`/`.joblib` for a non-sklearn-trained model,
            or if the input/output format is unrecognized.
        ImportError: If `.skl`/`.joblib` is requested without `joblib`.

    Example:
        convert("model.json", "model.cart")  # JSON -> compact binary
        convert("model.cart", "model.json")  # binary -> JSON
        convert("model.g2p.gz", "model.cart", input_format="jsonl")
    """
    require_distinct_paths([input_path], [output_path])
    ext_out, _ = resolve_format(output_path, output_format)

    model = _load_conversion_model(input_path, input_format)

    if ext_out in (".skl", ".joblib") and model._sklearn_model is None:
        raise ValueError(
            "Cannot export to sklearn format. "
            "Model must be trained with trainer='sklearn' or loaded from .skl/.joblib."
        )

    model.export(output_path, format=output_format)
    return ConversionResult(type(model).__name__, input_path, output_path)


def _load_conversion_model(
    path: str, format: str | None
) -> DecisionTree | RandomForest:
    """Decode once, dispatch by artifact kind, then reuse model application."""
    ext, data = _read_model_artifact(path, format)
    if ext == ".cart":
        if data.get("is_xgboost"):
            raise ValueError("XGBoost artifacts require XGBoostTree or Predictor")
        model: DecisionTree | RandomForest = (
            RandomForest() if data.get("is_forest") else DecisionTree()
        )
        model._apply_cart_data(data)
        return model
    if ext in (".skl", ".joblib"):
        estimator = data
        from sklearn.ensemble import IsolationForest as SkIsolationForest

        if isinstance(estimator, SkIsolationForest):
            raise ValueError("IsolationForest conversion requires its dedicated API")
        model = RandomForest() if hasattr(estimator, "estimators_") else DecisionTree()
        model._apply_sklearn_model(estimator)
        return model
    if not isinstance(data, dict):
        raise ValueError("model must be an object")
    model = RandomForest() if "trees" in data else DecisionTree()
    model._apply_loaded_data(data)
    return model


def _detect_is_forest(path: str, format: str | None = None) -> bool:
    """
    Detect if a model file contains a (supervised) forest or a single tree.

    Args:
        path: Path to model file.
        format: Optional explicit format override (e.g. ``"jsonl"``) used when
            the file lives under a non-standard suffix.

    Returns:
        True if the model is a `RandomForest`, False if it is a `DecisionTree`.

    Raises:
        ValueError: If the file is an `IsolationForest` export (use
            `IsolationForest.load_model()` directly), if the `.cart` magic
            bytes are missing, or if the extension/format is unrecognized.
        ImportError: If `.skl`/`.joblib` is requested without `joblib`.
    """
    ext, _ = resolve_format(path, format)

    if ext == ".cart":
        # IsolationForest deliberately exposes no `_export_cart` and its
        # public `export()` rejects every extension except .json/.pkl, so
        # any `.cart` file with FLAG_IS_FOREST is by construction a
        # supervised RandomForest. No isolation-vs-supervised disambiguation
        # is needed here.
        with open_file_binary(path, "rb") as f:
            header = f.read(_HEADER_SIZE)
        if header[:4] != _MAGIC:
            raise ValueError(f"Invalid model file (missing CART magic): {path}")
        flags = struct.unpack_from("<H", header, _OFF_FLAGS)[0]
        return bool(flags & _FLAG_IS_FOREST)

    if ext in (".json", ".jsonl", ".pkl", ".pickle"):
        # Read just enough to classify
        if ext in (".pkl", ".pickle"):
            with open_file_binary(path, "rb") as f:
                data: dict = pickle.load(f)
        elif ext == ".jsonl":
            with open_file(path, "r") as f:
                data = json.loads(f.readline())
        else:
            with open_file(path, "r") as f:
                data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("model must be an object")
        # IsolationForest exports also use a top-level "trees" key, so check
        # the explicit marker before falling through to the supervised path.
        if data.get("isolation_forest"):
            raise ValueError(
                f"{path} is an IsolationForest export; use "
                "IsolationForest.load_model() / .export() directly."
            )
        return "trees" in data  # supervised forest, otherwise DecisionTree

    if ext in (".skl", ".joblib"):
        sklearn_model = require_joblib().load(path)
        # sklearn IsolationForest also exposes estimators_; rule it out first
        try:
            from sklearn.ensemble import IsolationForest as _SkIsolationForest
        except ImportError:  # joblib without sklearn is unusual but possible
            _SkIsolationForest = None
        if _SkIsolationForest is not None and isinstance(
            sklearn_model, _SkIsolationForest
        ):
            raise ValueError(
                f"{path} is a sklearn IsolationForest; use "
                "IsolationForest.load_model() / .export() directly."
            )
        return hasattr(sklearn_model, "estimators_")

    raise ValueError(f"Unknown format: {ext}")


__all__ = [
    # Core models
    "DecisionTree",
    "IsolationForest",
    "RandomForest",
    "XGBoostTree",
    # Types and type aliases
    "ClassificationLeaf",
    "DecisionNode",
    "FeatureSpec",
    "LeafNode",
    "RegressionLeaf",
    "TreeNode",
    # Trainers
    "Native",
    "Trainer",
    # Constants
    "CRITERION_ENTROPY",
    "CRITERION_GINI",
    "DEFAULT_MIN_DIST_ENTROPY",
    "DTYPE_BOOL",
    "DTYPE_FLOAT",
    "DTYPE_INT",
    "DTYPE_STR",
    "PROB_HIGH_CONFIDENCE",
    "TASK_AUTO",
    "TASK_CLASSIFICATION",
    "TASK_REGRESSION",
    "TYPE_CAT",
    "TYPE_NUM",
    # Evaluation
    "confusion_matrix",
    "cross_validate",
    "evaluate_predictions",
    "evaluate_tree",
    "per_class_metrics",
    "regression_metrics",
    # Tree utilities
    "count_leaves",
    "count_nodes",
    "max_depth",
    "tree_stats",
    # Inference runner
    "Predictor",
    "get_vocabulary",
    "is_oov",
    "load_model",
    "predict",
    "predict_batch",
    "read_cart_metadata",
    # Bundling and conversion
    "bundle",
    "convert",
    "ConversionResult",
    "TrainingResult",
    "TrainingSettings",
    "train_file",
    "train_model",
]

__version__ = "0.5.0"
