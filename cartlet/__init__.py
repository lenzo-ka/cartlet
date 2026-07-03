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

from .evaluation import (
    confusion_matrix,
    cross_validate,
    evaluate_predictions,
    evaluate_tree,
    per_class_metrics,
)
from .forest import RandomForest
from .io import open_file, open_file_binary
from .io.bytes import bundle
from .io.cart_format import FLAG_IS_FOREST, HEADER_SIZE, MAGIC, OFF_FLAGS
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
from .xgboost import XGBoostTree


def convert(
    input_path: str,
    output_path: str,
    *,
    input_format: str | None = None,
    output_format: str | None = None,
) -> None:
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
    ext_out, _ = resolve_format(output_path, output_format)

    is_forest = _detect_is_forest(input_path, format=input_format)
    model: DecisionTree | RandomForest = RandomForest() if is_forest else DecisionTree()
    model.load_model(input_path, format=input_format)

    if ext_out in (".skl", ".joblib") and model._sklearn_model is None:
        raise ValueError(
            "Cannot export to sklearn format. "
            "Model must be trained with trainer='sklearn' or loaded from .skl/.joblib."
        )

    model.export(output_path, format=output_format)


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
            header = f.read(HEADER_SIZE)
        if header[:4] != MAGIC:
            raise ValueError(f"Invalid model file (missing CART magic): {path}")
        flags = struct.unpack_from("<H", header, OFF_FLAGS)[0]
        return bool(flags & FLAG_IS_FOREST)

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
]

__version__ = "0.5.0"
