#!/usr/bin/env python
"""
Inference runner for cartlet models (.cart binary format).

This module provides lightweight inference for deployed models.

Usage:
    from cartlet.runner import load_model, predict, predict_batch

    model = load_model("model.cart")
    result = predict(model, ["red", "large"])
    results = predict_batch(model, [["red", "large"], ["blue", "small"]])

Or as standalone script:
    python runner.py model.cart '["feature1", "feature2", ...]'
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter
from typing import Any, cast

from .io.cart_format import (
    FEAT_MASK,
    FLAG_HAS_DISTRIBUTIONS,
    FLAG_IS_FOREST,
    FLAG_IS_REGRESSION,
    FLAG_IS_XGBOOST,
    HEADER_SIZE,
    INDEX_MASK,
    LEAF_CLASS,
    LEAF_CLASS_DIST,
    LEAF_FLAG,
    MAGIC,
    OP_EQ,
    OP_LT,
    OP_MASK,
    OP_SHIFT,
    OP_SWITCH,
    SIZE_DIST_ENTRY,
    SIZE_FEAT_HEADER,
    SIZE_LEAF,
    SIZE_U16,
    TYPE_MASK,
    VERSION,
    decode_varint,
)
from .types import (
    BINARY_CLASSIFICATION_THRESHOLD,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    CaseTable,
    ModelData,
)

# Safety limit to prevent stack overflow from maliciously crafted models
# This prevents infinite recursion in corrupted or adversarial tree structures
MAX_TREE_DEPTH = 10000

# .cart header sanity caps. These are intentionally well above practical limits
# (sklearn caps at a few thousand features/classes) so legitimate models always
# load, but reject obviously corrupted or adversarial header counts early.
_CART_MAX_FEATURES = 10_000
_CART_MAX_CLASSES = 100_000
_CART_MAX_TREES = 100_000
_CART_MAX_NODES = 10_000_000

__all__ = [
    "load_model",
    "predict",
    "predict_batch",
    "get_vocabulary",
    "is_oov",
    "Predictor",
]


def load_model(path: str) -> ModelData:
    """
    Load a trained model from a ``.cart`` file.

    This is the zero-dependency loader used by :class:`Predictor` and the
    standalone bundled runner: it only understands the compact ``.cart``
    binary format and pulls in no model classes.

    For ``.json`` / ``.jsonl`` / ``.pkl`` / ``.skl`` use
    :meth:`DecisionTree.load_model` (or :meth:`RandomForest.load_model`),
    which dispatches on extension at the cost of a full library import.

    Args:
        path: Path to model file (.cart).

    Returns:
        ModelData dict with model data for prediction.

    Raises:
        ValueError: If the model file is invalid or not in ``.cart`` format.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Transparently gunzip a gzipped model (e.g. .cart.gz), detected by the
    # gzip magic bytes. Keeps parity with the bundled runner, which does the
    # same, so the package Predictor is not a surprise downgrade.
    if data[:2] == b"\x1f\x8b":
        import gzip

        data = gzip.decompress(data)

    return cast(ModelData, _load_cart_from_bytes(data))


def _load_cart_from_bytes(data: bytes) -> dict[str, Any]:
    """Load model from bytes, return model dict."""
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"File too small ({len(data)} bytes), minimum header is {HEADER_SIZE} bytes"
        )

    try:
        pos = 0

        # Header
        magic = data[pos : pos + 4]
        pos += 4
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")

        version, flags, n_features, n_classes, n_trees = struct.unpack_from(
            "<HHHHH", data, pos
        )
        pos += 10

        if version != VERSION:
            raise ValueError(
                f"Unsupported format version {version} (expected {VERSION})"
            )

        (
            n_decisions,
            n_leaves,
            n_floats,
            n_cat_vals,
            n_dists,
            n_case_tables,
            metadata_len,
        ) = struct.unpack_from("<IIIHHHH", data, pos)
        pos += 20

        if n_features > _CART_MAX_FEATURES:
            raise ValueError(f"Unreasonable n_features: {n_features}")
        if n_classes > _CART_MAX_CLASSES:
            raise ValueError(f"Unreasonable n_classes: {n_classes}")
        if n_trees > _CART_MAX_TREES:
            raise ValueError(f"Unreasonable n_trees: {n_trees}")
        if n_decisions > _CART_MAX_NODES:
            raise ValueError(f"Unreasonable n_decisions: {n_decisions}")
        if n_leaves > _CART_MAX_NODES:
            raise ValueError(f"Unreasonable n_leaves: {n_leaves}")

        is_forest = bool(flags & FLAG_IS_FOREST)
        is_regression = bool(flags & FLAG_IS_REGRESSION)
        has_distributions = bool(flags & FLAG_HAS_DISTRIBUTIONS)
        is_xgboost = bool(flags & FLAG_IS_XGBOOST)

        # String table
        strings, pos = _parse_string_table(data, pos)

        # Feature table
        features, pos = _parse_feature_table(data, pos, n_features, strings)

        # Class table
        class_labels, pos = _parse_class_table(data, pos, n_classes, strings)

        # Float pool
        floats = list(struct.unpack_from(f"<{n_floats}f", data, pos))
        pos += 4 * n_floats

        # Cat value pool
        cat_vals = list(struct.unpack_from(f"<{n_cat_vals}H", data, pos))
        pos += SIZE_U16 * n_cat_vals

        # Tree offsets (varint encoded)
        tree_offsets = []
        for _ in range(n_trees):
            off, pos = decode_varint(data, pos)
            tree_offsets.append(off)

        # Decision nodes (variable size)
        decisions, pos = _parse_decision_nodes(data, pos, n_decisions)

        # Leaf nodes (3 bytes each - no padding)
        leaves, pos = _parse_leaf_nodes(data, pos, n_leaves)

        # Distributions (if FLAG_HAS_DISTRIBUTIONS)
        distributions, pos = _parse_distributions(data, pos, n_dists, has_distributions)

        # Case tables (for OP_SWITCH nodes)
        case_tables, pos = _parse_case_tables(data, pos, n_case_tables)

        # Trailing metadata blob (JSON, may carry XGBoost base_score etc.).
        # Length 0 is the common case for plain DecisionTree/RandomForest exports.
        metadata: dict[str, Any] = {}
        if metadata_len:
            meta_blob = data[pos : pos + metadata_len]
            pos += metadata_len
            try:
                metadata = json.loads(meta_blob.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Unparseable metadata never blocks inference — features/trees
                # were already parsed.
                metadata = {}

        return {
            "meta": {
                "features": features,
                "task": TASK_REGRESSION if is_regression else TASK_CLASSIFICATION,
                "metadata": metadata,
            },
            "class_labels": class_labels,
            "floats": floats,
            "cat_vals": cat_vals,
            "strings": strings,
            "decisions": decisions,
            "leaves": leaves,
            "distributions": distributions,
            "case_tables": case_tables,
            "tree_offsets": tree_offsets,
            "is_regression": is_regression,
            "is_forest": is_forest,
            "is_xgboost": is_xgboost,
            "has_distributions": has_distributions,
            "n_trees": n_trees if is_forest else 1,
            "version": version,
        }

    except struct.error as e:
        raise ValueError(f"Malformed .cart file: {e}") from e
    except IndexError as e:
        raise ValueError(f"Truncated .cart file: {e}") from e


def _parse_string_table(data: bytes, pos: int) -> tuple[list[str], int]:
    """Parse string table from binary data."""
    (n_strings,) = struct.unpack_from("<H", data, pos)
    pos += SIZE_U16
    string_offsets = struct.unpack_from(f"<{n_strings}H", data, pos)
    pos += SIZE_U16 * n_strings

    strings = []
    string_data_start = pos
    end = string_data_start
    for off in string_offsets:
        start = string_data_start + off
        term = data.index(b"\x00", start)
        strings.append(data[start:term].decode("utf-8"))
        # Advance past the null terminator using the byte position, not the
        # decoded character count (which is wrong for any non-ASCII string).
        if term + 1 > end:
            end = term + 1

    return strings, end


def _parse_feature_table(
    data: bytes, pos: int, n_features: int, strings: list[str]
) -> tuple[list[dict], int]:
    """Parse feature table from binary data."""
    features = []
    for _ in range(n_features):
        name_idx, type_flags, n_cat = struct.unpack_from("<HBB", data, pos)
        pos += SIZE_FEAT_HEADER
        cat_indices = list(struct.unpack_from(f"<{n_cat}H", data, pos))
        pos += SIZE_U16 * n_cat
        feat_type = "cat" if (type_flags & TYPE_MASK) == 0 else "num"
        features.append(
            {
                "name": strings[name_idx],
                "type": feat_type,
                "values": [strings[ci] for ci in cat_indices],
            }
        )
    return features, pos


def _parse_class_table(
    data: bytes, pos: int, n_classes: int, strings: list[str]
) -> tuple[list[str], int]:
    """Parse class table from binary data."""
    class_labels = []
    for _ in range(n_classes):
        (ci,) = struct.unpack_from("<H", data, pos)
        pos += SIZE_U16
        class_labels.append(strings[ci])
    return class_labels, pos


def _parse_decision_nodes(
    data: bytes, pos: int, n_decisions: int
) -> tuple[list[tuple], int]:
    """Parse decision nodes (variable size)."""
    decisions = []
    for _ in range(n_decisions):
        feat_op, val = struct.unpack_from("<BH", data, pos)
        pos += 3
        feat = feat_op & FEAT_MASK
        op = (feat_op & OP_MASK) >> OP_SHIFT
        if op == OP_SWITCH:
            decisions.append((feat, op, val, 0, 0))
        else:
            left, pos = decode_varint(data, pos)
            right, pos = decode_varint(data, pos)
            decisions.append((feat, op, val, left, right))
    return decisions, pos


def _parse_leaf_nodes(data: bytes, pos: int, n_leaves: int) -> tuple[list[tuple], int]:
    """Parse leaf nodes (3 bytes each)."""
    leaves = []
    for _ in range(n_leaves):
        leaf_type, val = struct.unpack_from("<BH", data, pos)
        pos += SIZE_LEAF
        leaves.append((leaf_type, val))
    return leaves, pos


def _parse_distributions(
    data: bytes, pos: int, n_dists: int, has_distributions: bool
) -> tuple[list[list[tuple[int, float]]], int]:
    """Parse probability distributions."""
    distributions: list[list[tuple[int, float]]] = []
    if has_distributions:
        for _ in range(n_dists):
            (n_entries,) = struct.unpack_from("<H", data, pos)
            pos += SIZE_U16
            dist: list[tuple[int, float]] = []
            for _ in range(n_entries):
                class_idx, prob = struct.unpack_from("<Hf", data, pos)
                pos += SIZE_DIST_ENTRY
                dist.append((class_idx, prob))
            distributions.append(dist)
    return distributions, pos


def _parse_case_tables(
    data: bytes, pos: int, n_case_tables: int
) -> tuple[list[dict], int]:
    """Parse case tables for OP_SWITCH nodes."""
    case_tables: list[dict[str, Any]] = []
    for _ in range(n_case_tables):
        (n_cases,) = struct.unpack_from("<H", data, pos)
        pos += SIZE_U16
        default_child, pos = decode_varint(data, pos)
        cases: list[tuple[int, int]] = []
        for _ in range(n_cases):
            (cat_val_idx,) = struct.unpack_from("<H", data, pos)
            pos += SIZE_U16
            child_idx, pos = decode_varint(data, pos)
            cases.append((cat_val_idx, child_idx))
        case_tables.append({"default": default_child, "cases": cases})
    return case_tables, pos


def predict(
    model: ModelData,
    vector: list[Any],
    return_dist: bool = False,
) -> Any:
    """
    Make a prediction using a loaded model.

    Args:
        model: Loaded model dict from load_model()
        vector: Feature vector (may contain None for missing values)
        return_dist: If True and model has distributions, return dict of class->prob

    Returns:
        Prediction value (class label, regression value, or distribution dict)

    Note:
        Missing values: When a feature value is None or the vector is shorter than
        expected, comparisons fail and the tree takes the "no" branch (right
        child); switch/case nodes take their default branch. This ensures
        deterministic behavior with incomplete inputs.
    """
    if model.get("is_xgboost"):
        return _predict_xgboost(model, vector, return_dist)
    if model["is_forest"]:
        return _predict_forest(model, vector, return_dist)
    return _predict_tree(model, vector, 0, return_dist)


def _predict_tree(
    model: ModelData, vector: list[Any], tree_idx: int, return_dist: bool = False
) -> Any:
    """Predict using a single tree."""
    return _predict_tree_recursive(
        model["tree_offsets"][tree_idx],
        vector,
        model["decisions"],
        model["leaves"],
        model["floats"],
        model["cat_vals"],
        model["strings"],
        model.get("distributions", []),
        model.get("case_tables", []),
        model["meta"]["features"],
        len(vector),
        return_dist,
    )


def _predict_tree_recursive(
    idx: int,
    vector: list[Any],
    decisions: list[tuple[int, int, int, int, int]],
    leaves: list[tuple[int, int]],
    floats: list[float],
    cat_vals: list[int],
    strings: list[str],
    distributions: list[list[tuple[int, float]]],
    case_tables: list[CaseTable],
    features: list[Any],
    n_input_features: int,
    return_dist: bool = False,
) -> Any:
    """Core tree traversal logic."""
    for _ in range(MAX_TREE_DEPTH):
        # Check if leaf (high bit set)
        if idx & LEAF_FLAG:
            leaf_idx = idx & INDEX_MASK
            if leaf_idx >= len(leaves):
                raise RuntimeError(f"Invalid leaf index: {leaf_idx}")
            leaf_type, val = leaves[leaf_idx]
            if leaf_type == LEAF_CLASS:
                if val >= len(strings):
                    raise RuntimeError(f"Invalid string index in leaf: {val}")
                return strings[val]
            elif leaf_type == LEAF_CLASS_DIST:
                # Leaf with distribution
                if val >= len(distributions):
                    raise RuntimeError(f"Invalid distribution index in leaf: {val}")
                dist_data = distributions[val]
                if return_dist:
                    return {strings[ci]: prob for ci, prob in dist_data}
                if not dist_data or dist_data[0][0] >= len(strings):
                    raise RuntimeError("Invalid class index in distribution")
                return strings[dist_data[0][0]]
            else:  # LEAF_FLOAT
                if val >= len(floats):
                    raise RuntimeError(f"Invalid float index in leaf: {val}")
                return floats[val]

        # Decision node
        if idx >= len(decisions):
            raise RuntimeError(f"Invalid decision index: {idx}")
        feat, op, val, left, right = decisions[idx]

        # A feature index past the end of the input vector is treated the same
        # as an explicit missing (None) value: numeric/categorical comparisons
        # fail (go right) and switch nodes take their default branch. This keeps
        # the two runners in sync and fixes switch nodes, whose right placeholder
        # is 0 (which would otherwise jump traversal to decision node 0).
        feat_val = None if feat >= n_input_features else vector[feat]

        if op == OP_LT:
            # Numeric comparison. Coerce the value to float regardless of the
            # declared feature type; a non-numeric value at a numeric node
            # fails the comparison and goes right (matches the bundled runner).
            if val >= len(floats):
                raise RuntimeError(f"Invalid float index in decision: {val}")
            threshold = floats[val]
            go_left = False
            if feat_val is not None:
                try:
                    go_left = float(feat_val) <= threshold
                except (TypeError, ValueError):
                    go_left = False
            idx = left if go_left else right
        elif op == OP_EQ:
            # Categorical comparison
            if val >= len(cat_vals):
                raise RuntimeError(f"Invalid cat_val index in decision: {val}")
            cat_idx = cat_vals[val]
            if cat_idx >= len(strings):
                raise RuntimeError(f"Invalid string index in cat_vals: {cat_idx}")
            cat_str = strings[cat_idx]
            go_left = False if feat_val is None else (str(feat_val) == cat_str)
            idx = left if go_left else right
        elif op == OP_SWITCH:
            # Case table lookup
            if val >= len(case_tables):
                raise RuntimeError(f"Invalid case_table index in decision: {val}")
            table = case_tables[val]
            idx = table["default"]
            if feat_val is not None:
                feat_str = str(feat_val)
                for cat_val_idx, child_idx in table["cases"]:
                    if cat_val_idx >= len(cat_vals):
                        continue
                    actual_cat_idx = cat_vals[cat_val_idx]
                    if actual_cat_idx >= len(strings):
                        continue
                    if strings[actual_cat_idx] == feat_str:
                        idx = child_idx
                        break

    raise RuntimeError("Max tree depth exceeded (possible corrupted model)")


def _predict_forest(
    model: ModelData, vector: list[Any], return_dist: bool = False
) -> Any:
    """Predict using forest (majority vote or mean)."""
    decisions = model["decisions"]
    leaves = model["leaves"]
    floats = model["floats"]
    cat_vals = model["cat_vals"]
    strings = model["strings"]
    distributions = model.get("distributions", [])
    case_tables = model.get("case_tables", [])
    features = model["meta"]["features"]
    n_input_features = len(vector)

    predictions = [
        _predict_tree_recursive(
            model["tree_offsets"][i],
            vector,
            decisions,
            leaves,
            floats,
            cat_vals,
            strings,
            distributions,
            case_tables,
            features,
            n_input_features,
            return_dist,
        )
        for i in range(model["n_trees"])
    ]

    if model["is_regression"]:
        return sum(predictions) / len(predictions)

    if return_dist:
        # Aggregate distributions
        combined: dict[Any, float] = {}
        for pred in predictions:
            if isinstance(pred, dict):
                for cls, prob in pred.items():
                    combined[cls] = combined.get(cls, 0.0) + prob
            else:
                combined[pred] = combined.get(pred, 0.0) + 1.0
        total = sum(combined.values())
        return {cls: count / total for cls, count in combined.items()}

    # Majority vote
    return Counter(predictions).most_common(1)[0][0]


def _sigmoid(x: float) -> float:
    """Sigmoid activation function (numerically stable for both branches)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _softmax(scores: list[float]) -> list[float]:
    """Softmax activation, return list of probabilities."""
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def _predict_xgboost(
    model: ModelData, vector: list[Any], return_dist: bool = False
) -> Any:
    """
    XGBoost prediction: additive model with sigmoid/softmax.

    For binary classification:
      raw_score = base_score + sum(tree outputs)
      probability = sigmoid(raw_score)

    For multiclass (K classes, K trees per round):
      raw_scores[k] = base_score + sum(trees for class k)
      probabilities = softmax(raw_scores)
    """
    n_trees = model["n_trees"]
    n_classes = len(model["class_labels"])
    class_labels = model["class_labels"]
    # base_score is the additive offset the trained booster used; for binary
    # classification XGBoost stores it in probability space, so we round-trip
    # through the inverse logit so the addition lives in raw-score space and
    # matches XGBoostTree.predict.
    meta_dict = cast(dict, model.get("meta", {}))
    metadata_blob = cast(dict, meta_dict.get("metadata", {}))
    raw_base_score = float(metadata_blob.get("base_score", 0.0))
    if not model["is_regression"] and n_classes == 2 and 0.0 < raw_base_score < 1.0:
        base_score = math.log(raw_base_score / (1.0 - raw_base_score))
    else:
        base_score = raw_base_score

    decisions = model["decisions"]
    leaves = model["leaves"]
    floats = model["floats"]
    cat_vals = model["cat_vals"]
    strings = model["strings"]
    distributions = model.get("distributions", [])
    case_tables = model.get("case_tables", [])
    features = model["meta"]["features"]
    n_input_features = len(vector)

    def _eval_one(t_idx: int) -> Any:
        return _predict_tree_recursive(
            model["tree_offsets"][t_idx],
            vector,
            decisions,
            leaves,
            floats,
            cat_vals,
            strings,
            distributions,
            case_tables,
            features,
            n_input_features,
        )

    if model["is_regression"]:
        raw_score = base_score
        for i in range(n_trees):
            raw_score += _eval_one(i)
        return raw_score

    if n_classes == 2:
        # Binary classification
        raw_score = base_score
        for i in range(n_trees):
            raw_score += _eval_one(i)
        prob = _sigmoid(raw_score)
        if return_dist:
            return {class_labels[0]: 1 - prob, class_labels[1]: prob}
        return (
            class_labels[1]
            if prob > BINARY_CLASSIFICATION_THRESHOLD
            else class_labels[0]
        )

    # Multiclass: K trees per round
    n_rounds = n_trees // n_classes
    scores = [base_score] * n_classes

    for round_idx in range(n_rounds):
        for class_idx in range(n_classes):
            tree_idx = round_idx * n_classes + class_idx
            scores[class_idx] += _eval_one(tree_idx)

    probs = _softmax(scores)
    if return_dist:
        return {class_labels[i]: probs[i] for i in range(n_classes)}

    best_idx = probs.index(max(probs))
    return class_labels[best_idx]


def predict_batch(
    model: ModelData,
    vectors: list[list[Any]],
    return_dist: bool = False,
) -> list[Any]:
    """
    Make predictions for multiple feature vectors.

    Args:
        model: Loaded model dict from load_model()
        vectors: List of feature vectors
        return_dist: If True and model has distributions, return dict of class->prob

    Returns:
        List of predictions
    """
    return [predict(model, v, return_dist=return_dist) for v in vectors]


def read_cart_metadata(source: str | bytes) -> dict[str, Any]:
    """
    Return the embedded metadata dict from a ``.cart`` file or bytes blob
    without keeping the full parsed model around.

    Args:
        source: Path to a ``.cart`` file, or raw ``.cart`` bytes.

    Returns:
        The metadata JSON object (possibly empty); raises ``ValueError`` on a
        malformed or non-``.cart`` input.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as f:
            data = f.read()
    model = _load_cart_from_bytes(data)
    return cast(dict, model.get("meta", {}).get("metadata", {}))


def get_vocabulary(model: ModelData, feature: int | str) -> set[Any] | None:
    """
    Get the vocabulary (known values) for a categorical feature.

    Args:
        model: Loaded model dict from load_model()
        feature: Feature index (int) or name (str)

    Returns:
        Set of known values, or None if not categorical
    """
    features = model.get("meta", {}).get("features", [])
    for i, feat in enumerate(features):
        if (isinstance(feature, int) and i == feature) or feat.get("name") == feature:
            if feat.get("type") == "cat" and "values" in feat:
                return set(feat["values"])
            return None
    return None


def is_oov(model: ModelData, feature: int | str, value: Any) -> bool:
    """
    Check if a value is out-of-vocabulary for a feature.

    Args:
        model: Loaded model dict from load_model()
        feature: Feature index (int) or name (str)
        value: Value to check

    Returns:
        True if value was not seen during training, False otherwise.
    """
    vocab = get_vocabulary(model, feature)
    if vocab is None:
        return False
    return value not in vocab


class Predictor:
    """
    Object-oriented wrapper for model inference.

    Examples:
        p = Predictor("model.cart")
        print(p.predict(["red", "large"]))
    """

    def __init__(self, model_source: str | bytes | ModelData):
        """
        Initialize predictor from file path, bytes, or ModelData.

        Args:
            model_source: Path to .cart file, bytes of .cart file, or ModelData dict
        """
        if isinstance(model_source, str):
            self.model = load_model(model_source)
        elif isinstance(model_source, bytes):
            self.model = cast(ModelData, _load_cart_from_bytes(model_source))
        else:
            self.model = model_source

    def predict(self, vector: list[Any], return_dist: bool = False) -> Any:
        """Make a prediction for a single feature vector."""
        return predict(self.model, vector, return_dist=return_dist)

    def predict_batch(
        self, vectors: list[list[Any]], return_dist: bool = False
    ) -> list[Any]:
        """Make predictions for multiple feature vectors."""
        return predict_batch(self.model, vectors, return_dist=return_dist)

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f["name"] for f in self.model["meta"].get("features", [])]

    @property
    def class_labels(self) -> list[str]:
        """Get list of class labels (classification only)."""
        return self.model.get("class_labels", [])

    @property
    def task(self) -> str:
        """Get model task (classification/regression)."""
        return self.model["meta"].get("task", TASK_CLASSIFICATION)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Embedded metadata dict from the model trailer. Includes whatever the
        exporter chose to persist (e.g. ``locale``, ``width``, ``cased``,
        ``join``, ``exceptions``, ``training_config``, XGBoost ``base_score``).
        Returns an empty dict when the model carries no metadata.
        """
        meta = cast(dict, self.model.get("meta", {}))
        return cast(dict, meta.get("metadata", {}))

    def get_vocabulary(self, feature: int | str) -> set[Any] | None:
        """
        Return the known values for a categorical feature, or None if the
        feature is numerical or unknown. Convenience wrapper around the
        module-level :func:`get_vocabulary`.
        """
        return get_vocabulary(self.model, feature)

    def is_oov(self, feature: int | str, value: Any) -> bool:
        """
        True if ``value`` was not seen for ``feature`` during training. Always
        returns False for numerical or unknown features. Convenience wrapper
        around the module-level :func:`is_oov`.
        """
        return is_oov(self.model, feature, value)

    def __repr__(self) -> str:
        n_trees = self.model.get("n_trees", 1)
        return f"Predictor(task={self.task}, n_trees={n_trees})"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} model.cart \'["feature1", "feature2", ...]\'')
        sys.exit(1)

    model_path = sys.argv[1]
    vector = json.loads(sys.argv[2])

    model = load_model(model_path)
    result = predict(model, vector)
    print(result)
