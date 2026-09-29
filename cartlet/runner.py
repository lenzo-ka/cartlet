#!/usr/bin/env python
"""
Inference runner for cartlet models (.cart binary format).

This module provides lightweight inference for deployed models.

Usage:
    from cartlet.runner import load_model, predict, predict_batch

    model = load_model("model.cart")
    result = predict(model, ["red", "large"])
    results = predict_batch(model, [["red", "large"], ["blue", "small"]])

Or as a module (relative imports require -m, not `python runner.py`):
    python -m cartlet.runner model.cart '["feature1", "feature2", ...]'
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter
from typing import Any, cast

from .io.cart_format import (
    CATEGORY_SET,
    DECISION_FLAGS_MASK,
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
    MISSING_LEFT,
    MISSING_MASK,
    MISSING_NONE,
    MISSING_RIGHT,
    OP_EQ,
    OP_LE,
    OP_LT,
    OP_MASK,
    OP_SHIFT,
    OP_SWITCH,
    SIZE_DECISION_HEADER,
    SIZE_DIST_ENTRY,
    SIZE_F64,
    SIZE_FEAT_HEADER,
    SIZE_HEADER_COUNTS1,
    SIZE_HEADER_COUNTS2,
    SIZE_LEAF,
    SIZE_U16,
    TYPE_MASK,
    VERSION,
    decode_feature_dtype,
    decode_feature_value,
    decode_varint,
)
from .types import (
    BINARY_CLASSIFICATION_THRESHOLD,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    CaseTable,
    ModelData,
    normalize_bool,
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
    "MissingFeatureError",
    "load_model",
    "predict",
    "predict_path",
    "predict_batch",
    "get_vocabulary",
    "is_oov",
    "Predictor",
]


class MissingFeatureError(ValueError):
    """A decision could not be made because its tested feature is missing."""


def _check_missing_policy(missing: str) -> None:
    if missing not in ("error", "right"):
        raise ValueError("missing must be 'error' or 'right'")


def _is_self_unequal(value: Any) -> bool:
    """Return whether a non-string scalar has a trustworthy ``x != x``."""
    if isinstance(value, str):
        return False
    try:
        result = value != value
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    result_type = type(result)
    if result_type.__module__.split(".", 1)[0] == "numpy" and result_type.__name__ in (
        "bool",
        "bool_",
    ):
        try:
            return bool(result)
        except Exception:
            return False
    return False


def _is_categorical_missing(value: Any) -> bool:
    """Return whether a value is missing at equality or switch nodes."""
    return value is None or _is_self_unequal(value)


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
    """Parse ``.cart`` bytes into the model dict consumed by ``predict``.

    Returned keys:
      - ``meta``: ``{"features": [...], "task": str, "metadata": dict}`` where
        each feature is ``{"name", "type", "values"}``.
      - ``class_labels``: list[str] (classification).
      - ``floats`` / ``cat_vals`` / ``strings``: the deduped value pools.
      - ``decisions`` / ``leaves`` / ``distributions`` / ``case_tables`` /
        ``category_sets``: the flat node tables. Case tables carry both raw
        ``cases`` and a resolved ``lookup``; category sets are resolved to
        strings once at load time.
      - ``tree_offsets``: per-tree start index into ``decisions``/``leaves``.
      - ``is_regression`` / ``is_forest`` / ``is_xgboost`` /
        ``has_distributions``: bool flags; ``n_trees``: int; ``version``: int.

    Note: the bundled runner (``cartlet/bundled/predict.py``) returns a
    deliberately flatter dict (feature list at the top level, not under
    ``meta``); the two are kept behaviourally in lockstep but not
    key-for-key identical.
    """
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
        pos += SIZE_HEADER_COUNTS1

        if version == 2:
            raise ValueError(
                "Unsupported format version 2: format 2 was written by Cartlet "
                "0.6.0; re-export from the training model or retrain"
            )
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
            n_category_sets,
            metadata_len,
        ) = struct.unpack_from("<IIIHHHHH", data, pos)
        pos += SIZE_HEADER_COUNTS2

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

        # Reject internally-inconsistent headers rather than silently
        # misparsing. A positive n_dists with the distribution flag clear would
        # desync the case-table offset (W1-L7); n_trees > 1 with neither the
        # forest nor xgboost flag set would make runners disagree on whether to
        # use tree 0 or aggregate (W1-L8).
        if n_dists > 0 and not has_distributions:
            raise ValueError(
                "Inconsistent .cart header: n_dists > 0 but "
                "FLAG_HAS_DISTRIBUTIONS is clear"
            )
        if n_trees > 1 and not is_forest and not is_xgboost:
            raise ValueError(
                "Inconsistent .cart header: n_trees > 1 but neither "
                "FLAG_IS_FOREST nor FLAG_IS_XGBOOST is set"
            )

        # String table
        strings, pos = _parse_string_table(data, pos)

        # Feature table
        features, pos = _parse_feature_table(data, pos, n_features, strings)

        # Class table
        class_labels, pos = _parse_class_table(data, pos, n_classes, strings)

        # Float pool
        floats = list(struct.unpack_from(f"<{n_floats}d", data, pos))
        pos += SIZE_F64 * n_floats

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
        case_tables, pos = _parse_case_tables(
            data, pos, n_case_tables, cat_vals, strings
        )

        category_sets, pos = _parse_category_sets(
            data, pos, n_category_sets, cat_vals, strings
        )

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
            "category_sets": category_sets,
            "bool_features": [feature["dtype"] == "bool" for feature in features],
            "tree_offsets": tree_offsets,
            "is_regression": is_regression,
            "is_forest": is_forest,
            "is_xgboost": is_xgboost,
            "has_distributions": has_distributions,
            "n_trees": n_trees,
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
        dtype = decode_feature_dtype(type_flags)
        features.append(
            {
                "name": strings[name_idx],
                "type": feat_type,
                "dtype": dtype,
                "values": [
                    decode_feature_value(strings[ci], dtype) for ci in cat_indices
                ],
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
        feat_op, missing_flags, val = struct.unpack_from("<BBH", data, pos)
        pos += SIZE_DECISION_HEADER
        feat = feat_op & FEAT_MASK
        op = (feat_op & OP_MASK) >> OP_SHIFT
        if missing_flags & ~DECISION_FLAGS_MASK:
            raise ValueError(f"Invalid decision flags: {missing_flags}")
        missing_direction = missing_flags & MISSING_MASK
        if missing_direction not in (MISSING_NONE, MISSING_LEFT, MISSING_RIGHT):
            raise ValueError(f"Invalid decision flags: {missing_flags}")
        if missing_flags & CATEGORY_SET and op != OP_EQ:
            raise ValueError("CATEGORY_SET flag requires OP_EQ")
        if op == OP_SWITCH:
            decisions.append((feat, op, missing_flags, val, 0, 0))
        else:
            left, pos = decode_varint(data, pos)
            right, pos = decode_varint(data, pos)
            decisions.append((feat, op, missing_flags, val, left, right))
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
                class_idx, prob = struct.unpack_from("<Hd", data, pos)
                pos += SIZE_DIST_ENTRY
                dist.append((class_idx, prob))
            distributions.append(dist)
    return distributions, pos


def _parse_case_tables(
    data: bytes,
    pos: int,
    n_case_tables: int,
    cat_vals: list[int],
    strings: list[str],
) -> tuple[list[dict], int]:
    """Parse case tables for OP_SWITCH nodes.

    Each table's cases are resolved once to a ``{category_string: child_idx}``
    lookup so switch traversal is an O(1) dict get per node instead of a linear
    scan (with per-entry ``cat_vals``/``strings`` dereferences) on every
    prediction row. First-match-wins is preserved via ``setdefault``.
    """
    case_tables: list[dict[str, Any]] = []
    for _ in range(n_case_tables):
        (n_cases,) = struct.unpack_from("<H", data, pos)
        pos += SIZE_U16
        default_child, pos = decode_varint(data, pos)
        cases: list[tuple[int, int]] = []
        lookup: dict[str, int] = {}
        for _ in range(n_cases):
            (cat_val_idx,) = struct.unpack_from("<H", data, pos)
            pos += SIZE_U16
            child_idx, pos = decode_varint(data, pos)
            cases.append((cat_val_idx, child_idx))
            if cat_val_idx >= len(cat_vals):
                continue
            actual_cat_idx = cat_vals[cat_val_idx]
            if actual_cat_idx >= len(strings):
                continue
            lookup.setdefault(strings[actual_cat_idx], child_idx)
        # "cases" retains the raw (cat_val_idx, child_idx) pairs for tree
        # rebuild (cart_format.rebuild_tree_from_cart); "lookup" is the resolved
        # O(1) prediction path.
        case_tables.append({"default": default_child, "cases": cases, "lookup": lookup})
    return case_tables, pos


def _parse_category_sets(
    data: bytes,
    pos: int,
    n_category_sets: int,
    cat_vals: list[int],
    strings: list[str],
) -> tuple[list[set[str]], int]:
    """Parse category sets and resolve each to strings once for O(1) lookup."""
    category_sets: list[set[str]] = []
    for _ in range(n_category_sets):
        (n_values,) = struct.unpack_from("<H", data, pos)
        pos += SIZE_U16
        values: set[str] = set()
        for _ in range(n_values):
            (cat_val_idx,) = struct.unpack_from("<H", data, pos)
            pos += SIZE_U16
            if cat_val_idx >= len(cat_vals) or cat_vals[cat_val_idx] >= len(strings):
                raise ValueError("Invalid categorical value index in category set")
            values.add(strings[cat_vals[cat_val_idx]])
        category_sets.append(values)
    return category_sets, pos


def predict(
    model: ModelData,
    vector: list[Any],
    return_dist: bool = False,
    *,
    missing: str = "error",
) -> Any:
    """
    Make a prediction using a loaded model.

    Args:
        model: Loaded model dict from load_model()
        vector: Feature vector
        return_dist: If True and model has distributions, return dict of class->prob
        missing: ``"error"`` (default) raises when a tested value is absent,
            ``None``, float-convertible to NaN at a numeric node, or a
            non-string self-unequal scalar at equality/switch nodes. ``"right"``
            routes it right (or to a switch default). Unlike 0.6.0, a
            non-string NaN does not match an equality or switch key ``"nan"``.

    Returns:
        Prediction value (class label, regression value, or distribution dict)

    Only features tested on evaluated paths are read. Caller indexing
    exceptions propagate unchanged.
    """
    _check_missing_policy(missing)
    if model.get("is_xgboost"):
        return _predict_xgboost(model, vector, return_dist, missing=missing)
    if model["is_forest"]:
        return _predict_forest(model, vector, return_dist, missing=missing)
    return _predict_tree(model, vector, 0, return_dist, missing=missing)


def _predict_tree(
    model: ModelData,
    vector: list[Any],
    tree_idx: int,
    return_dist: bool = False,
    *,
    missing: str = "error",
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
        model.get("category_sets", []),
        model["meta"]["features"],
        model["bool_features"],
        len(vector),
        return_dist,
        missing,
        tree_idx,
        False,
    )


def _predict_tree_recursive(
    idx: int,
    vector: list[Any],
    decisions: list[tuple[int, int, int, int, int, int]],
    leaves: list[tuple[int, int]],
    floats: list[float],
    cat_vals: list[int],
    strings: list[str],
    distributions: list[list[tuple[int, float]]],
    case_tables: list[CaseTable],
    category_sets: list[set[str]],
    features: list[Any],
    bool_features: list[bool],
    n_input_features: int,
    return_dist: bool = False,
    missing: str = "error",
    tree_idx: int = 0,
    xgboost: bool = False,
    path: list[dict[str, Any]] | None = None,
) -> Any:
    """Hot tree traversal; path collection uses a separate implementation."""
    if path is not None or xgboost:
        return _predict_tree_path_recursive(
            idx,
            vector,
            decisions,
            leaves,
            floats,
            cat_vals,
            strings,
            distributions,
            case_tables,
            category_sets,
            features,
            bool_features,
            n_input_features,
            return_dist,
            missing,
            tree_idx,
            xgboost,
            path,
        )
    for _ in range(MAX_TREE_DEPTH):
        if idx & LEAF_FLAG:
            leaf_idx = idx & INDEX_MASK
            if leaf_idx >= len(leaves):
                raise RuntimeError(f"Invalid leaf index: {leaf_idx}")
            leaf_type, val = leaves[leaf_idx]
            if leaf_type == LEAF_CLASS:
                if val >= len(strings):
                    raise RuntimeError(f"Invalid string index in leaf: {val}")
                label = strings[val]
                return {label: 1.0} if return_dist else label
            if leaf_type == LEAF_CLASS_DIST:
                if val >= len(distributions):
                    raise RuntimeError(f"Invalid distribution index in leaf: {val}")
                dist_data = distributions[val]
                if return_dist:
                    return {strings[ci]: prob for ci, prob in dist_data}
                if not dist_data or dist_data[0][0] >= len(strings):
                    raise RuntimeError("Invalid class index in distribution")
                return strings[dist_data[0][0]]
            if val >= len(floats):
                raise RuntimeError(f"Invalid float index in leaf: {val}")
            return floats[val]

        if idx >= len(decisions):
            raise RuntimeError(f"Invalid decision index: {idx}")
        feat, op, missing_flags, val, left, right = decisions[idx]
        feat_val = None if feat >= n_input_features else vector[feat]
        numeric_value = None
        if op in (OP_LE, OP_LT):
            is_missing = feat_val is None
            if not is_missing:
                try:
                    numeric_value = float(feat_val)
                except OverflowError:
                    if not bool_features[feat]:
                        raise
                except (TypeError, ValueError):
                    pass
                else:
                    is_missing = math.isnan(numeric_value)
        else:
            is_missing = _is_categorical_missing(feat_val)
        missing_direction = missing_flags & MISSING_MASK
        learned_missing = is_missing and missing_direction != MISSING_NONE
        if is_missing and not learned_missing and missing == "error":
            feature_name = features[feat].get("name", str(feat))
            raise MissingFeatureError(
                f"feature {feat} ({feature_name!r}) is missing at "
                f"tree {tree_idx} node {idx}"
            )
        if not is_missing and bool_features[feat]:
            feat_val = normalize_bool(feat_val)
            if op in (OP_LE, OP_LT):
                numeric_value = float(feat_val)

        if op in (OP_LE, OP_LT):
            if val >= len(floats):
                raise RuntimeError(f"Invalid float index in decision: {val}")
            threshold = floats[val]
            go_left = learned_missing and missing_direction == MISSING_LEFT
            if not is_missing and numeric_value is not None:
                go_left = (
                    numeric_value < threshold
                    if op == OP_LT
                    else numeric_value <= threshold
                )
            idx = left if go_left else right
        elif op == OP_EQ:
            is_category_set = bool(missing_flags & CATEGORY_SET)
            predicate: str | set[str]
            if is_category_set:
                if val >= len(category_sets):
                    raise RuntimeError(f"Invalid category_set index in decision: {val}")
                predicate = category_sets[val]
            else:
                if val >= len(cat_vals):
                    raise RuntimeError(f"Invalid cat_val index in decision: {val}")
                cat_idx = cat_vals[val]
                if cat_idx >= len(strings):
                    raise RuntimeError(f"Invalid string index in cat_vals: {cat_idx}")
                predicate = strings[cat_idx]
            go_left = (
                missing_direction == MISSING_LEFT
                if learned_missing
                else not is_missing
                and (
                    str(feat_val) in predicate
                    if is_category_set
                    else str(feat_val) == predicate
                )
            )
            idx = left if go_left else right
        elif op == OP_SWITCH:
            if val >= len(case_tables):
                raise RuntimeError(f"Invalid case_table index in decision: {val}")
            table = case_tables[val]
            if learned_missing and missing_direction == MISSING_LEFT:
                idx = table["cases"][0][1]
            elif is_missing:
                idx = table["default"]
            else:
                idx = table["lookup"].get(str(feat_val), table["default"])
    raise RuntimeError("Max tree depth exceeded (possible corrupted model)")


def _predict_tree_path_recursive(
    idx: int,
    vector: list[Any],
    decisions: list[tuple[int, int, int, int, int, int]],
    leaves: list[tuple[int, int]],
    floats: list[float],
    cat_vals: list[int],
    strings: list[str],
    distributions: list[list[tuple[int, float]]],
    case_tables: list[CaseTable],
    category_sets: list[set[str]],
    features: list[Any],
    bool_features: list[bool],
    n_input_features: int,
    return_dist: bool = False,
    missing: str = "error",
    tree_idx: int = 0,
    xgboost: bool = False,
    path: list[dict[str, Any]] | None = None,
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
                label = strings[val]
                result: Any = {label: 1.0} if return_dist else label
                return (result, leaf_idx) if path is not None else result
            elif leaf_type == LEAF_CLASS_DIST:
                # Leaf with distribution
                if val >= len(distributions):
                    raise RuntimeError(f"Invalid distribution index in leaf: {val}")
                dist_data = distributions[val]
                if return_dist:
                    result = {strings[ci]: prob for ci, prob in dist_data}
                    return (result, leaf_idx) if path is not None else result
                if not dist_data or dist_data[0][0] >= len(strings):
                    raise RuntimeError("Invalid class index in distribution")
                result = strings[dist_data[0][0]]
                return (result, leaf_idx) if path is not None else result
            else:  # LEAF_FLOAT
                if val >= len(floats):
                    raise RuntimeError(f"Invalid float index in leaf: {val}")
                result = floats[val]
                return (result, leaf_idx) if path is not None else result

        # Decision node
        if idx >= len(decisions):
            raise RuntimeError(f"Invalid decision index: {idx}")
        feat, op, missing_flags, val, left, right = decisions[idx]

        # A feature index past the input is missing. Numeric nodes also treat a
        # successful float conversion to NaN as missing; categorical nodes use
        # conservative scalar self-inequality. The "right" policy routes missing
        # comparisons right and switches to their default child.
        # Keep caller indexing outside conversion/comparison handlers: exceptions
        # raised by a lazy vector must propagate unchanged.
        feat_val = None if feat >= n_input_features else vector[feat]
        numeric_value = None
        if op in (OP_LE, OP_LT):
            is_missing = feat_val is None
            if not is_missing:
                try:
                    numeric_value = float(feat_val)
                except OverflowError:
                    if not bool_features[feat]:
                        raise
                except (TypeError, ValueError):
                    pass
                else:
                    is_missing = math.isnan(numeric_value)
        else:
            is_missing = _is_categorical_missing(feat_val)
        missing_direction = missing_flags & MISSING_MASK
        learned_missing = is_missing and missing_direction != MISSING_NONE
        if is_missing and not learned_missing and missing == "error":
            feature_name = features[feat].get("name", str(feat))
            raise MissingFeatureError(
                f"feature {feat} ({feature_name!r}) is missing at "
                f"tree {tree_idx} node {idx}"
            )
        if not is_missing and bool_features[feat]:
            feat_val = normalize_bool(feat_val)
            if op in (OP_LE, OP_LT):
                numeric_value = float(feat_val)
        if xgboost and not is_missing:
            numeric = numeric_value
            if numeric is None:
                try:
                    numeric = float(feat_val)
                except (TypeError, ValueError):
                    numeric = None
            if numeric is not None:
                try:
                    feat_val = struct.unpack("<f", struct.pack("<f", numeric))[0]
                except OverflowError as e:
                    raise ValueError(
                        f"XGBoost input {feat_val!r} exceeds float32 range"
                    ) from e
                if op in (OP_LE, OP_LT):
                    numeric_value = feat_val

        if op in (OP_LE, OP_LT):
            # Numeric comparison. Coerce the value to float regardless of the
            # declared feature type; a non-numeric value at a numeric node
            # fails the comparison and goes right (matches the bundled runner).
            if val >= len(floats):
                raise RuntimeError(f"Invalid float index in decision: {val}")
            threshold = floats[val]
            go_left = learned_missing and missing_direction == MISSING_LEFT
            if not is_missing and numeric_value is not None:
                go_left = (
                    (numeric_value < threshold)
                    if op == OP_LT
                    else (numeric_value <= threshold)
                )
            branch = "left" if go_left else "right"
            if path is not None:
                step = {
                    "node": idx,
                    "feature": feat,
                    "name": features[feat].get("name", str(feat)),
                    "op": "<" if op == OP_LT else "<=",
                    "value": threshold,
                    "branch": branch,
                }
                if learned_missing:
                    step["missing"] = True
                path.append(step)
            idx = left if go_left else right
        elif op == OP_EQ:
            # Categorical comparison
            is_category_set = bool(missing_flags & CATEGORY_SET)
            predicate: str | set[str]
            if is_category_set:
                if val >= len(category_sets):
                    raise RuntimeError(f"Invalid category_set index in decision: {val}")
                predicate = category_sets[val]
                path_value: str | list[str] = sorted(predicate)
            else:
                if val >= len(cat_vals):
                    raise RuntimeError(f"Invalid cat_val index in decision: {val}")
                cat_idx = cat_vals[val]
                if cat_idx >= len(strings):
                    raise RuntimeError(f"Invalid string index in cat_vals: {cat_idx}")
                cat_str = strings[cat_idx]
                predicate = cat_str
                path_value = cat_str
            go_left = (
                missing_direction == MISSING_LEFT
                if learned_missing
                else not is_missing
                and (
                    str(feat_val) in predicate
                    if is_category_set
                    else str(feat_val) == predicate
                )
            )
            branch = "left" if go_left else "right"
            if path is not None:
                step = {
                    "node": idx,
                    "feature": feat,
                    "name": features[feat].get("name", str(feat)),
                    "op": "in" if is_category_set else "=",
                    "value": path_value,
                    "branch": branch,
                }
                if learned_missing:
                    step["missing"] = True
                path.append(step)
            idx = left if go_left else right
        elif op == OP_SWITCH:
            # Case table lookup
            if val >= len(case_tables):
                raise RuntimeError(f"Invalid case_table index in decision: {val}")
            table = case_tables[val]
            next_idx = table["default"]
            branch = "default"
            if learned_missing and missing_direction == MISSING_LEFT:
                next_idx = table["cases"][0][1]
                branch = "case"
            elif not is_missing:
                key = str(feat_val)
                if key in table["lookup"]:
                    next_idx = table["lookup"][key]
                    branch = "case"
            if path is not None:
                step = {
                    "node": idx,
                    "feature": feat,
                    "name": features[feat].get("name", str(feat)),
                    "op": "switch",
                    "value": None,
                    "branch": branch,
                }
                if learned_missing:
                    step["missing"] = True
                path.append(step)
            idx = next_idx

    raise RuntimeError("Max tree depth exceeded (possible corrupted model)")


def _tree_path(
    model: ModelData, vector: list[Any], tree_idx: int, missing: str
) -> tuple[Any, dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    result, leaf_idx = _predict_tree_recursive(
        model["tree_offsets"][tree_idx],
        vector,
        model["decisions"],
        model["leaves"],
        model["floats"],
        model["cat_vals"],
        model["strings"],
        model.get("distributions", []),
        model.get("case_tables", []),
        model.get("category_sets", []),
        model["meta"]["features"],
        model["bool_features"],
        len(vector),
        False,
        missing,
        tree_idx,
        bool(model.get("is_xgboost")),
        steps,
    )
    return result, {"tree": tree_idx, "leaf": leaf_idx, "path": steps}


def predict_path(
    model: ModelData, vector: list[Any], *, missing: str = "error"
) -> dict[str, Any]:
    """Predict and return the model-global leaf ID and decisions evaluated."""
    _check_missing_policy(missing)
    n_trees = model["n_trees"]
    evaluated = [_tree_path(model, vector, i, missing) for i in range(n_trees)]
    values = [item[0] for item in evaluated]
    trees = [item[1] for item in evaluated]

    if model.get("is_xgboost"):
        prediction = _aggregate_xgboost(model, values, False)
    elif model["is_forest"]:
        prediction = _aggregate_forest(model, values, False)
    else:
        prediction = values[0]
    return {"prediction": prediction, "trees": trees}


def _predict_forest(
    model: ModelData,
    vector: list[Any],
    return_dist: bool = False,
    *,
    missing: str = "error",
) -> Any:
    """Predict using forest (majority vote or mean)."""
    decisions = model["decisions"]
    leaves = model["leaves"]
    floats = model["floats"]
    cat_vals = model["cat_vals"]
    strings = model["strings"]
    distributions = model.get("distributions", [])
    case_tables = model.get("case_tables", [])
    category_sets = model.get("category_sets", [])
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
            category_sets,
            features,
            model["bool_features"],
            n_input_features,
            return_dist,
            missing,
            i,
            False,
        )
        for i in range(model["n_trees"])
    ]

    return _aggregate_forest(model, predictions, return_dist)


def _aggregate_forest(
    model: ModelData, predictions: list[Any], return_dist: bool
) -> Any:
    """Aggregate per-tree forest outputs for both prediction APIs."""
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


def _xgboost_base_scores(value, n_classes, is_regression):
    """Resolve scalar/vector intercepts; binary values are probability-space."""
    count = 1 if is_regression or n_classes == 2 else n_classes
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(
                f"XGBoost base_score needs {count} values, got {len(value)}"
            )
        scores = [float(v) for v in value]
    else:
        scores = [float(value)] * count
    if not scores or not all(math.isfinite(v) for v in scores):
        raise ValueError("XGBoost base_score must contain finite values")
    if not is_regression and n_classes == 2 and 0.0 < scores[0] < 1.0:
        scores[0] = math.log(scores[0] / (1.0 - scores[0]))
    return scores


def _aggregate_xgboost(model: ModelData, values: Any, return_dist: bool = False) -> Any:
    """Aggregate XGBoost tree outputs for both prediction APIs."""
    n_classes = len(model["class_labels"])
    class_labels = model["class_labels"]
    meta_dict = cast(dict, model.get("meta", {}))
    metadata = cast(dict, meta_dict.get("metadata", {}))
    bases = _xgboost_base_scores(
        metadata.get("base_score", 0.0), n_classes, model["is_regression"]
    )
    if model["is_regression"]:
        raw = bases[0]
        for value in values:
            raw += value
        return raw
    if n_classes == 2:
        raw = bases[0]
        for value in values:
            raw += value
        probability = _sigmoid(raw)
        if return_dist:
            return {class_labels[0]: 1 - probability, class_labels[1]: probability}
        return (
            class_labels[1]
            if probability > BINARY_CLASSIFICATION_THRESHOLD
            else class_labels[0]
        )
    values = list(values)
    if len(values) % n_classes:
        raise ValueError(
            f"multiclass XGBoost model has {len(values)} trees, "
            f"not a multiple of {n_classes} classes"
        )
    scores = list(bases)
    for tree_idx, value in enumerate(values):
        scores[tree_idx % n_classes] += value
    probabilities = _softmax(scores)
    if return_dist:
        return {class_labels[i]: probabilities[i] for i in range(n_classes)}
    return class_labels[probabilities.index(max(probabilities))]


def _predict_xgboost(
    model: ModelData,
    vector: list[Any],
    return_dist: bool = False,
    *,
    missing: str = "error",
) -> Any:
    """
    XGBoost prediction: additive model with sigmoid/softmax.

    For binary classification:
      raw_score starts at base_score and adds each tree output in tree order
      probability = sigmoid(raw_score)

    For multiclass (K classes, K trees per round):
      raw_scores[k] starts at its base score and adds each round in tree order
      probabilities = softmax(raw_scores)
    """
    n_trees = model["n_trees"]
    decisions = model["decisions"]
    leaves = model["leaves"]
    floats = model["floats"]
    cat_vals = model["cat_vals"]
    strings = model["strings"]
    distributions = model.get("distributions", [])
    case_tables = model.get("case_tables", [])
    category_sets = model.get("category_sets", [])
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
            category_sets,
            features,
            model["bool_features"],
            n_input_features,
            False,
            missing,
            t_idx,
            True,
        )

    return _aggregate_xgboost(
        model, (_eval_one(tree_idx) for tree_idx in range(n_trees)), return_dist
    )


def predict_batch(
    model: ModelData,
    vectors: list[list[Any]],
    return_dist: bool = False,
    *,
    missing: str = "error",
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
    _check_missing_policy(missing)
    return [
        predict(model, v, return_dist=return_dist, missing=missing) for v in vectors
    ]


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

    def predict(
        self,
        vector: list[Any],
        return_dist: bool = False,
        *,
        missing: str = "error",
    ) -> Any:
        """Make a prediction for a single feature vector."""
        return predict(self.model, vector, return_dist=return_dist, missing=missing)

    def predict_path(
        self, vector: list[Any], *, missing: str = "error"
    ) -> dict[str, Any]:
        """Predict and return the decisions and model-global leaf IDs."""
        return predict_path(self.model, vector, missing=missing)

    def predict_batch(
        self,
        vectors: list[list[Any]],
        return_dist: bool = False,
        *,
        missing: str = "error",
    ) -> list[Any]:
        """Make predictions for multiple feature vectors."""
        return predict_batch(
            self.model, vectors, return_dist=return_dist, missing=missing
        )

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
