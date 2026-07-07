#!/usr/bin/env python
"""
Minimal Python predictor for .cart binary format.
Zero dependencies beyond Python stdlib.

This file is shipped standalone (and embedded by `cartlet bundle`) so it
intentionally duplicates the format constants and traversal logic that
otherwise live in `cartlet/io/cart_format.py` and `cartlet/runner.py`.
Keep the constants below (OP_*, LEAF_*, FLAG_*, HEADER_SIZE, EXPECTED_VERSION,
SIZE_*, MAX_TREE_DEPTH) in sync with `cartlet/io/cart_format.py`. The format
is versioned via `EXPECTED_VERSION`; bump both files together if the layout
changes.

Library usage:
    from predict import Predictor

    model = Predictor("model.cart")    # or Predictor("model.cart.gz")
    result = model.predict(["red", "large"])
    results = model.predict_batch([["red", "large"], ["blue", "small"]])
    dist = model.predict(["red", "large"], return_dist=True)

    # Model info
    print(model.n_features, model.n_classes, model.is_forest)

CLI usage:
    ./predict.py model.cart red large
    ./predict.py -m model.cart -f input.txt
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import Counter

# Decision node encoding (packed feat_op byte)
# Bits 0-5: feature index (max 63 inline)
# Bits 6-7: operation type
OP_SHIFT = 6
OP_MASK = 0xC0  # Upper 2 bits for op
FEAT_MASK = 0x3F  # Lower 6 bits = feature index
OP_LT = 0  # Numerical less-than
OP_EQ = 1  # Categorical equality
OP_SWITCH = 2  # Case table lookup

# Leaf node types
LEAF_CLASS = 0
LEAF_FLOAT = 1
LEAF_CLASS_DIST = 2

# Index flag: high bit = leaf
LEAF_FLAG = 0x80000000
INDEX_MASK = 0x7FFFFFFF

# Header flags
FLAG_IS_FOREST = 1 << 0
FLAG_IS_REGRESSION = 1 << 1
FLAG_HAS_DISTRIBUTIONS = 1 << 2
FLAG_IS_XGBOOST = 1 << 3

# Safety limit for tree traversal
MAX_TREE_DEPTH = 10000
EXPECTED_VERSION = 1

# Sigmoid output above which we emit the positive class label.
BINARY_THRESHOLD = 0.5

# Header
HEADER_SIZE = 34  # Minimum header size in bytes
TYPE_MASK = 0x03  # Feature type mask: 0=cat, 1=num

# Sanity limits for header values
_MAX_FEATURES = 10000
_MAX_CLASSES = 100000
_MAX_TREES = 100000
_MAX_NODES = 10_000_000


def decode_varint(data, pos):
    """Decode a varint from bytes, return (value, new_pos)."""
    result = 0
    shift = 0
    # A 32-bit value needs at most 5 bytes; cap the run so corrupt data with a
    # long chain of continuation bits raises instead of building a huge int.
    for _ in range(5):
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7
    raise ValueError("varint too long (corrupt .cart file)")


# =============================================================================
# Predictor class - the main library API
# =============================================================================


class Predictor:
    """
    Decision tree predictor for .cart binary format.

    Example:
        model = Predictor("model.cart")
        result = model.predict(["red", "large"])
    """

    def __init__(self, path_or_bytes=None):
        """
        Load a model from file path or bytes.

        Args:
            path_or_bytes: Path to .cart file, .cart.gz file, or raw bytes
        """
        self._model = None
        if path_or_bytes is not None:
            self.load(path_or_bytes)

    def load(self, path_or_bytes):
        """Load model from file path or bytes."""
        if isinstance(path_or_bytes, (bytes, bytearray)):
            self._model = load_cart_from_bytes(path_or_bytes)
        else:
            self._model = load_cart(path_or_bytes)
        return self

    @classmethod
    def from_bytes(cls, data):
        """Create Predictor from raw bytes."""
        return cls(data)

    @classmethod
    def from_embedded(cls):
        """Create Predictor from embedded model data (for bundled scripts)."""
        model_data = load_embedded()
        if model_data is None:
            raise ValueError("No embedded model found")
        obj = cls()
        obj._model = model_data
        return obj

    def predict(self, features, return_dist=False):
        """
        Predict for a single feature vector.

        Args:
            features: List of feature values
            return_dist: If True, return dict of {class: probability}

        Returns:
            Predicted class/value, or distribution dict if return_dist=True
        """
        if self._model is None:
            raise ValueError("No model loaded")
        return predict(self._model, features, return_dist=return_dist)

    def predict_batch(self, feature_rows, return_dist=False):
        """
        Predict for multiple feature vectors.

        Args:
            feature_rows: List of feature vectors
            return_dist: If True, return dicts of {class: probability}

        Returns:
            List of predictions
        """
        return [self.predict(row, return_dist=return_dist) for row in feature_rows]

    def info(self):
        """Return dict with model info."""
        if self._model is None:
            raise ValueError("No model loaded")
        return {
            "n_features": len(self._model["features"]),
            "n_classes": len(self._model["class_labels"]),
            "n_decisions": len(self._model["decisions"]),
            "n_leaves": len(self._model["leaves"]),
            "n_trees": self._model.get("n_trees", 1),
            "n_case_tables": len(self._model.get("case_tables", [])),
            "is_forest": self._model["is_forest"],
            "is_regression": self._model["is_regression"],
            "is_xgboost": self._model.get("is_xgboost", False),
            "has_distributions": self._model.get("has_distributions", False),
            "features": self._model["features"],
            "class_labels": self._model["class_labels"],
        }

    @property
    def feature_names(self):
        """Get list of feature names."""
        if self._model is None:
            return []
        return [f["name"] for f in self._model["features"]]

    @property
    def class_labels(self):
        """Get list of class labels (classification only)."""
        return self._model["class_labels"] if self._model else []

    @property
    def task(self):
        """Get model task (classification/regression)."""
        if self._model is None:
            return None
        return "regression" if self._model["is_regression"] else "classification"

    @property
    def n_features(self):
        return len(self._model["features"]) if self._model else 0

    @property
    def n_classes(self):
        return len(self._model["class_labels"]) if self._model else 0

    @property
    def is_forest(self):
        return self._model["is_forest"] if self._model else False

    @property
    def is_regression(self):
        return self._model["is_regression"] if self._model else False

    @property
    def is_xgboost(self):
        return self._model.get("is_xgboost", False) if self._model else False

    @property
    def metadata(self):
        """Embedded metadata dict from the model trailer (empty when none)."""
        if self._model is None:
            return {}
        return self._model.get("metadata", {})

    def get_vocabulary(self, feature):
        """
        Return the known values for a categorical feature, or None if the
        feature is numerical or unknown. ``feature`` may be an index or name.
        """
        if self._model is None:
            return None
        features = self._model.get("features", [])
        for i, feat in enumerate(features):
            if (isinstance(feature, int) and i == feature) or feat.get(
                "name"
            ) == feature:
                if feat.get("type") == "cat" and "values" in feat:
                    return set(feat["values"])
                return None
        return None

    def is_oov(self, feature, value):
        """
        True if ``value`` was not seen for ``feature`` during training. Always
        returns False for numerical or unknown features.
        """
        vocab = self.get_vocabulary(feature)
        if vocab is None:
            return False
        return value not in vocab

    def __repr__(self):
        if self._model is None:
            return "Predictor(uninitialized)"
        n_trees = self._model.get("n_trees", 1)
        return f"Predictor(task={self.task}, n_trees={n_trees})"


# =============================================================================
# Procedural API
# =============================================================================


def load_cart_from_bytes(data):
    """Load model from bytes, return model dict.

    Wraps the low-level parser so a corrupt or truncated file surfaces as a
    clear ValueError instead of a raw struct.error / IndexError.
    """
    try:
        return _load_cart_from_bytes_impl(data)
    except (struct.error, IndexError) as e:
        raise ValueError(f"Malformed or truncated .cart file: {e}") from e


def _load_cart_from_bytes_impl(data):
    """Load model from bytes, return model dict."""
    pos = 0

    # Header
    magic = data[pos : pos + 4]
    pos += 4
    if magic != b"CART":
        raise ValueError(f"Invalid magic: {magic!r}")

    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"File too small ({len(data)} bytes), minimum header is {HEADER_SIZE} bytes"
        )

    version, flags, n_features, n_classes, n_trees = struct.unpack_from(
        "<HHHHH", data, pos
    )
    pos += 10

    if version != EXPECTED_VERSION:
        raise ValueError(
            f"Unsupported format version {version} (expected {EXPECTED_VERSION})"
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

    # Sanity checks on header values
    if n_features > _MAX_FEATURES:
        raise ValueError(f"Unreasonable n_features: {n_features}")
    if n_classes > _MAX_CLASSES:
        raise ValueError(f"Unreasonable n_classes: {n_classes}")
    if n_trees > _MAX_TREES:
        raise ValueError(f"Unreasonable n_trees: {n_trees}")
    if n_decisions > _MAX_NODES:
        raise ValueError(f"Unreasonable n_decisions: {n_decisions}")
    if n_leaves > _MAX_NODES:
        raise ValueError(f"Unreasonable n_leaves: {n_leaves}")

    is_forest = bool(flags & FLAG_IS_FOREST)
    is_regression = bool(flags & FLAG_IS_REGRESSION)
    has_distributions = bool(flags & FLAG_HAS_DISTRIBUTIONS)
    is_xgboost = bool(flags & FLAG_IS_XGBOOST)

    # String table
    (n_strings,) = struct.unpack_from("<H", data, pos)
    pos += 2
    string_offsets = struct.unpack_from(f"<{n_strings}H", data, pos)
    pos += 2 * n_strings

    # Find end of string data by scanning for nulls
    strings = []
    string_data_start = pos
    end = string_data_start
    for off in string_offsets:
        start = string_data_start + off
        term = data.index(b"\x00", start)
        strings.append(data[start:term].decode("utf-8"))
        # Advance by byte position, not decoded character count (which is
        # wrong for any non-ASCII string).
        if term + 1 > end:
            end = term + 1
    pos = end

    # Feature table
    features = []
    for _ in range(n_features):
        name_idx, type_flags, n_cat = struct.unpack_from("<HBB", data, pos)
        pos += 4
        cat_indices = list(struct.unpack_from(f"<{n_cat}H", data, pos))
        pos += 2 * n_cat
        feat_type = "cat" if (type_flags & TYPE_MASK) == 0 else "num"
        features.append(
            {
                "name": strings[name_idx],
                "type": feat_type,
                "values": [strings[ci] for ci in cat_indices],
            }
        )

    # Class table
    class_labels = []
    for _ in range(n_classes):
        (ci,) = struct.unpack_from("<H", data, pos)
        pos += 2
        class_labels.append(strings[ci])

    # Float pool
    floats = list(struct.unpack_from(f"<{n_floats}f", data, pos))
    pos += 4 * n_floats

    # Cat value pool
    cat_vals = list(struct.unpack_from(f"<{n_cat_vals}H", data, pos))
    pos += 2 * n_cat_vals

    # Tree offsets (varint encoded)
    tree_offsets = []
    for _ in range(n_trees):
        off, pos = decode_varint(data, pos)
        tree_offsets.append(off)

    # Decision nodes (variable size)
    # OP_LT/OP_EQ: feat_op(1) + val(2) + left(varint) + right(varint)
    # OP_SWITCH: feat_op(1) + table_idx(2) (no left/right)
    decisions = []
    for _ in range(n_decisions):
        feat_op, val = struct.unpack_from("<BH", data, pos)
        pos += 3
        feat = feat_op & FEAT_MASK
        op = (feat_op & OP_MASK) >> OP_SHIFT
        if op == OP_SWITCH:
            # Switch node: val is table_idx, no left/right
            decisions.append((feat, op, val, 0, 0))
        else:
            left, pos = decode_varint(data, pos)
            right, pos = decode_varint(data, pos)
            decisions.append((feat, op, val, left, right))

    # Leaf nodes (3 bytes each - no padding)
    leaves = []
    for _ in range(n_leaves):
        leaf_type, val = struct.unpack_from("<BH", data, pos)
        pos += 3
        leaves.append((leaf_type, val))

    # Distributions (if FLAG_HAS_DISTRIBUTIONS)
    distributions = []
    if has_distributions:
        for _ in range(n_dists):
            (n_entries,) = struct.unpack_from("<H", data, pos)
            pos += 2
            dist = []
            for _ in range(n_entries):
                class_idx, prob = struct.unpack_from("<Hf", data, pos)
                pos += 6
                dist.append((class_idx, prob))
            distributions.append(dist)

    # Case tables (for OP_SWITCH nodes)
    case_tables = []
    for _ in range(n_case_tables):
        (n_cases,) = struct.unpack_from("<H", data, pos)
        pos += 2
        default_child, pos = decode_varint(data, pos)
        cases = []
        for _ in range(n_cases):
            (cat_val_idx,) = struct.unpack_from("<H", data, pos)
            pos += 2
            child_idx, pos = decode_varint(data, pos)
            cases.append((cat_val_idx, child_idx))
        case_tables.append({"default": default_child, "cases": cases})

    # Trailing metadata blob (JSON); carries XGBoost base_score and friends.
    metadata = {}
    if metadata_len:
        meta_blob = data[pos : pos + metadata_len]
        pos += metadata_len
        try:
            metadata = json.loads(meta_blob.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            metadata = {}

    return {
        "features": features,
        "class_labels": class_labels,
        "floats": floats,
        "cat_vals": cat_vals,
        "strings": strings,
        "decisions": decisions,
        "leaves": leaves,
        "distributions": distributions,
        "case_tables": case_tables,
        "tree_offsets": tree_offsets,
        "has_distributions": has_distributions,
        "is_regression": is_regression,
        "is_forest": is_forest,
        "is_xgboost": is_xgboost,
        "n_trees": n_trees,
        "metadata": metadata,
    }


def load_cart(path):
    """Load a .cart file, return model dict. Supports .cart.gz."""
    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    return load_cart_from_bytes(data)


def load_embedded():
    """Load model embedded in this script file."""
    if "_EMBEDDED_MODEL_B64" in globals():
        import base64

        model_data = base64.b64decode(globals()["_EMBEDDED_MODEL_B64"])
        return load_cart_from_bytes(model_data)
    return None


def predict_tree(model, row, tree_idx=0, return_dist=False):
    """Predict using a single tree."""
    return _predict_tree_recursive(
        model["tree_offsets"][tree_idx],
        row,
        model["decisions"],
        model["leaves"],
        model["floats"],
        model["cat_vals"],
        model["strings"],
        model.get("distributions", []),
        model.get("case_tables", []),
        model["features"],
        len(row),
        return_dist,
    )


def _predict_tree_recursive(
    idx,
    row,
    decisions,
    leaves,
    floats,
    cat_vals,
    strings,
    distributions,
    case_tables,
    features,
    n_input_features,
    return_dist=False,
):
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
                if val >= len(distributions):
                    raise RuntimeError(f"Invalid distribution index in leaf: {val}")
                dist_data = distributions[val]
                if return_dist:
                    return {strings[ci]: prob for ci, prob in dist_data}
                else:
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
        feat_val = None if feat >= n_input_features else row[feat]

        if op == OP_LT:
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
            if val >= len(cat_vals):
                raise RuntimeError(f"Invalid cat_val index in decision: {val}")
            cat_idx = cat_vals[val]
            if cat_idx >= len(strings):
                raise RuntimeError(f"Invalid string index in cat_vals: {cat_idx}")
            cat_str = strings[cat_idx]
            go_left = False if feat_val is None else (str(feat_val) == cat_str)
            idx = left if go_left else right
        elif op == OP_SWITCH:
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


def sigmoid(x):
    """Sigmoid activation function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # Numerically stable for negative x
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def softmax(scores):
    """Softmax activation, return list of probabilities."""
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def predict(model, row, return_dist=False):
    """
    Predict for a feature row.

    Args:
        model: Loaded model dict
        row: Feature values (may contain None for missing)
        return_dist: If True, return probability distribution

    Note:
        Missing values: When a feature is None or missing, comparisons fail
        and the tree takes the "no" branch (right child).
    """
    is_xgboost = model.get("is_xgboost", False)

    if is_xgboost:
        return predict_xgboost(model, row, return_dist=return_dist)

    n_trees = model.get("n_trees", len(model["tree_offsets"]))
    if not model["is_forest"] and n_trees == 1:
        return predict_tree(model, row, 0, return_dist=return_dist)

    # Forest: aggregate predictions
    decisions = model["decisions"]
    leaves = model["leaves"]
    floats = model["floats"]
    cat_vals = model["cat_vals"]
    strings = model["strings"]
    distributions = model.get("distributions", [])
    case_tables = model.get("case_tables", [])
    features = model["features"]
    n_input_features = len(row)

    predictions = [
        _predict_tree_recursive(
            model["tree_offsets"][i],
            row,
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
        for i in range(n_trees)
    ]

    if model["is_regression"]:
        return sum(predictions) / len(predictions)

    if return_dist:
        # Aggregate distributions
        combined = {}  # type: ignore
        for pred in predictions:
            if isinstance(pred, dict):
                for cls, prob in pred.items():
                    combined[cls] = combined.get(cls, 0.0) + prob
            else:
                combined[pred] = combined.get(pred, 0.0) + 1.0
        total = sum(combined.values())
        return {cls: count / total for cls, count in combined.items()}

    # Classification: majority vote
    return Counter(predictions).most_common(1)[0][0]


def predict_xgboost(model, row, return_dist=False):
    """
    XGBoost prediction: additive model with sigmoid/softmax.

    For binary classification:
      raw_score = base_score + sum(tree outputs)
      probability = sigmoid(raw_score)

    For multiclass (K classes, K trees per round):
      raw_scores[k] = base_score + sum(trees for class k)
      probabilities = softmax(raw_scores)
    """
    n_trees = model.get("n_trees", len(model["tree_offsets"]))
    n_classes = len(model["class_labels"])
    class_labels = model["class_labels"]
    # base_score round-trips from the metadata trailer. For binary classification
    # XGBoost stores it in probability space, so re-project to raw-score space.
    raw_base_score = float(model.get("metadata", {}).get("base_score", 0.0))
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
    features = model["features"]
    n_input_features = len(row)

    def _eval_one(t_idx):
        return _predict_tree_recursive(
            model["tree_offsets"][t_idx],
            row,
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
        # Regression: sum tree outputs
        raw_score = base_score
        for i in range(n_trees):
            raw_score += _eval_one(i)
        return raw_score

    if n_classes == 2:
        # Binary classification
        raw_score = base_score
        for i in range(n_trees):
            raw_score += _eval_one(i)
        prob = sigmoid(raw_score)
        if return_dist:
            return {class_labels[0]: 1 - prob, class_labels[1]: prob}
        return class_labels[1] if prob > BINARY_THRESHOLD else class_labels[0]

    # Multiclass: K trees per round, sum per class
    n_rounds = n_trees // n_classes
    scores = [base_score] * n_classes

    for round_idx in range(n_rounds):
        for class_idx in range(n_classes):
            tree_idx = round_idx * n_classes + class_idx
            scores[class_idx] += _eval_one(tree_idx)

    probs = softmax(scores)
    if return_dist:
        return {class_labels[i]: probs[i] for i in range(n_classes)}

    best_idx = probs.index(max(probs))
    return class_labels[best_idx]


def parse_line(line, delimiter, model_features):
    """Parse a line into feature values, converting numeric features."""
    parts = line.strip().split(delimiter)
    parsed = []
    for i, val in enumerate(parts):
        if i < len(model_features) and model_features[i]["type"] == "num":
            feat = model_features[i].get("name", i)
            try:
                parsed.append(float(val))
            except ValueError:
                raise ValueError(
                    f"expected a number for feature {feat!r}, got {val!r}"
                ) from None
        else:
            parsed.append(val)
    return parsed


def show_info(model):
    """Print model information."""
    features = model["features"]
    print(f"Features ({len(features)}):")
    for f in features:
        print(f"  {f['name']}: {f['type']}")
    if model["class_labels"]:
        print(f"Classes: {model['class_labels']}")
    print(f"Decisions: {len(model['decisions'])}")
    print(f"Leaves: {len(model['leaves'])}")
    case_tables = model.get("case_tables", [])
    if case_tables:
        print(f"Case tables: {len(case_tables)}")
    if model["is_forest"] or model.get("is_xgboost"):
        print(f"Trees: {model.get('n_trees', len(model['tree_offsets']))}")
    model_type = "regression" if model["is_regression"] else "classification"
    if model.get("is_xgboost"):
        model_type = f"XGBoost {model_type}"
    elif model["is_forest"]:
        model_type = f"Random Forest {model_type}"
    else:
        model_type = f"Decision Tree {model_type}"
    print(f"Type: {model_type}")


def main():
    embedded_model = load_embedded()
    has_embedded = embedded_model is not None

    parser = argparse.ArgumentParser(
        description="Predict using a .cart decision tree model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.cart red small           # Single prediction
  %(prog)s -m model.cart -f input.txt     # Batch from file
  %(prog)s -m model.cart -f -             # Batch from stdin
  %(prog)s model.cart red small --dist    # Return distribution (JSON)
  %(prog)s --info model.cart              # Show model info
""",
    )

    if has_embedded:
        parser.add_argument(
            "-m",
            "--model",
            metavar="FILE",
            help="Model file (.cart) - overrides embedded model",
        )
    else:
        parser.add_argument(
            "-m",
            "--model",
            metavar="FILE",
            help="Model file (.cart)",
        )

    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        help="Read features from file (one per line, use - for stdin)",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        metavar="CHAR",
        help="Field delimiter for file input (default: whitespace)",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Show model info and exit",
    )
    parser.add_argument(
        "--dist",
        action="store_true",
        help="Return probability distributions (classification only, JSON output)",
    )
    parser.add_argument(
        "args",
        nargs="*",
        metavar="ARG",
        help="[MODEL] FEATURE... - model file followed by feature values",
    )

    args = parser.parse_args()

    # Parse positional args: first arg might be model, rest are features
    positional = args.args
    model_file = args.model
    features = []

    if positional:
        first = positional[0]
        # Check if first positional looks like a model file
        if not model_file and not first.startswith("-"):
            if first.endswith(".cart") or os.path.isfile(first):
                model_file = first
                features = positional[1:]
            else:
                features = positional
        else:
            features = positional

    # Determine model to use
    if model_file:
        model = load_cart(model_file)
    elif has_embedded:
        model = embedded_model
    else:
        parser.error(
            "No model specified. Use -m MODEL or provide model.cart as first argument."
        )

    # Info mode
    if args.info:
        show_info(model)
        return 0

    # Determine delimiter
    delimiter = args.delimiter if args.delimiter else None

    # File/stdin mode
    if args.file:
        if args.file == "-":
            input_file = sys.stdin
        else:
            input_file = open(args.file, encoding="utf-8")  # noqa: SIM115

        try:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    if delimiter:
                        row = parse_line(line, delimiter, model["features"])
                    else:
                        row = parse_line(line, None, model["features"])
                        if len(row) == 1:
                            # Try tab, then comma
                            row = parse_line(line, "\t", model["features"])
                            if len(row) == 1:
                                row = parse_line(line, ",", model["features"])
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return 1
                result = predict(model, row, return_dist=args.dist)
                if args.dist:
                    print(json.dumps(result))
                else:
                    print(result)
        finally:
            if args.file != "-":
                input_file.close()
        return 0

    # Single prediction mode
    if not features:
        parser.error("No features provided. Use -f FILE for batch or provide features.")

    # Convert numeric features
    parsed_row = []
    for i, val in enumerate(features):
        if i < len(model["features"]) and model["features"][i]["type"] == "num":
            try:
                parsed_row.append(float(val))
            except ValueError:
                feat = model["features"][i].get("name", i)
                print(
                    f"Error: expected a number for feature {feat!r}, got {val!r}",
                    file=sys.stderr,
                )
                return 1
        else:
            parsed_row.append(val)

    result = predict(model, parsed_row, return_dist=args.dist)
    if args.dist:
        print(json.dumps(result))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
