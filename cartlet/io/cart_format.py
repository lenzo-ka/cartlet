"""
.cart binary format constants and shared utilities.

This module defines the binary format for decision trees and forests.
All multi-byte values are little-endian.

Format optimizations:
- Repeated counts and indices use 32-bit varints
- Probabilities use unsigned 16-bit fixed-point values
- Header counts use fixed-width unsigned 32-bit values
"""

import struct
from collections.abc import Mapping
from typing import Any

# Magic bytes
MAGIC = b"CART"
VERSION = 3

# Decision node encoding:
# - feat: varint (feature index)
# - op: 1 byte
# - flags: 1 byte (learned missing direction and categorical-set marker)
# - val: varint (index into floats, cat_vals, category_sets, or case_tables)
# - left: varint (1-5 bytes) - only for OP_LE/OP_LT/OP_EQ
# - right: varint (1-5 bytes) - only for OP_LE/OP_LT/OP_EQ
# Note: OP_SWITCH nodes have children in the case table instead
# Per-decision flags byte. Native CART nodes use MISSING_NONE.
MISSING_NONE = 0
MISSING_LEFT = 1
MISSING_RIGHT = 2
MISSING_MASK = 0x03
CATEGORY_SET = 0x04
DECISION_FLAGS_MASK = MISSING_MASK | CATEGORY_SET

# Operation types
OP_LE = 0  # Numerical less-than-or-equal (<=) comparison; left branch = "yes"
OP_EQ = 1  # Categorical equality comparison
OP_SWITCH = 2  # Case table lookup (disjunction / n-ary split)
OP_LT = 3  # Strict numeric comparison (<), used by XGBoost

# Leaf node types (stored in 1 byte)
LEAF_CLASS = 0  # Leaf: classification (string index)
LEAF_FLOAT = 1  # Leaf: regression (float index)
LEAF_CLASS_DIST = 2  # Leaf: classification with distribution (dist index)

# Index flag: high bit set = leaf index (used in varints too)
LEAF_FLAG = 0x80000000
INDEX_MASK = 0x7FFFFFFF

# Header flags (stored in 2 bytes)
FLAG_IS_FOREST = 1 << 0
FLAG_IS_REGRESSION = 1 << 1
FLAG_HAS_DISTRIBUTIONS = 1 << 2  # Distributions stored for nbest support
FLAG_IS_XGBOOST = 1 << 3  # XGBoost model (additive prediction)

# Feature type encoding (bits 0-1 of type_flags byte)
TYPE_CAT = 0
TYPE_NUM = 1
TYPE_MASK = 0x03

# Dtype encoding (bits 2-4 of type_flags byte)
DTYPE_MAP = {"str": 0, "int": 1, "float": 2, "bool": 3}


def decode_feature_dtype(type_flags: int) -> str:
    """Decode the dtype bits written in each feature-table entry."""
    dtypes = {code: name for name, code in DTYPE_MAP.items()}
    code = type_flags >> 2
    if code not in dtypes:
        raise ValueError(f"Invalid feature dtype code: {code}")
    return dtypes[code]


def decode_feature_value(value: str, dtype: str) -> Any:
    """Restore known vocabulary values in the feature's declared dtype."""
    if dtype == "int":
        return int(value)
    if dtype == "float":
        return float(value)
    if dtype == "bool":
        if value in {"True", "true", "TRUE", "1", "yes", "Yes", "YES"}:
            return True
        if value in {"False", "false", "FALSE", "0", "no", "No", "NO"}:
            return False
        raise ValueError(f"Invalid boolean vocabulary value: {value!r}")
    return value


# Header size (bytes)
HEADER_SIZE = 52
# Offset into header
OFF_VERSION = 4
OFF_FLAGS = 6
OFF_FEATURES = 8
OFF_CLASSES = 12
OFF_TREES = 16
OFF_DECISIONS = 20
OFF_LEAVES = 24
OFF_FLOATS = 28
OFF_CAT_VALS = 32
OFF_DISTS = 36
OFF_CASE_TABLES = 40
OFF_CATEGORY_SETS = 44
OFF_META_LEN = 48

# Breakdown:
#   magic:          4 bytes
#   version:        2 bytes (u16)
#   flags:          2 bytes (u16)
#   n_features:     4 bytes (u32)
#   n_classes:      4 bytes (u32)
#   n_trees:        4 bytes (u32)
#   n_decisions:    4 bytes (u32)
#   n_leaves:       4 bytes (u32)
#   n_floats:       4 bytes (u32)
#   n_cat_vals:     4 bytes (u32)
#   n_dists:        4 bytes (u32) -- number of distribution entries
#   n_case_tables:  4 bytes (u32) -- number of case tables
#   n_category_sets: 4 bytes (u32) -- number of categorical membership sets
#   metadata_len:   4 bytes (u32)
# Total: 4 + 2 + 2 + (11*4) = 52

# Fixed-width element sizes (bytes)
SIZE_F64 = 8  # Float64 preserves native thresholds and regression outputs
SIZE_Q16 = 2

# Header struct groups parsed after the 4-byte magic (see the breakdown above).
HEADER_FMT_COUNTS1 = "<HHIII"  # version, flags, n_features, n_classes, n_trees
HEADER_FMT_COUNTS2 = "<IIIIIIII"  # nodes/floats + cat/dist/case/set/meta
SIZE_HEADER_COUNTS1 = struct.calcsize(HEADER_FMT_COUNTS1)  # 16
SIZE_HEADER_COUNTS2 = struct.calcsize(HEADER_FMT_COUNTS2)  # 32
# Sanity: the magic + both count groups must equal the fixed header size.
assert 4 + SIZE_HEADER_COUNTS1 + SIZE_HEADER_COUNTS2 == HEADER_SIZE


# =============================================================================
# Varint encoding/decoding (protobuf-style, 7 bits per byte, MSB = continuation)
# =============================================================================


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a varint (1-5 bytes for 32-bit values)."""
    if value < 0:
        raise ValueError(f"varint cannot encode a negative value: {value}")
    if value > 0xFFFFFFFF:
        raise ValueError(f"varint value exceeds u32: {value}")
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint from bytes, return (value, new_pos)."""
    result = 0
    shift = 0
    # A 32-bit value needs at most 5 bytes; cap the run so corrupt data with a
    # long chain of continuation bits raises instead of spinning up a huge int.
    for _ in range(5):
        byte = data[pos]
        if shift == 28 and not (byte & 0x80) and byte > 0x0F:
            raise ValueError("varint exceeds u32 (corrupt .cart file)")
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7
    raise ValueError("varint too long (corrupt .cart file)")


def rebuild_tree_from_cart(
    model_data: Mapping[str, Any],
    feature_names: list[str],
    tree_idx: int = 0,
) -> Any:
    """
    Rebuild nested tree structure from .cart flat nodes.

    Args:
        model_data: Dict from runner.load_model() with decisions, leaves, etc.
        feature_names: List of feature names
        tree_idx: Index of tree to rebuild (0 for single tree)

    Returns:
        Nested tree structure: [feature, op, value, left, right] or leaf
    """
    decisions = model_data["decisions"]
    leaves = model_data["leaves"]
    floats = model_data["floats"]
    cat_vals = model_data["cat_vals"]
    strings = model_data["strings"]

    distributions = model_data.get("distributions", [])

    def rebuild(idx: int) -> Any:
        if idx & LEAF_FLAG:
            # Leaf node
            leaf_idx = idx & INDEX_MASK
            leaf_type, val = leaves[leaf_idx]
            if leaf_type == LEAF_CLASS:
                return strings[val]
            elif leaf_type == LEAF_CLASS_DIST:
                # Rebuild distribution dict from stored data
                dist_data = distributions[val]
                return {strings[class_idx]: prob for class_idx, prob in dist_data}
            else:  # LEAF_FLOAT
                return [floats[val], 0.0, 1]

        # Decision node
        node = decisions[idx]
        feat = node[0]
        op = node[1]
        feature_name = feature_names[feat] if feat < len(feature_names) else str(feat)

        if op in (OP_LE, OP_LT):
            _, _, missing_flags, val, left, right = node
            value = floats[val]
            left_tree = rebuild(left)
            right_tree = rebuild(right)
            feature: Any = feature_name
            if missing_flags == MISSING_LEFT:
                feature = {"feature": feature_name, "missing": "left"}
            elif missing_flags == MISSING_RIGHT:
                feature = {"feature": feature_name, "missing": "right"}
            return [
                feature,
                "<" if op == OP_LT else "<=",
                value,
                left_tree,
                right_tree,
            ]
        elif op == OP_EQ:
            _, _, missing_flags, val, left, right = node
            is_category_set = bool(missing_flags & CATEGORY_SET)
            missing_flags &= MISSING_MASK
            left_tree = rebuild(left)
            right_tree = rebuild(right)
            feature = feature_name
            if missing_flags == MISSING_LEFT:
                feature = {"feature": feature_name, "missing": "left"}
            elif missing_flags == MISSING_RIGHT:
                feature = {"feature": feature_name, "missing": "right"}
            if is_category_set:
                category_sets = model_data.get("category_sets", [])
                value = sorted(category_sets[val])
                return [feature, "in", value, left_tree, right_tree]
            value = strings[cat_vals[val]]
            return [feature, "=", value, left_tree, right_tree]
        elif op == OP_SWITCH:
            # Switch nodes retain placeholders for their unused child fields.
            missing_flags = node[2]
            table_idx = node[3]
            case_tables = model_data.get("case_tables", [])
            table = case_tables[table_idx]
            # Rebuild case table as dict: {value: subtree, ...}
            cases = {}
            for cat_val_idx, child_idx in table["cases"]:
                cat_val = strings[cat_vals[cat_val_idx]]
                cases[cat_val] = rebuild(child_idx)
            default_tree = rebuild(table["default"])
            feature = feature_name
            if missing_flags == MISSING_LEFT:
                feature = {"feature": feature_name, "missing": "left"}
            elif missing_flags == MISSING_RIGHT:
                feature = {"feature": feature_name, "missing": "right"}
            return [feature, "switch", cases, default_tree]
        else:
            raise ValueError(f"Unknown op type: {op}")

    tree_offset = model_data["tree_offsets"][tree_idx]
    return rebuild(tree_offset)
