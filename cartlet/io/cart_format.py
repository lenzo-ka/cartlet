"""
.cart binary format constants and shared utilities.

This module defines the binary format for decision trees and forests.
All multi-byte values are little-endian.

Format optimizations:
- Decision nodes use varint for child indices
- Leaf nodes are 3 bytes (type + val, no padding)
- feat + op packed into 1 byte (6 bits feat, 2 bits op)
"""

from typing import Any

# Magic bytes
MAGIC = b"CART"
VERSION = 1

# Decision node encoding:
# - feat_op: 1 byte (bits 0-5 = feature index, bits 6-7 = op)
# - val: 2 bytes (index into floats, cat_vals, or case_tables)
# - left: varint (1-5 bytes) - only for OP_LT/OP_EQ
# - right: varint (1-5 bytes) - only for OP_LT/OP_EQ
# Note: OP_SWITCH nodes have children in the case table instead
OP_SHIFT = 6  # Op is in bits 6-7
OP_MASK = 0xC0  # Upper 2 bits for op
FEAT_MASK = 0x3F  # Lower 6 bits for feature index (max 63 features inline)

# Operation types
OP_LT = 0  # Numerical less-than-or-equal (<=) comparison; left branch = "yes"
OP_EQ = 1  # Categorical equality comparison
OP_SWITCH = 2  # Case table lookup (disjunction / n-ary split)

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

# Header size (bytes)
HEADER_SIZE = 34
# Offset into header
OFF_VERSION = 4
OFF_FLAGS = 6
OFF_FEATURES = 8
OFF_CLASSES = 10
OFF_TREES = 12
OFF_DECISIONS = 14
OFF_LEAVES = 18
OFF_FLOATS = 22
OFF_CAT_VALS = 26
OFF_DISTS = 28
OFF_CASE_TABLES = 30
OFF_META_LEN = 32

# Breakdown:
#   magic:          4 bytes
#   version:        2 bytes (u16)
#   flags:          2 bytes (u16)
#   n_features:     2 bytes (u16)
#   n_classes:      2 bytes (u16)
#   n_trees:        2 bytes (u16)
#   n_decisions:    4 bytes (u32)
#   n_leaves:       4 bytes (u32)
#   n_floats:       4 bytes (u32)
#   n_cat_vals:     2 bytes (u16)
#   n_dists:        2 bytes (u16) -- number of distribution entries
#   n_case_tables:  2 bytes (u16) -- number of case tables
#   metadata_len:   2 bytes (u16)
# Total: 4 + (5*2) + (3*4) + (4*2) = 4 + 10 + 12 + 8 = 34

# Element sizes (bytes)
SIZE_U16 = 2
SIZE_LEAF = 3  # type(1) + val(2) - no padding
SIZE_DIST_ENTRY = 6  # class_idx(u16) + prob(f32)
SIZE_FEAT_HEADER = 4  # name_idx(u16) + type_flags(u8) + n_cat(u8)


# =============================================================================
# Varint encoding/decoding (protobuf-style, 7 bits per byte, MSB = continuation)
# =============================================================================


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a varint (1-5 bytes for 32-bit values)."""
    if value < 0:
        raise ValueError(f"varint cannot encode a negative value: {value}")
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
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7
    raise ValueError("varint too long (corrupt .cart file)")


def rebuild_tree_from_cart(
    model_data: dict,
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

        if op == OP_LT:
            _, _, val, left, right = node
            value = floats[val]
            left_tree = rebuild(left)
            right_tree = rebuild(right)
            return [feature_name, "<", value, left_tree, right_tree]
        elif op == OP_EQ:
            _, _, val, left, right = node
            value = strings[cat_vals[val]]
            left_tree = rebuild(left)
            right_tree = rebuild(right)
            return [feature_name, "=", value, left_tree, right_tree]
        elif op == OP_SWITCH:
            # Switch nodes are stored as (feat, op, table_idx, 0, 0); the two
            # trailing placeholders exist so every decision node is a 5-tuple.
            table_idx = node[2]
            case_tables = model_data.get("case_tables", [])
            table = case_tables[table_idx]
            # Rebuild case table as dict: {value: subtree, ...}
            cases = {}
            for cat_val_idx, child_idx in table["cases"]:
                cat_val = strings[cat_vals[cat_val_idx]]
                cases[cat_val] = rebuild(child_idx)
            default_tree = rebuild(table["default"])
            return [feature_name, "switch", cases, default_tree]
        else:
            raise ValueError(f"Unknown op type: {op}")

    tree_offset = model_data["tree_offsets"][tree_idx]
    return rebuild(tree_offset)
