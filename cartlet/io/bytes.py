"""
Binary format writer for decision trees and forests.

See cart_format.py for format specification and constants.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import struct
import tempfile
from typing import Any

from .cart_format import (
    DTYPE_MAP,
    FEAT_MASK,
    FLAG_HAS_DISTRIBUTIONS,
    FLAG_IS_FOREST,
    FLAG_IS_REGRESSION,
    FLAG_IS_XGBOOST,
    LEAF_CLASS,
    LEAF_CLASS_DIST,
    LEAF_FLAG,
    LEAF_FLOAT,
    MAGIC,
    OP_EQ,
    OP_LT,
    OP_SHIFT,
    OP_SWITCH,
    TYPE_CAT,
    TYPE_NUM,
    VERSION,
    encode_varint,
)


class ByteWriter:
    """
    Build-then-serialize encoder for the .cart binary format.

    The writer is used in two phases:

    1. Build phase: call `_flatten_node(tree_root, name_to_col)` once per tree.
       This walks the in-memory nested-list tree and interns every value into
       a deduped pool (strings, floats, cat_values, distributions, case_tables)
       while appending one entry to `decisions` or `leaves` for each node.
       Each call returns the root node's index (with `LEAF_FLAG` set if the
       root is a leaf). The returned index is recorded in `tree_offsets`, so a
       single writer can hold multiple trees (used for forests / XGBoost).

    2. Write phase: call `write(path, ...)` (or `write_tree_bytes` /
       `write_forest_bytes` for the common one-shot cases). This emits the
       fixed-size header, then each pool in the order defined by
       `cartlet/io/cart_format.py`, then the decision and leaf arrays.

    Pool deduplication matters: most trees reuse the same handful of strings,
    floats, and categorical values across many nodes. Each `_add_*` method
    returns the existing index when its value has been seen before, which is
    what keeps the .cart format compact even for large forests.

    See `cartlet/io/cart_format.py` for the byte layout and opcode encoding.
    """

    def __init__(self, store_distributions: bool = False, is_xgboost: bool = False):
        self.store_distributions = store_distributions
        self.is_xgboost = is_xgboost
        self.strings: list[str] = []
        self.string_to_idx: dict[str, int] = {}
        self.floats: list[float] = []
        self.float_to_idx: dict[float, int] = {}
        self.cat_values: list[int] = []  # string indices for categorical comparisons
        self.cat_val_to_idx: dict[int, int] = {}  # string_idx -> cat_values index
        # Distributions: list of [(class_idx, prob), ...]
        self.distributions: list[list[tuple[int, float]]] = []
        self.dist_to_idx: dict[tuple, int] = {}  # hashable dist -> index
        # Case tables: list of (default_child, [(cat_val_idx, child_idx), ...])
        self.case_tables: list[tuple[int, list[tuple[int, int]]]] = []
        # Separate arrays for decisions and leaves
        # For OP_LT/OP_EQ: (feat, op, val, left, right)
        # For OP_SWITCH: (feat, op, table_idx, 0, 0) - left/right unused
        self.decisions: list[tuple[int, int, int, int, int]] = []
        self.leaves: list[tuple[int, int]] = []  # type, val
        self.tree_offsets: list[int] = []  # index into decisions array

    def _add_string(self, s: str) -> int:
        """Add string to pool, return index."""
        if s in self.string_to_idx:
            return self.string_to_idx[s]
        idx = len(self.strings)
        self.strings.append(s)
        self.string_to_idx[s] = idx
        return idx

    def _add_float(self, f: float) -> int:
        """Add float to pool, return index."""
        if f in self.float_to_idx:
            return self.float_to_idx[f]
        idx = len(self.floats)
        self.floats.append(f)
        self.float_to_idx[f] = idx
        return idx

    def _add_cat_value(self, string_idx: int) -> int:
        """Add categorical value (as string index) to cat pool, return index."""
        if string_idx in self.cat_val_to_idx:
            return self.cat_val_to_idx[string_idx]
        idx = len(self.cat_values)
        self.cat_values.append(string_idx)
        self.cat_val_to_idx[string_idx] = idx
        return idx

    def _add_distribution(self, dist: dict) -> int:
        """Add distribution to pool, return index."""
        # Convert dict to sorted list of (class_idx, prob)
        items = []
        for cls, prob in sorted(dist.items(), key=lambda x: -x[1]):
            class_idx = self._add_string(str(cls))
            items.append((class_idx, float(prob)))

        # Check for duplicate
        key = tuple(items)
        if key in self.dist_to_idx:
            return self.dist_to_idx[key]

        idx = len(self.distributions)
        self.distributions.append(items)
        self.dist_to_idx[key] = idx
        return idx

    def _add_leaf(self, leaf_type: int, val: int) -> int:
        """Add leaf node, return its index with LEAF_FLAG set."""
        idx = len(self.leaves)
        self.leaves.append((leaf_type, val))
        return idx | LEAF_FLAG

    def _add_case_table(self, default_child: int, cases: list[tuple[int, int]]) -> int:
        """Add case table, return index."""
        idx = len(self.case_tables)
        self.case_tables.append((default_child, cases))
        return idx

    @staticmethod
    def _resolve_feat_idx(feature: Any, name_to_col: dict[str, int]) -> int:
        """
        Map a tree-node feature reference (name or numeric index) to a column.

        Raises ValueError on unknown names rather than silently aliasing to
        column 0, which used to mask schema mismatches between trainer output
        and the writer.
        """
        if isinstance(feature, int) and not isinstance(feature, bool):
            return feature
        if feature in name_to_col:
            return name_to_col[feature]
        if isinstance(feature, str) and feature.isdigit():
            return int(feature)
        raise ValueError(
            f"Unknown feature reference {feature!r}: not in column map and "
            "not a numeric index"
        )

    def _flatten_node(self, node: Any, name_to_col: dict[str, int]) -> int:
        """Recursively flatten a node, return its index."""
        # Leaf: string (class label)
        if isinstance(node, str):
            class_idx = self._add_string(node)
            return self._add_leaf(LEAF_CLASS, class_idx)

        # Leaf: dict (distribution)
        if isinstance(node, dict):
            if self.store_distributions and len(node) > 1:
                dist_idx = self._add_distribution(node)
                return self._add_leaf(LEAF_CLASS_DIST, dist_idx)
            # Collapse to best class only. Use the argmax rather than the
            # first key so the collapsed leaf agrees with the distribution's
            # top class regardless of dict insertion order.
            best_class = max(node.items(), key=lambda kv: kv[1])[0]
            class_idx = self._add_string(str(best_class))
            return self._add_leaf(LEAF_CLASS, class_idx)

        # Leaf: [mean, variance, n] - regression
        if (
            isinstance(node, list)
            and len(node) == 3
            and all(isinstance(x, (int, float)) for x in node)
        ):
            float_idx = self._add_float(float(node[0]))
            return self._add_leaf(LEAF_FLOAT, float_idx)

        # Decision node: [feature, op, value, left, right]
        if isinstance(node, list) and len(node) == 5:
            feature, op, value, left_node, right_node = node
            feat_idx = self._resolve_feat_idx(feature, name_to_col)

            if op == "<":
                val_idx = self._add_float(float(value))
                op_type = OP_LT
            else:  # op == "="
                str_idx = self._add_string(str(value))
                val_idx = self._add_cat_value(str_idx)
                op_type = OP_EQ

            # Reserve decision slot
            dec_idx = len(self.decisions)
            self.decisions.append((0, 0, 0, 0, 0))  # placeholder

            # Recursively process children
            left_idx = self._flatten_node(left_node, name_to_col)
            right_idx = self._flatten_node(right_node, name_to_col)

            # Fill in the decision node
            self.decisions[dec_idx] = (feat_idx, op_type, val_idx, left_idx, right_idx)
            return dec_idx

        # Switch node: [feature, "switch", cases_dict, default_node]
        # cases_dict = {value: subtree, ...} or [(value, subtree), ...]
        if isinstance(node, list) and len(node) == 4 and node[1] == "switch":
            feature, _, cases, default_node = node
            feat_idx = self._resolve_feat_idx(feature, name_to_col)

            # Reserve decision slot
            dec_idx = len(self.decisions)
            self.decisions.append((0, 0, 0, 0, 0))  # placeholder

            # Process default first
            default_idx = self._flatten_node(default_node, name_to_col)

            # Process cases
            cases_items = list(cases.items()) if isinstance(cases, dict) else cases

            case_list: list[tuple[int, int]] = []
            for val, subtree in cases_items:
                str_idx = self._add_string(str(val))
                cat_val_idx = self._add_cat_value(str_idx)
                child_idx = self._flatten_node(subtree, name_to_col)
                case_list.append((cat_val_idx, child_idx))

            table_idx = self._add_case_table(default_idx, case_list)

            # Fill in the switch decision node
            self.decisions[dec_idx] = (feat_idx, OP_SWITCH, table_idx, 0, 0)
            return dec_idx

        raise ValueError(f"Unknown node type: {type(node)}")

    def add_tree(self, model: Any, name_to_col: dict[str, int]) -> None:
        """Add a tree to the writer."""
        root_idx = self._flatten_node(model, name_to_col)
        self.tree_offsets.append(root_idx)

    # Fixed-width field limits imposed by the .cart format. Exceeding any of
    # these must raise a clear error rather than silently truncate (feature
    # index) or emit a cryptic struct.error (everything else).
    _MAX_FEAT_IDX = FEAT_MASK  # 6-bit packed feature index (0-63)
    _MAX_U16 = 0xFFFF

    def _validate_capacity(
        self,
        feature_info: list[tuple[int, int, int, list[int]]],
        string_offsets: list[int],
        class_labels: list[str],
    ) -> None:
        """Reject models whose pools overflow the format's fixed-width fields."""

        def _check(count: int, limit: int, what: str) -> None:
            if count > limit:
                raise ValueError(
                    f"{what} ({count}) exceeds the .cart format limit of {limit}. "
                    "Reduce the model size or use a full-fidelity format "
                    "(.json/.pkl)."
                )

        _check(len(feature_info), self._MAX_U16, "number of features")
        _check(len(class_labels), self._MAX_U16, "number of classes")
        _check(len(self.tree_offsets), self._MAX_U16, "number of trees")
        _check(len(self.cat_values), self._MAX_U16, "number of categorical values")
        _check(len(self.distributions), self._MAX_U16, "number of distributions")
        _check(len(self.case_tables), self._MAX_U16, "number of case tables")

        # String offsets are u16 into the string blob.
        if string_offsets:
            _check(max(string_offsets), self._MAX_U16, "string-table byte offset")

        # Per-feature categorical value count is a u8.
        for _name_idx, _type_flags, n_cat, _cat_indices in feature_info:
            _check(n_cat, 0xFF, "categorical values for a single feature")

        # Decision-node feature index is a packed 6-bit field; val is a u16.
        for feat, _op, val, _left, _right in self.decisions:
            _check(feat, self._MAX_FEAT_IDX, "feature index")
            _check(val, self._MAX_U16, "decision-node value index")

        # Leaf val is a u16 index into floats/strings/distributions.
        for _leaf_type, val in self.leaves:
            _check(val, self._MAX_U16, "leaf value index")

    def write(
        self,
        path: str,
        feature_specs: list[Any],
        class_labels: list[str],
        is_regression: bool,
        metadata: dict | None = None,
    ) -> None:
        """Write binary format to file."""
        # Prepare metadata JSON
        meta_bytes = b""
        if metadata:
            meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        # Build feature info and add feature names to string table
        feature_info: list[tuple[int, int, int, list[int]]] = []
        for spec in feature_specs:
            name_idx = self._add_string(spec.name)
            dtype_val = DTYPE_MAP.get(spec.dtype, 0)
            type_val = TYPE_NUM if spec.type == "num" else TYPE_CAT
            type_flags = type_val | (dtype_val << 2)

            # Add known categorical values
            cat_indices: list[int] = []
            if spec.values:
                for v in sorted(spec.values, key=str):
                    cat_indices.append(self._add_string(str(v)))

            feature_info.append((name_idx, type_flags, len(cat_indices), cat_indices))

        # Add class labels to string table
        class_indices = [self._add_string(str(c)) for c in class_labels]

        # Build string table
        string_data = b""
        string_offsets: list[int] = []
        for s in self.strings:
            string_offsets.append(len(string_data))
            string_data += s.encode("utf-8") + b"\x00"

        # Fail fast (before opening the file) if anything overflows a
        # fixed-width field, rather than truncating or raising struct.error.
        self._validate_capacity(feature_info, string_offsets, class_labels)

        # Build flags
        flags = 0
        if len(self.tree_offsets) > 1:
            flags |= FLAG_IS_FOREST
        if is_regression:
            flags |= FLAG_IS_REGRESSION
        if self.distributions:
            flags |= FLAG_HAS_DISTRIBUTIONS
        if self.is_xgboost:
            flags |= FLAG_IS_XGBOOST

        # Write file
        with open(path, "wb") as f:
            # Header (34 bytes)
            f.write(MAGIC)
            f.write(struct.pack("<H", VERSION))
            f.write(struct.pack("<H", flags))
            f.write(struct.pack("<H", len(feature_specs)))
            f.write(struct.pack("<H", len(class_labels)))
            f.write(struct.pack("<H", len(self.tree_offsets)))
            f.write(struct.pack("<I", len(self.decisions)))  # n_decisions
            f.write(struct.pack("<I", len(self.leaves)))  # n_leaves
            f.write(struct.pack("<I", len(self.floats)))  # n_floats
            f.write(struct.pack("<H", len(self.cat_values)))
            f.write(struct.pack("<H", len(self.distributions)))  # n_dists
            f.write(struct.pack("<H", len(self.case_tables)))  # n_case_tables
            f.write(struct.pack("<H", len(meta_bytes)))

            # String table
            f.write(struct.pack("<H", len(self.strings)))
            for off in string_offsets:
                f.write(struct.pack("<H", off))
            f.write(string_data)

            # Feature table
            for name_idx, type_flags, n_cat, cat_indices in feature_info:
                f.write(struct.pack("<H", name_idx))
                f.write(struct.pack("<B", type_flags))
                f.write(struct.pack("<B", n_cat))
                for ci in cat_indices:
                    f.write(struct.pack("<H", ci))

            # Class table
            for ci in class_indices:
                f.write(struct.pack("<H", ci))

            # Float pool
            for fv in self.floats:
                f.write(struct.pack("<f", fv))

            # Cat value pool (string indices)
            for si in self.cat_values:
                f.write(struct.pack("<H", si))

            # Tree offsets (always written - needed for single-leaf trees too)
            # Use varint for tree offsets too
            for off in self.tree_offsets:
                f.write(encode_varint(off))

            # Decision nodes (variable size)
            # OP_LT/OP_EQ: feat_op(1) + val(2) + left(varint) + right(varint)
            # OP_SWITCH: feat_op(1) + table_idx(2) (no left/right)
            for feat, op, val, left, right in self.decisions:
                feat_op = (feat & FEAT_MASK) | (op << OP_SHIFT)
                f.write(struct.pack("<BH", feat_op, val))
                if op != OP_SWITCH:
                    f.write(encode_varint(left))
                    f.write(encode_varint(right))

            # Leaf nodes (3 bytes each - no padding)
            for leaf_type, val in self.leaves:
                f.write(struct.pack("<BH", leaf_type, val))

            # Distributions (if FLAG_HAS_DISTRIBUTIONS)
            # Format: for each dist: n_entries(u16) + [class_idx(u16) + prob(f32)]...
            for dist in self.distributions:
                f.write(struct.pack("<H", len(dist)))
                for class_idx, prob in dist:
                    f.write(struct.pack("<Hf", class_idx, prob))

            # Case tables (for OP_SWITCH nodes)
            # Format: for each table: n_cases(u16) + default(varint) +
            #         [cat_val_idx(u16) + child(varint)]...
            for default_child, cases in self.case_tables:
                f.write(struct.pack("<H", len(cases)))
                f.write(encode_varint(default_child))
                for cat_val_idx, child_idx in cases:
                    f.write(struct.pack("<H", cat_val_idx))
                    f.write(encode_varint(child_idx))

            # Metadata
            if meta_bytes:
                f.write(meta_bytes)


def write_tree_bytes(
    path: str,
    model: Any,
    feature_specs: list[Any],
    name_to_col: dict[str, int],
    class_labels: list[str],
    is_regression: bool,
    metadata: dict | None = None,
    store_distributions: bool = False,
    is_xgboost: bool = False,
) -> None:
    """Write a single decision tree to binary format."""
    writer = ByteWriter(store_distributions=store_distributions, is_xgboost=is_xgboost)
    writer.add_tree(model, name_to_col)
    writer.write(path, feature_specs, class_labels, is_regression, metadata)


def write_forest_bytes(
    path: str,
    trees: list[Any],
    feature_specs: list[Any],
    name_to_col: dict[str, int],
    class_labels: list[str],
    is_regression: bool,
    metadata: dict | None = None,
    store_distributions: bool = False,
    is_xgboost: bool = False,
) -> None:
    """Write a random forest or XGBoost model to binary format."""
    writer = ByteWriter(store_distributions=store_distributions, is_xgboost=is_xgboost)
    for tree in trees:
        writer.add_tree(tree, name_to_col)
    writer.write(path, feature_specs, class_labels, is_regression, metadata)


def _strip_cli_code(code: bytes) -> bytes:
    """
    Strip CLI scaffolding from the bundled Python runner, leaving only the
    library-style code (``Predictor`` class, ``load_cart``, helpers).

    Removes ``def main(...)`` and the ``if __name__ == "__main__"`` block,
    so callers that ``import`` the bundled file get just the library API.
    """
    lines = code.decode("utf-8").split("\n")
    new_lines: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("def main(") or line.startswith('if __name__ == "__main__"'):
            skip = True
            continue
        if skip:
            # Resume on the next top-level def/class.
            is_top_level = line and not line.startswith((" ", "\t"))
            is_definition = line.startswith("def ") or line.startswith("class ")
            if is_top_level and is_definition:
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)

    while len(new_lines) > 1 and new_lines[-1].strip() == "":
        new_lines.pop()
    new_lines.append("")

    return "\n".join(new_lines).encode("utf-8")


def _read_or_convert_cart(model_path: str, model_format: str | None) -> bytes:
    """
    Return the raw `.cart` bytes for ``model_path``, converting via a
    temporary `.cart` file if the input is in a different format.

    Importing the model classes is deferred to this helper to keep
    `io/bytes.bundle` callable from low-level code without dragging in
    the supervised-model classes for already-`.cart` inputs.
    """
    with open(model_path, "rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        with open(model_path, "rb") as f:
            return f.read()

    from .. import DecisionTree, RandomForest, _detect_is_forest

    tmp_fd, tmp_cart = tempfile.mkstemp(suffix=".cart")
    os.close(tmp_fd)
    try:
        is_forest = _detect_is_forest(model_path, format=model_format)
        model = RandomForest() if is_forest else DecisionTree()
        model.load_model(model_path, format=model_format)
        model.export(tmp_cart)
        with open(tmp_cart, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_cart):
            os.unlink(tmp_cart)


def bundle(
    model_path: str | None,
    output_path: str,
    library_only: bool = False,
    embed_model: bool = True,
    model_format: str | None = None,
) -> None:
    """
    Bundle a model with the bundled Python runner into a single
    executable or library file.

    Inputs that are not already ``.cart`` (e.g. ``.json``, ``.jsonl``,
    ``.pkl``, ``.skl``, or a custom suffix described by ``model_format``) are
    transparently converted to a temporary ``.cart`` file before embedding, so
    callers do not need to pre-convert the model themselves.

    .. warning::
        Converting a ``.pkl``/``.pickle`` or ``.skl``/``.joblib`` input
        unpickles it, which executes arbitrary code embedded in the file. Only
        bundle pickle/joblib models from a trusted source. ``.cart``/``.json``/
        ``.jsonl`` inputs and the runners themselves never unpickle.

    The model is base64-encoded and inserted as a module-level constant; the
    resulting file has no dependencies beyond the standard library.

    Args:
        model_path: Path to model file (required unless embed_model=False).
        output_path: Path for bundled output file.
        library_only: If True, strip CLI code (main + ``__main__`` block) to
                      produce a smaller file suitable for importing as a
                      library.
        embed_model: If False, output runner code only without embedded model.
                     Use this to create a standalone library that loads models
                     at runtime via Predictor(path).
        model_format: Explicit format of the input model (e.g. ``"jsonl"``);
                      forwarded to the underlying loader when the file lives
                      under a non-standard suffix.

    Example:
        bundle("model.cart", "predictor.py")
        # Then: ./predictor.py feature1 feature2 ...

        bundle("model.json", "lib.py", library_only=True)
        # Bundles a JSON model (auto-converted to .cart) as a Python library.

        bundle("model.g2p.gz", "predictor.py", model_format="jsonl")
        # Bundles a file that is internally JSONL but uses a custom suffix.

        bundle(None, "cart.py", library_only=True, embed_model=False)
        # Then: from cart import Predictor; p = Predictor("model.cart"); p.predict([...])
    """
    runner_path = os.path.join(os.path.dirname(__file__), "..", "bundled", "predict.py")
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"Runner not found: {runner_path}")

    with open(runner_path, "rb") as f:
        runner_code = f.read()

    model_data: bytes | None = None
    if embed_model:
        if model_path is None:
            raise ValueError("model_path is required when embed_model=True")
        model_data = _read_or_convert_cart(model_path, model_format)

    if library_only:
        runner_code = _strip_cli_code(runner_code)

    runner_text = runner_code.decode("utf-8")

    if embed_model and model_data:
        model_b64 = base64.b64encode(model_data).decode("ascii")

        lines = runner_text.split("\n")
        insert_idx = len(lines)
        for i, line in enumerate(lines):
            if line.startswith('if __name__ == "__main__"'):
                insert_idx = i
                break

        embed_block = (
            "\n# " + "=" * 70 + "\n"
            "# Embedded model data (do not edit below this line)\n"
            "# " + "=" * 70 + "\n"
            f'_EMBEDDED_MODEL_B64 = "{model_b64}"\n'
        )

        lines.insert(insert_idx, embed_block)
        bundled_code = "\n".join(lines)
    else:
        bundled_code = runner_text

    with open(output_path, "wb") as f:
        f.write(bundled_code.encode("utf-8"))

    st = os.stat(output_path)
    os.chmod(output_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
