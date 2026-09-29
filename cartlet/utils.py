"""
Utility functions for cartlet.

This module contains:
- Tree introspection utilities (is_leaf, count_nodes, etc.)
- Logging helpers
"""

import logging
import math
from typing import Any

# =============================================================================
# Logging
# =============================================================================


def default_logger():
    """Simple fallback logger if none provided."""
    return logging.getLogger("cartlet")


# =============================================================================
# Tree structure utilities
# =============================================================================

# Nested-tree node shapes (list-based):
#   decision node = [feature, op, value, left, right]  (5 elements)
#   regression leaf = [mean, variance, n]              (3 numbers)
# (classification leaves are str or dict, not lists.)
DECISION_ARITY = 5
REGRESSION_LEAF_ARITY = 3


def is_leaf(node: Any) -> bool:
    """
    Check if node is a leaf.

    Leaf types:
    - str: classification label
    - dict: probability distribution
    - list of 3 numbers: regression [mean, var, n]
    """
    if isinstance(node, str):
        return True
    if isinstance(node, dict):
        return True
    if isinstance(node, list) and len(node) == REGRESSION_LEAF_ARITY:
        return all(isinstance(x, (int, float)) for x in node)
    return False


def is_decision_node(node: Any) -> bool:
    """
    Check if node is a decision node.

    Format: [feature, op, value, left, right]
    """
    return isinstance(node, list) and len(node) == DECISION_ARITY


def is_switch_node(node: Any) -> bool:
    """Return whether ``node`` is a nested categorical switch."""
    return isinstance(node, list) and len(node) == 4 and node[1] == "switch"


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


def is_missing_for_feature(value: Any, spec: Any) -> bool:
    """Apply node-equivalent missing detection from a feature specification."""
    if value is None:
        return True
    if getattr(spec, "type", None) == "num":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return math.isnan(numeric)
    return _is_self_unequal(value)


def get_children(node: Any) -> tuple[Any, Any]:
    """Get left and right children from a decision node."""
    return node[3], node[4]  # [feature, op, value, left, right]


def collapse_distributions(node: Any) -> Any:
    """
    Return a copy of ``node`` with every probability-distribution leaf reduced
    to its most likely class label (a bare string).

    Used by ``export(..., store_distributions=False)`` for the JSON/JSONL/
    pickle codecs so the produced file matches the documented behaviour (leaves
    store only the best class). Regression leaves (``[mean, var, n]``) and plain
    string leaves are returned unchanged.
    """
    if isinstance(node, dict):
        return max(node, key=lambda k: node[k])
    if is_decision_node(node):
        feature, op, value, left, right = node
        return [
            feature,
            op,
            value,
            collapse_distributions(left),
            collapse_distributions(right),
        ]
    return node


def count_nodes(node: Any) -> int:
    """
    Count total nodes in a decision tree.

    Args:
        node: Tree node

    Returns:
        Total number of nodes in tree
    """
    if is_leaf(node):
        return 1
    if is_decision_node(node):
        left, right = get_children(node)
        return 1 + count_nodes(left) + count_nodes(right)
    return 0


def count_leaves(node: Any) -> int:
    """
    Count leaf nodes in a decision tree.

    Args:
        node: Tree node

    Returns:
        Number of leaf nodes
    """
    if is_leaf(node):
        return 1
    if is_decision_node(node):
        left, right = get_children(node)
        return count_leaves(left) + count_leaves(right)
    return 0


def max_depth(node: Any, depth: int = 0) -> int:
    """
    Calculate maximum depth of a decision tree.

    Args:
        node: Tree node
        depth: Current depth

    Returns:
        Maximum depth from this node
    """
    if is_leaf(node):
        return depth
    if is_decision_node(node):
        left, right = get_children(node)
        return max(max_depth(left, depth + 1), max_depth(right, depth + 1))
    return depth


def _collect_stats(node: Any, depth: int = 0) -> tuple:
    """
    Collect all tree statistics in a single traversal.

    Args:
        node: Tree node
        depth: Current depth

    Returns:
        Tuple of (total_nodes, leaf_nodes, max_depth)
    """
    if is_leaf(node):
        return (1, 1, depth)
    if is_decision_node(node):
        left, right = get_children(node)
        left_stats = _collect_stats(left, depth + 1)
        right_stats = _collect_stats(right, depth + 1)
        return (
            1 + left_stats[0] + right_stats[0],
            left_stats[1] + right_stats[1],
            max(left_stats[2], right_stats[2]),
        )
    return (0, 0, depth)


def tree_stats(node: Any) -> dict:
    """
    Get statistics about a decision tree.

    Args:
        node: Tree root

    Returns:
        Dict with statistics
    """
    total_nodes, leaf_nodes, depth = _collect_stats(node)
    return {
        "decision_nodes": total_nodes - leaf_nodes,
        "leaf_nodes": leaf_nodes,
        "max_depth": depth,
        "total_nodes": total_nodes,
    }


def eval_tree(
    node: Any,
    vector: list[Any],
    name_to_col: dict[str, int],
    return_dist: bool = False,
    *,
    missing: str = "error",
    tree_idx: int = 0,
    feature_specs: list[Any] | None = None,
    indices: tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], int]]
    | None = None,
    id_trees: list[Any] | None = None,
) -> Any:
    """
    Evaluate a nested tree structure (used by DecisionTree.predict).

    This handles the in-memory tree representation used during training,
    not the .cart binary format.

    Args:
        node: Tree node (nested list/dict structure)
        vector: Feature values
        name_to_col: Mapping of feature names to column indices
        return_dist: Return distribution dict (for classification leaves)
        missing: Whether a tested missing value raises or follows the
            right/default branch. Numeric float-convertible NaNs and categorical
            non-string self-unequal scalars are missing.

    Returns:
        Prediction (class label, distribution, or regression value)
    """
    result, _, _ = _eval_tree(
        node,
        vector,
        name_to_col,
        return_dist,
        missing,
        tree_idx,
        feature_specs,
        indices,
        False,
        id_trees,
    )
    return result


def eval_tree_path(
    node: Any,
    vector: list[Any],
    name_to_col: dict[str, int],
    *,
    missing: str = "error",
    tree_idx: int = 0,
    feature_specs: list[Any] | None = None,
    indices: tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], int]],
) -> tuple[Any, int, list[dict[str, Any]]]:
    """Evaluate a nested tree and return its prediction, leaf ID, and path."""
    return _eval_tree(
        node,
        vector,
        name_to_col,
        False,
        missing,
        tree_idx,
        feature_specs,
        indices,
        True,
    )


def _eval_tree(
    node: Any,
    vector: list[Any],
    name_to_col: dict[str, int],
    return_dist: bool,
    missing: str,
    tree_idx: int,
    feature_specs: list[Any] | None,
    indices: tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], int]]
    | None = None,
    collect_path: bool = False,
    id_trees: list[Any] | None = None,
) -> tuple[Any, int, list[dict[str, Any]]]:
    if missing not in ("error", "right"):
        raise ValueError("missing must be 'error' or 'right'")
    current = node
    address: tuple[Any, ...] = ()
    path: list[dict[str, Any]] = []
    while is_decision_node(current) or is_switch_node(current):
        is_switch = is_switch_node(current)
        if is_switch:
            feature, op, cases, default = current
            value = None
            left = right = None
        else:
            feature, op, value, left, right = current
        if isinstance(feature, str):
            if feature not in name_to_col:
                raise KeyError(
                    f"Decision references unknown feature {feature!r}; "
                    f"known features: {sorted(name_to_col)}"
                )
            col = name_to_col[feature]
            name = feature
        else:
            col = int(feature)
            name = next((n for n, i in name_to_col.items() if i == col), str(col))

        # Deliberately outside every conversion/comparison handler.
        feat_val = vector[col] if col < len(vector) else None
        numeric_value = None
        if op in ("<=", "<"):
            is_missing = feat_val is None
            if not is_missing:
                try:
                    numeric_value = float(feat_val)
                except (TypeError, ValueError):
                    pass
                else:
                    is_missing = math.isnan(numeric_value)
        else:
            is_missing = feat_val is None or _is_self_unequal(feat_val)
        spec = None
        if feature_specs and col < len(feature_specs):
            spec = feature_specs[col]
            if getattr(spec, "dtype", None) == "bool" and not is_missing:
                from .types import normalize_bool

                feat_val = normalize_bool(feat_val)
                if op in ("<=", "<"):
                    numeric_value = float(feat_val)

        decision_id = indices[0][address] if indices else -1
        if is_missing and missing == "error":
            from .runner import MissingFeatureError

            # Ordinary prediction deliberately carries no node-ID index. Build
            # the current writer-order numbering only on this exceptional path
            # so in-place model edits cannot affect prediction correctness or
            # add tree-size work to successful predictions.
            if indices is None:
                if id_trees is None:
                    current_indices = build_tree_indices([node])[0]
                else:
                    current_indices = build_tree_indices(id_trees)[tree_idx]
                decision_id = current_indices[0][address]

            raise MissingFeatureError(
                f"feature {col} ({name!r}) is missing at "
                f"tree {tree_idx} node {decision_id}"
            )

        if op in ("<=", "<"):
            go_left = False
            if not is_missing and numeric_value is not None:
                threshold = float(value)
                go_left = (
                    (numeric_value < threshold)
                    if op == "<"
                    else (numeric_value <= threshold)
                )
            branch = "left" if go_left else "right"
            next_address = address + ((0 if go_left else 1),)
            current = left if go_left else right
        elif is_switch:
            items = list(cases.items()) if isinstance(cases, dict) else cases
            key = None if is_missing else str(feat_val)
            selected = None
            for case_idx, (case_value, subtree) in enumerate(items):
                if spec is not None and getattr(spec, "dtype", None) == "bool":
                    from .types import normalize_bool

                    case_value = normalize_bool(case_value)
                if key == str(case_value):
                    selected = (case_idx, subtree)
                    break
            if selected is None:
                branch = "default"
                next_address = address + ("default",)
                current = default
            else:
                branch = "case"
                next_address = address + (("case", selected[0]),)
                current = selected[1]
        else:
            predicate = value
            if spec is not None and getattr(spec, "dtype", None) == "bool":
                from .types import normalize_bool

                predicate = normalize_bool(predicate)
            go_left = not is_missing and str(feat_val) == str(predicate)
            branch = "left" if go_left else "right"
            next_address = address + ((0 if go_left else 1),)
            current = left if go_left else right
        if collect_path:
            path_value: str | float | None
            if is_switch:
                path_value = None
            elif op == "=":
                predicate = value
                if spec is not None and getattr(spec, "dtype", None) == "bool":
                    from .types import normalize_bool

                    predicate = normalize_bool(predicate)
                path_value = str(predicate)
            else:
                path_value = float(value)
            path.append(
                {
                    "node": decision_id,
                    "feature": col,
                    "name": name,
                    "op": op,
                    "value": path_value,
                    "branch": branch,
                }
            )
        address = next_address

    if isinstance(current, str):
        result: Any = {current: 1.0} if return_dist else current
    elif isinstance(current, dict):
        result = current if return_dist else max(current, key=lambda k: current[k])
    elif is_leaf(current):
        result = current[0]
    else:
        raise ValueError(f"Unknown node type: {type(current)}")
    leaf_id = indices[1][address] if indices else -1
    return result, leaf_id, path


def build_tree_indices(
    trees: list[Any],
) -> list[tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], int]]]:
    """Number nested nodes exactly as the .cart writer numbers its arrays."""
    decision_count = 0
    leaf_count = 0
    result = []

    for root in trees:
        decisions: dict[tuple[Any, ...], int] = {}
        leaves: dict[tuple[Any, ...], int] = {}

        def walk(
            current: Any,
            address: tuple[Any, ...],
            decisions: dict[tuple[Any, ...], int] = decisions,
            leaves: dict[tuple[Any, ...], int] = leaves,
        ) -> None:
            nonlocal decision_count, leaf_count
            if is_decision_node(current):
                decisions[address] = decision_count
                decision_count += 1
                walk(current[3], address + (0,))
                walk(current[4], address + (1,))
            elif is_switch_node(current):
                decisions[address] = decision_count
                decision_count += 1
                walk(current[3], address + ("default",))
                cases = current[2]
                items = list(cases.items()) if isinstance(cases, dict) else cases
                for case_idx, (_, subtree) in enumerate(items):
                    walk(subtree, address + (("case", case_idx),))
            else:
                leaves[address] = leaf_count
                leaf_count += 1

        walk(root, ())
        result.append((decisions, leaves))
    return result
