"""
Utility functions for cartlet.

This module contains:
- Tree introspection utilities (is_leaf, count_nodes, etc.)
- Logging helpers
"""

import logging
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
    if isinstance(node, list) and len(node) == 3:
        return all(isinstance(x, (int, float)) for x in node)
    return False


def is_decision_node(node: Any) -> bool:
    """
    Check if node is a decision node.

    Format: [feature, op, value, left, right]
    """
    return isinstance(node, list) and len(node) == 5


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

    Returns:
        Prediction (class label, distribution, or regression value)
    """
    # Leaf: string class label
    if isinstance(node, str):
        return {node: 1.0} if return_dist else node

    # Leaf: dict distribution
    if isinstance(node, dict):
        if return_dist:
            return node
        # Return class with highest probability
        return max(node, key=lambda k: node[k])

    # Leaf: regression [mean, variance, n]
    if (
        isinstance(node, list)
        and len(node) == 3
        and all(isinstance(x, (int, float)) for x in node)
    ):
        return node[0]  # Return mean

    # Decision node: [feature, op, value, left, right]
    if isinstance(node, list) and len(node) == 5:
        feature, op, value, left, right = node

        # Get feature index. Fail loudly on unknown feature names rather than
        # silently routing the comparison to column 0.
        if isinstance(feature, str):
            if feature not in name_to_col:
                raise KeyError(
                    f"Decision references unknown feature {feature!r}; "
                    f"known features: {sorted(name_to_col)}"
                )
            col = name_to_col[feature]
        else:
            col = int(feature)

        feat_val = vector[col] if col < len(vector) else None

        if op == "<":
            go_left = feat_val is not None and float(feat_val) <= float(value)
        else:  # op == "="
            go_left = feat_val is not None and str(feat_val) == str(value)

        next_node = left if go_left else right
        return eval_tree(next_node, vector, name_to_col, return_dist)

    raise ValueError(f"Unknown node type: {type(node)}")
