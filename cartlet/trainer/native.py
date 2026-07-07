"""
Native pure-Python decision tree trainer.

Supports:
- Categorical features (equality splits)
- Numerical features (threshold splits)
- Classification (entropy-based information gain)
- Regression (variance reduction)
- Instance weighting
- Reduced error pruning
"""

from __future__ import annotations

import math
import random
from time import time
from typing import TYPE_CHECKING, Any

from ..types import TYPE_NUM
from ..utils import is_leaf
from .base import Trainer, make_classification_distribution, normalize_importances

_PROGRESS_INTERVAL_SEC = 30

# Hard ceiling on native tree depth when no max_depth is set. Real trees are
# rarely deeper than a few dozen levels; hitting this means degenerate data.
# Kept well below CPython's default recursion limit so build/predict/prune
# recursion stays safe.
_MAX_NATIVE_DEPTH = 500

if TYPE_CHECKING:
    from ..tree import DecisionTree


def _count_leaf_correct(node: Any, tree: DecisionTree, val_rows: list[int]) -> int:
    """
    Count the validation rows in ``val_rows`` that ``node`` (a classification
    leaf) predicts correctly.

    Returns 0 for regression-shaped leaves; the caller short-circuits the
    classification path with ``is_regression`` before invoking this.
    """
    if not val_rows:
        return 0
    if isinstance(node, str):
        label: Any = node
    elif isinstance(node, dict):
        label = max(node, key=lambda k: node[k])
    else:
        return 0
    return sum(1 for i in val_rows if tree.y[i] == label)


def _partition_val_rows(
    val_rows: list[int],
    X: list[list[Any]],
    col: int,
    op: str,
    value: Any,
) -> tuple[list[int], list[int]]:
    """
    Split ``val_rows`` into the indices that go left vs. right of a single
    decision node, using the same rules as ``utils.eval_tree``.

    ``"<"`` is the CART convention: left when ``feat_val <= value``.
    ``"="`` is categorical equality. ``None`` / out-of-range columns route
    right, matching the runner.
    """
    left_rows: list[int] = []
    right_rows: list[int] = []
    if op == "<":
        threshold = float(value)
        for i in val_rows:
            row = X[i]
            feat_val = row[col] if col < len(row) else None
            if feat_val is not None and float(feat_val) <= threshold:
                left_rows.append(i)
            else:
                right_rows.append(i)
    else:  # "=" categorical equality
        target = str(value)
        for i in val_rows:
            row = X[i]
            feat_val = row[col] if col < len(row) else None
            if feat_val is not None and str(feat_val) == target:
                left_rows.append(i)
            else:
                right_rows.append(i)
    return left_rows, right_rows


class Native(Trainer):
    """
    Pure Python decision tree trainer.

    Zero external dependencies. Supports both categorical and numerical
    features with entropy-based (classification) or variance-based
    (regression) splitting.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        prune: bool = False,
        random_state: int | None = None,
        max_features: int | None = None,
        criterion: str = "entropy",
        extra_trees: bool = False,
    ):
        """
        Initialize the native trainer.

        Args:
            max_depth: Maximum tree depth (None = unlimited)
            prune: Whether to prune using validation data
            random_state: Random seed for reproducibility
            max_features: Max features to consider per split (None = all)
            criterion: Split criterion for classification ("entropy" or "gini")
            extra_trees: Use random splits instead of best splits (Extra-Trees)
        """
        if criterion not in ("entropy", "gini"):
            raise ValueError(
                f"Unknown criterion {criterion!r}; expected 'entropy' or 'gini'. "
                "(Regression always uses variance reduction regardless.)"
            )
        self.max_depth = max_depth
        self.prune = prune
        self.random_state = random_state
        self.max_features = max_features
        self.criterion = criterion
        self.extra_trees = extra_trees
        self._nodes_built = 0
        self._last_progress_time = 0.0
        self._tree_start_time = 0.0
        self._rng: random.Random | None = None
        self._feature_importances: dict[str, float] = {}
        self._total_samples = 0

    @property
    def supports_categorical(self) -> bool:
        return True

    @property
    def supports_pruning(self) -> bool:
        return True

    def train(
        self,
        tree: DecisionTree,
        train_rows: list[int],
        val_rows: list[int] | None = None,
    ) -> Any:
        """Build the decision tree using pure Python."""
        self._nodes_built = 0
        self._last_progress_time = time()
        self._tree_start_time = time()
        self._rng = (
            random.Random(self.random_state)
            if self.max_features or self.extra_trees
            else None
        )

        # Initialize feature importance tracking (MDI - Mean Decrease in Impurity)
        self._feature_importances = dict.fromkeys(tree.feature_names, 0.0)
        self._total_samples = sum(tree.counts[i] for i in train_rows)

        model = self._build_tree(tree, set(train_rows))

        if self.prune and val_rows:
            model = self._prune_with_validation(tree, model, train_rows, val_rows)

        # Store computed feature importances on the tree
        tree._feature_importances = self._normalize_importances()

        return model

    def _normalize_importances(self) -> dict[str, float]:
        """Normalize feature importances to sum to 1.0."""
        return normalize_importances(self._feature_importances)

    def _build_tree(
        self, tree: DecisionTree, row_ids: set[int], depth: int = 0
    ) -> str | dict[str, float] | list[Any]:
        """Recursively build a subtree for the given row indices."""
        self._nodes_built += 1
        current_time = time()
        if (
            tree.verbose
            and current_time - self._last_progress_time > _PROGRESS_INTERVAL_SEC
        ):
            elapsed = current_time - self._tree_start_time
            tree.logger.info(
                "  Progress: %d nodes built (%.1f sec elapsed)",
                self._nodes_built,
                elapsed,
            )
            self._last_progress_time = current_time

        total = self._sum_counts(tree, row_ids)

        # Guard against unbounded recursion when max_depth is None. A degenerate
        # chain (e.g. an ID-like feature that peels off one row at a time) would
        # otherwise recurse until Python's frame limit and crash with an opaque
        # RecursionError -- and even if built, such a tree could not be inferred
        # recursively. Fail early with actionable guidance instead.
        if depth >= _MAX_NATIVE_DEPTH:
            raise ValueError(
                f"Tree depth exceeded {_MAX_NATIVE_DEPTH} without a max_depth "
                "limit; this usually means near-duplicate rows or an ID-like "
                "feature. Set max_depth or increase min_samples_leaf/"
                "min_samples_split to bound the tree."
            )

        # Early stopping conditions
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(tree, row_ids)
        if total < tree.min_samples_split:
            return self._make_leaf(tree, row_ids)

        # For classification, check if node is pure
        if not tree._is_regression():
            cat_counts, _ = self._cat_counts(tree, row_ids)
            if len(cat_counts) == 1:  # Pure node
                return next(iter(cat_counts))

        # Find best split
        best_name, best_id, best_value, best_op, best_gain = self._best_gain_feat_val(
            tree, row_ids
        )
        if best_name is None or best_id is None or best_gain == 0:
            return self._make_leaf(tree, row_ids)

        # Split data based on operator
        # Note: CART convention - "<" operator means "<=", left branch for value <= threshold
        if best_op == "<":
            yes = {i for i in row_ids if tree.X[i][best_id] <= best_value}
        else:  # "=" (categorical equality)
            yes = {i for i in row_ids if tree.X[i][best_id] == best_value}
        no = row_ids - yes

        # Check split validity
        if not yes or not no:
            return self._make_leaf(tree, row_ids)
        if self._sum_counts(tree, yes) < tree.min_samples_leaf:
            return self._make_leaf(tree, row_ids)
        if self._sum_counts(tree, no) < tree.min_samples_leaf:
            return self._make_leaf(tree, row_ids)

        # Recurse
        left = self._build_tree(tree, yes, depth + 1)
        right = self._build_tree(tree, no, depth + 1)

        # If both branches are identical, collapse
        if left == right:
            return left

        # Track feature importance (MDI: weighted impurity decrease). Credited
        # only once the split is committed as a real decision node -- after the
        # validity checks and the identical-branch collapse above -- so rejected
        # splits do not leave phantom importance.
        # importance = (n_samples / total_samples) * impurity_decrease
        if best_name in self._feature_importances and self._total_samples > 0:
            weighted_importance = (total / self._total_samples) * best_gain
            self._feature_importances[best_name] += weighted_importance

        # Decision node: [feature, op, value, yes_branch, no_branch]
        return [best_name, best_op, best_value, left, right]

    def _prune_with_validation(
        self,
        tree: DecisionTree,
        node: Any,
        train_rows: list[int],
        val_rows: list[int],
    ) -> Any:
        """
        Reduced-error pruning using validation data.

        Standard bottom-up REP: each validation row is routed down a
        single root-to-leaf path (partitioned at each split as we
        recurse), and per-subtree correctness counts are accumulated by
        summing the children's counts rather than re-walking the whole
        validation set through every internal node. Total cost is
        roughly ``O((|train_rows| + |val_rows|) * tree_depth)`` plus a
        single leaf construction at each internal node visited, instead
        of the old ``O(internal_nodes * |val_rows| * avg_subtree_depth)``.

        The candidate leaf at each node is labelled from the *training*
        rows reaching it (the same weighted-majority rule used when the
        tree was grown); the validation set only decides whether
        collapsing to that leaf is at least as accurate as keeping the
        subtree. Scoring the leaf on the training-derived label rather
        than the validation majority avoids the optimistic bias of the
        earlier implementation, which both labelled and scored the leaf
        on the same validation rows.

        Regression trees are returned unchanged: there is no accuracy
        signal to prune against, matching the original behavior.
        """
        pruned, _ = self._prune_recursive(
            tree, node, list(train_rows), val_rows, tree._is_regression()
        )
        return pruned

    def _prune_recursive(
        self,
        tree: DecisionTree,
        node: Any,
        train_rows: list[int],
        val_rows: list[int],
        is_regression: bool,
    ) -> tuple[Any, int]:
        """
        Bottom-up pruning helper.

        ``train_rows`` and ``val_rows`` each contain only the indices
        that reach this subtree (both already filtered by the ancestor
        predicates above us). ``train_rows`` is used to label a candidate
        collapsed leaf; ``val_rows`` is used to score it.

        Returns ``(pruned_node, correct_count)`` where ``correct_count``
        is the number of rows in ``val_rows`` that ``pruned_node``
        predicts correctly. For regression the count is always 0 (the
        caller short-circuits before using it).
        """
        if is_leaf(node) or not isinstance(node, list) or len(node) != 5:
            if is_regression:
                return node, 0
            return node, _count_leaf_correct(node, tree, val_rows)

        feature, op, value, left, right = node

        if isinstance(feature, str):
            if feature not in tree.name_to_col:
                raise KeyError(
                    f"Decision references unknown feature {feature!r}; "
                    f"known features: {sorted(tree.name_to_col)}"
                )
            col = tree.name_to_col[feature]
        else:
            col = int(feature)

        left_train, right_train = _partition_val_rows(
            train_rows, tree.X, col, op, value
        )
        left_val, right_val = _partition_val_rows(val_rows, tree.X, col, op, value)

        pruned_left, left_correct = self._prune_recursive(
            tree, left, left_train, left_val, is_regression
        )
        pruned_right, right_correct = self._prune_recursive(
            tree, right, right_train, right_val, is_regression
        )

        if pruned_left == pruned_right and is_leaf(pruned_left):
            return pruned_left, left_correct + right_correct

        current_node = [feature, op, value, pruned_left, pruned_right]

        if is_regression:
            return current_node, 0

        if not val_rows:
            return current_node, 0

        current_correct = left_correct + right_correct

        # Candidate replacement leaf: labelled from the training rows that
        # reach this node (same weighted-majority rule as when the tree was
        # grown), then scored on the held-out validation rows. Labelling and
        # scoring on independent sets is what keeps REP unbiased.
        leaf = self._make_classification_leaf(tree, train_rows)
        leaf_correct = _count_leaf_correct(leaf, tree, val_rows)

        if leaf_correct >= current_correct:
            return leaf, leaf_correct

        return current_node, current_correct

    # --- Helper methods ---

    def _sum_counts(self, tree: DecisionTree, row_ids) -> int:
        """Sum instance counts for given row indices."""
        return sum(tree.counts[i] for i in row_ids)

    def _cat_counts(self, tree: DecisionTree, row_ids) -> tuple[dict[Any, int], int]:
        """Accumulate counts for each category using instance weights."""
        counts: dict[Any, int] = {}
        total = 0
        for i in row_ids:
            label = tree.y[i]
            counts[label] = counts.get(label, 0) + tree.counts[i]
            total += tree.counts[i]
        return counts, total

    def _entropy_for_rows(self, tree: DecisionTree, row_ids) -> float:
        """Calculate entropy for a set of rows (classification)."""
        counts, total = self._cat_counts(tree, row_ids)
        if total == 0:
            return 0.0
        probs = [count / total for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _gini_for_rows(self, tree: DecisionTree, row_ids) -> float:
        """Calculate Gini impurity for a set of rows (classification)."""
        counts, total = self._cat_counts(tree, row_ids)
        if total == 0:
            return 0.0
        probs = [count / total for count in counts.values()]
        return 1.0 - sum(p * p for p in probs)

    def _mean_for_rows(self, tree: DecisionTree, row_ids) -> tuple[float, float, int]:
        """Calculate weighted mean, variance, and count for regression."""
        if not row_ids:
            return 0.0, 0.0, 0

        total_weight = sum(tree.counts[i] for i in row_ids)
        if total_weight == 0:
            return 0.0, 0.0, 0

        weighted_sum = sum(tree.y[i] * tree.counts[i] for i in row_ids)
        mean = weighted_sum / total_weight

        variance = (
            sum(tree.counts[i] * (tree.y[i] - mean) ** 2 for i in row_ids)
            / total_weight
        )

        return mean, variance, total_weight

    def _variance_for_rows(self, tree: DecisionTree, row_ids) -> float:
        """Calculate weighted variance for a set of rows (regression)."""
        _, variance, _ = self._mean_for_rows(tree, row_ids)
        return variance

    def _impurity_for_rows(self, tree: DecisionTree, row_ids) -> float:
        """Calculate impurity (entropy/gini for classification, variance for regression)."""
        if tree._is_regression():
            return self._variance_for_rows(tree, row_ids)
        if self.criterion == "gini":
            return self._gini_for_rows(tree, row_ids)
        return self._entropy_for_rows(tree, row_ids)

    def _best_gain_categorical(
        self,
        tree: DecisionTree,
        row_ids: set[int],
        feat_id: int,
        impurity0: float | None = None,
    ) -> tuple[Any, float]:
        """Find the best value for a categorical feature (equality split)."""
        if impurity0 is None:
            impurity0 = self._impurity_for_rows(tree, row_ids)

        # Group rows by feature value (single pass)
        value_ids: dict[Any, set[int]] = {}
        value_counts: dict[Any, int] = {}
        total = 0

        for i in row_ids:
            value = tree.X[i][feat_id]
            if value not in value_ids:
                value_ids[value] = set()
                value_counts[value] = 0
            value_ids[value].add(i)
            value_counts[value] += tree.counts[i]
            total += tree.counts[i]

        # Find best split value
        best_gain = 0.0
        best_value = None
        min_leaf = tree.min_samples_leaf

        for value, inside in value_ids.items():
            count = value_counts[value]
            outside_count = total - count
            # Skip values that would leave either side below min_samples_leaf,
            # so a valid alternative category can still be chosen instead of
            # collapsing the node post-hoc.
            if count < min_leaf or outside_count < min_leaf:
                continue
            outside = row_ids - inside

            # Calculate weighted impurity after split
            impurity1 = count / total * self._impurity_for_rows(tree, inside)
            impurity1 += (
                (total - count) / total * self._impurity_for_rows(tree, outside)
            )

            gain = impurity0 - impurity1
            if gain > best_gain:
                best_value = value
                best_gain = gain

        return best_value, best_gain

    def _best_gain_numerical(
        self,
        tree: DecisionTree,
        row_ids: set[int],
        feat_id: int,
        impurity0: float | None = None,
    ) -> tuple[float | None, float]:
        """Find the best threshold for a numerical feature (threshold split).

        Single sorted sweep maintaining running aggregates for the left/right
        partitions, so each candidate threshold is O(classes) (classification)
        or O(1) (regression) instead of rescanning every row. This is O(n log n)
        per feature per node rather than the previous O(n^2). The split-selection
        semantics are preserved exactly: same sorted order, same boundaries,
        strict ``gain > best_gain`` (first/lowest threshold wins ties), and the
        same positive-weight guard on both sides.
        """
        if impurity0 is None:
            impurity0 = self._impurity_for_rows(tree, row_ids)

        values_with_ids = [(tree.X[i][feat_id], i, tree.counts[i]) for i in row_ids]
        values_with_ids.sort(key=lambda x: x[0])

        if tree._is_regression():
            return self._best_gain_numerical_regression(
                tree, values_with_ids, impurity0
            )
        return self._best_gain_numerical_classification(
            tree, values_with_ids, impurity0
        )

    @staticmethod
    def _impurity_from_counts(
        counts: dict[Any, float], total: float, gini: bool
    ) -> float:
        """Entropy or Gini from already-accumulated weighted class counts."""
        if total <= 0:
            return 0.0
        if gini:
            return 1.0 - sum((c / total) ** 2 for c in counts.values())
        acc = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                acc -= p * math.log2(p)
        return acc

    def _best_gain_numerical_classification(
        self,
        tree: DecisionTree,
        values_with_ids: list[tuple[Any, int, int]],
        impurity0: float,
    ) -> tuple[float | None, float]:
        # Right side starts with every row; move rows left as the sweep advances.
        right_counts: dict[Any, float] = {}
        total = 0
        for _value, i, w in values_with_ids:
            label = tree.y[i]
            right_counts[label] = right_counts.get(label, 0) + w
            total += w
        if total == 0:
            return None, 0.0

        gini = self.criterion == "gini"
        min_leaf = tree.min_samples_leaf
        left_counts: dict[Any, float] = {}
        left_count = 0
        best_gain = 0.0
        best_threshold: float | None = None
        prev_value = None

        for value, idx, count in values_with_ids:
            if prev_value is not None and value != prev_value:
                right_count = total - left_count
                # Skip candidates that would leave either side below
                # min_samples_leaf, so a valid alternative threshold can still
                # win instead of the node collapsing to a leaf post-hoc.
                if left_count >= min_leaf and right_count >= min_leaf:
                    imp_left = self._impurity_from_counts(left_counts, left_count, gini)
                    imp_right = self._impurity_from_counts(
                        right_counts, right_count, gini
                    )
                    impurity1 = (
                        left_count / total * imp_left + right_count / total * imp_right
                    )
                    gain = impurity0 - impurity1
                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = (prev_value + value) / 2

            label = tree.y[idx]
            left_counts[label] = left_counts.get(label, 0) + count
            right_counts[label] -= count
            if right_counts[label] == 0:
                del right_counts[label]
            left_count += count
            prev_value = value

        return best_threshold, best_gain

    def _best_gain_numerical_regression(
        self,
        tree: DecisionTree,
        values_with_ids: list[tuple[Any, int, int]],
        impurity0: float,
    ) -> tuple[float | None, float]:
        # Running weighted sums let us derive each partition's variance as
        # E[y^2] - E[y]^2 in O(1) per split point.
        total_w = 0.0
        total_wy = 0.0
        total_wyy = 0.0
        for _value, i, w in values_with_ids:
            y = tree.y[i]
            total_w += w
            total_wy += w * y
            total_wyy += w * y * y
        if total_w == 0:
            return None, 0.0

        min_leaf = tree.min_samples_leaf
        left_w = 0.0
        left_wy = 0.0
        left_wyy = 0.0
        best_gain = 0.0
        best_threshold: float | None = None
        prev_value = None

        for value, idx, count in values_with_ids:
            if prev_value is not None and value != prev_value:
                right_w = total_w - left_w
                # Skip candidates violating min_samples_leaf (see the
                # classification sweep) so a valid split isn't discarded.
                if left_w >= min_leaf and right_w >= min_leaf:
                    var_left = max(0.0, left_wyy / left_w - (left_wy / left_w) ** 2)
                    right_wy = total_wy - left_wy
                    right_wyy = total_wyy - left_wyy
                    var_right = max(
                        0.0, right_wyy / right_w - (right_wy / right_w) ** 2
                    )
                    impurity1 = (
                        left_w / total_w * var_left + right_w / total_w * var_right
                    )
                    gain = impurity0 - impurity1
                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = (prev_value + value) / 2

            y = tree.y[idx]
            left_w += count
            left_wy += count * y
            left_wyy += count * y * y
            prev_value = value

        return best_threshold, best_gain

    def _random_split_numerical(
        self,
        tree: DecisionTree,
        row_ids: set[int],
        feat_id: int,
        impurity0: float,
    ) -> tuple[float | None, float]:
        """Pick a random threshold between min and max for a numerical feature."""
        values = [tree.X[i][feat_id] for i in row_ids]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return None, 0.0

        assert self._rng is not None
        threshold = self._rng.uniform(min_val, max_val)

        total = sum(tree.counts[i] for i in row_ids)
        left = {i for i in row_ids if tree.X[i][feat_id] <= threshold}
        right = row_ids - left
        if not left or not right:
            return None, 0.0

        left_count = sum(tree.counts[i] for i in left)
        right_count = total - left_count
        if left_count < tree.min_samples_leaf or right_count < tree.min_samples_leaf:
            return None, 0.0
        impurity1 = left_count / total * self._impurity_for_rows(
            tree, left
        ) + right_count / total * self._impurity_for_rows(tree, right)
        return threshold, impurity0 - impurity1

    def _random_split_categorical(
        self,
        tree: DecisionTree,
        row_ids: set[int],
        feat_id: int,
        impurity0: float,
    ) -> tuple[Any, float]:
        """Pick a random value for a categorical feature."""
        values = list({tree.X[i][feat_id] for i in row_ids})
        if len(values) <= 1:
            return None, 0.0

        assert self._rng is not None
        value = self._rng.choice(values)

        total = sum(tree.counts[i] for i in row_ids)
        inside = {i for i in row_ids if tree.X[i][feat_id] == value}
        outside = row_ids - inside
        if not inside or not outside:
            return None, 0.0

        in_count = sum(tree.counts[i] for i in inside)
        out_count = total - in_count
        if in_count < tree.min_samples_leaf or out_count < tree.min_samples_leaf:
            return None, 0.0
        impurity1 = in_count / total * self._impurity_for_rows(
            tree, inside
        ) + out_count / total * self._impurity_for_rows(tree, outside)
        return value, impurity0 - impurity1

    def _best_gain_feat_val(
        self, tree: DecisionTree, row_ids: set[int]
    ) -> tuple[str | None, int | None, Any | None, str, float]:
        """Find the best feature and value/threshold for splitting."""
        # Weighted sample total, consistent with the min_samples_split gate in
        # _build_tree (and with cartlet's weighted min_samples_leaf semantics).
        # Using raw len(row_ids) here disagreed for instance-weighted data.
        if self._sum_counts(tree, row_ids) < tree.min_samples_split:
            return None, None, None, "=", 0.0

        impurity0 = self._impurity_for_rows(tree, row_ids)

        best_feat_id = None
        best_val = None
        best_op = "="
        best_gain = 0.0

        # Select features to consider
        all_features = list(range(len(tree.feature_names)))
        if self.max_features and self._rng and self.max_features < len(all_features):
            features_to_try = self._rng.sample(all_features, self.max_features)
        else:
            features_to_try = all_features

        for feat_id in features_to_try:
            feat_type = tree._feature_type(feat_id)

            if feat_type == TYPE_NUM:
                if self.extra_trees:
                    val, gain = self._random_split_numerical(
                        tree, row_ids, feat_id, impurity0
                    )
                else:
                    val, gain = self._best_gain_numerical(
                        tree, row_ids, feat_id, impurity0
                    )
                op = "<"
            else:
                if self.extra_trees:
                    val, gain = self._random_split_categorical(
                        tree, row_ids, feat_id, impurity0
                    )
                else:
                    val, gain = self._best_gain_categorical(
                        tree, row_ids, feat_id, impurity0
                    )
                op = "="

            if gain > best_gain:
                best_feat_id = feat_id
                best_val = val
                best_op = op
                best_gain = gain

        if best_feat_id is None:
            return None, None, None, "=", 0.0

        return (
            tree.feature_names[best_feat_id],
            best_feat_id,
            best_val,
            best_op,
            best_gain,
        )

    def _make_leaf(
        self, tree: DecisionTree, row_ids
    ) -> str | dict[str, float] | list[float]:
        """Create a leaf node."""
        if tree._is_regression():
            return self._make_regression_leaf(tree, row_ids)
        return self._make_classification_leaf(tree, row_ids)

    def _make_classification_leaf(
        self, tree: DecisionTree, row_ids
    ) -> str | dict[str, float]:
        """Create a classification leaf node."""
        cat_counts, total = self._cat_counts(tree, row_ids)

        if not cat_counts:
            return "-"

        # Build sorted (class, prob) list
        items = [(cat, count / total) for cat, count in cat_counts.items()]
        items.sort(key=lambda x: x[1], reverse=True)

        # Early exit if not storing distributions
        if not tree.store_distributions:
            return items[0][0]

        # Check entropy threshold (native-specific)
        if len(items) > 1:
            probs = [prob for _, prob in items]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            if entropy < tree.min_dist_entropy:
                return items[0][0]

        return make_classification_distribution(
            items, tree.store_distributions, tree.min_confidence
        )

    def _make_regression_leaf(self, tree: DecisionTree, row_ids) -> list[float]:
        """Create a regression leaf node: [mean, variance, n]."""
        mean, variance, n = self._mean_for_rows(tree, row_ids)
        return [mean, variance, n]
