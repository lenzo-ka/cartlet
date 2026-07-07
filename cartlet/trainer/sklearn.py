"""
Scikit-learn trainer for decision trees.

Optional module - requires scikit-learn to be installed.
Provides faster training using sklearn's optimized C implementation.

Automatically handles categorical features via one-hot encoding,
then converts back to our native format with proper equality splits.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from ..types import PROB_HIGH_CONFIDENCE, TYPE_CAT
from .base import Trainer, make_classification_distribution, normalize_importances

if TYPE_CHECKING:
    from ..tree import DecisionTree
    from ..types import FeatureSpec

# Lazy import sklearn to make it optional
_sklearn_available: bool | None = None


def _check_sklearn() -> bool:
    """Check if sklearn is available, cache result."""
    global _sklearn_available
    if _sklearn_available is None:
        _sklearn_available = importlib.util.find_spec("sklearn") is not None
    return _sklearn_available


# =============================================================================
# Standalone encoding/conversion utilities (used by Sklearn trainer and RandomForest)
# =============================================================================


def encode_categorical(
    X: list[list[Any]],
    feature_names: list[str],
    feature_specs: list[FeatureSpec],
) -> tuple[list[list[float]], list[str], list[int], dict[int, list[Any]]]:
    """
    One-hot encode categorical features for sklearn.

    Args:
        X: Training data
        feature_names: Feature names
        feature_specs: Feature specifications

    Returns:
        Tuple of (encoded_X, encoded_feature_names, cat_columns, cat_values)
    """
    cat_columns: list[int] = []
    cat_values: dict[int, list[Any]] = {}

    # Identify categorical columns and collect their values
    for col, spec in enumerate(feature_specs):
        if spec.type == TYPE_CAT:
            cat_columns.append(col)
            values = sorted({row[col] for row in X}, key=str)
            cat_values[col] = values

    # Build encoded feature names
    encoded_names: list[str] = []
    for col, name in enumerate(feature_names):
        if col in cat_columns:
            for val in cat_values[col]:
                encoded_names.append(f"{name}={val}")
        else:
            encoded_names.append(name)

    # Encode data
    encoded_X: list[list[float]] = []
    for row in X:
        encoded_row: list[float] = []
        for col, val in enumerate(row):
            if col in cat_columns:
                for cat_val in cat_values[col]:
                    encoded_row.append(1.0 if val == cat_val else 0.0)
            else:
                encoded_row.append(float(val))
        encoded_X.append(encoded_row)

    return encoded_X, encoded_names, cat_columns, cat_values


def convert_sklearn_tree(
    sklearn_tree,
    feature_names: list[str],
    encoded_names: list[str],
    cat_columns: list[int],
    cat_values: dict[int, list[Any]],
    classes: Any | None = None,
    is_regression: bool = False,
    store_distributions: bool = True,
    min_confidence: float = PROB_HIGH_CONFIDENCE,
) -> Any:
    """
    Convert sklearn tree structure to our nested list format.

    Args:
        sklearn_tree: Trained sklearn tree
        feature_names: Original feature names
        encoded_names: Encoded feature names (after one-hot). Retained for
            signature compatibility; the origin of each encoded column is now
            reconstructed from feature_names/cat_columns/cat_values instead of
            parsing these strings.
        cat_columns: Indices of categorical columns
        cat_values: Mapping of column index to list of category values
        classes: Class labels (for classification)
        is_regression: Whether this is a regression tree
        store_distributions: Whether to store probability distributions
        min_confidence: Collapse distributions above this probability

    Returns:
        Tree in our nested list format
    """
    sk_tree = sklearn_tree.tree_

    # Explicit encoded-column -> origin map, built in the SAME order as
    # encode_categorical expands columns. Each entry is either
    # (False, orig_name, None) for a numeric passthrough column or
    # (True, orig_name, category_value) for a one-hot column. This replaces
    # parsing "=" out of the encoded name string, which misfired when a
    # feature name or category value legitimately contained "=".
    encoded_meta: list[tuple[bool, Any, Any]] = []
    for col, name in enumerate(feature_names):
        if col in cat_columns:
            for val in cat_values[col]:
                encoded_meta.append((True, name, val))
        else:
            encoded_meta.append((False, name, None))

    def convert_node(node_id: int) -> Any:
        if sk_tree.children_left[node_id] == -1:
            if is_regression or classes is None:
                mean = float(sk_tree.value[node_id, 0, 0])
                n = int(sk_tree.n_node_samples[node_id])
                # sklearn's regression impurity is the node MSE (variance),
                # matching native's [mean, variance, n] leaf shape.
                variance = float(sk_tree.impurity[node_id])
                return [mean, variance, n]

            # Classification: build class distribution
            values = sk_tree.value[node_id, 0]
            total = values.sum()
            if total == 0:
                return "-"

            probs = values / total
            items = [
                (str(classes[idx]), float(prob))
                for idx, prob in enumerate(probs)
                if prob > 0
            ]
            items.sort(key=lambda x: x[1], reverse=True)

            return make_classification_distribution(
                items, store_distributions, min_confidence
            )

        # Decision node
        feat_idx = sk_tree.feature[node_id]
        threshold = float(sk_tree.threshold[node_id])
        is_cat, orig_name, cat_value = encoded_meta[feat_idx]

        left = convert_node(sk_tree.children_left[node_id])
        right = convert_node(sk_tree.children_right[node_id])

        if is_cat:
            # One-hot column "orig_name == cat_value" with threshold 0.5:
            # left (<=0.5) = NOT equal, right (>0.5) = EQUAL. Original name and
            # value objects are carried through directly, preserving their type.
            return [orig_name, "=", cat_value, right, left]
        # Numerical: standard threshold split
        return [orig_name, "<", threshold, left, right]

    return convert_node(0)


def map_feature_importances(
    sklearn_importances: list[float],
    feature_names: list[str],
    cat_columns: list[int],
    cat_values: dict[int, list[Any]],
) -> dict[str, float]:
    """
    Map sklearn feature importances back to original features.

    Args:
        sklearn_importances: Importances from sklearn (one per encoded feature)
        feature_names: Original feature names
        cat_columns: Indices of categorical columns
        cat_values: Mapping of column index to list of category values

    Returns:
        Dict mapping original feature name to importance
    """
    importances: dict[str, float] = dict.fromkeys(feature_names, 0.0)

    encoded_idx = 0
    for col, name in enumerate(feature_names):
        if col in cat_columns:
            # Sum importances for all one-hot columns
            for _ in cat_values[col]:
                importances[name] += sklearn_importances[encoded_idx]
                encoded_idx += 1
        else:
            importances[name] = sklearn_importances[encoded_idx]
            encoded_idx += 1

    return normalize_importances(importances)


class Sklearn(Trainer):
    """
    Scikit-learn based decision tree trainer.

    Uses sklearn's optimized C implementation for faster training.
    Automatically one-hot encodes categorical features and converts
    the resulting tree back to native format with equality splits.

    Requires: scikit-learn (pip install scikit-learn)
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        criterion: str | None = None,
    ):
        """
        Initialize the sklearn trainer.

        Args:
            max_depth: Maximum depth of tree (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            random_state: Random seed for reproducibility
            criterion: Split criterion for classification ("entropy" or "gini").
                None uses sklearn's default. cartlet's values map directly onto
                sklearn's, so it is forwarded verbatim for classifiers and
                ignored for regressors.
        """
        if not _check_sklearn():
            raise ImportError(
                "scikit-learn is required for Sklearn trainer. "
                "Install with: pip install scikit-learn"
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion

    @property
    def supports_categorical(self) -> bool:
        return True  # We handle encoding automatically

    def train(
        self,
        tree: DecisionTree,
        train_rows: list[int],
        val_rows: list[int] | None = None,
    ) -> Any:
        """Build the decision tree using sklearn."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # One-hot encode categorical features
        X_subset = [tree.X[i] for i in train_rows]
        X_encoded, encoded_names, cat_cols, cat_vals = encode_categorical(
            X_subset, tree.feature_names, tree.feature_specs
        )
        y_train = [tree.y[i] for i in train_rows]
        weights = [tree.counts[i] for i in train_rows]

        # Choose classifier or regressor
        is_regression = tree._is_regression()
        if is_regression:
            sklearn_tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        else:
            clf_kwargs: dict[str, Any] = {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            }
            if self.criterion is not None:
                clf_kwargs["criterion"] = self.criterion
            sklearn_tree = DecisionTreeClassifier(**clf_kwargs)

        # Train
        sklearn_tree.fit(X_encoded, y_train, sample_weight=weights)

        # Store sklearn model for potential export
        tree.set_sklearn_model(sklearn_tree)

        # Store feature importances (mapped to original features)
        tree.set_feature_importances(
            map_feature_importances(
                list(sklearn_tree.feature_importances_),
                tree.feature_names,
                cat_cols,
                cat_vals,
            )
        )

        # Convert to our format
        classes = sklearn_tree.classes_ if hasattr(sklearn_tree, "classes_") else None
        return convert_sklearn_tree(
            sklearn_tree,
            tree.feature_names,
            encoded_names,
            cat_cols,
            cat_vals,
            classes=classes,
            is_regression=is_regression,
            store_distributions=tree.store_distributions,
            min_confidence=tree.min_confidence,
        )

    @classmethod
    def from_sklearn(
        cls,
        sklearn_tree,
        feature_names: list[str],
        task: str = "auto",
        store_distributions: bool = True,
        min_confidence: float = PROB_HIGH_CONFIDENCE,
    ) -> tuple[Any, dict]:
        """
        Import a pre-trained sklearn tree.

        Args:
            sklearn_tree: Trained sklearn DecisionTreeClassifier/Regressor
            feature_names: Names for each feature
            task: "classification", "regression", or "auto"
            store_distributions: Whether to store distributions at leaves
            min_confidence: Collapse distributions above this probability

        Returns:
            Tuple of (model, config_dict) for use with DecisionTree

        Example:
            from sklearn.tree import DecisionTreeClassifier
            from cartlet import DecisionTree
            from cartlet.trainer import Sklearn

            # Train with sklearn
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)

            # Import to cartlet
            model, config = Sklearn.from_sklearn(
                clf, feature_names=["age", "income", "score"]
            )
            dt = DecisionTree(**config)
            dt.model = model

            # Now use zero-dependency inference
            dt.predict([25, 50000, 0.8])
        """
        if not _check_sklearn():
            raise ImportError("scikit-learn required")

        from sklearn.tree import DecisionTreeClassifier

        is_classifier = isinstance(sklearn_tree, DecisionTreeClassifier)
        sk_tree = sklearn_tree.tree_
        classes = sklearn_tree.classes_ if is_classifier else None

        # Determine task
        if task == "auto":
            task = "classification" if is_classifier else "regression"

        def convert_node(node_id: int) -> Any:
            if sk_tree.children_left[node_id] == -1:
                if is_classifier and classes is not None:
                    values = sk_tree.value[node_id, 0]
                    total = values.sum()
                    if total == 0:
                        return "-"

                    # Build sorted (class, prob) list
                    probs = values / total
                    items = [
                        (str(classes[idx]), float(prob))
                        for idx, prob in enumerate(probs)
                        if prob > 0
                    ]
                    items.sort(key=lambda x: x[1], reverse=True)

                    return make_classification_distribution(
                        items, store_distributions, min_confidence
                    )
                else:
                    mean = float(sk_tree.value[node_id, 0, 0])
                    n = int(sk_tree.n_node_samples[node_id])
                    variance = float(sk_tree.impurity[node_id])
                    return [mean, variance, n]

            feat_idx = sk_tree.feature[node_id]
            threshold = float(sk_tree.threshold[node_id])
            feat_name = feature_names[feat_idx]

            left = convert_node(sk_tree.children_left[node_id])
            right = convert_node(sk_tree.children_right[node_id])

            return [feat_name, "<", threshold, left, right]

        model = convert_node(0)

        config = {
            "features": [
                {"name": name, "dtype": "float", "type": "num"}
                for name in feature_names
            ],
            "task": task,
            "store_distributions": store_distributions,
        }

        return model, config
