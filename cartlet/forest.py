"""
Random Forest for Classification and Regression.

An ensemble of decision trees with bootstrap sampling and random feature subsets.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .base import BaseModel
from .io.bytes import write_forest_bytes
from .io.cart_format import rebuild_tree_from_cart
from .io.utils import open_file_binary, write_with_optional_gzip
from .runner import _load_cart_from_bytes
from .trainer import Native
from .trainer.base import normalize_importances
from .tree import DecisionTree
from .types import (
    _MAX_RANDOM_SEED,
    CRITERION_ENTROPY,
    DEFAULT_N_ESTIMATORS,
    TASK_AUTO,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TYPE_NUM,
)
from .utils import collapse_distributions

# Verbose log cadence: print progress every Nth tree.
_VERBOSE_TREE_INTERVAL = 10


class RandomForest(BaseModel):
    """
    Random Forest classifier/regressor.

    An ensemble of decision trees, each trained on a bootstrap sample
    with random feature subsets at each split.

    Examples:
        # Classification
        rf = RandomForest(feature_names=["color", "size"])
        rf.load_data(X, y)
        rf.train()
        prediction = rf.predict(["red", "large"])

        # With sklearn backend (faster)
        rf.train(trainer="sklearn")
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_features: str | int | float = "sqrt",
        bootstrap: bool = True,
        extra_trees: bool = False,
        features: list[dict[str, Any]] | None = None,
        feature_names: list[str] | None = None,
        target: dict[str, Any] | None = None,
        task: str = TASK_AUTO,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = CRITERION_ENTROPY,
        verbose: bool = False,
        logger=None,
    ):
        """
        Initialize a random forest.

        Args:
            n_estimators: Number of trees in the forest
            max_features: Features to consider at each split:
                - "sqrt": sqrt(n_features)
                - "log2": log2(n_features)
                - int: exact number
                - float: fraction of features
            bootstrap: Whether to use bootstrap sampling
            extra_trees: Use random splits instead of best splits (Extra-Trees)
            features: Feature specs (same as DecisionTree)
            feature_names: Simple feature names (same as DecisionTree)
            target: Target spec (same as DecisionTree)
            task: "classification", "regression", or "auto"
            max_depth: Maximum depth per tree (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            criterion: Split criterion for classification ("entropy" or "gini")
            verbose: Enable verbose output
            logger: Custom logger
        """
        super().__init__(
            features=features,
            feature_names=feature_names,
            target=target,
            task=task,
            verbose=verbose,
            logger=logger,
        )

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.extra_trees = extra_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

        # Store feature config for creating trees
        self._features = features
        self._feature_names = feature_names
        self._target = target
        self._task = task
        self._logger = logger

        # Trained trees
        self.trees: list[DecisionTree] = []

    def load_data(
        self,
        X: list[list[Any]],
        y: list[Any],
        counts: list[int] | None = None,
    ) -> None:
        """
        Load training data.

        Rows are shallow-copied so subsequent in-place mutation of `X` by the
        caller will not affect the forest's training set (matches
        `DecisionTree.load_data` semantics).

        Args:
            X: Feature vectors (list of lists of feature values).
            y: Target values.
            counts: Optional instance weights (default: all 1).

        Raises:
            ValueError: If `len(X) != len(y)`.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        if counts is not None and len(counts) != len(y):
            raise ValueError(
                f"counts and y must have same length: {len(counts)} != {len(y)}"
            )

        # Build a reference tree; DecisionTree.load_data copies + bool-normalizes
        # X, stringifies classification labels, and derives feature specs. Reuse
        # its normalized data so every backend (native per-tree and the sklearn
        # one-hot path) sees identical, normalized inputs -- otherwise bool
        # features trained via sklearn split on raw True/False and never match
        # the 0/1 the predictor produces.
        ref_tree = self._make_tree()
        ref_tree.load_data(X, y, counts)
        self.X = ref_tree.X
        self.y = ref_tree.y
        self.counts = ref_tree.counts
        self.feature_names = ref_tree.feature_names
        self.feature_specs = ref_tree.feature_specs
        self._detected_task = ref_tree._detected_task

    def _make_tree(self) -> DecisionTree:
        """Create a new tree with our configuration."""
        return DecisionTree(
            features=self._features,
            feature_names=self._feature_names,
            target=self._target,
            task=self._task,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            store_distributions=False,  # Forests aggregate, don't need distributions
            verbose=False,
            logger=self._logger,
        )

    def _get_max_features(self) -> int:
        """Calculate number of features to consider at each split.

        Raises ValueError on unrecognized strings or out-of-range numbers
        rather than silently falling back to "use all features" (which masked
        typos like "auto" and non-positive counts).
        """
        n_features = len(self.feature_names)
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(math.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(math.log2(n_features)))
            raise ValueError(
                f"Invalid max_features {mf!r}. Use 'sqrt', 'log2', an int, "
                "a float in (0, 1], or None."
            )
        # bool is an int subclass but is not a valid feature count.
        if isinstance(mf, bool):
            raise ValueError(f"Invalid max_features {mf!r}.")
        if isinstance(mf, int):
            if mf < 1:
                raise ValueError(f"max_features int must be >= 1, got {mf}.")
            return min(mf, n_features)
        if isinstance(mf, float):
            if not (0.0 < mf <= 1.0):
                raise ValueError(f"max_features float must be in (0, 1], got {mf}.")
            return max(1, int(mf * n_features))
        raise ValueError(f"Invalid max_features {mf!r}.")

    def train(
        self,
        trainer: str | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> dict[str, Any]:
        """
        Train the random forest.

        Args:
            trainer: "native" (default) or "sklearn"
            random_state: Random seed for reproducibility
            n_jobs: Parallel jobs for sklearn (None/1 = sequential, -1 = all cores)

        Returns:
            Dict with training info
        """
        if not self.X:
            raise ValueError("No training data loaded. Call load_data() first.")

        if trainer == "sklearn":
            return self._train_sklearn(random_state, n_jobs=n_jobs)

        # n_jobs ignored for native due to Python's GIL
        return self._train_native(random_state)

    def _train_native(self, random_state: int | None = None) -> dict[str, Any]:
        """
        Train using native Python implementation.

        Note: n_jobs is not supported for native trainer due to Python's GIL.
        For parallel training, use trainer="sklearn" with n_jobs.
        """

        rng = random.Random(random_state)
        n_samples = len(self.X)
        max_features = self._get_max_features()

        if n_samples == 0:
            raise ValueError("Cannot train forest with zero samples")

        self.trees = []
        try:
            for i in range(self.n_estimators):
                if self.verbose and (i + 1) % _VERBOSE_TREE_INTERVAL == 0:
                    self.logger.info("Training tree %d/%d...", i + 1, self.n_estimators)

                # Bootstrap sample
                if self.bootstrap:
                    indices = [rng.randint(0, n_samples - 1) for _ in range(n_samples)]
                else:
                    indices = list(range(n_samples))

                seed = (
                    rng.randint(0, _MAX_RANDOM_SEED)
                    if random_state is not None
                    else None
                )
                tree = self._train_single_tree(indices, seed, max_features)
                self.trees.append(tree)
        except KeyboardInterrupt:
            if self.verbose:
                self.logger.info("\nInterrupted after %d trees.", len(self.trees))
            raise

        return {"n_estimators": len(self.trees)}

    def _train_single_tree(
        self, indices: list[int], seed: int | None, max_features: int
    ) -> DecisionTree:
        """Train a single tree on bootstrap sample."""
        X_sample = [self.X[i] for i in indices]
        y_sample = [self.y[i] for i in indices]
        counts_sample = [self.counts[i] for i in indices]

        tree = self._make_tree()
        tree.load_data(X_sample, y_sample, counts_sample)

        tree_trainer = Native(
            max_depth=self.max_depth,
            prune=False,
            random_state=seed,
            max_features=max_features,
            criterion=self.criterion,
            extra_trees=self.extra_trees,
        )

        tree.train(trainer=tree_trainer)
        return tree

    def _train_sklearn(
        self, random_state: int | None = None, n_jobs: int | None = None
    ) -> dict[str, Any]:
        """
        Train using sklearn RandomForest with automatic categorical encoding.

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (None/1 = sequential, -1 = all cores)
        """
        try:
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for sklearn trainer. "
                "Install with: pip install scikit-learn"
            ) from e

        from .trainer.sklearn import convert_sklearn_tree, encode_categorical

        # One-hot encode categorical features
        X_encoded, encoded_names, cat_cols, cat_vals = encode_categorical(
            self.X, self.feature_names, self.feature_specs
        )

        is_regression = self._is_regression()
        # Honor extra_trees (random splits) just like the native backend, rather
        # than silently training a plain random forest.
        if self.extra_trees:
            sklearn_cls = ExtraTreesRegressor if is_regression else ExtraTreesClassifier
        else:
            sklearn_cls = (
                RandomForestRegressor if is_regression else RandomForestClassifier
            )

        rf_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        # Forward the split criterion for classification. cartlet's "entropy"/
        # "gini" are also valid sklearn values; sklearn regressors use a
        # different criterion family, so only pass it for classifiers.
        if not is_regression:
            rf_kwargs["criterion"] = self.criterion
        sklearn_rf = sklearn_cls(**rf_kwargs)

        # Pass instance weights through to sklearn (previously dropped).
        sklearn_rf.fit(X_encoded, self.y, sample_weight=self.counts)

        # Store sklearn model for potential export
        self._sklearn_model = sklearn_rf

        # Get forest-level classes (individual trees only have indices)
        forest_classes = (
            sklearn_rf.classes_ if hasattr(sklearn_rf, "classes_") else None
        )

        # Convert each tree
        self.trees = []
        for sklearn_tree in sklearn_rf.estimators_:
            tree = self._make_tree()
            tree.feature_names = self.feature_names
            tree.feature_specs = self.feature_specs
            tree._rebuild_name_to_col()

            # Convert sklearn tree to our format
            tree.model = convert_sklearn_tree(
                sklearn_tree,
                self.feature_names,
                encoded_names,
                cat_cols,
                cat_vals,
                classes=forest_classes,
                is_regression=is_regression,
                store_distributions=False,  # Forests don't need distributions
            )
            self.trees.append(tree)

        return {"n_estimators": len(self.trees)}

    def predict(self, vector: list[Any], **kwargs: Any) -> Any:
        """
        Predict for a feature vector.

        Args:
            vector: Feature vector

        Returns:
            Prediction (majority vote for classification, mean for regression)
        """
        if not self.trees:
            raise ValueError("Forest not trained. Call train() first.")

        predictions = [tree.predict(vector) for tree in self.trees]

        if self._is_regression():
            # In regression mode, predictions are numeric (mypy can't infer from runtime check)
            values = [float(p) for p in predictions]  # type: ignore[arg-type]
            return sum(values) / len(values)

        # Majority vote for classification
        votes = Counter(predictions)
        return votes.most_common(1)[0][0]

    def predict_proba(self, vector: list[Any]) -> dict[Any, float]:
        """
        Get class probabilities (classification only).

        Args:
            vector: Feature vector

        Returns:
            Dict mapping class -> probability
        """
        if not self.trees:
            raise ValueError("Forest not trained. Call train() first.")

        if self._is_regression():
            raise ValueError("predict_proba not available for regression")

        predictions = [tree.predict(vector) for tree in self.trees]
        votes = Counter(predictions)
        total = len(predictions)
        return {cls: count / total for cls, count in votes.items()}

    @property
    def feature_importances_(self) -> dict[str, float]:
        """
        Compute feature importances averaged across all trees.

        Returns:
            Dict mapping feature name -> importance (sums to 1.0)
        """
        if not self.trees:
            raise ValueError("Forest not trained. Call train() first.")

        # Accumulate importances from all trees
        total_importances: dict[str, float] = dict.fromkeys(self.feature_names, 0.0)

        for tree in self.trees:
            tree_imp = tree.feature_importances_
            for name, imp in tree_imp.items():
                total_importances[name] += imp

        return normalize_importances(total_importances)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "bootstrap": self.bootstrap,
            "criterion": self.criterion,
            "extra_trees": self.extra_trees,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "n_estimators": self.n_estimators,
        }

    def _export_cart(
        self,
        path: str,
        metadata: dict | None = None,
        use_gzip: bool = False,
        store_distributions: bool = False,
    ) -> None:
        """Export to compact binary format."""
        if not self.trees:
            raise ValueError("No forest to export. Call train() first.")

        def _write(dest: str) -> None:
            write_forest_bytes(
                dest,
                [tree.model for tree in self.trees],
                self.feature_specs,
                self.name_to_col,
                self._class_labels,
                self._is_regression(),
                metadata,
                store_distributions=store_distributions,
            )

        write_with_optional_gzip(path, use_gzip, _write)

    def _build_export_dict(
        self, metadata: dict | None = None, store_distributions: bool = True
    ) -> dict:
        """Build dictionary for JSON/pickle export."""
        trees = [tree.model for tree in self.trees]
        if not store_distributions:
            trees = [collapse_distributions(t) for t in trees]
        return {
            "trees": trees,
            "feature_specs": self._serialize_feature_specs(),
            "feature_names": self.feature_names,
            "task": self._effective_task(),
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "class_labels": self._class_labels,
            "metadata": metadata or {},
        }

    def _load_cart(self, path: str, use_gzip: bool = False) -> dict:
        """Load from compact binary format."""
        with open_file_binary(path, "rb") as f:
            raw_data = f.read()
        model_data = _load_cart_from_bytes(raw_data)

        # Restore config
        self._apply_config_from_cart(model_data)

        self.n_estimators = model_data.get("n_trees", len(model_data["tree_offsets"]))
        self.bootstrap = True  # Not stored in .cart, assume default

        # Rebuild trees from flat nodes
        self.trees = []
        for tree_idx in range(len(model_data["tree_offsets"])):
            tree = self._make_tree()
            tree.feature_names = self.feature_names
            tree.feature_specs = self.feature_specs
            tree.name_to_col = self.name_to_col
            tree.model = self._rebuild_tree_from_cart(model_data, tree_idx)
            self.trees.append(tree)

        return {
            "features": model_data["meta"]["features"],
            "task": model_data["meta"]["task"],
        }

    def _load_sklearn(self, path: str, use_gzip: bool = False) -> dict:
        """Load sklearn model - converts to cartlet format for inference."""
        from .trainer.sklearn import convert_sklearn_tree

        sklearn_rf, feature_names, feature_specs = self._read_sklearn_for_load(path)
        self.feature_names = feature_names
        self.feature_specs = feature_specs
        self.n_estimators = len(sklearn_rf.estimators_)
        self.bootstrap = sklearn_rf.bootstrap

        # Determine task
        from sklearn.ensemble import RandomForestClassifier

        is_classifier = isinstance(sklearn_rf, RandomForestClassifier)
        task = TASK_CLASSIFICATION if is_classifier else TASK_REGRESSION
        self.task = task

        # Get forest-level classes
        forest_classes = sklearn_rf.classes_ if is_classifier else None

        # Convert each tree
        name_to_col = self._rebuild_name_to_col()
        self.trees = []
        for sklearn_tree in sklearn_rf.estimators_:
            tree = self._make_tree()
            tree.feature_names = feature_names
            tree.feature_specs = self.feature_specs
            tree.name_to_col = name_to_col
            tree.model = convert_sklearn_tree(
                sklearn_tree,
                feature_names,
                feature_names,  # No encoding, same names
                [],  # No categorical columns
                {},  # No categorical values
                classes=forest_classes,
                is_regression=not is_classifier,
                store_distributions=False,
            )
            self.trees.append(tree)

        return {
            "features": [{"name": name, "type": TYPE_NUM} for name in feature_names],
            "task": task,
        }

    def _apply_loaded_data(self, data: dict) -> dict:
        """Apply loaded data from JSON/pickle to instance."""
        self.n_estimators = data.get("n_estimators", len(data.get("trees", [])))
        self.bootstrap = data.get("bootstrap", True)

        super()._apply_loaded_data(data)

        # Rebuild trees
        self.trees = []
        for tree_model in data.get("trees", []):
            tree = self._make_tree()
            tree.feature_names = self.feature_names
            tree.feature_specs = self.feature_specs
            tree.name_to_col = self.name_to_col
            tree.model = tree_model
            self.trees.append(tree)

        return data

    def _rebuild_tree_from_cart(self, model_data: dict, tree_idx: int) -> Any:
        """Rebuild nested tree structure from .cart flat nodes for one tree."""
        return rebuild_tree_from_cart(model_data, self.feature_names, tree_idx)

    def __repr__(self) -> str:
        status = f"{len(self.trees)} trees" if self.trees else "untrained"
        task = self._effective_task()
        kind = "ExtraTrees" if self.extra_trees else "RandomForest"
        return f"{kind}(n_estimators={self.n_estimators}, task={task}, {status})"
