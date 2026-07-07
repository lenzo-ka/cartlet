"""
Decision Tree for Classification and Regression (CART).

A clean implementation supporting:
- Categorical features (equality splits)
- Numerical features (threshold splits)
- Classification (entropy/information gain)
- Regression (variance reduction)
- Instance weighting
- Pruning
- Probability distributions at leaves
- N-best predictions with confidence scores
"""

from __future__ import annotations

import random
from time import time
from typing import Any

from .base import BaseModel
from .evaluation import evaluate_tree
from .io.bytes import write_tree_bytes
from .io.cart_format import rebuild_tree_from_cart
from .io.utils import open_file_binary, write_with_optional_gzip
from .runner import _load_cart_from_bytes
from .trainer import Native, Trainer
from .trainer.base import normalize_importances
from .types import (
    CRITERION_ENTROPY,
    DEFAULT_MIN_DIST_ENTROPY,
    DEFAULT_VALIDATION_SPLIT,
    DTYPE_BOOL,
    DTYPE_STR,
    PROB_HIGH_CONFIDENCE,
    TASK_AUTO,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TYPE_CAT,
    TYPE_NUM,
    FeatureSpec,
    is_likely_regression,
    normalize_bool,
)
from .utils import (
    collapse_distributions,
    count_nodes,
    eval_tree,
)
from .utils import (
    max_depth as compute_max_depth,
)

# Default beam width returned by `predict_nbest`. 5 mirrors the conventional
# top-K reported in ranking benchmarks; callers can override per call.
_DEFAULT_NBEST = 5


class DecisionTree(BaseModel):
    """
    A decision tree for classification and regression.

    Supports:
    - Categorical features: equality splits (feature == value)
    - Numerical features: threshold splits (feature <= threshold)
    - Classification: entropy-based information gain
    - Regression: variance reduction

    Examples:
        # Classification with feature schema
        dt = DecisionTree(
            features=[
                {"name": "color", "dtype": "str", "type": "cat"},
                {"name": "size", "dtype": "str", "type": "cat"},
            ],
            task="classification"
        )
        dt.load_data(X, y)
        dt.train()

        # Regression with numerical features
        dt = DecisionTree(
            features=[
                {"name": "age", "dtype": "int", "type": "num"},
                {"name": "income", "dtype": "float", "type": "num"},
            ],
            task="regression"
        )
        dt.load_data(X, y)  # y contains numerical targets
        dt.train()

        # Simple: just feature names (all categorical strings)
        dt = DecisionTree(feature_names=["color", "size"])
    """

    def __init__(
        self,
        features: list[dict[str, Any]] | None = None,
        feature_names: list[str] | None = None,
        target: dict[str, Any] | None = None,
        task: str = TASK_AUTO,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        store_distributions: bool = True,
        min_dist_entropy: float = DEFAULT_MIN_DIST_ENTROPY,
        min_confidence: float = PROB_HIGH_CONFIDENCE,
        criterion: str = CRITERION_ENTROPY,
        verbose: bool = False,
        logger=None,
    ):
        """
        Initialize a decision tree.

        Args:
            features: List of input feature specs, e.g.:
                [{"name": "age", "dtype": "int", "type": "num"}, ...]
            feature_names: Simple alternative: list of names (all categorical strings)
            target: Output/target feature spec, e.g.:
                {"name": "price", "dtype": "float"} for regression
                {"name": "class", "dtype": "str"} for classification
            task: "classification", "regression", or "auto" (detect from target/y)
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            store_distributions: Store probability distributions at leaves
            min_dist_entropy: Minimum entropy to store distribution
            min_confidence: If best-class probability exceeds this, store only the
                class label instead of the full distribution (default 0.95).
                Set to 1.0 to always keep distributions.
            criterion: Split criterion for classification ("entropy" or "gini")
            verbose: Enable verbose output
            logger: Custom logger (uses default if None)
        """
        super().__init__(
            features=features,
            feature_names=feature_names,
            target=target,
            task=task,
            verbose=verbose,
            logger=logger,
        )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.store_distributions = store_distributions
        self.min_dist_entropy = min_dist_entropy
        self.min_confidence = min_confidence
        self.criterion = criterion

        # Trained model
        self.model: Any = None

    def _feature_type(self, feat_idx: int) -> str:
        """Get the type (cat/num) for a feature."""
        if feat_idx < len(self.feature_specs):
            return self.feature_specs[feat_idx].type or TYPE_CAT
        return TYPE_CAT

    def load_data(
        self,
        X: list[list[Any]],
        y: list[Any],
        counts: list[int] | None = None,
    ) -> None:
        """
        Load training data.

        Args:
            X: Feature vectors (list of lists of feature values)
            y: Target values (strings for classification, numbers for regression)
            counts: Optional instance weights (default: all 1)
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        if counts is not None and len(counts) != len(y):
            raise ValueError(
                f"counts and y must have same length: {len(counts)} != {len(y)}"
            )

        self.X = [row[:] for row in X]  # Copy to avoid mutating input
        self.y = list(y)  # Copy so caller mutations don't reach into the model
        self.counts = list(counts) if counts is not None else [1] * len(y)

        # Normalize bool features to 0/1 and collect known values for categorical features
        if self.feature_specs:
            for col, spec in enumerate(self.feature_specs):
                if spec.dtype == DTYPE_BOOL:
                    for row in self.X:
                        if col < len(row):
                            row[col] = normalize_bool(row[col])
                # Track known values for categorical features
                if spec.type == TYPE_CAT:
                    spec.values = {row[col] for row in self.X if col < len(row)}

        if not self.feature_names and X:
            # Auto-generate feature names and specs
            self.feature_names = [str(i) for i in range(len(X[0]))]
            self.feature_specs = [
                FeatureSpec(name=name, dtype=DTYPE_STR, type=TYPE_CAT)
                for name in self.feature_names
            ]
            self._rebuild_name_to_col()

        # Auto-detect task if needed. Use the unique-count/ratio heuristic so a
        # small set of integer *class* labels (e.g. y=[0, 1, 0, 1]) is treated
        # as classification rather than regression; all-numeric targets with
        # many distinct values still resolve to regression.
        if self.task == TASK_AUTO and y:
            self._detected_task = (
                TASK_REGRESSION if is_likely_regression(y) else TASK_CLASSIFICATION
            )

            if self.verbose:
                self.logger.info("Auto-detected task: %s", self._detected_task)

        # Classification labels are compared and stored as strings everywhere
        # (class_labels, .cart export, both runners, sklearn-converted leaves).
        # Stringify here so the in-memory native tree agrees with the exported
        # model and predictions have a single, stable label type.
        if self._effective_task() == TASK_CLASSIFICATION:
            self.y = [str(v) for v in self.y]

    def _split_data(
        self,
        validation_split: float,
        test_split: float,
        random_state: int | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Split row indices into train/validation/test sets.

        Args:
            validation_split: Fraction for validation
            test_split: Fraction for test
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_rows, val_rows, test_rows)
        """
        all_rows = list(range(len(self.y)))

        if validation_split <= 0 and test_split <= 0:
            return all_rows, [], []

        rng = random.Random(random_state)
        rng.shuffle(all_rows)
        test_size = int(len(all_rows) * test_split)
        val_size = int(len(all_rows) * validation_split)

        test_rows = all_rows[:test_size] if test_size > 0 else []
        val_rows = all_rows[test_size : test_size + val_size] if val_size > 0 else []
        train_rows = all_rows[test_size + val_size :]

        return train_rows, val_rows, test_rows

    def train(
        self,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        prune: bool = False,
        random_state: int | None = None,
        trainer: str | Trainer | None = None,
    ) -> dict[str, float]:
        """
        Train the decision tree.

        Args:
            validation_split: Fraction held out for reduced-error pruning.
                When ``prune=True`` and this is left at 0.0, a default of
                DEFAULT_VALIDATION_SPLIT is used so pruning actually happens.
            test_split: Fraction for test (evaluation)
            prune: Whether to prune tree using validation data. Only the native
                backend prunes; with the sklearn backend ``prune=True`` is a
                no-op and a warning is logged (no data is held out).
            random_state: Random seed for reproducibility (used when pruning)
            trainer: Trainer to use - "native" (default), "sklearn", or a Trainer instance

        Returns:
            Task-appropriate test-set metrics from `evaluate_tree`. Always
            includes a ``task`` key (``"classification"`` or ``"regression"``).
            Classification adds ``accuracy``, ``correct``, ``total``;
            regression adds ``mse``, ``mae``, ``rmse``, ``total``. Returns an
            empty dict when ``test_split`` is 0 (no test rows produced).

        Raises:
            ValueError: If `load_data` has not been called.
        """
        if not self.X:
            raise ValueError("No training data loaded. Call load_data() first.")

        if self.verbose:
            self.logger.info("Building tree from %d observations", len(self.X))

        # Get trainer instance up front so pruning support can inform the split.
        trainer_instance = self._get_trainer(trainer, prune, random_state)

        if self.verbose:
            self.logger.info("Using trainer: %s", trainer_instance.name)

        # Resolve the effective validation split for pruning. Historically
        # ``prune=True`` with the default ``validation_split=0.0`` produced an
        # empty validation set and silently never pruned; fall back to
        # DEFAULT_VALIDATION_SPLIT so pruning actually happens. Backends that
        # do not honour validation rows (e.g. sklearn) warn instead of holding
        # out data pointlessly.
        effective_val_split = validation_split
        if prune:
            if not trainer_instance.supports_pruning:
                self.logger.warning(
                    "The %s backend does not support pruning; prune=True is "
                    "ignored (no validation data is held out).",
                    trainer_instance.name,
                )
                effective_val_split = 0.0
            elif effective_val_split <= 0.0:
                effective_val_split = DEFAULT_VALIDATION_SPLIT
                if self.verbose:
                    self.logger.info(
                        "prune=True with no validation_split; using default "
                        "%.3g for pruning.",
                        effective_val_split,
                    )

        do_prune = prune and effective_val_split > 0.0

        # Split data
        train_rows, val_rows, test_rows = self._split_data(
            effective_val_split if do_prune else 0.0,
            test_split,
            random_state,
        )
        if self.verbose and (val_rows or test_rows):
            self.logger.info(
                "Data split: %d train, %d validation, %d test",
                len(train_rows),
                len(val_rows),
                len(test_rows),
            )

        # Build tree
        tree_start = time()
        self.model = trainer_instance.train(
            self,
            train_rows,
            val_rows if do_prune else None,
        )
        tree_time = time() - tree_start

        nodes = count_nodes(self.model)

        if self.verbose:
            self.logger.info(
                "Tree built: %d nodes (%.1f sec, %.1f nodes/sec)",
                nodes,
                tree_time,
                nodes / tree_time if tree_time > 0 else 0,
            )

        if not test_rows:
            return {}

        X_test = [self.X[i] for i in test_rows]
        y_test = [self.y[i] for i in test_rows]
        metrics = evaluate_tree(self, X_test, y_test)
        if self.verbose:
            if metrics["task"] == TASK_CLASSIFICATION:
                self.logger.info("Test accuracy: %.2f%%", metrics["accuracy"] * 100)
            else:
                self.logger.info(
                    "Test RMSE: %.4f (MAE %.4f)", metrics["rmse"], metrics["mae"]
                )
        return metrics

    def _get_trainer(self, trainer, prune: bool, random_state: int | None):
        """
        Get a trainer instance from string name or return as-is if already a Trainer.

        Args:
            trainer: Trainer name ("native", "sklearn") or Trainer instance
            prune: Whether pruning is enabled
            random_state: Random seed for reproducibility

        Returns:
            Trainer instance

        Raises:
            ValueError: If trainer name is unknown
        """
        if trainer is None or trainer == "native":
            return Native(
                max_depth=self.max_depth,
                prune=prune,
                random_state=random_state,
                criterion=self.criterion,
            )
        if trainer == "sklearn":
            from .trainer import Sklearn

            return Sklearn(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=random_state,
                criterion=self.criterion,
            )
        if isinstance(trainer, Trainer):
            return trainer
        raise ValueError(
            f"Unknown trainer: {trainer!r}. "
            f"Available options: 'native', 'sklearn', or a Trainer instance."
        )

    def predict(
        self,
        vector: list[Any],
        return_dist: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Any | dict[str, float] | float:
        """
        Predict for a feature vector.

        Args:
            vector: Feature vector
            return_dist: If True, return distribution when available (classification)
            strict: If True, raise ValueError for OOV categorical values

        Returns:
            Classification: category (str) or distribution (dict)
            Regression: predicted value (float)

        Raises:
            ValueError: If strict=True and OOV categorical value encountered
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Normalize bool features
        normalized = self._normalize_vector(vector)

        # Check for OOV values in strict mode
        if strict:
            oov_features = self._check_oov(normalized)
            if oov_features:
                raise ValueError(f"OOV values for features: {oov_features}")

        # Evaluate using the nested-list tree interpreter from utils
        return eval_tree(self.model, normalized, self.name_to_col, return_dist)

    def _check_oov(self, vector: list[Any]) -> list[tuple[str, Any]]:
        """Check for OOV categorical values in a vector."""
        oov: list[tuple[str, Any]] = []
        if not self.feature_specs:
            return oov
        for col, spec in enumerate(self.feature_specs):
            if (
                col < len(vector)
                and spec.values is not None
                and vector[col] not in spec.values
            ):
                oov.append((spec.name, vector[col]))
        return oov

    def _normalize_vector(self, vector: list[Any]) -> list[Any]:
        """
        Normalize bool features in a vector to 0/1.

        Args:
            vector: Feature vector to normalize

        Returns:
            Normalized vector with bool features converted to 0/1
        """
        if not self.feature_specs:
            return vector
        result = vector[:]
        for col, spec in enumerate(self.feature_specs):
            if col < len(result) and spec.dtype == DTYPE_BOOL:
                result[col] = normalize_bool(result[col])
        return result

    def predict_with_confidence(self, vector: list[Any]) -> tuple[Any, float]:
        """
        Predict with confidence score (classification only).

        Args:
            vector: Feature vector

        Returns:
            Tuple of `(prediction, confidence)`. Confidence is 1.0 when the
            leaf stores only the best class, otherwise the probability of the
            argmax class in the leaf distribution. Returns `("-", 0.0)` for
            non-classification leaves.
        """
        dist = self.predict(vector, return_dist=True)
        if isinstance(dist, str):
            return dist, 1.0
        if isinstance(dist, dict):
            # Explicit argmax avoids depending on dict insertion order matching
            # the writer's probability-sorted order.
            best = max(dist, key=dist.__getitem__)
            return best, dist[best]
        return "-", 0.0

    def predict_nbest(
        self, vector: list[Any], n: int = _DEFAULT_NBEST
    ) -> list[tuple[Any, float]]:
        """
        Get n-best predictions with scores.

        Args:
            vector: Feature vector
            n: Number of alternatives to return

        Returns:
            List of (category, probability) tuples, sorted by probability
        """
        dist = self.predict(vector, return_dist=True)

        if isinstance(dist, str):
            return [(dist, 1.0)]

        if isinstance(dist, dict):
            items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            return items[:n]

        return [("-", 0.0)]

    @property
    def feature_importances_(self) -> dict[str, float]:
        """
        Feature importances based on Mean Decrease in Impurity (MDI).

        Computed during training as the weighted impurity decrease at each split.
        This matches sklearn's `feature_importances_` attribute.

        Returns:
            Dict mapping feature name -> importance (sums to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Use pre-computed importances from training if available
        if self._feature_importances:
            return self._feature_importances

        # Fallback: estimate from tree structure (for loaded models)
        return self._estimate_importances_from_tree()

    def _estimate_importances_from_tree(self) -> dict[str, float]:
        """Estimate feature importances from tree structure (for loaded models)."""
        importances: dict[str, float] = dict.fromkeys(self.feature_names, 0.0)

        def count_splits(node: Any, depth: int = 0) -> None:
            if not isinstance(node, list) or len(node) != 5:
                return  # Leaf node

            feature, op, value, left, right = node

            # Weight by depth (higher nodes affect more samples)
            weight = 1.0 / (depth + 1)
            if feature in importances:
                importances[feature] += weight

            count_splits(left, depth + 1)
            count_splits(right, depth + 1)

        count_splits(self.model)

        return normalize_importances(importances)

    def get_depth(self) -> int:
        """Get the depth of the tree."""
        if self.model is None:
            return 0
        return compute_max_depth(self.model)

    def _export_cart(
        self,
        path: str,
        metadata: dict | None = None,
        use_gzip: bool = False,
        store_distributions: bool = False,
    ) -> None:
        """Export to compact binary format."""
        if self.model is None:
            raise ValueError("No model to export. Call train() first.")

        def _write(dest: str) -> None:
            write_tree_bytes(
                dest,
                self.model,
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
        model = self.model
        if not store_distributions:
            model = collapse_distributions(model)
        return {
            "model": model,
            "feature_specs": self._serialize_feature_specs(),
            "feature_names": self.feature_names,
            "task": self._effective_task(),
            "class_labels": self._class_labels,
            "metadata": metadata or {},
        }

    def _load_cart(self, path: str, use_gzip: bool = False) -> dict:
        """Load from compact binary format."""
        with open_file_binary(path, "rb") as f:
            data = f.read()
        model_data = _load_cart_from_bytes(data)

        # XGBoost .cart files carry K trees per round and an additive
        # base_score that DecisionTree's single-tree shape can't represent.
        # Forests are handled by RandomForest. Direct callers should use
        # cartlet.runner.Predictor or XGBoostTree.load_model() instead.
        if model_data.get("is_xgboost"):
            raise ValueError(
                f"{path} is an XGBoost .cart export; load it with "
                "cartlet.Predictor / cartlet.load_model() or "
                "cartlet.XGBoostTree.load_model() instead of DecisionTree."
            )
        if model_data.get("is_forest"):
            raise ValueError(
                f"{path} is a RandomForest .cart export; "
                "load it with cartlet.RandomForest.load_model() instead."
            )

        self._apply_config_from_cart(model_data)
        self.model = self._rebuild_tree_from_cart(model_data)

        return {
            "features": model_data["meta"]["features"],
            "task": model_data["meta"]["task"],
        }

    def _load_sklearn(self, path: str, use_gzip: bool = False) -> dict:
        """Load sklearn model - converts to cartlet format for inference."""
        from .trainer.sklearn import Sklearn

        sklearn_model, feature_names, feature_specs = self._read_sklearn_for_load(path)
        model, config = Sklearn.from_sklearn(sklearn_model, feature_names)
        self.model = model
        self.feature_names = feature_names
        self.feature_specs = feature_specs
        self._rebuild_name_to_col()

        from sklearn.tree import DecisionTreeClassifier

        is_classifier = isinstance(sklearn_model, DecisionTreeClassifier)
        task = TASK_CLASSIFICATION if is_classifier else TASK_REGRESSION
        self.task = task
        self._detected_task = task

        return {
            "features": [{"name": name, "type": TYPE_NUM} for name in feature_names],
            "task": task,
        }

    def _apply_loaded_data(self, data: dict) -> dict:
        """Apply loaded data from JSON/pickle to instance."""
        self.model = data["model"]
        return super()._apply_loaded_data(data)

    def _rebuild_tree_from_cart(self, model_data: dict) -> Any:
        """Rebuild nested tree structure from .cart flat nodes."""
        return rebuild_tree_from_cart(model_data, self.feature_names, tree_idx=0)

    def __repr__(self) -> str:
        status = "trained" if self.model else "untrained"
        task = self._effective_task()
        return (
            f"DecisionTree(features={len(self.feature_names)}, task={task}, {status})"
        )
