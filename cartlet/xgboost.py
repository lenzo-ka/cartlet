"""
XGBoost integration for cartlet.

Provides XGBoostTree class that trains using XGBoost's sklearn API
and exports to .cart format for lightweight inference.

Requires: xgboost>=1.5.0 (for native categorical support)
"""

from __future__ import annotations

import json
import pickle
from typing import Any

from .base import BaseModel
from .io.bytes import write_forest_bytes
from .io.utils import write_with_optional_gzip
from .types import (
    BINARY_CLASSIFICATION_THRESHOLD,
    DEFAULT_N_ESTIMATORS,
    TASK_AUTO,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TYPE_CAT,
    FeatureSpec,
    infer_feature_specs,
    is_likely_regression,
)

_DEFAULT_BASE_SCORE = 0.5
_UNKNOWN_CATEGORY = -1
_DEFAULT_LEARNING_RATE = 0.1
_DEFAULT_MAX_DEPTH = 6
# Unknown class labels at train-encode time fall back to the first class index;
# this mirrors sklearn's behavior when a label hasn't been seen during fit.
_FALLBACK_LABEL_INDEX = 0


class XGBoostTree(BaseModel):
    """
    XGBoost gradient boosted tree classifier/regressor.

    Trains using XGBoost's sklearn-compatible API, extracts trees,
    and exports to .cart format for lightweight cross-language inference.

    Examples:
        # Classification
        xgb = XGBoostTree(feature_names=["color", "size"])
        xgb.load_data(X, y)
        xgb.train()
        prediction = xgb.predict(["red", "large"])

        # Export to .cart
        xgb.export("model.cart")

        # Export to native XGBoost format
        xgb.export("model.xgb")
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        features: list[dict[str, Any]] | None = None,
        feature_names: list[str] | None = None,
        target: dict[str, Any] | None = None,
        task: str = TASK_AUTO,
        verbose: bool = False,
        **xgb_params,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage
            max_depth: Maximum depth per tree
            features: Feature specs
            feature_names: Simple feature names
            target: Target spec
            task: "classification", "regression", or "auto"
            verbose: Enable verbose output
            **xgb_params: Additional XGBoost parameters
        """
        super().__init__(
            features=features,
            feature_names=feature_names,
            target=target,
            task=task,
            verbose=verbose,
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.xgb_params = xgb_params

        # XGBoost-specific state
        self.trees: list[Any] = []
        self.class_labels: list[str] = []
        self.base_score: float = _DEFAULT_BASE_SCORE
        self._xgb_model: Any = None  # Raw XGBoost Booster
        self._warned_missing_direction = False

    def load_data(
        self,
        X: list[list[Any]],
        y: list[Any],
        counts: list[int] | None = None,
    ) -> None:
        """Load training data.

        Returns None, matching ``DecisionTree.load_data`` and
        ``RandomForest.load_data`` (previously returned ``self``).
        """
        self.X = list(X)
        self.y = list(y)
        self.counts = counts or [1] * len(y)
        self._infer_features()

    def _infer_features(self) -> None:
        """Infer feature types from data."""
        if not self.X:
            return

        n_features = len(self.X[0])

        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(n_features)]
            self._rebuild_name_to_col()

        if self.feature_specs:
            for i, spec in enumerate(self.feature_specs):
                if spec.type == TYPE_CAT:
                    spec.values = {str(row[i]) for row in self.X}
        else:
            inferred = infer_feature_specs(
                self.X,
                self.feature_names,
                exclude_bool_from_numeric=True,
                include_values=True,
                force_float_numeric=True,
            )
            self.feature_specs = [
                FeatureSpec(
                    name=spec["name"],
                    dtype=spec["dtype"],
                    type=spec["type"],
                    values=spec.get("values"),
                )
                for spec in inferred
            ]

        # Infer task type. Use the unique-count/ratio heuristic so integer
        # class labels are not mistaken for a regression target (see
        # DecisionTree.load_data for the same reasoning).
        if self.task == TASK_AUTO:
            self._detected_task = (
                TASK_REGRESSION if is_likely_regression(self.y) else TASK_CLASSIFICATION
            )

        if self._effective_task() == TASK_CLASSIFICATION:
            self.class_labels = sorted({str(v) for v in self.y})

    def train(self, random_state: int | None = None) -> dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            random_state: Random seed for reproducibility

        Returns:
            Dict with training info
        """
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost>=1.5.0"
            ) from err

        # Prepare data with native categorical support
        X_train, cat_features = self._prepare_data(self.X)

        # Create DMatrix with categorical support. Pass instance weights
        # through (previously stored but dropped) so weighted training matches
        # the other backends.
        dtrain = xgb.DMatrix(
            X_train,
            label=self._encode_labels(self.y),
            weight=self.counts if self.counts else None,
            enable_categorical=True,
            feature_names=self.feature_names,
            feature_types=[
                "c" if i in cat_features else "q"
                for i in range(len(self.feature_names))
            ],
        )

        # XGBoost parameters
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "verbosity": 1 if self.verbose else 0,
            **self.xgb_params,
        }

        is_classification = self._effective_task() == TASK_CLASSIFICATION
        if is_classification:
            n_classes = len(self.class_labels)
            if n_classes == 2:
                params["objective"] = "binary:logistic"
            else:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
        else:
            params["objective"] = "reg:squarederror"

        if random_state is not None:
            params["seed"] = random_state

        # Train
        self._xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=self.verbose,
        )

        # Extract trees
        self.trees = self._extract_trees()
        self.base_score = self._extract_base_score()

        return {
            "n_trees": len(self.trees),
            "n_estimators": self.n_estimators,
        }

    def _extract_base_score(self) -> float:
        """
        Read the trained booster's actual ``base_score`` (the additive offset
        applied to every prediction before tree contributions).

        XGBoost exposes the calibrated value via ``save_config`` under
        ``learner.learner_model_param.base_score``; the legacy
        ``Booster.attr("base_score")`` is ``None`` for default training, so
        we used to fall back to ``0.5`` and silently miss the real offset
        (e.g. the mean of the target for regression).
        """
        if self._xgb_model is None:
            return _DEFAULT_BASE_SCORE
        try:
            cfg = json.loads(self._xgb_model.save_config())
            raw = cfg["learner"]["learner_model_param"]["base_score"]
        except (KeyError, ValueError, TypeError):
            raw = self._xgb_model.attr("base_score")
        if raw is None:
            return _DEFAULT_BASE_SCORE
        # XGBoost may serialize values like "[3E0]" (hex-encoded float).
        text = str(raw).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return float.fromhex("0x" + text[1:-1].replace("E", "p"))
            except ValueError:
                pass
        return float(text)

    def _prepare_data(self, X: list[list[Any]]) -> tuple[list[list[Any]], set[int]]:
        """
        Prepare data for XGBoost with native categorical support.

        Returns:
            Tuple of (X_prepared, categorical_feature_indices)
        """
        cat_features: set[int] = set()
        cat_mappings: dict[int, dict[str, int]] = {}

        # Build category mappings
        for i, spec in enumerate(self.feature_specs):
            if spec.type == TYPE_CAT:
                cat_features.add(i)
                spec_values = sorted(spec.values) if spec.values else []
                cat_mappings[i] = {str(v): idx for idx, v in enumerate(spec_values)}

        # Convert categorical to integers (XGBoost native categorical)
        X_prep: list[list[float]] = []
        for row in X:
            new_row: list[float] = []
            for i, val in enumerate(row):
                if i in cat_features:
                    mapping = cat_mappings[i]
                    new_row.append(float(mapping.get(str(val), _UNKNOWN_CATEGORY)))
                else:
                    new_row.append(float(val))
            X_prep.append(new_row)

        return X_prep, cat_features

    def _encode_labels(self, y: list[Any]) -> list[float]:
        """Encode class labels to numeric."""
        if self._is_regression():
            return [float(v) for v in y]

        label_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        return [float(label_to_idx.get(str(v), _FALLBACK_LABEL_INDEX)) for v in y]

    def _extract_trees(self) -> list[Any]:
        """
        Extract trees from XGBoost model and convert to our format.

        Returns:
            List of tree structures in cartlet format
        """
        trees_json = self._xgb_model.get_dump(dump_format="json")
        trees = []

        for tree_json in trees_json:
            tree_data = json.loads(tree_json)
            tree = self._convert_xgb_node(tree_data)
            trees.append(tree)

        return trees

    def _convert_xgb_node(self, node: dict) -> Any:
        """
        Convert XGBoost JSON node to cartlet format.

        XGBoost node format:
            Decision: {"nodeid": 0, "split": "f0", "split_condition": 0.5,
                       "yes": 1, "no": 2, "children": [...]}
            Leaf: {"nodeid": 0, "leaf": 0.234}

        For native categorical:
            split_condition is a list of category indices (e.g., [0, 2])
            "yes" branch is taken when value IS in the set
        """
        # Leaf node
        if "leaf" in node:
            # XGBoost leaves are always floats
            return [float(node["leaf"]), 0.0, 1]

        # Decision node
        split_feature = node["split"]
        split_condition = node.get("split_condition", 0)

        # Find children
        children = node.get("children", [])
        yes_id = node.get("yes", 0)
        no_id = node.get("no", 1)

        # cartlet's .cart format routes a missing/None feature to the right
        # ("no") branch unconditionally. XGBoost stores a per-node "missing"
        # direction; when it points at the "yes" branch, exported predictions
        # can diverge from Booster.predict on inputs with missing features.
        # The format can't encode per-node missing direction, so warn once.
        missing_id = node.get("missing")
        if (
            missing_id is not None
            and missing_id != no_id
            and not self._warned_missing_direction
        ):
            self.logger.warning(
                "XGBoost model routes missing values to the 'yes' branch at "
                "some nodes; the .cart format always routes missing values "
                "right, so predictions on inputs with missing features may "
                "diverge from Booster.predict."
            )
            self._warned_missing_direction = True

        yes_child = None
        no_child = None
        for child in children:
            if child["nodeid"] == yes_id:
                yes_child = self._convert_xgb_node(child)
            elif child["nodeid"] == no_id:
                no_child = self._convert_xgb_node(child)

        if yes_child is None or no_child is None:
            raise ValueError(f"Could not find children for node: {node}")

        # Get feature name
        feat_name = split_feature

        # Detect categorical: split_condition is a list
        is_categorical = isinstance(split_condition, list)

        if is_categorical:
            categories = split_condition
            if len(categories) == 1:
                cat_val = self._get_category_value(feat_name, categories[0])
                return [feat_name, "=", cat_val, yes_child, no_child]
            cases = {}
            for cat_idx in categories:
                cat_val = self._get_category_value(feat_name, cat_idx)
                cases[cat_val] = yes_child
            return [feat_name, "switch", cases, no_child]

        return [feat_name, "<", float(split_condition), yes_child, no_child]

    def _get_category_value(self, feat_name: str, cat_idx: int) -> str:
        """Get category string value from index."""
        for spec in self.feature_specs:
            if spec.name == feat_name and spec.values:
                sorted_values = sorted(spec.values)
                if 0 <= cat_idx < len(sorted_values):
                    return str(sorted_values[cat_idx])
        return str(cat_idx)

    def _raw_predict(self, vector: list[Any]) -> Any:
        """Run the underlying booster on a single row; returns its raw output."""
        if self._xgb_model is None:
            raise ValueError("Model not trained. Call train() first.")

        import xgboost as xgb

        X_prep, _ = self._prepare_data([vector])
        dtest = xgb.DMatrix(
            X_prep,
            enable_categorical=True,
            feature_names=self.feature_names,
        )
        return self._xgb_model.predict(dtest)[0]

    def _require_class_labels(self) -> None:
        """Fail clearly if a classification predict is attempted without labels.

        ``XGBoostTree.load()`` / ``_load_pickle`` restore only the raw Booster;
        feature specs and class labels are not persisted there, so a
        classification predict would otherwise raise an opaque IndexError.
        """
        if not self.class_labels:
            raise ValueError(
                "This XGBoost model has no class labels. Models loaded via "
                "XGBoostTree.load() or from a pickle do not carry feature specs "
                "or class labels; set feature_specs and class_labels (e.g. via "
                "load_data + train) before a classification predict."
            )

    def predict(self, vector: list[Any], **kwargs: Any) -> Any:
        """
        Predict for a single feature vector.

        Args:
            vector: Feature values

        Returns:
            Prediction (class label or float)

        Note:
            Missing values: When a feature is None or missing, comparisons fail
            and the tree takes the "no" branch (right child).
        """
        pred = self._raw_predict(vector)

        if self._is_regression():
            return float(pred)

        self._require_class_labels()
        n_classes = len(self.class_labels)
        if n_classes == 2:
            pred_class = 1 if pred > BINARY_CLASSIFICATION_THRESHOLD else 0
        else:
            pred_class = int(pred.argmax())

        return self.class_labels[pred_class]

    def predict_proba(self, vector: list[Any]) -> dict[str, float]:
        """
        Get prediction probabilities.

        Args:
            vector: Feature values

        Returns:
            Dict of {class_label: probability}
        """
        if self._is_regression():
            raise ValueError("predict_proba only available for classification")

        self._require_class_labels()
        pred = self._raw_predict(vector)

        n_classes = len(self.class_labels)
        if n_classes == 2:
            prob_1 = float(pred)
            return {
                self.class_labels[0]: 1 - prob_1,
                self.class_labels[1]: prob_1,
            }
        return {self.class_labels[i]: float(pred[i]) for i in range(n_classes)}

    # =========================================================================
    # Export methods (override BaseModel abstracts)
    # =========================================================================

    def export(
        self,
        path: str,
        metadata: dict | None = None,
        store_distributions: bool = False,
        format: str | None = None,
    ) -> None:
        """
        Export model to file.

        Supported formats:
            .cart  - Compact binary format for the minimal runner
            .xgb / .ubj - Native XGBoost format
            .json  - Native XGBoost JSON (requires a loaded booster)

        Args:
            path: Output file path.
            metadata: Optional metadata dict (embedded into the .cart trailer).
            store_distributions: Store class distributions (.cart only).
            format: Explicit format override; one of ``"cart"``, ``"xgb"``,
                ``"ubj"``, ``"xgb-json"``. When set, bypasses
                extension-based dispatch (``.gz`` on ``path`` still toggles
                gzip for ``"cart"``).

        Raises:
            ValueError: If the resolved format is not a supported
                XGBoostTree export target (e.g. ``.pkl``/``.skl`` are
                rejected here rather than silently writing ``.cart`` bytes
                into the wrong file).
        """
        resolved = self._resolve_export_format(path, format)
        if resolved == "xgb-native":
            self._export_xgb_native(path)
            return
        use_gzip = path.endswith(".gz")
        self._export_cart(path, metadata, use_gzip, store_distributions)

    def _resolve_export_format(self, path: str, format: str | None) -> str:
        """
        Return either ``"cart"`` or ``"xgb-native"``.

        Raises ValueError for anything else (e.g. ``.pkl``, ``.skl``, ``.jsonl``)
        rather than silently writing the wrong codec into the file.
        """
        if format is not None:
            fmt = format.lower().lstrip(".")
            if fmt in {"xgb", "ubj", "xgb-json"}:
                return "xgb-native"
            if fmt == "cart":
                return "cart"
            raise ValueError(
                f"Unsupported XGBoostTree format {format!r}. "
                "Use one of: cart, xgb, ubj, xgb-json"
            )

        if path.endswith((".xgb", ".ubj")):
            return "xgb-native"
        if path.endswith(".json") and self._xgb_model is not None:
            return "xgb-native"

        bare = path[:-3] if path.endswith(".gz") else path
        if bare.endswith(".cart"):
            return "cart"

        raise ValueError(
            f"XGBoostTree cannot export to {path!r}. "
            "Use .cart / .cart.gz, .xgb / .ubj, or pass format='cart'/'xgb'. "
            "Pickle, JSON Lines, and sklearn formats are not supported here."
        )

    def _build_export_dict(
        self, metadata: dict | None = None, store_distributions: bool = True
    ) -> dict:
        """Build dictionary for JSON/pickle export.

        ``store_distributions`` is accepted to match the base signature;
        XGBoost leaves are raw scores, so there are no class distributions to
        strip.
        """
        return {
            "trees": self.trees,
            "feature_specs": self._serialize_feature_specs(),
            "feature_names": self.feature_names,
            "task": self._effective_task(),
            "class_labels": self.class_labels,
            "base_score": self.base_score,
            "metadata": metadata or {},
        }

    def _export_cart(
        self,
        path: str,
        metadata: dict | None,
        use_gzip: bool,
        store_distributions: bool,
    ) -> None:
        """Export to .cart binary format."""
        if not self.trees:
            raise ValueError("No trees to export. Call train() first.")

        meta = dict(metadata) if metadata else {}
        meta.update(
            {
                "model_type": "xgboost",
                "base_score": self.base_score,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
            }
        )

        def _write(dest: str) -> None:
            write_forest_bytes(
                dest,
                self.trees,
                self.feature_specs,
                self.name_to_col,
                self.class_labels,
                is_regression=self._is_regression(),
                metadata=meta,
                store_distributions=store_distributions,
                is_xgboost=True,
            )

        write_with_optional_gzip(path, use_gzip, _write)

    def _export_xgb_native(self, path: str) -> None:
        """Export to native XGBoost format (.xgb, .ubj, or .json)."""
        if self._xgb_model is None:
            raise ValueError("No model to export. Call train() first.")
        self._xgb_model.save_model(path)

    # =========================================================================
    # Load methods (override BaseModel abstracts)
    # =========================================================================

    def _load_cart(self, path: str, use_gzip: bool) -> dict:
        """Load from .cart - not fully supported, use runner instead."""
        raise NotImplementedError(
            "Loading .cart into XGBoostTree not supported. "
            "Use cartlet.runner.load_model() for inference."
        )

    def _load_json(self, path: str, use_gzip: bool = False) -> dict:
        """Load from XGBoost JSON format."""
        return self._load_xgb_native(path)

    def _load_jsonl(self, path: str, use_gzip: bool = False) -> dict:
        """JSONL not supported for XGBoost."""
        raise NotImplementedError("JSONL load not supported for XGBoost")

    def _load_pickle(self, path: str, use_gzip: bool = False) -> dict:
        """Load from pickle."""
        with open(path, "rb") as f:
            self._xgb_model = pickle.load(f)
        return {"loaded": True}

    def _load_sklearn(self, path: str, use_gzip: bool = False) -> dict:
        """sklearn load not applicable for XGBoost."""
        raise NotImplementedError("sklearn load not supported for XGBoost. Use .xgb")

    def _load_xgb_native(self, path: str) -> dict:
        """Load from native XGBoost format."""
        import xgboost as xgb

        self._xgb_model = xgb.Booster()
        self._xgb_model.load_model(path)
        return {"loaded": True}

    @classmethod
    def load(cls, path: str) -> XGBoostTree:
        """
        Load an XGBoost model from file.

        Args:
            path: Path to model file (.xgb or .json)

        Returns:
            XGBoostTree instance
        """
        import xgboost as xgb

        model = cls()
        model._xgb_model = xgb.Booster()
        model._xgb_model.load_model(path)

        # Note: feature specs and class labels need to be provided separately
        # when loading from native format
        return model
