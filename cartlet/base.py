"""
Base class for models in cartlet.
"""

from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable

from .io.cart_format import MAGIC as CART_MAGIC
from .io.utils import open_file, open_file_binary, require_joblib, resolve_format
from .types import (
    DTYPE_STR,
    TASK_AUTO,
    TASK_REGRESSION,
    TYPE_CAT,
    TYPE_NUM,
    VALID_TASKS,
    FeatureSpec,
    normalize_feature_spec,
    resolve_task,
)
from .utils import default_logger


class BaseModel(ABC):
    """
    Base class for cartlet supervised tree models.

    Concrete subclasses must implement `predict`, `_export_cart`,
    `_build_export_dict`, `_load_cart`, and `_load_sklearn`. JSON/JSONL/pickle
    export and load are provided here in terms of `_build_export_dict` /
    `_apply_loaded_data` and can be overridden if a subclass needs custom
    format handling.

    Two concrete subclasses follow the full BaseModel contract:
    `DecisionTree` and `RandomForest`. `XGBoostTree` reuses this base for the
    public API surface (constructor args, evaluation, task detection) but
    overrides export/load to support XGBoost's native formats; in particular,
    `_load_cart` reconstructs an internal XGBoost booster rather than the
    nested-list tree used by the other subclasses.

    `IsolationForest` does **not** subclass `BaseModel`; it follows a similar
    but separate API for unsupervised anomaly detection.
    """

    def __init__(
        self,
        features: list[dict[str, Any]] | None = None,
        feature_names: list[str] | None = None,
        target: dict[str, Any] | None = None,
        task: str = TASK_AUTO,
        verbose: bool = False,
        logger=None,
    ):
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {VALID_TASKS}")

        self.feature_specs: list[FeatureSpec] = []
        self.feature_names: list[str] = []
        self.name_to_col: dict[str, int] = {}

        if features:
            self.feature_specs = [normalize_feature_spec(f) for f in features]
            self.feature_names = [f.name for f in self.feature_specs]
        elif feature_names:
            self.feature_names = feature_names
            self.feature_specs = [
                FeatureSpec(name=name, dtype=DTYPE_STR, type=None)
                for name in feature_names
            ]

        if self.feature_names:
            self._rebuild_name_to_col()

        self.target_spec: FeatureSpec | None = None
        if target:
            self.target_spec = (
                target if isinstance(target, FeatureSpec) else FeatureSpec(**target)
            )

        self.task = task
        self._detected_task: str | None = None
        self.verbose = verbose
        self.logger = logger or default_logger()

        # Training data
        self.X: list[list[Any]] = []
        self.y: list[Any] = []
        self.counts: list[int] = []

        # Original sklearn model
        self._sklearn_model: Any = None

        # Feature importances (set by trainer)
        self._feature_importances: dict[str, float] = {}

    def _rebuild_name_to_col(self) -> dict[str, int]:
        """Rebuild name-to-column index mapping from feature_names."""
        self.name_to_col = {name: idx for idx, name in enumerate(self.feature_names)}
        return self.name_to_col

    @property
    def _class_labels(self) -> list[str]:
        """Sorted unique string class labels from training targets."""
        if not self.y:
            return []
        return sorted({str(v) for v in self.y}, key=str)

    def _effective_task(self) -> str:
        """Return the effective task (classification or regression)."""
        target_type = self.target_spec.type if self.target_spec else None
        return resolve_task(self.task, self._detected_task, target_type)

    def _is_regression(self) -> bool:
        """Check if this is a regression task."""
        return self._effective_task() == TASK_REGRESSION

    def _serialize_feature_specs(self) -> list[dict[str, Any]]:
        """Serialize feature_specs to a list of dicts for export."""
        return [
            {
                "name": s.name,
                "dtype": s.dtype,
                "type": s.type,
                "values": sorted(s.values, key=str) if s.values else None,
            }
            for s in self.feature_specs
        ]

    @abstractmethod
    def predict(self, vector: list[Any], **kwargs) -> Any:
        """Predict for a single feature vector."""
        ...

    def predict_batch(
        self,
        vectors: list[list[Any]],
        **kwargs,
    ) -> list[Any]:
        """Predict for multiple feature vectors."""
        return [self.predict(v, **kwargs) for v in vectors]

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        path: str,
        metadata: dict | None = None,
        store_distributions: bool = True,
        format: str | None = None,
    ) -> None:
        """
        Export trained model to file.

        Format is detected from the file extension; supports `.cart`, `.json`,
        `.jsonl`, `.pkl`/`.pickle`, `.skl`/`.joblib`, with an optional `.gz`
        suffix for gzip compression.

        Args:
            path: Destination path.
            metadata: Optional metadata dict embedded into the exported model.
            store_distributions: If False, leaves store only the best class
                (smaller file, but `predict_nbest` will return only the top).
            format: Explicit format override (e.g. ``"jsonl"``, ``"cart"``).
                When set, the file extension is ignored for codec selection;
                gzip is still inferred from a trailing ``.gz`` on ``path``.
                Use this to write a model under a custom suffix
                (e.g. ``model.g2p.gz``) while keeping the chosen wire format.

        Raises:
            ValueError: If the extension or ``format`` is not recognized.
            ImportError: If `.skl`/`.joblib` is requested without `joblib`.
        """
        ext, use_gzip = resolve_format(path, format)

        dispatch: dict[str, Callable] = {
            ".cart": self._export_cart,
            ".json": self._export_json,
            ".jsonl": self._export_jsonl,
            ".pkl": self._export_pickle,
            ".pickle": self._export_pickle,
            ".skl": self._export_sklearn,
            ".joblib": self._export_sklearn,
        }

        if ext not in dispatch:
            raise ValueError(
                f"Unknown format: {ext}. Use .cart, .json, .jsonl, .pkl, or .skl"
            )

        method = dispatch[ext]
        if ext in (".skl", ".joblib"):
            method(path, use_gzip=use_gzip)
        else:
            method(
                path,
                metadata=metadata,
                use_gzip=use_gzip,
                store_distributions=store_distributions,
            )

    @abstractmethod
    def _export_cart(
        self,
        path: str,
        metadata: dict | None,
        use_gzip: bool,
        store_distributions: bool,
    ): ...

    @abstractmethod
    def _build_export_dict(self, metadata: dict | None = None) -> dict: ...

    def _export_json(
        self,
        path: str,
        metadata: dict | None = None,
        use_gzip: bool = False,
        store_distributions: bool = True,
    ) -> None:
        """Export to JSON format."""
        data = self._build_export_dict(metadata)
        with open_file(path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_jsonl(
        self,
        path: str,
        metadata: dict | None = None,
        use_gzip: bool = False,
        store_distributions: bool = True,
    ) -> None:
        """Export to JSON Lines format."""
        data = self._build_export_dict(metadata)
        with open_file(path, "w") as f:
            f.write(json.dumps(data) + "\n")

    def _export_pickle(
        self,
        path: str,
        metadata: dict | None = None,
        use_gzip: bool = False,
        store_distributions: bool = True,
    ) -> None:
        """Export to pickle format."""
        data = self._build_export_dict(metadata)
        with open_file_binary(path, "wb") as f:
            pickle.dump(data, f)

    def _export_sklearn(self, path: str, use_gzip: bool = False) -> None:
        """Export sklearn model (requires training with trainer='sklearn')."""
        if self._sklearn_model is None:
            raise ValueError(
                "No sklearn model available. Train with trainer='sklearn' first."
            )
        compress = 3 if use_gzip else 0
        require_joblib().dump(self._sklearn_model, path, compress=compress)

    def _read_sklearn_for_load(
        self, path: str
    ) -> tuple[Any, list[str], list[FeatureSpec]]:
        """
        Common boilerplate for `_load_sklearn`: load the joblib file, derive
        feature names (from `feature_names_in_` if present, otherwise `f0..fN`)
        and build numeric FeatureSpec entries. Returns the raw sklearn model
        plus the derived feature names and specs; subclasses are responsible
        for the model-specific reconstruction that follows.
        """
        sklearn_model = require_joblib().load(path)
        self._sklearn_model = sklearn_model
        n_features = sklearn_model.n_features_in_
        if hasattr(sklearn_model, "feature_names_in_"):
            feature_names = list(sklearn_model.feature_names_in_)
        else:
            feature_names = [f"f{i}" for i in range(n_features)]
        feature_specs = [
            FeatureSpec(name=name, dtype=DTYPE_STR, type=TYPE_NUM)
            for name in feature_names
        ]
        return sklearn_model, feature_names, feature_specs

    # =========================================================================
    # Load
    # =========================================================================

    def load_model(self, path: str, format: str | None = None) -> dict:
        """
        Load a trained model from file.

        Format is detected from the file extension (`.cart`, `.json`, `.jsonl`,
        `.pkl`/`.pickle`, `.skl`/`.joblib`, with optional `.gz`). Files with
        an unknown extension but a leading `CART` magic are still accepted as
        binary.

        Args:
            path: Path to the model file.
            format: Explicit format override (e.g. ``"jsonl"``, ``"cart"``).
                When set, the file extension is ignored for codec selection;
                gzip is still inferred from a trailing ``.gz`` on ``path``.
                Use this to read a model that lives under a custom suffix
                (e.g. ``model.g2p.gz``).

        Returns:
            The raw decoded model dict (already applied to `self`); useful
            mainly for introspection.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If neither the extension nor ``format`` resolves to a
                known codec and the file does not start with the `CART` magic.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        ext, use_gzip = resolve_format(path, format)

        dispatch: dict[str, Callable] = {
            ".cart": self._load_cart,
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
            ".pkl": self._load_pickle,
            ".pickle": self._load_pickle,
            ".skl": self._load_sklearn,
            ".joblib": self._load_sklearn,
        }

        if ext in dispatch:
            return dispatch[ext](path, use_gzip=use_gzip)

        with open(path, "rb") as f:
            magic = f.read(len(CART_MAGIC))
        if magic == CART_MAGIC:
            return self._load_cart(path, use_gzip=False)

        raise ValueError(
            f"Unknown format: {ext}. Use .cart, .json, .jsonl, .pkl, or .skl"
        )

    @abstractmethod
    def _load_cart(self, path: str, use_gzip: bool) -> dict: ...

    def _load_json(self, path: str, use_gzip: bool = False) -> dict:
        """Load from JSON format."""
        with open_file(path, "r") as f:
            data = json.load(f)
        return self._apply_loaded_data(data)

    def _load_jsonl(self, path: str, use_gzip: bool = False) -> dict:
        """Load from JSON Lines format."""
        with open_file(path, "r") as f:
            data = json.loads(f.readline())
        return self._apply_loaded_data(data)

    def _load_pickle(self, path: str, use_gzip: bool = False) -> dict:
        """Load from pickle format."""
        with open_file_binary(path, "rb") as f:
            data = pickle.load(f)
        return self._apply_loaded_data(data)

    @abstractmethod
    def _load_sklearn(self, path: str, use_gzip: bool = False) -> dict: ...

    # =========================================================================
    # Config / data restoration
    # =========================================================================

    def _apply_config_from_cart(self, model_data: dict) -> None:
        """Apply configuration from loaded .cart model."""
        meta = model_data["meta"]
        features = meta.get("features", [])

        self.feature_specs = []
        for f in features:
            spec = FeatureSpec(
                name=f["name"],
                dtype=DTYPE_STR,  # .cart doesn't store dtype
                type=f.get("type"),
            )
            if spec.type == TYPE_CAT and "values" in f:
                spec.values = set(f["values"])
            self.feature_specs.append(spec)

        self.feature_names = [f.name for f in self.feature_specs]
        self._rebuild_name_to_col()

        task = meta.get("task", TASK_AUTO)
        if task in VALID_TASKS:
            self.task = task
            self._detected_task = task

    def _apply_loaded_data(self, data: dict) -> dict:
        """Apply loaded data from JSON/pickle to instance."""
        self.feature_names = data.get("feature_names", [])
        self._rebuild_name_to_col()

        self.feature_specs = []
        for spec_dict in data.get("feature_specs", []):
            spec = FeatureSpec(
                name=spec_dict["name"],
                dtype=spec_dict.get("dtype", DTYPE_STR),
                type=spec_dict.get("type"),
            )
            if spec_dict.get("values"):
                spec.values = set(spec_dict["values"])
            self.feature_specs.append(spec)

        task = data.get("task", TASK_AUTO)
        if task in VALID_TASKS:
            self.task = task
            self._detected_task = task

        return data
