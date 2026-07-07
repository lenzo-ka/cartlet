"""
Isolation Forest for anomaly detection.

Identifies anomalies by how quickly data points are isolated by random splits.
Anomalies are isolated in fewer splits (shorter path length = more anomalous).
"""

from __future__ import annotations

import json
import math
import pickle
import random
from typing import Any

from .io.utils import open_file, open_file_binary, resolve_format
from .types import _MAX_RANDOM_SEED, DEFAULT_N_ESTIMATORS
from .utils import default_logger

_EULER_MASCHERONI = 0.5772156649

# Default subsample size used by the original Isolation Forest paper. Picked so
# the average path length c(n) is well-defined for typical anomaly workloads.
DEFAULT_MAX_SAMPLES = 256

# Anomaly score midpoint: scores at or below this look "normal"; scores above
# are progressively more anomalous. Used both as the cold-start fallback in
# `predict` and as the centering term in `decision_function`.
ANOMALY_SCORE_MIDPOINT = 0.5

# Verbose log cadence: print progress every Nth tree.
_VERBOSE_TREE_INTERVAL = 10


def _c(n: int) -> float:
    """Average path length of unsuccessful search in BST with n elements."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + _EULER_MASCHERONI) - 2.0 * (n - 1) / n


class IsolationForest:
    """
    Isolation Forest for anomaly detection.

    Builds an ensemble of random trees on subsamples. Anomalies have
    shorter average path lengths across trees.

    Examples:
        ifo = IsolationForest(n_estimators=100)
        ifo.load_data(X)
        ifo.train()
        score = ifo.predict([1.0, 2.0])  # 0-1, higher = more anomalous
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_samples: int | float | str = DEFAULT_MAX_SAMPLES,
        max_depth: int | None = None,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        logger=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.feature_names: list[str] = feature_names or []
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger or default_logger()

        self.trees: list[Any] = []
        self._n_samples_used = 0
        self.X: list[list[float]] = []

    def load_data(self, X: list[list[Any]]) -> None:
        """Load training data (no labels — unsupervised)."""
        self.X = [[float(v) for v in row] for row in X]
        if not self.feature_names and X:
            self.feature_names = [f"f{i}" for i in range(len(X[0]))]

    def _resolve_max_samples(self, n: int) -> int:
        """Resolve ``max_samples`` (int / float fraction / "auto") to a count.

        - ``"auto"``: ``min(256, n)`` (matches sklearn's IsolationForest).
        - ``float`` in (0, 1]: that fraction of ``n`` (at least 1).
        - ``int``: capped at ``n``.
        """
        ms = self.max_samples
        if isinstance(ms, str):
            if ms != "auto":
                raise ValueError(
                    f"max_samples={ms!r}; expected 'auto', an int, or a "
                    "float in (0, 1]."
                )
            return min(256, n)
        if isinstance(ms, bool):  # bool is an int subclass; reject explicitly
            raise ValueError("max_samples must be a number or 'auto', not bool.")
        if isinstance(ms, float):
            if not 0.0 < ms <= 1.0:
                raise ValueError(
                    f"max_samples={ms}; float must be in (0, 1] (a fraction of n)."
                )
            return max(1, int(ms * n))
        if isinstance(ms, int):
            if ms <= 0:
                raise ValueError(f"max_samples={ms}; int must be positive.")
            return min(ms, n)
        raise ValueError(
            f"max_samples={ms!r}; expected 'auto', an int, or a float in (0, 1]."
        )

    def train(self) -> dict[str, Any]:
        """Build the isolation forest."""
        if not self.X:
            raise ValueError("No training data loaded. Call load_data() first.")

        n = len(self.X)
        sub_size = self._resolve_max_samples(n)
        self._n_samples_used = sub_size

        depth_limit = self.max_depth or math.ceil(math.log2(max(sub_size, 2)))

        rng = random.Random(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % _VERBOSE_TREE_INTERVAL == 0:
                self.logger.info(
                    "Building isolation tree %d/%d...", i + 1, self.n_estimators
                )
            indices = rng.sample(range(n), sub_size) if sub_size < n else list(range(n))
            subsample = [self.X[j] for j in indices]
            seed = rng.randint(0, _MAX_RANDOM_SEED)
            tree = self._build_tree(subsample, depth_limit, random.Random(seed))
            self.trees.append(tree)

        return {"n_estimators": len(self.trees), "max_samples": sub_size}

    def _build_tree(
        self,
        X: list[list[float]],
        depth_limit: int,
        rng: random.Random,
        depth: int = 0,
    ) -> Any:
        """Recursively build an isolation tree with random splits."""
        n = len(X)
        if n <= 1 or depth >= depth_limit:
            return n  # leaf: number of samples

        n_features = len(X[0])
        feat_id = rng.randint(0, n_features - 1)

        values = [row[feat_id] for row in X]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return n  # can't split on constant feature

        threshold = rng.uniform(min_val, max_val)

        left = [row for row in X if row[feat_id] <= threshold]
        right = [row for row in X if row[feat_id] > threshold]

        if not left or not right:
            return n

        left_tree = self._build_tree(left, depth_limit, rng, depth + 1)
        right_tree = self._build_tree(right, depth_limit, rng, depth + 1)

        # [feature_index, threshold, left_child, right_child]
        return [feat_id, threshold, left_tree, right_tree]

    def _path_length(self, vector: list[float], node: Any, depth: int = 0) -> float:
        """Compute path length for a sample through one tree."""
        # Leaf node: integer storing number of samples
        if isinstance(node, int):
            return depth + _c(node)

        feat_id, threshold, left, right = node
        if vector[feat_id] <= threshold:
            return self._path_length(vector, left, depth + 1)
        return self._path_length(vector, right, depth + 1)

    def predict(self, vector: list[Any], **kwargs: Any) -> float:
        """
        Anomaly score for a single sample.

        Returns:
            Float in [0, 1]. Values > 0.5 suggest anomalies.
            Close to 1.0 = definite anomaly. Close to 0.0 = normal.
        """
        if not self.trees:
            raise ValueError("Model not trained. Call train() first.")

        if self.feature_names and len(vector) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {len(vector)}."
            )

        fvec = [float(v) for v in vector]
        mean_path = sum(self._path_length(fvec, tree) for tree in self.trees) / len(
            self.trees
        )

        cn = _c(self._n_samples_used)
        if cn == 0:
            return ANOMALY_SCORE_MIDPOINT
        return 2.0 ** (-mean_path / cn)

    def predict_batch(self, vectors: list[list[Any]]) -> list[float]:
        """Anomaly scores for multiple samples."""
        return [self.predict(v) for v in vectors]

    def decision_function(self, vector: list[Any]) -> float:
        """
        Raw anomaly score (sklearn convention: negative = more anomalous).

        Returns:
            Negative values for anomalies, positive for normal points.
        """
        return -(self.predict(vector) - ANOMALY_SCORE_MIDPOINT)

    # -- Export / Import (JSON and pickle) --

    def _tree_to_nested(self, node: Any) -> Any:
        """Convert internal tree to serializable nested list format."""
        if isinstance(node, int):
            return {"leaf": node}
        feat_id, threshold, left, right = node
        return {
            "feature": feat_id,
            "threshold": threshold,
            "left": self._tree_to_nested(left),
            "right": self._tree_to_nested(right),
        }

    def _tree_from_nested(self, data: Any) -> Any:
        """Reconstruct internal tree from serialized format."""
        if "leaf" in data:
            return data["leaf"]
        return [
            data["feature"],
            data["threshold"],
            self._tree_from_nested(data["left"]),
            self._tree_from_nested(data["right"]),
        ]

    def _build_export_dict(self, metadata: dict | None = None) -> dict:
        return {
            "isolation_forest": True,
            "trees": [self._tree_to_nested(t) for t in self.trees],
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "n_samples_used": self._n_samples_used,
            "metadata": metadata or {},
        }

    def export(
        self,
        path: str,
        metadata: dict | None = None,
        format: str | None = None,
    ) -> None:
        """
        Export model to JSON or pickle.

        Args:
            path: Destination path; extension is used for codec selection
                when ``format`` is None.
            metadata: Optional metadata dict embedded into the exported model.
            format: Explicit codec name (``"json"`` or ``"pkl"``); bypasses
                extension detection. Use this to write under a custom suffix.
        """
        ext, _ = resolve_format(path, format)

        if ext == ".json":
            self._export_json(path, metadata)
            return
        if ext == ".pkl":
            self._export_pickle(path, metadata)
            return
        raise ValueError(f"IsolationForest supports .json and .pkl export, not {ext}")

    def _export_json(self, path: str, metadata: dict | None = None) -> None:
        data = self._build_export_dict(metadata)
        with open_file(path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_pickle(self, path: str, metadata: dict | None = None) -> None:
        data = self._build_export_dict(metadata)
        with open_file_binary(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path: str, format: str | None = None) -> dict:
        """
        Load model from JSON or pickle.

        Args:
            path: Path to model file.
            format: Explicit codec name (``"json"`` or ``"pkl"``); bypasses
                extension detection.
        """
        ext, _ = resolve_format(path, format)

        if ext == ".json":
            return self._load_json(path)
        if ext == ".pkl":
            return self._load_pickle(path)
        raise ValueError(f"IsolationForest supports .json and .pkl loading, not {ext}")

    def _load_json(self, path: str) -> dict:
        with open_file(path, "r") as f:
            data = json.load(f)
        return self._apply_loaded_data(data)

    def _load_pickle(self, path: str) -> dict:
        with open_file_binary(path, "rb") as f:
            data = pickle.load(f)
        return self._apply_loaded_data(data)

    def _apply_loaded_data(self, data: dict) -> dict:
        self.feature_names = data.get("feature_names", [])
        self.n_estimators = data.get("n_estimators", 0)
        self.max_samples = data.get("max_samples", DEFAULT_MAX_SAMPLES)
        self._n_samples_used = data.get("n_samples_used", self.max_samples)
        self.trees = [self._tree_from_nested(t) for t in data.get("trees", [])]
        return data

    def __repr__(self) -> str:
        status = f"{len(self.trees)} trees" if self.trees else "untrained"
        return f"IsolationForest(n_estimators={self.n_estimators}, {status})"
