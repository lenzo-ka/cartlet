"""
Base trainer class for decision trees.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..types import (
    DEFAULT_MIN_DIST_ENTROPY,
    PROB_HIGH_CONFIDENCE,
    PROB_MIN_THRESHOLD,
)

if TYPE_CHECKING:
    from ..tree import DecisionTree


class _IndexedSequence(Sequence[Any]):
    """Read-only indexed view over a sequence without copying its values."""

    def __init__(self, values: Sequence[Any], indices: Sequence[int]):
        self._values = values
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._values[i] for i in self._indices[index]]
        return self._values[self._indices[index]]


def make_classification_distribution(
    class_probs: list[tuple[Any, float]],
    store_distributions: bool = True,
    min_confidence: float = PROB_HIGH_CONFIDENCE,
    min_dist_entropy: float = DEFAULT_MIN_DIST_ENTROPY,
) -> Any:
    """
    Build a classification leaf value from class probabilities.

    Args:
        class_probs: List of (class_label, probability) tuples, sorted by prob desc
        store_distributions: Whether to store full distributions
        min_confidence: If best-class probability exceeds this, lossily store
            only the class label. The collapsed leaf reports probability 1.0.
            The default 1.0 disables this confidence gate.
        min_dist_entropy: If the distribution's entropy (bits) is below this,
            lossily collapse to the best class, which reports probability 1.0.
            Applied consistently across backends so native and sklearn leaves
            agree. The default 0.0 disables the gate.

    Returns:
        Best class label (str) or distribution dict
    """
    if not class_probs:
        return "-"

    best_class, best_prob = class_probs[0]

    if not store_distributions:
        return best_class

    if len(class_probs) == 1 or best_prob > min_confidence:
        return best_class

    if min_dist_entropy > 0.0:
        entropy = -sum(p * math.log2(p) for _, p in class_probs if p > 0)
        if entropy < min_dist_entropy:
            return best_class

    dist = {cls: prob for cls, prob in class_probs if prob >= PROB_MIN_THRESHOLD}

    if len(dist) <= 1:
        return best_class

    return dist


def normalize_importances(importances: dict[str, float]) -> dict[str, float]:
    """Normalize feature importances to sum to 1.0."""
    total = sum(importances.values())
    if total > 0:
        return {k: v / total for k, v in importances.items()}
    return importances


class Trainer(ABC):
    """
    Abstract base class for decision tree trainer backends.

    Subclasses implement the actual tree-building algorithm.
    """

    @abstractmethod
    def train(
        self,
        tree: DecisionTree,
        train_rows: Sequence[int],
        val_rows: list[int] | None = None,
    ) -> Any:
        """
        Build a decision tree model.

        Args:
            tree: DecisionTree instance with loaded data and config
            train_rows: Row indices for training
            val_rows: Row indices for validation/pruning, if any

        Returns:
            Tree model structure
        """
        ...

    @property
    def supports_categorical(self) -> bool:
        """Whether this trainer supports categorical (equality) splits."""
        return False

    @property
    def supports_pruning(self) -> bool:
        """Whether this trainer honours ``val_rows`` for reduced-error pruning.

        Backends that ignore the validation rows (e.g. sklearn) return False so
        callers can warn instead of silently holding out data and never pruning.
        """
        return False

    @property
    def name(self) -> str:
        """Human-readable name for this trainer."""
        return self.__class__.__name__
