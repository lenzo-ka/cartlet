"""
Base trainer class for decision trees.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..types import PROB_HIGH_CONFIDENCE, PROB_MIN_THRESHOLD

if TYPE_CHECKING:
    from ..tree import DecisionTree


def make_classification_distribution(
    class_probs: list[tuple[Any, float]],
    store_distributions: bool = True,
    min_confidence: float = PROB_HIGH_CONFIDENCE,
    min_dist_entropy: float = 0.0,
) -> Any:
    """
    Build a classification leaf value from class probabilities.

    Args:
        class_probs: List of (class_label, probability) tuples, sorted by prob desc
        store_distributions: Whether to store full distributions
        min_confidence: If best-class probability exceeds this, store only the
            class label instead of the full distribution.  Set to 1.0 to always
            keep distributions.
        min_dist_entropy: If the distribution's entropy (bits) is below this,
            collapse to the best class. Applied consistently across backends so
            native and sklearn leaves agree (0.0 disables the gate).

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
        train_rows: list[int],
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
