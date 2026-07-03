"""
Base trainer class for decision trees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..types import PROB_HIGH_CONFIDENCE, PROB_MIN_THRESHOLD

if TYPE_CHECKING:
    from ..tree import DecisionTree


def make_classification_distribution(
    class_probs: list[tuple[Any, float]],
    store_distributions: bool = True,
    min_confidence: float = PROB_HIGH_CONFIDENCE,
) -> Any:
    """
    Build a classification leaf value from class probabilities.

    Args:
        class_probs: List of (class_label, probability) tuples, sorted by prob desc
        store_distributions: Whether to store full distributions
        min_confidence: If best-class probability exceeds this, store only the
            class label instead of the full distribution.  Set to 1.0 to always
            keep distributions.

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
    def name(self) -> str:
        """Human-readable name for this trainer."""
        return self.__class__.__name__
