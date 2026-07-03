"""
Decision tree trainer backends.

Provides different backends for building decision trees:
- Native: Pure Python, supports categorical + numerical features
- Sklearn: Uses scikit-learn (optional), auto-encodes categorical features
"""

from .base import Trainer
from .native import Native

__all__ = [
    "Native",
    "Sklearn",
    "Trainer",
    # Utility functions for sklearn integration
    "convert_sklearn_tree",
    "encode_categorical",
    "map_feature_importances",
]

# Names resolved lazily from the sklearn backend module (see __getattr__).
_LAZY_SKLEARN_NAMES = frozenset(
    {"Sklearn", "encode_categorical", "convert_sklearn_tree", "map_feature_importances"}
)


def __getattr__(name: str):
    """Lazily resolve sklearn-backed names (PEP 562).

    Deferring the import keeps scikit-learn optional: importing
    ``cartlet.trainer`` never imports the sklearn backend until one of these
    names is accessed. Crucially this returns the *real* class/function object
    (not a wrapper), so ``Sklearn.from_sklearn(...)`` and
    ``isinstance(x, Sklearn)`` work as documented.
    """
    if name in _LAZY_SKLEARN_NAMES:
        from . import sklearn as _sklearn_mod

        return getattr(_sklearn_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SKLEARN_NAMES)
