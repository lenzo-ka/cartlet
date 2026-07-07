"""
Type definitions and constants for cartlet.

This module contains:
- Feature type constants (DTYPE_*, TYPE_*)
- Task type constants (TASK_*)
- Tree node type aliases
- FeatureSpec dataclass
- ModelData TypedDict for loaded .cart models
- Bool normalization utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

# =============================================================================
# Tree node type aliases
# =============================================================================

# Classification leaf: class label (str) or distribution (dict)
ClassificationLeaf = str | dict[str, float]

# Regression leaf: [mean, variance, n_samples]
RegressionLeaf = list[float]

# Any leaf node
LeafNode = str | dict[str, float] | list[float]

# Decision node: [feature_name, operator, value, left_child, right_child]
# where operator is "<" for numerical or "=" for categorical splits
# Note: This is a forward reference since TreeNode references itself
DecisionNode = list[Any]  # [str, str, Any, TreeNode, TreeNode]

# Complete tree node (leaf or decision)
TreeNode = LeafNode | DecisionNode


# =============================================================================
# Model data schema (for loaded .cart files)
# =============================================================================


class FeatureInfo(TypedDict):
    """Feature metadata from .cart file."""

    name: str
    type: str  # "cat" or "num"
    values: list[str]  # known categorical values


class ModelMeta(TypedDict, total=False):
    """Model metadata."""

    features: list[FeatureInfo]
    task: str  # "classification" or "regression"
    target: FeatureInfo | None
    training: dict[str, Any] | None
    min_samples_split: int | None
    min_samples_leaf: int | None
    store_distributions: bool | None
    format: dict[str, str] | None
    # Embedded JSON blob from the .cart trailer. Anything the exporter chose to
    # persist via `metadata=` (locale, training_config, XGBoost base_score,
    # consumer-defined keys, ...). Empty dict when nothing was embedded.
    metadata: dict[str, Any]


class CaseTable(TypedDict):
    """Case table for OP_SWITCH nodes."""

    default: int  # default child index
    cases: list[tuple[int, int]]  # [(cat_val_idx, child_idx), ...] (for rebuild)
    lookup: dict[str, int]  # {category_string: child_idx} (O(1) prediction path)


class ModelData(TypedDict):
    """
    Schema for loaded .cart model data.

    This is returned by runner.load_model() and used by predict().
    """

    # Metadata
    meta: ModelMeta

    # String pools
    strings: list[str]
    class_labels: list[str]

    # Value pools
    floats: list[float]
    cat_vals: list[int]  # indices into strings

    # Tree structure
    decisions: list[tuple[int, int, int, int, int]]  # (feat, op, val, left, right)
    leaves: list[tuple[int, int]]  # (type, val)
    tree_offsets: list[int]

    # Optional structure
    distributions: list[list[tuple[int, float]]]  # [(class_idx, prob), ...]
    case_tables: list[CaseTable]

    # Flags
    is_regression: bool
    is_forest: bool
    is_xgboost: bool
    has_distributions: bool
    n_trees: int


# =============================================================================
# Feature type constants
# =============================================================================

DTYPE_BOOL = "bool"
DTYPE_FLOAT = "float"
DTYPE_INT = "int"
DTYPE_STR = "str"
VALID_DTYPES = {DTYPE_BOOL, DTYPE_FLOAT, DTYPE_INT, DTYPE_STR}

TYPE_CAT = "cat"  # Categorical: equality splits
TYPE_NUM = "num"  # Numerical: threshold splits
VALID_TYPES = {TYPE_CAT, TYPE_NUM}

CRITERION_ENTROPY = "entropy"
CRITERION_GINI = "gini"


# =============================================================================
# Task type constants
# =============================================================================

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASK_AUTO = "auto"
VALID_TASKS = {TASK_CLASSIFICATION, TASK_REGRESSION, TASK_AUTO}

_MAX_RANDOM_SEED = 2**31


# =============================================================================
# Probability thresholds (for leaf distributions)
# =============================================================================

# If best class probability exceeds this, store only the class (not full distribution)
PROB_HIGH_CONFIDENCE = 0.95

# Minimum probability to include in distribution (filter out noise)
# Note: Keep very small to preserve rare alignments in G2P models
PROB_MIN_THRESHOLD = 1e-8

# Decision boundary for binary classifiers when working from a calibrated
# probability of class 1 (sklearn / xgboost binary heads).
BINARY_CLASSIFICATION_THRESHOLD = 0.5


# =============================================================================
# Task inference thresholds
# =============================================================================

# If target has more than this many unique values, likely regression
TASK_INFER_MAX_CLASSES = 20

# If unique values exceed this fraction of total samples, likely regression
TASK_INFER_UNIQUE_RATIO = 0.5


# =============================================================================
# Default hyperparameters
# =============================================================================

DEFAULT_N_ESTIMATORS = 100
DEFAULT_VALIDATION_SPLIT = 0.05
DEFAULT_TEST_SPLIT = 0.05
DEFAULT_MIN_DIST_ENTROPY = 0.1
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1


# =============================================================================
# Bool normalization
# =============================================================================

_BOOL_TRUE = {True, "1", "true", "True", "TRUE", "yes", "Yes", "YES"}
_BOOL_FALSE = {False, "0", "false", "False", "FALSE", "no", "No", "NO"}


def normalize_bool(value: Any) -> int:
    """Normalize boolean-like values to 0 or 1."""
    if value in _BOOL_TRUE:
        return 1
    if value in _BOOL_FALSE:
        return 0
    raise ValueError(f"Cannot convert {value!r} to bool")


# =============================================================================
# FeatureSpec dataclass
# =============================================================================


@dataclass
class FeatureSpec:
    """
    Specification for a feature (input or output).

    Defaults:
        dtype="bool" -> type="cat" (booleans are categorical by default)
        dtype="str" -> type="cat" (strings are categorical by default)
        dtype="int" -> type="num" (integers are numerical by default)
        dtype="float" -> type="num" (floats are numerical by default)

    Override type explicitly when needed (e.g., int grades as categorical).
    """

    name: str
    dtype: str = DTYPE_STR  # bool, str, int, float
    type: str | None = None  # cat or num; None = infer from dtype
    values: set[Any] | None = None  # known values for categorical features

    def __post_init__(self):
        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype: {self.dtype}. Must be one of {VALID_DTYPES}"
            )

        # Infer type from dtype if not specified
        if self.type is None:
            if self.dtype in (DTYPE_BOOL, DTYPE_STR):
                self.type = TYPE_CAT  # bool/str default to categorical
            else:
                self.type = TYPE_NUM  # int/float default to numerical

        if self.type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {self.type}. Must be one of {VALID_TYPES}")


# =============================================================================
# Task and feature inference utilities
# =============================================================================


def resolve_task(
    task: str,
    detected_task: str | None = None,
    target_type: str | None = None,
) -> str:
    """
    Resolve the effective task type.

    Args:
        task: User-specified task (TASK_AUTO, TASK_CLASSIFICATION, or TASK_REGRESSION)
        detected_task: Task detected from data (if auto-detection was performed)
        target_type: Target feature type (TYPE_CAT or TYPE_NUM) for inference

    Returns:
        Resolved task (TASK_CLASSIFICATION or TASK_REGRESSION)
    """
    if task != TASK_AUTO:
        return task
    if detected_task:
        return detected_task
    if target_type == TYPE_NUM:
        return TASK_REGRESSION
    return TASK_CLASSIFICATION


def is_likely_regression(y: list[Any]) -> bool:
    """
    Determine if target values suggest regression rather than classification.

    Returns True if:
    - All values are numeric (int/float, not bool)
    - AND either:
      - More than TASK_INFER_MAX_CLASSES unique values, OR
      - Unique values exceed TASK_INFER_UNIQUE_RATIO of total samples
    """
    # Single pass: check type and collect unique values simultaneously
    unique: set[Any] = set()
    for v in y:
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            return False
        unique.add(v)
    n_unique = len(unique)
    return (
        n_unique > TASK_INFER_MAX_CLASSES or n_unique > len(y) * TASK_INFER_UNIQUE_RATIO
    )


def normalize_feature_spec(
    spec: dict[str, Any] | str | FeatureSpec,
    name: str | None = None,
) -> FeatureSpec:
    """
    Normalize various feature spec formats to a FeatureSpec object.

    Args:
        spec: Can be:
            - FeatureSpec object (returned as-is)
            - dict with keys: name, dtype, type, values
            - str: shorthand type ("num", "numeric", "numerical", "cat", "categorical")
        name: Feature name (required if spec is str or dict without 'name')

    Returns:
        FeatureSpec object
    """
    if isinstance(spec, FeatureSpec):
        return spec

    if isinstance(spec, str):
        spec_type = spec.lower()
        if spec_type in ("num", "numerical", "numeric"):
            return FeatureSpec(name=name or "", dtype=DTYPE_FLOAT, type=TYPE_NUM)
        return FeatureSpec(name=name or "", dtype=DTYPE_STR, type=TYPE_CAT)

    if isinstance(spec, dict):
        spec_name = spec.get("name", name) or ""
        spec_dtype = spec.get("dtype", DTYPE_STR)
        split_type = str(spec["type"]) if "type" in spec else None
        spec_values = spec.get("values")
        fs = FeatureSpec(name=spec_name, dtype=spec_dtype, type=split_type)
        if spec_values:
            fs.values = (
                set(spec_values) if isinstance(spec_values, list) else spec_values
            )
        return fs

    raise ValueError(f"Invalid feature spec: {spec}")


def infer_feature_specs(
    X: list[list[Any]],
    feature_names: list[str],
    *,
    exclude_bool_from_numeric: bool = False,
    include_values: bool = False,
    force_float_numeric: bool = False,
) -> list[dict[str, Any]]:
    """
    Infer feature specifications from data.

    Args:
        X: Feature data (list of rows).
        feature_names: Names for each column.
        exclude_bool_from_numeric: If True, columns containing any `bool` are
            treated as categorical (XGBoostTree semantics). Defaults to False,
            so Python `bool` counts as numeric for feature inference.
        include_values: If True, populate the `values` key on categorical
            specs with the set of stringified column values (used when the
            caller plans to validate OOV inputs).
        force_float_numeric: If True, numeric columns get `dtype="float"`
            regardless of whether the values were ints. Useful when the
            downstream trainer always casts numeric features to float.

    Returns:
        List of feature spec dicts with `name`, `dtype`, `type`, and
        optionally `values` (when `include_values=True`).
    """
    if not X:
        return []

    specs: list[dict[str, Any]] = []
    for col, name in enumerate(feature_names):
        values = [row[col] for row in X if col < len(row)]
        if exclude_bool_from_numeric:
            all_numeric = all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
            )
        else:
            all_numeric = all(isinstance(v, (int, float)) for v in values)

        if not all_numeric:
            spec: dict[str, Any] = {"name": name, "dtype": DTYPE_STR, "type": TYPE_CAT}
            if include_values:
                spec["values"] = {str(v) for v in values}
            specs.append(spec)
            continue

        if force_float_numeric or any(isinstance(v, float) for v in values):
            dtype = DTYPE_FLOAT
        else:
            dtype = DTYPE_INT
        specs.append({"name": name, "dtype": dtype, "type": TYPE_NUM})

    return specs
