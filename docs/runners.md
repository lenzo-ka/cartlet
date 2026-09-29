# Runner Philosophy

Cartlet provides a single minimal, dependency-free Python runner for inference.
This document describes the design philosophy and architecture.

---

## Core Principles

### 1. Library First, CLI Optional

The runner is primarily a **library** that can be imported and used
programmatically. The CLI is a thin convenience wrapper.

```
┌─────────────────────────────────────┐
│            CLI (optional)           │
│  Parses args, calls library, prints │
├─────────────────────────────────────┤
│           Library (core)            │
│  Predictor class / load + predict   │
└─────────────────────────────────────┘
```

This means:
- All functionality is accessible via the library API.
- The CLI has no special powers; it just calls the library.
- Users can strip the CLI for embedded/import use cases.

### 2. Zero Dependencies

The runner is a **single file** with no external dependencies beyond Python's
standard library:

| Runner | Dependencies |
|--------|--------------|
| `cartlet/bundled/predict.py` | Python 3.11+ stdlib |

This enables deployment in constrained environments (locked-down servers,
air-gapped networks) where installing third-party packages is difficult or
impossible.

### 3. Standalone & Copyable

The runner file can be copied into any project and used immediately. No
installation, no `pip install`, no build system required.

```bash
# From the Cartlet repository root, copy the runner beside your model
cp cartlet/bundled/predict.py ~/my_project/
cd ~/my_project
python predict.py model.cart red large
```

---

## Library API

```python
from predict import Predictor

# Load model
model = Predictor("model.cart")  # from file
model = Predictor(raw_bytes)  # from bytes
model = Predictor("model.cart.gz")  # gzip supported

# Predict
result = model.predict(["red", "large"])
results = model.predict_batch([["red", "large"], ["blue", "small"]])
attribution = model.predict_path(["red", "large"])

# With distribution (classification only, if model has distributions)
dist = model.predict(["red", "large"], return_dist=True)
# {"apple": 0.8, "ball": 0.2}

# Model metadata
model.n_features  # number of features
model.n_classes  # number of classes (classification)
model.feature_names  # list of feature names
model.class_labels  # list of class labels
model.is_forest  # True if random forest
model.is_regression  # True if regression task
model.is_xgboost  # True if XGBoost model
model.metadata  # dict of embedded JSON metadata
```

---

## CLI Interface

### Basic Usage

```bash
# Single prediction from command-line arguments
./predict.py model.cart red large
```

### Batch Prediction

```bash
# From file (one vector per line)
./predict.py model.cart -f input.csv
./predict.py model.cart -f input.tsv

# From stdin
echo "red large" | ./predict.py model.cart -f -
```

### Common Options

```bash
./predict.py model.cart --help        # Show help
./predict.py model.cart --info        # Print model metadata
./predict.py -m model.cart -f in.csv  # Explicit model flag

# Return probability distributions (JSON output)
./predict.py model.cart red large --dist
# {"apple": 0.82, "ball": 0.18}
```

---

## Bundling

Cartlet can bundle a model with the runner to create standalone executables:

```bash
# Full bundle: library + CLI + embedded model
cartlet bundle model.cart predict.py
# Result: ./predict.py red large  (model is embedded)

# Library only: strip CLI, keep embedded model
cartlet bundle model.cart predict.py --library-only
# Result: importable module with embedded model

# No model: runner without embedded data
cartlet bundle --no-model predict.py
# Result: runner that requires model path at runtime
```

Inputs that are not already `.cart` (e.g. `.json`, `.jsonl`, `.pkl`, `.skl`, or
a custom suffix) are transparently converted to a temporary `.cart` file before
embedding, so callers do not need to pre-convert the model.

> **Security:** converting a `.pkl`/`.pickle` or `.skl`/`.joblib` input
> unpickles it, executing any code embedded in the file. Only bundle
> pickle/joblib models from a trusted source. `.cart`/`.json`/`.jsonl` inputs
> and the runners themselves never unpickle.

### How Bundling Works

The bundled file contains:
1. Runner source code (library + optional CLI).
2. Embedded model data as a base64 string constant.

At load time, the runner:
1. Checks for embedded data in its own source.
2. If found and no external model specified, uses embedded data.
3. If external model specified via `-m`, uses that instead.

---

## Missing Values

All prediction entry points accept `missing="error"` or `missing="right"`.
The default is `"error"`. If an evaluated decision tests a missing value,
prediction raises `MissingFeatureError` naming the feature, tree, and decision
node. Missing values in features that the evaluated paths do not test are
irrelevant. `None` and an index at or beyond the vector length are always
missing. At a numeric node, any value that successfully converts with `float()`
and produces NaN is missing, including the string `"nan"`. At equality and
switch nodes, a non-string scalar is missing when comparing it with itself using
`!=` produces a Boolean true result. This covers Python and NumPy scalar NaNs
without making NumPy a runner dependency. Training rejects nonfinite values, so
no model can learn a NaN category.

Self-inequality is deliberately conservative: an exception from `__ne__`, or a
result that is neither a built-in Boolean nor NumPy's scalar Boolean, does not
establish missingness. The value continues through ordinary comparison or bool
normalization, which may then reject it. Strings never use this test, so a
literal `"nan"` remains an ordinary category at equality and switch nodes.

For compatibility with 0.6.0, `missing="right"` makes numeric and categorical
comparisons take the right branch and switch nodes take their default branch.
CLI callers use `--missing {error,right}`. Empty delimited fields are parsed as
`None`; absent named fields are also missing. Non-numeric strings at numeric
nodes retain their established right-branch behavior.

`XGBoostTree.predict` uses the native Booster and its learned missing direction.
Its `.cart` exports use the explicit runner policy because the binary format
does not store those directions.

## Decision paths

`predict_path(model, vector, *, missing="error")` and
`Predictor.predict_path(vector, *, missing="error")` return the ordinary
prediction plus one path record per evaluated tree. Decision and leaf IDs are
the model-global `.cart` array indexes documented in
[the binary format](cart_format.md#stable-node-ids). Training-side
`DecisionTree` and `RandomForest` expose the same result. For XGBoost, use one
of the `.cart` runners for path attribution. A step's predicate `value` is the
writer-canonical value: numeric thresholds are floats, equality values are
strings, and bool-dtype equality values are normalized to `"0"` or `"1"`.

## Lazy feature access

Prediction indexes a vector only through `len(vector)` and `vector[i]`, where
`i` is an integer feature index tested along an evaluated path. It does not
iterate, slice, copy, or convert the whole vector, and reads each decision's
feature at most once. This applies to trees, forests, and XGBoost `.cart`
models in both runners, and to non-strict in-process tree and forest prediction.
Exceptions raised by `vector[i]` propagate unchanged. `strict=True` is the
documented exception: out-of-vocabulary validation must inspect every present
feature. It skips absent and missing values, but normalizes every present bool
and rejects unrecognized bool or out-of-vocabulary values before traversal.
Missing values are then handled during traversal by `missing`, exactly as in
non-strict prediction. Thus strict validation takes precedence for invalid
present values, while the missing policy takes precedence for missing values.
The executable contracts are covered by `tests/test_lazy_features.py` and
`tests/test_missing_scalar_contract.py`.

When a tested feature has bool dtype, runners first check it for missingness,
then normalize the value at that read
using the same accepted true/false spellings as in-process prediction. An
unrecognized bool value raises `ValueError`; an untested bool feature is never
read or normalized. Accepted true strings are `"1"`, `"true"`, `"True"`,
`"TRUE"`, `"yes"`, `"Yes"`, and `"YES"`; accepted false strings are `"0"`,
`"false"`, `"False"`, `"FALSE"`, `"no"`, `"No"`, and `"NO"`. Boolean values
and numeric `1` and `0` are also accepted.

Those spellings describe values as the prediction library receives them, after
any CLI field parsing. The package `cartlet predict` command leaves JSONL value
types intact. For delimited CSV, TSV, and SSV input, it maps an empty field to
`None`; for every other field, regardless of the model feature type, it tries
`float(field)` when the text contains a period and `int(field)` otherwise,
leaving the field as a string if conversion fails. Thus a delimited field such
as `01` reaches bool normalization as the integer `1` and is accepted even
though the library rejects the literal string `"01"`.

The bundled CLI uses the model split type instead: for file input and positional
arguments it converts fields for numerical features with `float()` and leaves
categorical fields, including bool-dtype categorical fields, as strings. In
file input only, an empty field becomes `None`; a positional empty string stays
an empty string. These CLI conversions do not change the library APIs or their
accepted bool values.

---

## Relationship to `cartlet` Package

| Component | Purpose | Dependencies |
|-----------|---------|--------------|
| `cartlet/bundled/predict.py` | Standalone runner | None (stdlib only) |
| `cartlet/runner.py` | Package module | Part of cartlet |
| `cartlet predict` | Full CLI | Full cartlet package |

For most users:
- **Training & export**: use the `cartlet` package or CLI.
- **Inference in production**: copy `cartlet/bundled/predict.py`, or bundle a
  standalone script with `cartlet bundle`.

`cartlet/runner.py` is a lighter-weight module for users who have `cartlet`
installed and want inference without importing the training model classes. It
exposes both the functional `load_model()` + `predict()` API and a `Predictor`
class equivalent to the bundled runner's.

## Input and export contracts

The package and standalone CART loaders recognize gzip by content, including a
compressed file without a `.gz` suffix. Invalid non-numeric values at numeric
nodes take the right branch in both nested-model and exported inference.
XGBoost inputs and thresholds use float32 precision to match DMatrix. Strict
XGBoost nodes use `<`; native CART nodes use `<=`. Multiclass XGBoost
metadata may carry one finite raw intercept per class; binary intercepts remain
in probability space and are converted to a logit for additive prediction.

Model and vector writers replace path outputs atomically after successful
serialization. NUL-containing pooled strings and nonfinite or out-of-range
float64 values are rejected instead of emitting invalid binary models.

Vector JSONL uses nonempty object records. `write_vectors` assigns one-indexed
column names when no header is supplied. All rows must match the header width;
JSONL keys must be unique. Batch and streaming readers reject invalid JSON or
record shapes with physical line diagnostics. Labeled records must contain a
non-null target; absent feature values remain `None`. Blank lines are ignored.
This vector schema is separate from model JSONL serialization.
