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
| `cartlet/bundled/predict.py` | Python 3.9+ stdlib |

This enables deployment in constrained environments (locked-down servers,
air-gapped networks) where installing third-party packages is difficult or
impossible.

### 3. Standalone & Copyable

The runner file can be copied into any project and used immediately. No
installation, no `pip install`, no build system required.

```bash
# Just copy and use
cp bundled/predict.py ~/my_project/
cd ~/my_project
python predict.py model.cart red large
```

---

## Library API

```python
from predict import Predictor

# Load model
model = Predictor("model.cart")       # from file
model = Predictor(raw_bytes)          # from bytes
model = Predictor("model.cart.gz")    # gzip supported

# Predict
result = model.predict(["red", "large"])
results = model.predict_batch([["red", "large"], ["blue", "small"]])

# With distribution (classification only, if model has distributions)
dist = model.predict(["red", "large"], return_dist=True)
# {"apple": 0.8, "ball": 0.2}

# Model metadata
model.n_features      # number of features
model.n_classes       # number of classes (classification)
model.feature_names   # list of feature names
model.class_labels    # list of class labels
model.is_forest       # True if random forest
model.is_regression   # True if regression task
model.is_xgboost      # True if XGBoost model
model.metadata        # dict of embedded JSON metadata
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

When a feature value is `None`, missing, or out of bounds:

- **Numeric comparisons** (`<=`): comparison fails → go right (value > threshold).
- **Categorical comparisons** (`==`): comparison fails → go right (value ≠ target).
- **Switch/case tables**: use default branch.

This policy is consistent across the runner and the in-process predictor and
ensures deterministic behaviour even with incomplete input vectors.

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
