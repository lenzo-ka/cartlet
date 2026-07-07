# Cartlet

A little CART -- decision trees for classification and regression.

This module comes from a couple of motivations -- to more easily do some
of the things from Festival/Festvox voice building using tools like wagon
(in Edinburgh Speech Tools), and to re-vivify a method of decision-tree
grapheme-to-phoneme implementation, without a lot of requirements.

Train decision trees, random forests, or XGBoost trees, and deploy them
on a tiny dependency-free Python runtime.

## Features

- **Classification & Regression**: Full CART support
- **Random Forests**: Ensemble learning with configurable trees
- **XGBoost**: Gradient boosted trees with native categorical support
- **Categorical features**: Equality splits (`feature == value`)
- **Numerical features**: Threshold splits (`feature <= value`)
- **Information gain**: Entropy or Gini for classification, variance for regression
- **Instance weighting**: Supports weighted training examples
- **Pruning**: Reduced Error Pruning with validation data
- **Probability distributions**: Store distributions at leaves
- **N-best predictions**: Multiple predictions with confidence scores
- **CLI**: Full command-line interface for training/prediction
- **Config presets**: Built-in presets for common training configurations
- **Compact binary format**: Efficient `.cart` format for deployment
- **Minimal runner**: Zero-dependency Python runner for inference
- **Format conversion**: Convert between `.cart`, JSON, JSONL, Pickle, and sklearn formats

## Installation

Requires Python 3.11+.

```bash
pip install cartlet
```

Optional sklearn backend for faster training:
```bash
pip install cartlet[sklearn]
```

Optional XGBoost support:
```bash
pip install cartlet[xgboost]
```

## Quick Start

### Python API

```python
from cartlet import DecisionTree, RandomForest

# Classification
dt = DecisionTree(feature_names=["color", "size"])
dt.load_data([["red", "small"], ["blue", "large"]], ["apple", "ball"])
dt.train()
print(dt.predict(["red", "small"]))  # "apple"

# Regression
dt = DecisionTree(task="regression", feature_names=["sqft"])
dt.load_data([[1000], [2000], [3000]], [100000, 200000, 300000])
dt.train()
print(dt.predict([1500]))  # ~150000

# Random Forest
rf = RandomForest(n_estimators=100, feature_names=["x", "y"])
rf.load_data(X, y)
rf.train()
print(rf.predict([1, 2]))

# XGBoost (requires xgboost>=1.5.0)
from cartlet import XGBoostTree

xgb = XGBoostTree(n_estimators=100, feature_names=["color", "size"])
xgb.load_data([["red", "small"], ["blue", "large"]], ["apple", "ball"])
xgb.train()
print(xgb.predict(["red", "small"]))  # "apple"
print(xgb.predict_proba(["red", "small"]))  # {"apple": 0.8, "ball": 0.2}

# Export to various formats
xgb.export("model.cart")  # Compact binary for the runner
xgb.export("model.xgb")   # Native XGBoost format
```

### CLI

```bash
# Train
cartlet train data.csv -o model.cart

# Train with options
cartlet train data.csv -o model.cart \
    -F              # RandomForest \
    -n 100          # 100 trees \
    -D 5            # max depth 5 \
    -P              # pruning \
    -S 0.2          # 20% test split

# Train with config preset
cartlet train data.csv -o model.cart -c fast      # Quick training
cartlet train data.csv -o model.cart -c accurate  # Best accuracy

# Predict
cartlet predict model.cart input.csv -m append -f json

# Evaluate
cartlet eval model.cart test.csv

# Model stats
cartlet stats model.cart -J  # JSON output

# Convert between formats
cartlet convert model.cart model.json
cartlet convert model.json model.pkl
```

**Input/Output formats:** CSV, TSV, SSV (space-separated), JSON, JSONL

## Model Formats

Cartlet supports multiple model formats for different use cases:

| Format | Extension | Use Case |
|--------|-----------|----------|
| `.cart` | Binary | Compact, cross-language, deployment |
| `.cart.gz` | Compressed binary | Even smaller |
| `.json` | JSON | Human-readable, full fidelity |
| `.jsonl` | JSON Lines | Streaming, full fidelity |
| `.pkl` | Pickle | Python-only, full fidelity |
| `.skl` / `.joblib` | Sklearn | sklearn interoperability |

The `.cart` binary format is optimized for size:
- Varint encoding for node indices (1-5 bytes vs fixed 4)
- Packed feature+op byte (supports up to 64 features inline)
- 3-byte leaf nodes (no padding)

```python
# Export to different formats
dt.export("model.cart")      # Compact binary (default)
dt.export("model.cart.gz")   # Compressed binary
dt.export("model.json")      # JSON (full tree structure)
dt.export("model.jsonl")     # JSON Lines
dt.export("model.pkl")       # Pickle
dt.export("model.skl")       # sklearn-compatible (if trained with sklearn)

# Load from any format
dt.load_model("model.cart")
dt.load_model("model.json")

# Custom file suffixes (skip extension detection)
dt.export("model.g2p.gz", format="jsonl")           # write JSONL under .g2p.gz
dt.load_model("model.g2p.gz", format="jsonl")       # read it back
convert("model.g2p.gz", "model.cart", input_format="jsonl")
bundle("model.g2p.gz", "predictor.py", model_format="jsonl")
```

### Distributions in .cart

By default, `.cart` files store class distributions for `predict_nbest` support:

```python
# Default: store distributions (supports nbest)
dt.export("model.cart")

# Without distributions (smaller file, no nbest)
dt.export("model.cart", store_distributions=False)
```

## API Reference

### DecisionTree

```python
dt = DecisionTree(
    features=[{"name": "age", "dtype": "int", "type": "num"}],  # Feature specs
    feature_names=["age", "color"],  # Or just names (all categorical)
    task="auto",                # "classification", "regression", or "auto"
    max_depth=None,             # Max tree depth (None = unlimited)
    min_samples_split=2,        # Min samples to split
    min_samples_leaf=1,         # Min samples in leaf
    criterion="entropy",        # "entropy" or "gini"
    store_distributions=True,   # Keep full probability distributions at leaves
    min_dist_entropy=DEFAULT_MIN_DIST_ENTROPY,  # Below this, collapse to best class
    min_confidence=PROB_HIGH_CONFIDENCE,        # Above this best-prob, collapse too
)

dt.load_data(X, y, counts=None)  # Load training data (optional weights)
dt.train(trainer="native", prune=False, validation_split=0.0)

dt.predict(vector)                    # Single prediction
dt.predict_batch(vectors)             # Batch prediction
dt.predict_with_confidence(vector)    # (prediction, confidence)
dt.predict_nbest(vector, n=5)         # Top n predictions

dt.export("model.cart")               # Save (default: .cart)
dt.load_model("model.cart")           # Load
```

Distribution storage knobs (`store_distributions`, `min_dist_entropy`,
`min_confidence`) only apply to classification trees. They trade `.cart` file
size and `predict_nbest` fidelity for predictability:

| Setting | Effect |
|---------|--------|
| `store_distributions=False` | Leaves store only the best class; `predict_nbest` will return 1 result. |
| `store_distributions=True`, low `min_confidence` | Almost every leaf keeps its full distribution (largest models). |
| `store_distributions=True`, `min_confidence=1.0` | Always keep distributions, never collapse. |
| `min_dist_entropy=0.0` | Never use entropy as a collapse trigger. |

### RandomForest

```python
rf = RandomForest(
    n_estimators=100,      # Number of trees
    max_features="sqrt",   # Features per split: "sqrt", "log2", int, or None
    bootstrap=True,        # Sample with replacement
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

rf.load_data(X, y)
rf.train(random_state=42)

rf.predict(vector)
rf.predict_batch(vectors)
rf.predict_proba(vector)        # Class probabilities
rf.feature_importances_         # Feature importance dict

rf.export("forest.cart")
rf.load_model("forest.cart")
```

### Zero-Dependency Inference

For deployment without training dependencies. All inference helpers are also
re-exported at the package root so callers do not need to know the internal
module layout:

```python
from cartlet import load_model, predict, predict_batch

model = load_model("model.cart")
result = predict(model, [1, 2, 3])
results = predict_batch(model, [[1, 2, 3], [4, 5, 6]])
```

For an object-oriented entry point:

```python
from cartlet import Predictor

p = Predictor("model.cart")
p.predict([1, 2, 3])
p.predict_batch([[1, 2, 3], [4, 5, 6]])
p.feature_names      # list of feature names
p.class_labels       # list of class labels (classification)
p.task               # "classification" or "regression"
```

### Vocabulary inspection and OOV handling

For categorical features, the `.cart` file stores the set of values seen
during training. Callers building wrapper inference (e.g. one prediction per
position in a sliding-window text model) can query and react to
out-of-vocabulary values:

```python
from cartlet import get_vocabulary, is_oov, load_model

model = load_model("model.cart")

vocab = get_vocabulary(model, "color")  # set, or None if not categorical
get_vocabulary(model, 0)                # by feature index also works

if is_oov(model, "color", "chartreuse"):
    # decide how to handle: skip, substitute, fall back, etc.
    ...
```

The same helpers are also exposed as instance methods on `Predictor`, so
callers using the OO API don't have to drop back to the functional form:

```python
from cartlet import Predictor

p = Predictor("model.cart")
p.get_vocabulary("color")              # same as get_vocabulary(p.model, ...)
p.is_oov("color", "chartreuse")        # same as is_oov(p.model, ...)
```

Numerical features return `None` from `get_vocabulary` and `False` from
`is_oov` (all real numbers are in-vocabulary).

### Embedded model metadata

`.export(..., metadata={"locale": "en", ...})` writes a JSON trailer that
round-trips through `.cart` and is surfaced on `Predictor.metadata`:

```python
from cartlet import Predictor, read_cart_metadata

p = Predictor("model.cart")
p.metadata  # {"locale": "en", ...}; {} when nothing was embedded

# Without loading the full model:
read_cart_metadata("model.cart")        # works on a path
read_cart_metadata(open("model.cart", "rb").read())  # or bytes
```

### Minimal Runner

For embedded or constrained environments, a minimal Python runner in
`cartlet/bundled/predict.py` provides inference with no dependencies beyond
the standard library.

The runner produces **identical predictions** to sklearn-trained models
(verified by automated tests).

**Missing values**: when a feature is `None` or missing, comparisons fail and
the tree takes the "no" branch (right child).

```python
from predict import Predictor

p = Predictor("model.cart")
print(p.predict(["red", "small"]))

# Or from command line:
# python predict.py model.cart red small
```

### Runner throughput

`benchmarks/runner_throughput.py` measures steady-state throughput
(predict 10k feature vectors over stdin) for the bundled Python runner.
Numbers below are an indicative single-host snapshot (Apple Silicon, macOS);
run locally for your own platform via `make bench`.

| dataset       | features | depth | pred/s |
|---------------|---------:|------:|-------:|
| iris          | 4        | 8     | ~195k  |
| wine          | 13       | 8     | ~165k  |
| breast_cancer | 30       | 8     | ~125k  |
| diabetes      | 10       | 8     | ~130k  |
| breast_cancer | 30       | 4     | ~125k  |
| breast_cancer | 30       | 16    | ~120k  |

A pytest smoke test (`tests/test_runner_perf_smoke.py`) asserts the runner
clears a generous floor (200 pred/s on iris) so order-of-magnitude
regressions break CI.

### Bundled Executables

Create standalone executables with embedded models:

```bash
# CLI
cartlet bundle model.cart predict.py            # Python executable
cartlet bundle model.json predict.py            # auto-converted to .cart

# Python API
from cartlet.io.bytes import bundle
bundle("model.cart", "predict.py")
bundle("model.json", "predict.py")              # auto-converted
```

The bundled executables contain the model data and can run without external files:
```bash
./predict.py red small  # Uses embedded model
./predict.py -m other.cart red small  # Override with external model
```

#### Library-only and No-model Options

```bash
# Library-only: strip CLI code for import use
cartlet bundle model.cart lib.py --library-only
# Then: from lib import Predictor; p = Predictor(); p.predict([...])

# No-model: output runner without embedded model (load at runtime)
cartlet bundle --no-model --library-only cart.py
# Then: from cart import Predictor; p = Predictor("model.cart")
```

### Evaluation

```python
from cartlet import (
    confusion_matrix,
    cross_validate,
    evaluate_predictions,
    evaluate_tree,
    per_class_metrics,
)

# Cross-validation (task-aware: returns accuracy for classification, mse for regression)
results = cross_validate(DecisionTree, X, y, n_folds=5)
print(f"{results['metric']}: {results['mean']:.4f} +/- {results['std']:.4f}")

# Metrics on pre-computed predictions
metrics = evaluate_predictions(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Task-aware metrics on a trained model
metrics = evaluate_tree(model, X_test, y_test)
# Always includes "task"; dispatch on it instead of probing keys:
#   classification -> {"task": "classification", "accuracy", "correct", "total"}
#   regression     -> {"task": "regression", "mse", "mae", "rmse", "total"}

# Per-class
for cls, m in per_class_metrics(y_true, y_pred).items():
    print(f"{cls}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

# Confusion matrix as a dict-of-dicts
cm = confusion_matrix(y_true, y_pred)
```

### IsolationForest

Anomaly detection. Trains an unsupervised forest of random binary trees;
samples that isolate quickly (short average path length) get higher scores.

```python
from cartlet import IsolationForest

ifo = IsolationForest(n_estimators=100, max_samples=256)
ifo.load_data(X)
ifo.train(random_state=42)
score = ifo.predict([1.0, 2.0])  # higher = more anomalous (range ~0 to 1)
```

Persisted via `.json`/`.jsonl`/`.pkl`. Not interchangeable with `convert()`
or the `.cart` runner (use `ifo.export(path)` / `IsolationForest.load_model(path)`).

### XGBoostTree

Thin wrapper around an XGBoost model that conforms to the cartlet API
and can export to the `.cart` binary format for the zero-dependency runner.

```python
from cartlet import XGBoostTree

xgb = XGBoostTree(feature_names=[...], task="classification")
xgb.load_data(X, y).train(n_estimators=10, max_depth=4)
xgb.export("model.cart")            # cross-language inference
xgb.export("model.xgb")             # native XGBoost format
```

Requires `xgboost`. See the [XGBoost section](#xgboosttree-1) for the full
constructor signature.

### Format conversion

```python
from cartlet import convert

convert("model.json", "model.cart")     # JSON -> binary
convert("model.cart", "model.pkl")      # binary -> pickle
convert("model.json", "model.cart.gz")  # JSON -> gzipped binary
```

See the [convert CLI](#convert) for the full list of supported extensions
and limitations.

### Tree utilities

```python
from cartlet import count_leaves, count_nodes, max_depth, tree_stats

count_nodes(tree.model)   # total internal + leaf nodes
count_leaves(tree.model)  # leaf count
max_depth(tree.model)     # depth of the longest root-to-leaf path
tree_stats(tree.model)    # {"nodes", "leaves", "depth"} in one call
```

### Constants

The package exposes typed string constants so callers can avoid stringly-typed
arguments, plus a few numeric defaults referenced by public API parameters:

| Group | Values |
|-------|--------|
| Tasks | `TASK_AUTO`, `TASK_CLASSIFICATION`, `TASK_REGRESSION` |
| Split criteria | `CRITERION_ENTROPY`, `CRITERION_GINI` |
| Feature types | `TYPE_CAT`, `TYPE_NUM` |
| Feature dtypes | `DTYPE_BOOL`, `DTYPE_INT`, `DTYPE_FLOAT`, `DTYPE_STR` |
| Distribution thresholds | `PROB_HIGH_CONFIDENCE`, `DEFAULT_MIN_DIST_ENTROPY` |

Type aliases for nested tree structures (`TreeNode`, `DecisionNode`,
`LeafNode`, `ClassificationLeaf`, `RegressionLeaf`) and the `FeatureSpec`
dataclass are also exported.

## Embedding cartlet in another library

Libraries that build domain-specific models on top of cartlet (g2p, anomaly
detection, etc.) generally follow this pattern:

```python
from cartlet import (
    CRITERION_ENTROPY,
    DecisionTree,
    PROB_HIGH_CONFIDENCE,
    bundle,
    get_vocabulary,
    is_oov,
    load_model,
    predict,
)

# 1. Train: build a DecisionTree (or RandomForest) with your own feature
#    vectorizer, using the typed constants for tunable knobs.
dt = DecisionTree(
    feature_names=vectorizer.feature_names,
    criterion=CRITERION_ENTROPY,
    store_distributions=True,
    min_confidence=PROB_HIGH_CONFIDENCE,
)
dt.load_data(X, y, counts)
dt.train()
dt.export("model.cart")

# 2. Inspect: query the trained vocabulary for OOV handling at inference time.
model = load_model("model.cart")
known = get_vocabulary(model, "center_letter")
if is_oov(model, "center_letter", letter):
    ...

# 3. Predict: use the functional API or the Predictor class.
phones = [predict(model, vec) for vec in vectors]

# 4. Bundle: emit a standalone Python executable with the model embedded,
#    so end users do not need cartlet installed.
bundle("model.cart", "predict.py")
```

All of the above are re-exported at the package root - subpath imports like
`from cartlet.runner import ...` continue to work but are not required.

## CLI Reference

### train

```
cartlet train DATA [-o MODEL] [-c CONFIG] [-t TARGET] [-d DELIM] [-H] [-N NAMES]
                    [-X FEATURES] [-T TASK] [-F] [--extra-trees] [--isolation-forest]
                    [-n N] [-D N] [-s N] [-l N] [-C {entropy,gini}] [-S FRAC]
                    [-e FILE] [-V FRAC] [-E FILE] [-P] [-R SEED]
                    [-B {native,sklearn}] [-j N] [--no-distributions] [-v]

  DATA              Training data (CSV/TSV/JSONL)
  -o, --output      Output model file (.cart, .json, .jsonl, .pkl, .skl/.joblib)
  -c, --config      Config preset or file (see below)
  --save-config     Save current args to config file (.yaml or .json)
  -t, --target      Target column (default: last)
  -d, --delimiter   Input column delimiter (auto-detect)
  -H, --no-header   Data has no header row
  -N, --column-names  Column names when no header (comma-separated)
  -X, --features    Feature specs as JSON file or inline
  -T, --task        Task type: auto, classification, regression
  -F, --forest      Train RandomForest
  --extra-trees     Train ExtraTrees forest (random splits, implies --forest)
  --isolation-forest  Train IsolationForest for anomaly detection (unsupervised)
  -n, --n-estimators  Trees in forest (default: 100)
  -D, --max-depth   Maximum tree depth
  -s, --min-samples-split  Min samples to split (default: 2)
  -l, --min-samples-leaf   Min samples in leaf (default: 1)
  -C, --criterion   Split criterion: entropy or gini (default: entropy)
  -S, --test-split  Fraction for test eval
  -e, --test-file   Separate test file
  -V, --validation-split  Fraction for pruning validation
  -E, --validation-file   Separate validation data file
  -P, --prune       Enable pruning (auto 5% validation if -V not set)
  -R, --random-seed Random seed
  -B, --trainer     Backend: native or sklearn
  -j, --n-jobs      Parallel jobs for forest training
  --no-distributions  Omit distributions in .cart (smaller, no nbest)
  -v, --verbose     Verbose output
```

**Config presets** (`-c/--config`):

| Preset | Description |
|--------|-------------|
| `defaults` | All default values (template) |
| `fast` | Quick training: max_depth=10, min_samples_split=10 |
| `accurate` | Best accuracy: forest with 100 trees |
| `small` | Smaller model: max_depth=8, min_samples_split=20 |
| `forest` | Default forest: 50 trees |
| `forest-large` | Large forest: 200 trees with sklearn backend |
| `sklearn` | Use sklearn backend |
| `g2p` | Tuned for grapheme-to-phoneme tasks |
| `extra-trees` | Extra-Trees: random splits, no bootstrap |

Or provide a path to a YAML/JSON config file. CLI args override preset values.

### predict

```
cartlet predict MODEL DATA [-o FILE] [-t TARGET] [-d DELIM] [--output-delimiter DELIM]
                           [-H] [-m MODE] [-p NAME] [-f FORMAT]

  MODEL             Model file (.cart only; use `cartlet convert` for other formats)
  DATA              Input data (CSV/TSV/JSONL)
  -o, --output      Output file (default: stdout)
  -t, --target      Target column (default: last)
  -d, --delimiter   Input column delimiter (auto-detect)
  --output-delimiter  Output column delimiter
  -H, --no-header   Data has no header row
  -m, --mode        Output mode: values, append, inplace
  -p, --prediction-column  Column name (default: "prediction")
  -f, --output-format  Output format: csv, tsv, ssv, json, jsonl
```

### evaluate

```
cartlet eval MODEL DATA [-o FILE] [-J] [-t TARGET] [-d DELIM] [-H] [-N NAMES] [-v]

  MODEL             Model file (.cart only; use `cartlet convert` for other formats)
  DATA              Test data with labels
  -o, --output      Output to file (default: stdout)
  -J, --json        Output as JSON (machine-readable)
  -t, --target      Target column
  -d, --delimiter   Column delimiter
  -H, --no-header   Data has no header row
  -N, --column-names  Column names when no header
  -v, --verbose     Show per-class metrics
```

### stats

```
cartlet stats MODEL [-J] [-v]

  MODEL             Model file (.cart only)
  -J, --json        JSON output (machine-readable)
  -v, --verbose     Detailed statistics
```

### convert

```
cartlet convert INPUT OUTPUT [--input-format FMT] [--output-format FMT]

  INPUT             Input model file
  OUTPUT            Output model file (format inferred from extension)
  --input-format    Override input format detection (cart|json|jsonl|pkl|skl)
  --output-format   Override output format selection
```

Supported conversions:
- `.cart` to/from `.json`, `.jsonl`, `.pkl`/`.pickle`
- `.json` to/from `.jsonl`, `.pkl`, `.cart`
- `.skl`/`.joblib` (requires `joblib`; export requires sklearn-trained model)
- Append `.gz` for compression (Python only)
- Custom suffixes: `cartlet convert model.g2p.gz model.cart --input-format jsonl`
- IsolationForest models use their own `.export()` / `.load_model()` paths,
  not `convert`

### bundle

```
cartlet bundle [MODEL] OUTPUT [--library-only] [--no-model] [--model-format FMT]

  MODEL             Model file (any format; auto-converted to .cart)
  OUTPUT            Output file path
  --library-only    Omit CLI code, produce library-only output for import use
  --no-model        Output runner code only without embedded model
  --model-format    Override input model format when the extension is custom
```

Examples:
```bash
cartlet bundle model.cart predict.py              # Python executable
cartlet bundle model.json predict.py              # JSON in -> auto-convert -> bundle
cartlet bundle model.g2p.gz predict.py --model-format jsonl
cartlet bundle model.cart lib.py --library-only   # Library with model
cartlet bundle --no-model --library-only cart.py  # Library, no model
```

### inspect

Infer feature types from data and output a spec file:

```
cartlet inspect DATA [-o FILE] [-t TARGET] [-H] [-N NAMES] [-f FORMAT]

  DATA              Data file to inspect
  -o, --output      Save spec to file (default: stdout)
  -t, --target      Target column (default: last)
  -H, --no-header   Data has no header row
  -N, --column-names  Column names when no header
  -f, --format      Output format: simple, full, array
```

**Workflow:**
```bash
# 1. Inspect data and save specs
cartlet inspect data.csv -o specs.json

# 2. Edit specs.json if needed (e.g., change "num" to "cat")

# 3. Train with edited specs
cartlet train data.csv -X specs.json -o model.cart
```

**Output formats:**
```bash
# Simple (default) - edit-friendly
cartlet inspect data.csv
# {"age": "num", "color": "cat", "_target": "cat (label)"}

# Full - with dtypes
cartlet inspect data.csv -f full
# {"age": {"dtype": "int", "type": "num"}, ...}

# Array - complete spec
cartlet inspect data.csv -f array
# {"features": [...], "target": {...}}
```

## Feature Schema

```python
{"name": "age", "dtype": "int", "type": "num"}   # Numerical integer
{"name": "color", "dtype": "str", "type": "cat"} # Categorical string
{"name": "rating", "dtype": "int", "type": "cat"}  # Categorical integer
```

| Field   | Values                | Default     | Description |
|---------|----------------------|-------------|-------------|
| `name`  | string               | required    | Feature name |
| `dtype` | `str`, `int`, `float`| `str`       | Data type |
| `type`  | `cat`, `num`         | *inferred*  | Split type |

### CLI Feature Specs

Provide explicit feature types via `-X/--features`:

```bash
# Inline JSON (simple: just type)
cartlet train data.csv -X '{"age": "num", "color": "cat"}'

# Inline JSON (full specs)
cartlet train data.csv -X '[{"name": "age", "dtype": "int", "type": "num"}]'

# From JSON file
cartlet train data.csv -X features.json
```

**features.json** supports three equivalent shapes:

Simple (type only):
```json
{"age": "num", "color": "cat", "size": "cat"}
```

Object with full specs:
```json
{"age": {"dtype": "int", "type": "num"}, "color": {"dtype": "str", "type": "cat"}}
```

Array form:
```json
[{"name": "age", "dtype": "int", "type": "num"}, {"name": "color", "type": "cat"}]
```

## Architecture

For contributors, the data flow is:

```
Training data (CSV/TSV/JSONL)
        │
        ▼
  load_training_data        cartlet/io/loader.py
        │  (X, y, feature_names)
        ▼
  DecisionTree / RandomForest / XGBoostTree
        │                   cartlet/{tree,forest,xgboost}.py
        │  load_data()
        ▼
  Trainer.train()           cartlet/trainer/{native,sklearn}.py
        │                   produces a nested-list tree:
        │                   [feature, op, value, left, right]
        ▼
  (in-memory model)
        │
        │  export()         cartlet/base.py dispatches by extension
        ▼
  ByteWriter                cartlet/io/bytes.py
        │                   builds string/float/cat/distribution pools,
        │                   flattens nodes to decision/leaf arrays
        ▼
  .cart bytes               format spec in cartlet/io/cart_format.py
        │                   (magic + 30-byte header + pools + tables)
        ▼
  runner.load_model         cartlet/runner.py
  runner.predict            (or zero-dep cartlet/bundled/predict.py)
```

Key files for a new contributor to read, in order:

1. `cartlet/types.py` — feature/task constants, `FeatureSpec`, `ModelData`.
2. `cartlet/tree.py` — `DecisionTree` (in-memory tree shape, predict, export).
3. `cartlet/trainer/native.py` — pure-Python CART training, the canonical algorithm.
4. `cartlet/io/cart_format.py` — the binary format spec (header layout, opcodes).
5. `cartlet/io/bytes.py` — `ByteWriter` builds the pools then serializes.
6. `cartlet/runner.py` — flat-array tree traversal for inference.
7. `cartlet/bundled/predict.py` — same algorithm with zero imports beyond stdlib, intentionally duplicating constants from `cart_format.py` for portable deployment.

## Why "Cartlet"?

It's a little CART (Classification And Regression Trees).

## License

MIT License - see [LICENSE](LICENSE) for details.
