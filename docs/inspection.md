# Held-out importance and path inspection

Cartlet provides two complementary inspection tools:

- `permutation_importance` measures how much a trained tree or forest depends
  on held-out columns.
- `leaf_paths` exports every structural root-to-leaf route. `decisive_leaves`
  selects classification paths by predicted class and optional support or
  purity thresholds.

These APIs inspect an existing model. They do not retrain it, choose a split,
or recover training observations that were not stored in the model artifact.

## Permutation importance

```python
from cartlet import permutation_importance

report = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=7,
)
for item in report["importances"]:
    print(item["name"], item["mean"], item["std"])
```

The baseline and every shuffled score use held-out rows supplied by the caller.
Classification reports the increase in error rate; regression reports the
increase in mean squared error. A positive value means shuffling made the model
worse. Zero means no measured change, and a negative estimate is possible when
a particular finite shuffle happens to improve the held-out score.

Classification targets are converted to strings before scoring, matching the
model's label representation. Held-out integer labels `0` and `1` therefore
compare correctly with predictions `"0"` and `"1"`.

Set `random_state` to an integer for repeatable permutations. It accepts only
`int` or `None`; booleans and other values are errors. The input rows are
copied and never mutated.

Groups move related columns together with one donor-row permutation:

```python
report = permutation_importance(
    model,
    X_test,
    y_test,
    feature_groups={"coordinates": ["latitude", "longitude"]},
    random_state=7,
)
```

Group members can be feature names or zero-based indexes. A group defines one
reported unit; members must be unique within that group. When groups are not
supplied, every model feature is measured separately.

Every prediction uses the selected missing policy. `missing="error"` raises
`MissingFeatureError` if an evaluated decision lacks a value.
`missing="right"` applies the prediction compatibility route; an XGBoost
learned missing direction still takes precedence.

### Out-of-bag importance

Bootstrap-trained random forests can measure Breiman OOB importance without a
separate held-out set:

```python
report = forest.oob_permutation_importance(
    n_repeats=10,
    random_state=7,
)
```

For each tree, Cartlet finds the training rows absent from that tree's bootstrap
sample, computes its baseline loss once, and shuffles each feature or group only
within those OOB rows. The report's `baseline` and `oob_baseline` are the mean
per-tree OOB loss. Each repeat value is the mean per-tree rise in loss, and the
reported mean and sample standard deviation summarize those repeat values.
Classification uses error rate and regression uses mean squared error, as in
held-out importance. Rows are unweighted, matching the held-out API; feature
groups, seeds, and missing-value policies have the same validation and meaning.

Bootstrap membership and training rows stay only in memory. OOB importance is
therefore available only on a `RandomForest` trained in the current process with
`bootstrap=True`. Loaded models and forests trained with `bootstrap=False` must
use held-out `permutation_importance`. There is no CLI command for OOB importance
because model artifacts intentionally do not retain training rows or bootstrap
membership.

## Structural paths

```python
from cartlet import decisive_leaves, leaf_paths

export = leaf_paths(model)
approved = decisive_leaves(model, "approved", min_purity=0.95)
```

The export is JSON-compatible:

```text
{
  "task": "classification",
  "features": [{"name": "age", "dtype": "int", "type": "num"}],
  "numeric_input_float32": false,
  "trees": [{"tree": 0, "leaves": [...]}]
}
```

The feature schema is part of the routing contract. Bool values must be
normalized according to the exported `dtype`. XGBoost-derived `.cart` models
set `numeric_input_float32` to `true`; consumers must convert numeric inputs to
float32 before comparing them with an exported threshold.

Each leaf record contains:

- `tree`, the tree index, and `leaf`, the model-global ID used by
  `predict_path`;
- `path`, the ordered conditions for one complete route;
- `prediction` and, for ordinary classification leaves, `predicted_class`;
- `class_distribution`, `class_counts`, `support`, and `purity` when the model
  representation actually retains them.

Paths, not leaf IDs, are the unit of export. A nested or loader-admitted flat
structure can route different branches to the same leaf ID, in which case the
export contains multiple path records carrying that ID. Binary conditions
retain their split operator and identify the selected `left` or `right` branch.
Switch values reaching one
child are one sorted `in` condition; the default route is `not in` with all
reachable case values. Switches are first-match-wins, so a later duplicate case
value is unreachable and omitted. A split's learned `missing_direction` appears
only on the condition that accepts a missing value; all other conditions for
that split report `null`. For a switch, `"left"` marks the condition containing
the first stored case and `"right"` marks the default condition.

For classification, a stored distribution supplies `class_distribution` and
its winning probability supplies model-stored `purity`. A collapsed label does
not imply purity and has `null` for both the distribution and purity. Cartlet
does not currently retain classification class counts or leaf support, so
those model-stored fields are `null`.

For regression, `support` means effective training weight, not row count. Full
nested and JSON models retain it. Compact `.cart` regression leaves retain only
the prediction, so support is `null` after `.cart` reload. XGBoost tree leaves
are additive contributions rather than independently predicted classes; their
class, distribution, purity, and support fields are `null`, and
`decisive_leaves` rejects XGBoost-derived models.

## Empirical path support

Supply data to measure how current rows use the exported paths:

```python
export = leaf_paths(model, X_test, y_test)
selected = decisive_leaves(
    model,
    "approved",
    X_test,
    y_test,
    min_support=20,
    min_purity=0.9,
)
```

Rows are routed with the model's own `predict_path`. `data_support` is the
number of supplied rows reaching a path. For classification data with `y`,
`data_class_counts` counts canonical string labels and `data_purity` is the
fraction equal to that path's predicted class. These fields remain separate
from model-stored `support` and `purity`; empirical observations never fill an
unavailable model statistic.

When `X` is supplied, `decisive_leaves` applies thresholds to `data_support`
and `data_purity`. Otherwise it uses model-stored fields. An unknown field does
not satisfy a requested threshold. In particular, a data purity threshold
without classification labels selects nothing.

## Command line

```bash
cartlet importance model.cart test.csv --target label --random-seed 7
cartlet importance model.cart test.csv --groups groups.json -f tsv

cartlet leaves model.cart
cartlet leaves model.cart --data test.csv
cartlet leaves model.cart --data test.csv --target label \
  --class approved --min-support 20 --min-purity 0.9
```

Both commands write JSON by default and support TSV with `-f tsv`. Use
`-o FILE` to write the machine-readable result to a file. `leaves --data`
accepts CSV, TSV, and object-record JSONL; `--target` is optional, so an
unlabeled feature-only file can provide `data_support`. Named columns are
aligned to the model schema like `evaluate` and `importance`.
