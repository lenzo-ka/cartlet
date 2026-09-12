# Operational training API

`train_model` owns validation, splitting, training, evaluation, and optional
export. `train_file` adds tabular reading and named test-column alignment. The
installed `cartlet train` command delegates to these functions.

```python
from cartlet import TrainingSettings, train_model

result = train_model(
    [["red"], ["blue"], ["red"], ["blue"]],
    ["R", "B", "R", "B"],
    feature_names=["color"],
    settings=TrainingSettings(test_split=0, random_state=7),
)
assert result.model.predict(["red"]) == "R"
report = result.to_dict()
```

Calls are quiet. With `output=None`, they write nothing. Supply `output` to
export using the existing model codecs. `TrainingResult` contains the live
model and a JSON-compatible report with effective task, settings, metrics,
statistics, warnings, and output path. `source_samples` counts active rows after
zero-weight rows are removed; training, validation, and test populations are
reported separately. Explicit test data is additional to the source population.

The default settings use a native decision tree and reserve 5% of source rows
for testing, rounding down. Pruning is off. When enabled for native
classification trees, the requested validation fraction is relative to the
source population, rounded down; the tree owns the actual validation split.
Explicit zero validation with supported pruning is rejected. Pruning off holds
out no validation rows. Native depth zero creates one leaf; sklearn requires
positive depth and at least two samples per split. Direct model constructors
and training calls use the same parameter validator as TrainingSettings. Unsupported
supervised pruning returns a warning and holds out no validation rows.
Isolation training is unlabeled and uses every input column unless `train_file`
receives an explicit target column to exclude. It does not hold out test rows
and rejects pruning, supervised task settings, weights, and labeled test data. train_model rejects supplied targets for isolation;
train_file can explicitly exclude a label column from the unlabeled input.

`train_file` accepts feature specifications as Python mappings, ordered
specifications, inline JSON, or a JSON file. Named test columns are aligned to
training columns; missing or duplicate columns are errors. Without a header,
columns remain positional. Input and output aliases, including hard links and
symlinks, are rejected before writing.

```sh
cartlet train data.csv --target label --random-seed 7 --json -o model.cart
```

`--json` prints one structured report. Without it, the CLI prints a concise
summary and evaluation metrics; `--verbose` also prints model statistics.

`convert` similarly returns `ConversionResult` with `to_dict()`. It decodes the
input once, preserves available distributions, and rejects isolation and
XGBoost artifacts that require their dedicated model APIs. Binary exports that
omitted distributions cannot recover them through conversion.
