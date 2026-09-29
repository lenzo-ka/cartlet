# Model and data contracts

Cartlet is in alpha. Version 0.6.0 introduces these contracts and model format 2;
no compatibility layer is provided for pre-version-2 model artifacts.

## Saved models

The binary `.cart` format is version 2. Numeric values and probabilities use
float64 storage so native tree thresholds survive export without float32
rounding. Numeric decision operators are explicit: `<=` is inclusive and `<`
is strict. XGBoost's strict comparisons retain its float32 input semantics.
The package and standalone runners use the same operators and stored feature
dtypes. See [the binary specification](cart_format.md).

DecisionTree and RandomForest JSON, JSONL, and pickle envelopes declare
`schema_version: 2`. Loading an older or unversioned envelope raises a clear
error instead of interpreting its old `<` nodes as strict comparisons.
Retrain with the new release, or use the release that wrote the artifact to
inspect it and perform an explicit conversion of its representation. The new
release does not automatically migrate old artifacts. Replace copied standalone
runners together with the models they load.

Decision and leaf attribution uses the existing model-format-2 array indexes;
adding `predict_path` does not change the binary format. See
[stable node IDs](cart_format.md#stable-node-ids).

A failed load leaves the existing model usable. A successful load replaces its
training data and backend provenance; loading JSON over a sklearn-trained model
cannot leave an old estimator available for `.skl` export.

Writers serialize to a sibling temporary file and replace the destination only
on success. Parent directories must already exist. Embedded NUL characters
cannot be represented in the binary string table and are rejected. Operations
with distinct input and output artifacts reject aliases, including existing
symlinks and hard links.

## Training data and named columns

Training requires nonempty, rectangular rows of finite scalar strings or
numbers, with aligned targets and weights. Missing or nested training values
are rejected. Weights must be finite, nonnegative, and have a finite positive
total; zero-weight rows are omitted from fitting. Loaded observations are copied
so caller mutations do not change the stored training data.

Prediction has a separate missing-input policy. The default, `missing="error"`,
raises `MissingFeatureError` only when an evaluated decision tests a missing
value. `None` and a feature beyond the vector length are always missing. A
numeric-node value is missing when `float(value)` is NaN, so the string `"nan"`
is missing there. At categorical equality and switch decisions, a non-string
scalar whose self-inequality returns a trusted Boolean true is missing; a string
`"nan"` remains a category. Training rejects nonfinite values, so no model
learns a NaN. The
compatibility policy `missing="right"` routes those values right, or to a
switch default. Native XGBoost in-process prediction continues to use the
Booster's learned missing directions; use a `.cart` runner for path attribution.
Bool-dtype inputs are checked for missingness and then normalized only when
their feature is tested, with the
same accepted values in nested prediction and both `.cart` runners.

With `strict=True`, OOV validation inspects every present value, normalizes
bools, and rejects unrecognized bool and OOV values. It skips absent and missing
values; traversal then applies the selected missing policy exactly as in
non-strict prediction. This precedence is the same for trees and forests.

Named prediction and evaluation inputs are aligned to model feature names.
Reordering CSV or JSONL columns therefore does not change predictions or
accuracy. Missing or duplicate named features are errors; explicitly headerless
inputs use positional columns.

Test and validation fractions must be finite and nonnegative, individually
less than one, and sum to less than one. The admitted training population must
remain nonempty. Unsupported pruning does not silently withhold training rows.

JSONL datasets use nonempty object records with unique named fields. The writer
requires column names; positional array records are not a supported alternative.
Batch and streaming readers reject malformed records with line diagnostics.
Training targets must be present. CSV/TSV/SSV support headerless positional records.
Tabular readers ignore empty CSV records, including those before the header or
first headerless row; quoted empty fields are still records. Ragged records are
skipped with a warning. Batch readers reject empty or header-only input, while
`iter_vectors` yields nothing; a file containing only ragged data records yields
an empty batch. Headers and values are NFC-normalized. `load_training_data`
converts numeric strings; `read_vectors` and `iter_vectors` preserve them.

Tabular batch loading consumes and processes records incrementally, without
retaining a second complete collection of raw rows. Returned feature and target
lists still occupy memory proportional to the dataset; use `iter_vectors` when
the consumer can process one record at a time.

Configuration files use the same accepted option names and values as the CLI.
Unknown keys, invalid scalar values, and malformed YAML/JSON are errors; explicit
CLI arguments override valid configuration values.

## Evaluation

Evaluation helpers require equal target and prediction lengths. Cross-validation
requires aligned feature and target populations and allocates fold remainders
across folds rather than concentrating them in the final fold. Its reported mean
remains the mean of the per-fold scores.
