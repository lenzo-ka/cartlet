# Changelog

## Unreleased

Cartlet is in alpha. Before 1.0, a `0.X.0` release may change APIs and model
formats incompatibly. Keep the package and standalone runner used to export and
load a model on the same release; retrain or re-export models when a release
requires it.

### Breaking changes

- Model format 2 preserves float64 numeric values and feature dtypes. Numeric
  tree operators now mean exactly `<` or `<=`; older binary runners reject the
  new version. JSON/JSONL/pickle tree and forest envelopes require
  `schema_version: 2`. See [model contracts](docs/model_contracts.md) for migration.
- Training rejects malformed rows, nonfinite data, invalid weights and splits,
  invalid direct-model parameters, and inconsistent saved-model schemas.
  Zero-weight rows are omitted. Explicit zero validation with supported pruning
  is an error; disabled or unsupported pruning does not hold out rows.
- Native and sklearn minimum-sample limits count positive-weight rows; frequency
  weights still affect impurity and leaf statistics.
- JSONL datasets require object records and named columns; malformed records and
  absent training targets are errors in batch and streaming readers.
- Replacement training data invalidates stale XGBoost state. Native XGBoost
  artifacts require validated Cartlet metadata when loaded through its API.

### Correctness and efficiency

- Stabilize regression split statistics under target offsets and extreme finite
  feature bounds; choose varying columns for isolation-tree splits.
- Preserve XGBoost strict float32 comparisons and multiclass intercepts through
  export, plus metadata and single-class behavior through native roundtrips.
- Use sparse categorical encoding for sklearn training and decode conversions
  and bundle inputs once.
- Align named evaluation/test columns, validate metric populations, and balance
  cross-validation folds.
- Preserve live models on failed loads and existing outputs on failed writes;
  reject input/output path aliases and unrepresentable binary strings.
- Keep package and standalone numeric, gzip, and typed-vocabulary behavior aligned.

### Library and CLI

- Add quiet `train_model` and `train_file` workflows with `TrainingSettings`,
  `TrainingResult.to_dict()`, and a CLI `train --json` report.
- Return serializable conversion provenance through `ConversionResult.to_dict()`.
- Validate configuration values and report actual training/test/validation counts.

### Documentation and development

- Correct the numeric regression quick start and shell continuation example.
- Link runnable dataset tutorials and correct the standalone runner copy path.
- Clean up temporary output from the deployment tutorial.
- Identify the package as alpha in distribution metadata.
- Use the BSD 2-Clause license, matching the updated project license.
- Require the Python 3.14 and optional-backend test jobs before the CI build.
- Pin Ruff across development and hooks; CI installs the same development tools
  and checks Python snippets in Markdown as well as source files.
- Keep the Python 3.11 source type check consistent when optional backends with
  newer stub grammars are installed; backend behavior remains covered by tests.
- Check the release tag against the package version before building a publication.

## 0.5.0

Baseline preceding this cleanup. See the
[release and source](https://github.com/lenzo-ka/cartlet/releases/tag/v0.5.0)
for the existing implementation.
