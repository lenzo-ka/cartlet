# Changelog

Cartlet is in alpha. Before 1.0, a `0.X.0` release may change APIs and model
formats incompatibly. Keep the package and standalone runner used to export and
load a model on the same release; retrain or re-export models when a release
requires it.

## Unreleased

### Breaking changes

- `.cart` model format 3 adds a one-byte flags field to each decision record,
  and JSON/JSONL/pickle tree envelopes use `schema_version: 3`. Format 2,
  written by Cartlet 0.6.0, is refused; re-export from the training model or
  retrain.

- Classification leaves keep their class distribution by default:
  `min_confidence=1.0` and `min_dist_entropy=0.0`. Only class probabilities
  below `PROB_MIN_THRESHOLD` (1e-8) are dropped. Explicit lower confidence or
  higher entropy thresholds remain lossy compression controls. Default-trained
  `.cart` files may be larger.
- Prediction now raises `MissingFeatureError` by default when an evaluated
  decision tests a missing value: `None`, an index past the vector, a value
  whose float conversion is NaN at a numeric decision, or a non-string NaN at an
  equality or switch decision. Untested features are never read. Pass `missing="right"` (or CLI
  `--missing right`) for the 0.6.0 right/default routing under the new missing
  definition. Exception: at equality or switch decisions, 0.6.0 could match a
  non-string NaN to a stored `"nan"` category; it is now missing and therefore
  takes the right/default branch.

### Added

- Add `predict_path` to nested tree and forest models and to both `.cart`
  runners, with stable model-global decision and leaf IDs.
- Guarantee lazy feature reads during non-strict prediction: only features
  tested along evaluated paths are indexed, once per decision test.

### Fixed

- Preserve XGBoost's learned left/right missing direction in nested trees and
  `.cart` exports, so missing-value predictions and paths match the Booster.

- Return `{label: 1.0}` from both `.cart` runners when `return_dist=True`
  reaches a bare classification leaf, matching in-process prediction.
- Reduce training memory: build the sklearn categorical CSR in preallocated
  arrays instead of per-value Python lists, and train through indexed views of
  the loaded rows instead of further row copies. Trained models are unchanged;
  on a large categorical sklearn fit, peak memory roughly halves.
- Normalize bool-dtype features when evaluated by both `.cart` runners, keeping
  raw bool and accepted string inputs aligned with in-process prediction.
- Canonicalize hand-authored bool-dtype predicates during nested prediction to
  match `.cart` export, and reject switch cases with duplicate canonical keys.
- Evaluate switch nodes in nested trees.
- Reject multiclass XGBoost `.cart` models whose tree count is not a multiple of
  the class count instead of ignoring the remainder trees.

### Development

- Ruff is a floor (`>=0.16.7`) in the development extra and `required-version`,
  so newer Ruff releases run the checks.

## 0.6.0 — 2026-09-13

### Breaking changes

- Model format 2 preserves float64 numeric values and feature dtypes. Numeric
  tree operators now mean exactly `<` or `<=`; older binary runners reject the
  new version. JSON/JSONL/pickle tree and forest envelopes require
  `schema_version: 2`. See [model contracts](docs/model_contracts.md) for migration.
- Training APIs reject malformed rows, nonfinite data, invalid weights and splits,
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

- Process CSV/TSV/SSV records incrementally during batch loading to reduce peak
  memory. Share record validation with streaming readers; ignore blank records
  before headers consistently and warn when skipping ragged records.
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

- Add reproducible generated-data benchmarks for numeric, categorical, and forest
  workloads, with JSON/TSV reports covering loading, training, export, model
  loading, and prediction. Preserve source/fixture hashes, settings, and versions
  in [recorded measurement snapshots](docs/scalability_results.md).
- Document the measured tabular loader memory reduction and its limits; returned
  batch data still occupies memory proportional to dataset size.
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
