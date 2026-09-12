# Scalability measurement snapshots

These are historical measurements of two clean source revisions, not promises
about every dataset, machine, or future release. The JSON files retain all timing
samples, exact settings, fixture hashes, source hashes, and dependency versions.
Use the [benchmark methodology and recipes](benchmarks.md) to make a new
comparison when the implementation changes. Artifact commits follow the measured
source revisions; the reports intentionally keep those original identities.

## Tabular loader memory

The comparison uses 100,000 generated rows, 12 features, seed 42, and a categorical
vocabulary of 128 values per feature. Before/after settings and fixture hashes
match within each format. Both revisions used Python 3.12.11 on Darwin arm64;
full version details are in each report. The table was calculated from the raw
`peak_python_bytes` fields (decimal MB = bytes / 1,000,000).

| Format | Case | Loader | Before MB | After MB | Reduction |
|---|---|---|---:|---:|---:|
| TSV | numeric | `read_vectors` | 106.03 | 86.04 | 18.9% |
| TSV | numeric | `load_training_data` | 134.82 | 48.84 | 63.8% |
| TSV | categorical | `read_vectors` | 107.23 | 87.24 | 18.6% |
| TSV | categorical | `load_training_data` | 107.22 | 87.24 | 18.6% |
| JSONL | numeric | `read_vectors` | 86.02 | 86.02 | 0.0% |
| JSONL | numeric | `load_training_data` | 48.82 | 48.82 | 0.0% |
| JSONL | categorical | `read_vectors` | 87.22 | 87.22 | 0.0% |
| JSONL | categorical | `load_training_data` | 87.22 | 87.22 | 0.0% |

The TSV numeric training loader reduced its peak by 63.8% in this measurement by
converting rows as they arrive. Raw vectors and categorical training still retain
string values in their outputs, so their proportional reduction is smaller.
JSONL already processed records incrementally and its measured memory is unchanged.
Batch results still grow with dataset size. These peaks are Python allocations
tracked during one call, **not total process memory or RSS**.

Timing samples are archived without a speedup or regression claim: observed
wall times also varied in the unchanged JSONL path. The memory observation and
incremental-consumption regression test justify the targeted loading change;
timing conclusions would need more controlled repeated comparisons.

### Source identities and raw reports

- Before revision: `f89459896607cbcebbd1c8ed07a944329571338f`;
  source SHA-256: `cdc6274d0481d963cfde12aefc7630abfae674a96ac88966285558e16c35a612`.
- After revision: `7f3d32d40ec60c2a079a88f161bc2ab4f96bcda8`;
  source SHA-256: `2e8a2920d3b94f3b70fd3d1dd2326e519fa8d18f4ec38a55f9c8e94c37176402`.

- [TSV before](benchmark-results/loaders-before-tsv.json)
- [TSV after](benchmark-results/loaders-after-tsv.json)
- [JSONL before](benchmark-results/loaders-before-jsonl.json)
- [JSONL after](benchmark-results/loaders-after-jsonl.json)

## Baseline workflow observations

These workflow measurements are all from the **before** revision above. They
establish reproducible workload examples, not results of the loader optimization
or comparisons of whole-process native/sklearn memory. The small native and
sklearn cases use 500 rows, 6 features, depth 6, 16 trees for forests, and 2,000
requested predictions. The larger native forest uses 10,000 rows, 12 features,
depth 10, 64 trees, and 10,000 predictions. Multiple settings differ between small
and large cases, so these observations do not establish a scaling curve or
complexity class. Each timer has three samples; the table uses their median.

| Snapshot | Case | Train seconds | Train peak Python MB | Predict / second | Model bytes |
|---|---|---:|---:|---:|---:|
| Small native | numeric | 0.007 | 0.29 | 1,795,801 | 490 |
| Small native | categorical | 0.098 | 0.43 | 1,184,571 | 4,036 |
| Small native | forest | 0.053 | 1.44 | 92,591 | 6,361 |
| Small sklearn | numeric | 0.002 | 0.30 | 1,779,624 | 490 |
| Small sklearn | categorical | 0.003 | 0.52 | 1,178,319 | 4,036 |
| Small sklearn | forest | 0.011 | 0.33 | 85,074 | 6,878 |
| Larger native forest | forest | 10.727 | 122.23 | 15,421 | 173,073 |

Training/predictor label agreement was checked on 100 source rows per case before
these measurements. Reports also contain separate export, model-loading, and
prediction allocation measurements. Prediction throughput excludes loading and
uses repeated source rows; it is not held-out accuracy.

- [Small native workflow](benchmark-results/scale-native.json)
- [Small sklearn workflow](benchmark-results/scale-sklearn.json)
- [Larger native forest workflow](benchmark-results/scale-forest-large.json)

The small-workflow and loader commands are in the [comparison recipes](benchmarks.md#comparison-recipes).
To reproduce the larger forest at a recorded source revision:

```sh
python -m benchmarks.scalability --cases forest --rows 10000 --features 12 \
  --trees 64 --depth 10 --predictions 10000 --repeats 3 --format json > forest-large.json
```

These results support keeping the benchmark suite and the targeted loader
optimization. A metadata-only model parser remains deferred; these measurements
do not establish a need for another parsing path.
