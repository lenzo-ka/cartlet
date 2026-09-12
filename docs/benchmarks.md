# Reproducible scalability benchmarks

Run the generated-data benchmark from a source checkout:

```sh
python -m benchmarks.scalability > results.tsv
python -m benchmarks.scalability --format json > results.json
```

No extra packages are needed for the default native backend. Install the
`sklearn` extra to compare that backend. The benchmark lives in the developer
`benchmarks` package, which is not included in the runtime wheel. Existing
`make bench` / `benchmarks.runner_throughput` remains the public-dataset,
bundled-runner benchmark; it includes process startup and pipe I/O. This
benchmark measures the in-process `.cart` predictor instead.

## Workloads and stages

Fixtures are generated locally from a seed, with no downloaded corpus. All
workloads are classification tasks. Numeric and forest cases use the same
uniform numeric inputs and a two-feature label rule, allowing tree/forest
comparisons. The categorical case samples named categories and derives labels
from the first category. `--categories` is the vocabulary size available to
each feature, not a promise that every value occurs in a finite sample. These
workloads exercise costs; their prediction accuracy does not establish quality
on real-world data.

| Stage | Measured operation |
|---|---|
| `read_vectors` | Batch vector reader, including file open and parsing |
| `load_training_data` | Training loader, including file open, parsing and conversion |
| `train` | Quiet `train_model` workflow on already loaded data, including validation and model construction |
| `export` | `.cart` serialization and atomic file replacement |
| `load` | `Predictor` construction from the `.cart` file |
| `predict` | `Predictor.predict_batch` on prebuilt inputs, including the returned prediction list |

Training uses an explicit classification task, seed, maximum depth and feature
specifications. Splits and pruning are disabled; forest size is configurable.
Sklearn forests use one worker. Native categorical search uses the default
exact strategy. `--trees` affects only the forest case. Binary export preserves
distributions. For stages that need an exported model, the benchmark verifies
training-model/predictor label agreement on up to 100 source rows outside the
measurements and reports `parity_rows`; disagreement fails the run.

Generated JSONL records contain the same numeric strings as TSV cells. The two
loader APIs intentionally do different work: `read_vectors` preserves JSONL
value types, while `load_training_data` converts numeric strings. Compare each
API against itself across revisions; their numbers are not interchangeable.

## Measurement and interpretation

Each stage gets one untimed warmup, then `--repeats` untraced `perf_counter`
measurements. The report retains every sample and its median. Garbage
collection runs before each call, outside the timer; normal garbage collection
remains enabled during the call. Imports, fixture generation, prerequisite
model training/loading and input-batch construction are outside the measured
stage. Results stay alive until that call's measurement ends. Files are warm
in the OS cache; this is not cold-disk or process-startup timing. Export includes
normal atomic replacement, without claiming durable-storage latency.

One additional call runs under `tracemalloc`, separately from timing.
`peak_python_bytes` is the peak additional Python allocation tracked during
that call, in bytes. It excludes preexisting inputs/models and untracked native
backend buffers. It is **not total process memory or RSS**, and cannot support
whole-process memory comparisons between native and sklearn. For the loaders,
it includes output retention and intermediate Python allocations. For
training, the already loaded dataset is excluded. The memory call is a single
observation, not a median. An existing caller-owned trace is refused rather
than changed.

Prediction throughput is requested predictions divided by median batch time;
the batch cycles source rows. It excludes model loading and is not held-out
accuracy. Small smoke defaults establish runnable measurements; sweep sizes
and tree counts before making scaling claims. Do not infer a complexity class
from a single size or use noisy wall times as CI pass/fail thresholds.

JSON is the archival form: settings, timing samples, measurement definitions,
dataset SHA-256 and byte size, model byte size, package versions, OS/architecture,
Git revision/dirty state and a source-content SHA-256. The source hash covers
relative filenames and contents of `cartlet/**/*.py` plus this benchmark;
it distinguishes modified source without emitting local paths. A dirty hash
identifies content but cannot reconstruct it: retain the diff or use a clean
commit for published comparisons. The imported Cartlet must be from the same
checkout as the benchmark; an adjacent installed checkout is refused.

TSV has a header and one row per case/stage with unadorned numeric values.
Timing, memory and throughput are columns 3, 4 and 5, respectively. For example:

```sh
python -m benchmarks.scalability > results.tsv
(head -n 1 results.tsv; tail -n +2 results.tsv | sort -t "$(printf '\t')" -k4,4nr) > by-memory.tsv
```

## Comparison recipes

Use the same interpreter, hardware, settings and fixture hashes at both clean
source revisions. Keep JSON reports with the measured revision; an artifact
commit may follow it. Run from each checkout so the subject-import guard holds.
Loader-only selection does not train models, making larger input sizes useful:

```sh
python -m benchmarks.scalability --rows 100000 --features 12 \
  --cases numeric categorical --stages read_vectors load_training_data \
  --repeats 3 --format json > loaders-tsv.json
python -m benchmarks.scalability --rows 100000 --features 12 \
  --cases numeric categorical --stages read_vectors load_training_data \
  --data-format jsonl --repeats 3 --format json > loaders-jsonl.json
```

For a modest full workflow comparison, including a 16-tree forest:

```sh
python -m benchmarks.scalability --rows 500 --features 6 --categories 128 \
  --trees 16 --depth 6 --repeats 3 --format json > native.json
python -m benchmarks.scalability --rows 500 --features 6 --categories 128 \
  --trees 16 --depth 6 --repeats 3 --backend sklearn --format json > sklearn.json
```

Increase `--rows`, `--features`, `--categories` or `--trees` independently to
isolate a cost. Exact categorical search and large forests can be expensive.
Available settings and defaults come from `python -m benchmarks.scalability
--help`.

The same workflow is available quietly from Python:

```python
from benchmarks.scalability import BenchmarkSettings, render_tsv, run_benchmarks

report = run_benchmarks(
    BenchmarkSettings(rows=1000, features=12, repeats=3),
    cases=("numeric", "categorical"),
    stages=("read_vectors", "load_training_data"),
)
print(render_tsv(report), end="")
```

## Recorded measurements

See [scalability measurement snapshots](scalability_results.md) for the pinned
loader before/after comparison and baseline native/sklearn workflow observations,
with their raw JSON reports. Re-run the recipes at clean source revisions to
refresh conclusions; historical reports retain the revisions they measured.
