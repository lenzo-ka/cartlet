"""Check benchmark subjects and measurement contracts, not speed thresholds."""

import csv
import io
import json
import tracemalloc
from dataclasses import replace

import pytest

from benchmarks.scalability import (
    CASES,
    STAGES,
    BenchmarkSettings,
    generate_dataset,
    main,
    measure,
    render_tsv,
    run_benchmarks,
)
from cartlet.io import load_training_data

SMALL = BenchmarkSettings(
    rows=24, features=3, categories=16, trees=3, depth=3, predictions=30, repeats=1
)


@pytest.mark.parametrize("data_format", ["tsv", "jsonl"])
@pytest.mark.parametrize("case", CASES)
def test_generated_fixture_is_reproducible_and_loadable(tmp_path, case, data_format):
    settings = replace(SMALL, data_format=data_format)
    path = tmp_path / f"data.{data_format}"
    digest = generate_dataset(case, settings, path)
    original = path.read_bytes()
    assert generate_dataset(case, settings, path) == digest
    assert path.read_bytes() == original
    X, y, names, target = load_training_data(str(path))
    assert len(X) == len(y) == settings.rows
    assert names == ["x0", "x1", "x2"]
    assert target == "label"
    assert all(len(row) == settings.features for row in X)
    if case == "categorical":
        assert len({row[0] for row in X}) > 5
        assert all(value.startswith("category_") for row in X for value in row)
        assert y == [int(row[0].split("_")[1]) % 3 for row in X]
    else:
        assert all(
            isinstance(value, float) and 0 <= value < 1 for row in X for value in row
        )
        assert y == [int(row[0] + row[-1] > 1) for row in X]
    assert generate_dataset(case, replace(settings, seed=43), path) != digest


def test_all_stage_report_checks_prediction_parity_and_serializes():
    report = run_benchmarks(SMALL)
    assert len(report["results"]) == len(CASES) * len(STAGES)
    assert len(report["provenance"]["source_sha256"]) == 64
    json.dumps(report, allow_nan=False)
    for result in report["results"]:
        assert result["parity_rows"] == SMALL.rows
        assert len(result["seconds"]) == SMALL.repeats
        assert result["median_seconds"] > 0
        assert result["peak_python_bytes"] > 0
        if result["stage"] == "predict":
            assert (
                result["predictions_per_second"]
                == SMALL.predictions / result["median_seconds"]
            )
    rows = list(csv.DictReader(io.StringIO(render_tsv(report)), delimiter="\t"))
    assert len(rows) == len(report["results"])
    assert rows[0]["rows"] == str(SMALL.rows)
    assert rows[0]["backend"] == "native"


def test_loader_only_does_not_train(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("loader-only benchmark attempted training")

    monkeypatch.setattr("benchmarks.scalability.train_model", fail)
    report = run_benchmarks(
        SMALL, cases=["numeric"], stages=["read_vectors", "load_training_data"]
    )
    assert all(result["model_bytes"] is None for result in report["results"])
    assert all(result["parity_rows"] == 0 for result in report["results"])


def test_measure_runs_timing_without_tracing_and_stops_own_trace():
    tracing = []

    def operation():
        tracing.append(tracemalloc.is_tracing())
        return bytearray(4096)

    result = measure(operation, 2)
    assert tracing == [False, False, False, True]
    assert result["peak_python_bytes"] >= 4096
    assert not tracemalloc.is_tracing()


def test_existing_trace_is_not_disturbed():
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="existing tracemalloc"):
            measure(lambda: None, 1)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_cli_default_tsv_and_invalid_input(capsys):
    assert (
        main(
            [
                "--rows",
                "5",
                "--repeats",
                "1",
                "--cases",
                "numeric",
                "--stages",
                "read_vectors",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("case\tstage\tmedian_seconds\t")
    with pytest.raises(SystemExit) as error:
        main(["--rows", "0"])
    assert error.value.code == 2


def test_wrong_checkout_refused(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.scalability.cartlet.__file__", "/somewhere/cartlet/__init__.py"
    )
    with pytest.raises(RuntimeError, match="same source checkout"):
        run_benchmarks(SMALL)
