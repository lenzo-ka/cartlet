"""
Smoke tests for the runnable scripts in ``examples/``.

Each example exposes a ``run(*, random_state=...) -> dict`` function. The
tests here invoke it directly with a fixed seed and assert that the
returned metrics are in the expected ballpark. This keeps the examples
honest (they will fail CI if they regress) and exercises the same paths a
user copying the example would hit, without re-implementing the workflow
inside the test file.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest

# The example scripts import sklearn (via examples.common); skip the whole
# module when it is unavailable so the no-optional-deps CI job can collect.
pytest.importorskip("sklearn")

from examples import (  # noqa: E402
    breast_cancer_binary,
    diabetes_regression,
    iris_decision_tree,
    iris_runner_deploy,
    wine_random_forest,
)
from examples.common import (  # noqa: E402
    available_datasets,
    build_example_parser,
    load_dataset,
    print_classification_report,
    print_regression_report,
)

SEED = 42


class TestSharedHelpers:
    """``examples.common`` should give consistent splits and clean reports."""

    def test_load_dataset_classification_round_trip(self):
        ds = load_dataset("iris", random_state=SEED)
        assert ds.name == "iris"
        assert ds.task == "classification"
        assert ds.class_names == ["setosa", "versicolor", "virginica"]
        assert len(ds.feature_names) == 4
        assert len(ds.X_train) + len(ds.X_test) == 150
        spec_names = [s["name"] for s in ds.feature_specs]
        assert spec_names == ds.feature_names
        assert all(s["type"] == "num" for s in ds.feature_specs)

    def test_load_dataset_regression(self):
        ds = load_dataset("diabetes", random_state=SEED)
        assert ds.task == "regression"
        assert ds.class_names is None
        assert all(isinstance(v, float) for v in ds.y_train[:5])

    def test_load_dataset_split_is_deterministic(self):
        a = load_dataset("wine", random_state=SEED)
        b = load_dataset("wine", random_state=SEED)
        assert a.y_test == b.y_test

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("not-a-dataset")

    def test_available_datasets(self):
        assert set(available_datasets()) == {
            "iris",
            "wine",
            "breast_cancer",
            "diabetes",
        }

    def test_parser_has_shared_flags(self):
        parser = build_example_parser("test")
        args = parser.parse_args([])
        assert args.random_state == 42
        assert args.test_fraction == 0.3
        assert args.output is None
        assert args.quiet is False

    def test_report_helpers_emit_expected_fields(self):
        buf = io.StringIO()
        print_classification_report(
            {"accuracy": 0.95, "correct": 19, "total": 20}, file=buf
        )
        out = buf.getvalue()
        assert "Accuracy" in out and "19/20" in out

        buf = io.StringIO()
        print_regression_report(
            {"mse": 1.0, "mae": 0.8, "rmse": 1.0, "r2": 0.5, "total": 5},
            file=buf,
        )
        out = buf.getvalue()
        assert "MSE" in out and "R^2" in out


class TestIrisDecisionTreeExample:
    def test_run_meets_accuracy_floor(self):
        metrics = iris_decision_tree.run(random_state=SEED)
        assert metrics["total"] == 45
        assert metrics["accuracy"] > 0.9

    def test_run_exports_when_requested(self, tmp_path):
        out = tmp_path / "iris.cart"
        iris_decision_tree.run(random_state=SEED, output=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_main_with_quiet_returns_zero(self, capsys):
        rc = iris_decision_tree.main(["--quiet", "--random-state", str(SEED)])
        assert rc == 0
        assert capsys.readouterr().out == ""


class TestWineRandomForestExample:
    def test_run_meets_accuracy_floor(self):
        result = wine_random_forest.run(random_state=SEED, n_estimators=20)
        assert result["accuracy"] > 0.9
        importances = result["feature_importances"]
        assert abs(sum(importances.values()) - 1.0) < 1e-6
        assert max(importances.values()) > 0.05


class TestBreastCancerBinaryExample:
    def test_run_exercises_proba_helpers(self):
        result = breast_cancer_binary.run(random_state=SEED)
        assert result["accuracy"] > 0.85
        assert result["forest_accuracy"] > 0.85
        assert abs(result["forest_proba_sum"] - 1.0) < 1e-6
        assert result["nbest_first_label"] in {"malignant", "benign"}


class TestDiabetesRegressionExample:
    def test_run_returns_regression_metrics(self):
        metrics = diabetes_regression.run(random_state=SEED, max_depth=5)
        assert metrics["task"] == "regression"
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert abs(metrics["rmse"] - metrics["mse"] ** 0.5) < 1e-9
        assert -1.0 < metrics["r2"] <= 1.0


class TestIrisRunnerDeployExample:
    def test_export_then_predictor_agrees(self):
        result = iris_runner_deploy.run(random_state=SEED)
        assert result["accuracy"] > 0.9
        # .cart stores thresholds as float32; allow a few boundary flips.
        assert result["agreement"] > 0.95
        assert result["metadata"]["model"] == "iris-decision-tree"
        assert result["metadata"]["random_state"] == SEED


class TestExamplesRunFromCLI:
    """Each ``main()`` should accept argv and exit 0 without crashing."""

    @pytest.mark.parametrize(
        "module",
        [
            iris_decision_tree,
            wine_random_forest,
            breast_cancer_binary,
            diabetes_regression,
            iris_runner_deploy,
        ],
    )
    def test_main_quiet_returns_zero(self, module):
        saved = sys.argv
        try:
            sys.argv = [module.__name__]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = module.main(["--quiet", "--random-state", str(SEED)])
            assert rc == 0
        finally:
            sys.argv = saved
