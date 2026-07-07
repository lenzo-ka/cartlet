"""
End-to-end workflow tests using bundled scikit-learn datasets.

These tests exercise the full pipeline (load -> train -> export -> reload ->
predict -> evaluate) against realistic public datasets that ship with sklearn,
so no network access or auth is required.

Datasets used:
- Iris (150 samples, 4 numerical features, 3 classes): classic multiclass
- Wine (178 samples, 13 numerical features, 3 classes): higher-dimensional
- Breast cancer (569 samples, 30 numerical features, 2 classes): binary
- Diabetes (442 samples, 10 numerical features, continuous target): regression

Gaps these tests fill vs the existing suite:
- test_sklearn_parity only compares native vs sklearn trainers; here we
  exercise export/reload via the runner, the CLI, and the task-aware
  evaluate_tree / cross_validate paths added in the last commit.
"""

from __future__ import annotations

import csv
import json
import sys

import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import (  # noqa: E402
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from cartlet import (  # noqa: E402
    DecisionTree,
    Predictor,
    RandomForest,
    bundle,
    convert,
    cross_validate,
    evaluate_predictions,
    evaluate_tree,
    read_cart_metadata,
)
from cartlet.cli import main as cli_main  # noqa: E402
from cartlet.runner import load_model, predict_batch  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


def _numeric_specs(names: list[str]) -> list[dict]:
    """Build numerical-feature specs for a list of feature names."""
    return [{"name": n, "dtype": "float", "type": "num"} for n in names]


@pytest.fixture
def iris_split():
    """Iris dataset, 70/30 train/test, target as species-name strings."""
    data = load_iris()
    target_names = list(data.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data.tolist(),
        [target_names[int(c)] for c in data.target.tolist()],
        test_size=0.3,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


@pytest.fixture
def diabetes_split():
    """Diabetes regression dataset, 70/30 split, target as floats."""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data.tolist(),
        [float(v) for v in data.target.tolist()],
        test_size=0.3,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    """Write a CSV file with header + rows."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def iris_csv(tmp_path):
    """Write iris dataset to a CSV with species-name labels (forces string targets)."""
    data = load_iris()
    target_names = list(data.target_names)
    header = [name.replace(" ", "_") for name in data.feature_names] + ["species"]
    rows = [
        [*[float(v) for v in row], target_names[int(label)]]
        for row, label in zip(data.data.tolist(), data.target.tolist(), strict=False)
    ]
    path = tmp_path / "iris.csv"
    _write_csv(str(path), header, rows)
    return str(path)


# =============================================================================
# Iris (classification) end-to-end via Python API
# =============================================================================


class TestIrisWorkflow:
    """Full train -> export -> reload pipeline on iris."""

    def test_native_decision_tree_accuracy(self, iris_split):
        X_train, X_test, y_train, y_test, names = iris_split
        dt = DecisionTree(features=_numeric_specs(names), task="classification")
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)
        metrics = evaluate_predictions(y_test, dt.predict_batch(X_test))
        assert metrics["accuracy"] > 0.9, (
            f"decision tree on iris should easily beat 0.9, got {metrics['accuracy']:.3f}"
        )

    def test_sklearn_trainer_accuracy(self, iris_split):
        X_train, X_test, y_train, y_test, names = iris_split
        dt = DecisionTree(features=_numeric_specs(names), task="classification")
        dt.load_data(X_train, y_train)
        dt.train(trainer="sklearn", random_state=42)
        metrics = evaluate_predictions(y_test, dt.predict_batch(X_test))
        assert metrics["accuracy"] > 0.9

    @pytest.mark.parametrize("ext", [".cart", ".json", ".jsonl", ".pkl"])
    def test_round_trip_via_runner(self, iris_split, tmp_path, ext):
        """Train -> export -> runner.load_model -> runner.predict matches in-memory predictions."""
        X_train, X_test, y_train, _, names = iris_split
        dt = DecisionTree(features=_numeric_specs(names), task="classification")
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        in_memory = dt.predict_batch(X_test)

        out = tmp_path / f"model{ext}"
        dt.export(str(out))
        assert out.exists() and out.stat().st_size > 0

        # The runner only knows about the .cart format; for JSON/JSONL/pkl we
        # round-trip back into a fresh DecisionTree instead.
        if ext == ".cart":
            model_data = load_model(str(out))
            from_disk = predict_batch(model_data, X_test)
        else:
            reloaded = DecisionTree()
            reloaded.load_model(str(out))
            from_disk = reloaded.predict_batch(X_test)

        # .cart stores numeric thresholds as float32, so a few samples sitting
        # exactly on a split boundary can flip. JSON/JSONL/pkl are lossless.
        n = len(in_memory)
        agreements = sum(
            str(a) == str(b) for a, b in zip(in_memory, from_disk, strict=False)
        )
        if ext == ".cart":
            assert agreements / n > 0.95, (
                f"{ext} round-trip agreement too low: {agreements}/{n}"
            )
        else:
            assert agreements == n, (
                f"predictions diverged after {ext} round-trip ({agreements}/{n} agreed)"
            )

    def test_cross_validate_classification(self, iris_split):
        X_train, _, y_train, _, names = iris_split
        result = cross_validate(
            DecisionTree,
            X_train,
            y_train,
            n_folds=5,
            random_state=42,
            features=_numeric_specs(names),
            task="classification",
        )
        assert result["metric"] == "accuracy"
        assert result["n_folds"] == 5
        assert result["metric"] == "accuracy"
        assert len(result["scores"]) == 5
        assert result["mean"] > 0.85, (
            f"5-fold CV on iris should beat 0.85, got {result['mean']:.3f}"
        )


# =============================================================================
# Wine (multiclass classification) with forest
# =============================================================================


class TestWineForest:
    """RandomForest on wine, including feature importances."""

    @pytest.fixture
    def wine_split(self):
        data = load_wine()
        target_names = list(data.target_names)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.tolist(),
            [target_names[int(c)] for c in data.target.tolist()],
            test_size=0.3,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test, list(data.feature_names)

    def test_forest_classification_accuracy(self, wine_split):
        X_train, X_test, y_train, y_test, names = wine_split
        rf = RandomForest(
            n_estimators=20,
            features=_numeric_specs(names),
            task="classification",
        )
        rf.load_data(X_train, y_train)
        rf.train(random_state=42)
        metrics = evaluate_predictions(y_test, rf.predict_batch(X_test))
        assert metrics["accuracy"] > 0.9, (
            f"20-tree forest on wine should beat 0.9, got {metrics['accuracy']:.3f}"
        )

    def test_feature_importances_well_formed(self, wine_split):
        X_train, _, y_train, _, names = wine_split
        rf = RandomForest(
            n_estimators=20,
            features=_numeric_specs(names),
            task="classification",
        )
        rf.load_data(X_train, y_train)
        rf.train(random_state=42)

        importances = rf.feature_importances_
        assert set(importances.keys()) == set(names)
        assert all(v >= 0 for v in importances.values())
        # Importances are normalized
        assert abs(sum(importances.values()) - 1.0) < 1e-6
        # At least one feature should be informative
        assert max(importances.values()) > 0.05

    def test_forest_round_trip_cart(self, wine_split, tmp_path):
        X_train, X_test, y_train, _, names = wine_split
        rf = RandomForest(
            n_estimators=10, features=_numeric_specs(names), task="classification"
        )
        rf.load_data(X_train, y_train)
        rf.train(random_state=42)

        in_memory = rf.predict_batch(X_test)
        out = tmp_path / "forest.cart"
        rf.export(str(out))

        model_data = load_model(str(out))
        from_disk = predict_batch(model_data, X_test)
        assert from_disk == in_memory


# =============================================================================
# Breast cancer (binary classification) - nbest / proba
# =============================================================================


class TestBreastCancerProba:
    """Exercise probability / nbest paths on a binary classifier."""

    @pytest.fixture
    def bc_split(self):
        data = load_breast_cancer()
        target_names = list(data.target_names)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.tolist(),
            [target_names[int(c)] for c in data.target.tolist()],
            test_size=0.3,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test, list(data.feature_names)

    @pytest.fixture
    def bc_labels(self):
        return set(load_breast_cancer().target_names)

    def test_predict_with_confidence(self, bc_split, bc_labels):
        X_train, X_test, y_train, y_test, names = bc_split
        dt = DecisionTree(
            features=_numeric_specs(names),
            task="classification",
            store_distributions=True,
            min_confidence=1.0,  # always keep distributions
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        label, conf = dt.predict_with_confidence(X_test[0])
        assert label in bc_labels
        assert 0.0 <= conf <= 1.0

    def test_predict_nbest(self, bc_split, bc_labels):
        X_train, X_test, y_train, _, names = bc_split
        dt = DecisionTree(
            features=_numeric_specs(names),
            task="classification",
            store_distributions=True,
            min_confidence=1.0,
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        nbest = dt.predict_nbest(X_test[0], n=2)
        assert 1 <= len(nbest) <= 2
        # Sorted descending by probability
        if len(nbest) == 2:
            assert nbest[0][1] >= nbest[1][1]
        # Labels are valid class names
        for label, prob in nbest:
            assert label in bc_labels
            assert 0.0 <= prob <= 1.0

    def test_forest_predict_proba_sums_to_one(self, bc_split, bc_labels):
        X_train, X_test, y_train, _, names = bc_split
        rf = RandomForest(
            n_estimators=10,
            features=_numeric_specs(names),
            task="classification",
        )
        rf.load_data(X_train, y_train)
        rf.train(random_state=42)

        proba = rf.predict_proba(X_test[0])
        assert set(proba.keys()) <= bc_labels
        assert abs(sum(proba.values()) - 1.0) < 1e-6


# =============================================================================
# Diabetes (regression) - exercises the task-aware evaluate_tree / cross_validate
# =============================================================================


class TestDiabetesRegression:
    """Regression workflow on diabetes, including the task-aware eval helpers."""

    def test_decision_tree_regression(self, diabetes_split):
        X_train, X_test, y_train, y_test, names = diabetes_split
        dt = DecisionTree(
            features=_numeric_specs(names),
            task="regression",
            max_depth=5,  # cap depth to control variance for the test
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        preds = dt.predict_batch(X_test)
        # All predictions are finite numbers near the target range
        assert all(isinstance(p, (int, float)) for p in preds)
        assert min(preds) >= 0 and max(preds) <= 400

    def test_evaluate_tree_returns_regression_metrics(self, diabetes_split):
        """evaluate_tree should detect regression and return mse/mae/rmse, not accuracy."""
        X_train, X_test, y_train, y_test, names = diabetes_split
        dt = DecisionTree(
            features=_numeric_specs(names), task="regression", max_depth=5
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        metrics = evaluate_tree(dt, X_test, y_test)
        assert set(metrics.keys()) == {"task", "mse", "mae", "rmse", "total"}
        assert metrics["task"] == "regression"
        assert metrics["total"] == len(X_test)
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert abs(metrics["rmse"] - metrics["mse"] ** 0.5) < 1e-9

    def test_cross_validate_regression(self, diabetes_split):
        """cross_validate should use mse for regression."""
        X_train, _, y_train, _, names = diabetes_split
        result = cross_validate(
            DecisionTree,
            X_train,
            y_train,
            n_folds=3,
            random_state=42,
            features=_numeric_specs(names),
            task="regression",
            max_depth=5,
        )
        assert result["metric"] == "mse"
        assert result["n_folds"] == 3
        assert len(result["scores"]) == 3
        assert all(s > 0 for s in result["scores"])

    def test_regression_round_trip_cart(self, diabetes_split, tmp_path):
        X_train, X_test, y_train, _, names = diabetes_split
        dt = DecisionTree(
            features=_numeric_specs(names), task="regression", max_depth=5
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)

        in_memory = dt.predict_batch(X_test)
        out = tmp_path / "reg.cart"
        dt.export(str(out))

        model_data = load_model(str(out))
        from_disk = predict_batch(model_data, X_test)
        # .cart stores numeric values as float32, so allow ~1e-5 relative error
        # plus a small absolute floor for values that round to zero.
        assert len(from_disk) == len(in_memory)
        for a, b in zip(in_memory, from_disk, strict=False):
            assert abs(float(a) - float(b)) <= 1e-5 * abs(float(a)) + 1e-3


# =============================================================================
# CLI end-to-end using iris.csv
# =============================================================================


def _run_cli(args: list[str]) -> int:
    """Invoke cartlet.cli.main(args), restoring sys.argv afterwards."""
    saved = sys.argv
    try:
        sys.argv = ["cartlet", *args]
        return cli_main(args)
    finally:
        sys.argv = saved


class TestIrisCLI:
    """End-to-end CLI flows on a real CSV dumped from iris."""

    def test_train_then_predict_then_eval(self, iris_csv, tmp_path):
        model_path = tmp_path / "iris.cart"
        rc = _run_cli(
            [
                "train",
                iris_csv,
                "-o",
                str(model_path),
                "-t",
                "species",
                "-R",
                "42",
                "-S",
                "0.3",
            ]
        )
        assert rc == 0
        assert model_path.exists() and model_path.stat().st_size > 0

        # Predict on the same CSV (sanity: produces same number of rows)
        pred_out = tmp_path / "preds.csv"
        rc = _run_cli(
            [
                "predict",
                str(model_path),
                iris_csv,
                "-o",
                str(pred_out),
                "-m",
                "append",
                "-p",
                "predicted",
            ]
        )
        assert rc == 0
        with open(pred_out, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # 1 header row + 150 samples
        assert len(rows) == 151
        assert "predicted" in rows[0]

        # Evaluate against same labels
        eval_out = tmp_path / "eval.json"
        rc = _run_cli(
            [
                "eval",
                str(model_path),
                iris_csv,
                "-o",
                str(eval_out),
                "-t",
                "species",
                "-J",
            ]
        )
        assert rc == 0
        with open(eval_out, encoding="utf-8") as f:
            results = json.load(f)
        assert results["task"] == "classification"
        assert results["accuracy"] > 0.9

    def test_inspect_detects_numeric_features(self, iris_csv, tmp_path, capsys):
        rc = _run_cli(["inspect", iris_csv, "-t", "species", "-f", "simple"])
        assert rc == 0
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        # All four iris features are numerical floats
        assert out["sepal_length_(cm)"] == "num"
        assert out["sepal_width_(cm)"] == "num"
        assert out["petal_length_(cm)"] == "num"
        assert out["petal_width_(cm)"] == "num"
        # Target string says cat (the species column is integer-coded as strings)
        assert "_target" in out

    def test_convert_cart_to_json_and_back(self, iris_csv, tmp_path):
        cart_in = tmp_path / "iris.cart"
        rc = _run_cli(
            ["train", iris_csv, "-o", str(cart_in), "-t", "species", "-R", "42"]
        )
        assert rc == 0

        json_path = tmp_path / "iris.json"
        rc = _run_cli(["convert", str(cart_in), str(json_path)])
        assert rc == 0
        assert json_path.exists()

        cart_out = tmp_path / "iris-roundtrip.cart"
        rc = _run_cli(["convert", str(json_path), str(cart_out)])
        assert rc == 0

        # Predictions should match between original and round-tripped .cart
        orig = load_model(str(cart_in))
        rt = load_model(str(cart_out))
        data = load_iris()
        sample_vectors = data.data.tolist()[:20]
        assert predict_batch(orig, sample_vectors) == predict_batch(rt, sample_vectors)


# =============================================================================
# Custom suffix support via format= override (the phonebox `.g2p` case)
# =============================================================================


class TestCustomSuffixWorkflow:
    """Mirrors the phonebox flow: write a JSONL model under a custom suffix
    and read it back without temporary symlinks."""

    def _train_iris(self, iris_split):
        X_train, _, y_train, _, names = iris_split
        dt = DecisionTree(
            features=_numeric_specs(names), task="classification", max_depth=4
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)
        return dt

    def test_export_load_with_custom_suffix(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        custom_path = tmp_path / "model.g2p.gz"
        dt.export(str(custom_path), format="jsonl")
        assert custom_path.exists()

        loaded = DecisionTree()
        loaded.load_model(str(custom_path), format="jsonl")

        X_test = iris_split[1]
        assert dt.predict_batch(X_test) == loaded.predict_batch(X_test)

    def test_convert_from_custom_suffix(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        custom_path = tmp_path / "model.g2p.gz"
        dt.export(str(custom_path), format="jsonl")

        cart_out = tmp_path / "model.cart"
        convert(str(custom_path), str(cart_out), input_format="jsonl")
        assert cart_out.exists()

        loaded = load_model(str(cart_out))
        X_test = iris_split[1]
        assert dt.predict_batch(X_test) == predict_batch(loaded, X_test)

    def test_bundle_accepts_jsonl(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        json_model = tmp_path / "model.jsonl"
        dt.export(str(json_model))

        bundle_out = tmp_path / "bundled.py"
        bundle(str(json_model), str(bundle_out))
        assert bundle_out.exists()
        assert bundle_out.stat().st_size > 0

    def test_bundle_accepts_custom_suffix(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        custom_path = tmp_path / "model.g2p"
        dt.export(str(custom_path), format="jsonl")

        bundle_out = tmp_path / "bundled.py"
        bundle(str(custom_path), str(bundle_out), model_format="jsonl")
        assert bundle_out.exists()

    def test_predictor_get_vocabulary_and_is_oov(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        cart_path = tmp_path / "iris.cart"
        dt.export(str(cart_path))

        p = Predictor(str(cart_path))
        # Iris features are all numeric so vocabulary is None.
        for name in p.feature_names:
            assert p.get_vocabulary(name) is None
            assert p.is_oov(name, "anything") is False

    def test_unknown_format_raises(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        with pytest.raises(ValueError, match="Unknown format"):
            dt.export(str(tmp_path / "x.weird"), format="not-a-format")


class TestMetadataRoundTrip:
    """The .cart trailer carries arbitrary JSON metadata; loaders should
    surface it instead of dropping it on the floor."""

    def _train_iris(self, iris_split):
        X_train, _, y_train, _, names = iris_split
        dt = DecisionTree(
            features=_numeric_specs(names), task="classification", max_depth=4
        )
        dt.load_data(X_train, y_train)
        dt.train(random_state=42)
        return dt

    def test_predictor_exposes_metadata(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        cart_path = tmp_path / "iris.cart"
        meta = {"locale": "en", "width": 3, "training_config": {"max_depth": 4}}
        dt.export(str(cart_path), metadata=meta)

        p = Predictor(str(cart_path))
        assert p.metadata == meta

    def test_read_cart_metadata_helper(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        cart_path = tmp_path / "iris.cart"
        meta = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        dt.export(str(cart_path), metadata=meta)

        assert read_cart_metadata(str(cart_path)) == meta
        with open(cart_path, "rb") as f:
            blob = f.read()
        assert read_cart_metadata(blob) == meta

    def test_metadata_absent_returns_empty(self, iris_split, tmp_path):
        dt = self._train_iris(iris_split)
        cart_path = tmp_path / "iris.cart"
        dt.export(str(cart_path))
        assert Predictor(str(cart_path)).metadata == {}
        assert read_cart_metadata(str(cart_path)) == {}
