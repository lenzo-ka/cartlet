"""Tests for the minimal runner and bundling."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from cartlet import TASK_REGRESSION, DecisionTree, RandomForest
from cartlet.io.bytes import bundle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_PY = os.path.join(REPO_ROOT, "cartlet", "bundled", "predict.py")


def _load_bundled_module():
    """Import cartlet/bundled/predict.py as a module."""
    spec = importlib.util.spec_from_file_location("bundled_predict", PREDICT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPythonRunner:
    """Tests for bundled/predict.py runner."""

    @pytest.fixture
    def model_path(self, tmp_path):
        """Create a trained model file."""
        dt = DecisionTree()
        X = [["a", 1.0], ["b", 2.0], ["a", 3.0], ["b", 4.0]]
        y = ["yes", "no", "yes", "no"]
        dt.load_data(X, y)
        dt.train()

        path = tmp_path / "model.cart"
        dt.export(str(path))
        return str(path)

    def test_predict_single(self, model_path):
        """Predict a single vector via CLI."""
        result = subprocess.run(
            [
                sys.executable,
                "cartlet/bundled/predict.py",
                "-m",
                model_path,
                "a",
                "1.0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "yes" in result.stdout

    def test_predict_from_stdin(self, model_path, tmp_path):
        """Predict from stdin."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("a,1.0\nb,2.0\n")

        with open(input_file) as stdin:
            result = subprocess.run(
                [
                    sys.executable,
                    "cartlet/bundled/predict.py",
                    "-m",
                    model_path,
                    "-f",
                    "-",
                ],
                stdin=stdin,
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 2

    @pytest.fixture
    def numeric_model_path(self, tmp_path):
        """A model whose sole feature is explicitly numeric."""
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
        )
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = ["low", "low", "high", "high"]
        dt.load_data(X, y)
        dt.train()
        path = tmp_path / "num.cart"
        dt.export(str(path))
        return str(path)

    def test_non_numeric_value_clean_error_single(self, numeric_model_path):
        """A non-numeric value for a numeric feature -> clean error, not a
        raw traceback (W1-L9)."""
        result = subprocess.run(
            [
                sys.executable,
                "cartlet/bundled/predict.py",
                "-m",
                numeric_model_path,
                "notanumber",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "notanumber" in result.stderr
        assert "Traceback" not in result.stderr

    def test_non_numeric_value_clean_error_batch(self, numeric_model_path, tmp_path):
        """Batch mode also reports a clean error on a bad numeric cell."""
        input_file = tmp_path / "bad.txt"
        input_file.write_text("notanumber\n")
        result = subprocess.run(
            [
                sys.executable,
                "cartlet/bundled/predict.py",
                "-m",
                numeric_model_path,
                "-f",
                str(input_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "Traceback" not in result.stderr

    def test_predict_from_file(self, model_path, tmp_path):
        """Predict from file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("a,1.0\nb,2.0\n")

        result = subprocess.run(
            [
                sys.executable,
                "cartlet/bundled/predict.py",
                "-m",
                model_path,
                "-f",
                str(input_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 2

    def test_info_flag(self, model_path):
        """Show model info."""
        result = subprocess.run(
            [sys.executable, "cartlet/bundled/predict.py", "-m", model_path, "-i"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "features" in result.stdout.lower() or "class" in result.stdout.lower()


class TestPythonRunnerLibrary:
    """Tests for bundled/predict.py as a library."""

    def test_predictor_class(self, tmp_path):
        """Use Predictor class directly."""
        dt = DecisionTree()
        X = [["a", 1.0], ["b", 2.0]]
        y = ["yes", "no"]
        dt.load_data(X, y)
        dt.train()

        path = tmp_path / "model.cart"
        dt.export(str(path))

        predict_module = _load_bundled_module()
        predictor = predict_module.Predictor(str(path))
        assert predictor.predict(["a", 1.0]) == "yes"
        assert predictor.predict(["b", 2.0]) == "no"

    def test_predictor_parity_helpers(self, tmp_path):
        """
        Bundled Predictor mirrors cartlet.runner.Predictor for metadata /
        get_vocabulary / is_oov so users that import the bundled file get
        the same OO surface phonebox-style callers expect.
        """
        dt = DecisionTree(
            features=[{"name": "color", "dtype": "str", "type": "cat"}],
        )
        dt.load_data([["red"], ["blue"], ["red"], ["green"]], ["a", "b", "a", "c"])
        dt.train()

        path = tmp_path / "model.cart"
        dt.export(str(path), metadata={"locale": "en"})

        predict_module = _load_bundled_module()
        predictor = predict_module.Predictor(str(path))

        assert predictor.metadata == {"locale": "en"}
        assert predictor.get_vocabulary("color") == {"red", "blue", "green"}
        assert predictor.get_vocabulary(0) == {"red", "blue", "green"}
        assert predictor.get_vocabulary("missing") is None
        assert predictor.is_oov("color", "red") is False
        assert predictor.is_oov("color", "chartreuse") is True
        assert predictor.is_oov("missing", "anything") is False


class TestBundling:
    """Tests for bundling models into executables."""

    @pytest.fixture
    def model_path(self, tmp_path):
        """Create a trained model file."""
        dt = DecisionTree()
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]
        y = ["a", "a", "b", "b"]
        dt.load_data(X, y)
        dt.train()

        path = tmp_path / "model.cart"
        dt.export(str(path))
        return str(path)

    def test_bundle_python(self, model_path, tmp_path):
        """Bundle model into Python executable."""
        output = tmp_path / "bundled.py"
        bundle(model_path, str(output))

        assert os.path.exists(str(output))
        assert os.path.getsize(str(output)) > 0

        # Run bundled executable
        result = subprocess.run(
            [sys.executable, str(output), "1.0", "2.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() in ["a", "b"]

    def test_bundle_python_explicit_model(self, model_path, tmp_path):
        """Bundled Python can still load external model."""
        output = tmp_path / "bundled.py"
        bundle(model_path, str(output))

        # Create a different model
        dt2 = DecisionTree()
        X = [[1.0], [2.0]]
        y = ["x", "y"]
        dt2.load_data(X, y)
        dt2.train()
        alt_path = tmp_path / "alt.cart"
        dt2.export(str(alt_path))

        # Run with explicit model
        result = subprocess.run(
            [sys.executable, str(output), "-m", str(alt_path), "1.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() in ["x", "y"]


class TestCrossRunnerConsistency:
    """Tests for consistency across runners."""

    @pytest.fixture
    def model_and_data(self, tmp_path):
        """Create model and test vectors."""
        dt = DecisionTree()
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0], [5.0, 5.0]]
        y = ["a", "a", "b", "b", "a"]
        dt.load_data(X, y)
        dt.train()

        path = tmp_path / "model.cart"
        dt.export(str(path))

        # Create test vectors
        test_vectors = [[1.0, 2.0], [3.0, 1.0], [5.0, 5.0]]
        return str(path), test_vectors

    def test_python_vs_library(self, model_and_data):
        """Python runner matches library predictions."""
        from cartlet.runner import load_model, predict_batch

        model_path, test_vectors = model_and_data

        # Library predictions
        model = load_model(model_path)
        lib_preds = predict_batch(model, test_vectors)

        # Runner predictions
        for vec, expected in zip(test_vectors, lib_preds, strict=False):
            args = [sys.executable, "cartlet/bundled/predict.py", "-m", model_path] + [
                str(v) for v in vec
            ]
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == str(expected)

    def test_forest_and_dist_parity(self, tmp_path):
        """Random forest and distribution parity."""
        rf = RandomForest(n_estimators=3, feature_names=["color", "size"])
        X = [["red", "small"], ["blue", "large"]] * 10
        y = ["apple", "ball"] * 10
        rf.load_data(X, y)
        rf.train()

        path = str(tmp_path / "forest.cart")
        rf.export(path)

        bundled_module = _load_bundled_module()

        from cartlet.runner import Predictor as LibPredictor

        lib_p = LibPredictor(path)
        std_p = bundled_module.Predictor(path)

        test_vec = ["red", "small"]
        assert lib_p.predict(test_vec) == std_p.predict(test_vec)
        assert lib_p.predict(test_vec, return_dist=True) == std_p.predict(
            test_vec, return_dist=True
        )
        assert lib_p.feature_names == std_p.feature_names
        assert lib_p.class_labels == std_p.class_labels

    def test_regression_and_missing_parity(self, tmp_path):
        """Regression and missing value parity."""
        dt = DecisionTree(feature_names=["a", "b"], task=TASK_REGRESSION)
        X = [[1.0, 1.0], [2.0, 2.0]]
        y = [10.0, 20.0]
        dt.load_data(X, y)
        dt.train()

        path = str(tmp_path / "reg.cart")
        dt.export(path)

        bundled_module = _load_bundled_module()

        from cartlet.runner import Predictor as LibPredictor

        lib_p = LibPredictor(path)
        std_p = bundled_module.Predictor(path)

        # Missing value (short vector)
        test_vec = [1.0]
        assert abs(lib_p.predict(test_vec) - std_p.predict(test_vec)) < 1e-6

        # Explicit None
        test_vec = [1.0, None]
        assert abs(lib_p.predict(test_vec) - std_p.predict(test_vec)) < 1e-6


class TestOOVCategorical:
    """Tests for out-of-vocabulary categorical values."""

    @pytest.fixture
    def cat_model(self, tmp_path):
        """Classification tree with categorical features."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        dt.load_data(X, y)
        dt.train()

        path = str(tmp_path / "model.cart")
        dt.export(path)
        return path

    def test_oov_via_library_runner(self, cat_model):
        """OOV categorical value produces a valid prediction via library runner."""
        from cartlet.runner import load_model, predict

        model = load_model(cat_model)
        result = predict(model, ["green", "small"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_oov_via_standalone_cli(self, cat_model):
        """OOV categorical value produces a valid prediction via standalone CLI."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", cat_model, "green", "small"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() in ["apple", "ball", "box"]

    def test_oov_consistent_across_runners(self, cat_model):
        """OOV values produce identical predictions in library and standalone runners."""
        from cartlet.runner import load_model, predict

        model = load_model(cat_model)
        bundled = _load_bundled_module()

        for oov_vec in [["green", "small"], ["red", "huge"], ["green", "huge"]]:
            lib_result = predict(model, oov_vec)
            std_result = bundled.predict(bundled.load_cart(cat_model), oov_vec)
            assert lib_result == std_result, f"Mismatch for {oov_vec}"

    def test_is_oov_api(self, cat_model):
        """get_vocabulary and is_oov correctly identify OOV values."""
        from cartlet.runner import get_vocabulary, is_oov, load_model

        model = load_model(cat_model)

        vocab = get_vocabulary(model, "color")
        assert vocab is not None
        assert "red" in vocab
        assert "blue" in vocab
        assert "green" not in vocab

        assert is_oov(model, "color", "green")
        assert not is_oov(model, "color", "red")


class TestDistCLI:
    """Tests for --dist flag on standalone runner CLI."""

    @pytest.fixture
    def dist_model(self, tmp_path):
        """Classification tree with impure leaves that store distributions."""
        dt = DecisionTree(
            feature_names=["color", "size"],
            store_distributions=True,
            min_dist_entropy=0.0,
            max_depth=1,
        )
        X = [
            ["red", "small"],
            ["red", "small"],
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "banana", "apple", "cherry", "ball", "box"] * 10
        dt.load_data(X, y)
        dt.train()

        path = str(tmp_path / "model.cart")
        dt.export(path, store_distributions=True)
        return path

    def test_dist_single_prediction(self, dist_model):
        """--dist flag returns valid JSON distribution."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", dist_model, "--dist", "red", "small"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        dist = json.loads(result.stdout.strip())
        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert all(isinstance(v, float) for v in dist.values())

    def test_dist_batch_from_file(self, dist_model, tmp_path):
        """--dist flag works in batch mode with file input."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("red,small\nblue,large\n")

        result = subprocess.run(
            [
                sys.executable,
                PREDICT_PY,
                "-m",
                dist_model,
                "--dist",
                "-f",
                str(input_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            dist = json.loads(line)
            assert isinstance(dist, dict)

    def test_dist_parity_with_library(self, dist_model):
        """--dist output matches library runner return_dist=True."""
        from cartlet.runner import Predictor as LibPredictor

        bundled = _load_bundled_module()
        lib_p = LibPredictor(dist_model)
        std_p = bundled.Predictor(dist_model)

        test_vec = ["red", "small"]
        lib_dist = lib_p.predict(test_vec, return_dist=True)
        std_dist = std_p.predict(test_vec, return_dist=True)

        assert isinstance(lib_dist, dict)
        assert isinstance(std_dist, dict)
        assert set(lib_dist.keys()) == set(std_dist.keys())
        for k in lib_dist:
            assert abs(lib_dist[k] - std_dist[k]) < 1e-6


class TestRegressionCLI:
    """Tests for regression output via standalone runner CLI."""

    @pytest.fixture
    def reg_model(self, tmp_path):
        """Regression tree exported to .cart."""
        dt = DecisionTree(
            features=[
                {"name": "x", "dtype": "float", "type": "num"},
                {"name": "y", "dtype": "float", "type": "num"},
            ],
            task=TASK_REGRESSION,
        )
        X = [[float(i), float(i * 2)] for i in range(20)]
        y = [float(i * 3 + 1) for i in range(20)]
        dt.load_data(X, y)
        dt.train()

        path = str(tmp_path / "reg.cart")
        dt.export(path)
        return path

    def test_regression_single(self, reg_model):
        """Single regression prediction returns a valid float."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", reg_model, "5.0", "10.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        val = float(result.stdout.strip())
        assert isinstance(val, float)

    def test_regression_batch_from_file(self, reg_model, tmp_path):
        """Batch regression prediction from file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("1.0\t2.0\n5.0\t10.0\n15.0\t30.0\n")

        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", reg_model, "-f", str(input_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            float(line)  # Should not raise

    def test_regression_value_sanity(self, reg_model):
        """Regression prediction is in a reasonable range."""
        from cartlet.runner import load_model, predict

        model = load_model(reg_model)
        result = predict(model, [5.0, 10.0])
        assert 1.0 <= result <= 60.0


class TestErrorPaths:
    """Tests for error handling in the standalone runner."""

    def test_invalid_model_file(self, tmp_path):
        """Non-.cart file is rejected with a clear error."""
        bad = tmp_path / "garbage.cart"
        bad.write_bytes(b"NOT_CART_DATA_AT_ALL")

        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", str(bad), "a", "b"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_truncated_model_file(self, tmp_path):
        """Truncated .cart file is rejected."""
        bad = tmp_path / "truncated.cart"
        bad.write_bytes(b"CART" + b"\x00" * 10)

        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", str(bad), "a"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_nonexistent_model_file(self):
        """Missing model file produces an error."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", "/nonexistent/model.cart", "a"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_no_model_no_features(self):
        """No model and no features prints usage error."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_no_features_provided(self, tmp_path):
        """Model given but no features prints error."""
        dt = DecisionTree(feature_names=["x"])
        dt.load_data([["a"], ["b"]], ["1", "2"])
        dt.train()

        path = str(tmp_path / "model.cart")
        dt.export(path)

        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", path],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_library_load_invalid_bytes(self):
        """Library runner raises ValueError on invalid bytes."""
        from cartlet.runner import Predictor

        with pytest.raises(ValueError):
            Predictor(b"NOT_CART_DATA")

    def test_library_load_truncated_bytes(self):
        """Library runner raises ValueError on truncated bytes."""
        from cartlet.runner import Predictor

        with pytest.raises(ValueError):
            Predictor(b"CART" + b"\x00" * 5)


class TestForestRegression:
    """Tests for forest regression via the runner."""

    @pytest.fixture
    def forest_reg_model(self, tmp_path):
        """Random forest regression model."""
        rf = RandomForest(
            n_estimators=5,
            features=[
                {"name": "x", "dtype": "float", "type": "num"},
                {"name": "y", "dtype": "float", "type": "num"},
            ],
            task=TASK_REGRESSION,
        )
        X = [[float(i), float(i**2)] for i in range(30)]
        y = [float(i * 2 + 0.5) for i in range(30)]
        rf.load_data(X, y)
        rf.train(random_state=42)

        path = str(tmp_path / "forest_reg.cart")
        rf.export(path)
        return path

    def test_forest_regression_via_library(self, forest_reg_model):
        """Forest regression prediction via library runner."""
        from cartlet.runner import load_model, predict

        model = load_model(forest_reg_model)
        result = predict(model, [5.0, 25.0])
        assert isinstance(result, float)

    def test_forest_regression_via_cli(self, forest_reg_model):
        """Forest regression prediction via standalone CLI."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", forest_reg_model, "5.0", "25.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        val = float(result.stdout.strip())
        assert isinstance(val, float)

    def test_forest_regression_parity(self, forest_reg_model):
        """Library and standalone runners agree on forest regression."""
        from cartlet.runner import load_model, predict

        model = load_model(forest_reg_model)
        bundled = _load_bundled_module()

        for vec in [[5.0, 25.0], [10.0, 100.0], [20.0, 400.0]]:
            lib_result = predict(model, vec)
            std_result = bundled.predict(bundled.load_cart(forest_reg_model), vec)
            assert abs(lib_result - std_result) < 1e-6, f"Mismatch for {vec}"


class TestNoDistributions:
    """Tests for models exported with store_distributions=False."""

    @pytest.fixture
    def no_dist_model(self, tmp_path):
        """Classification tree exported without distributions."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        dt.load_data(X, y)
        dt.train()

        path = str(tmp_path / "no_dist.cart")
        dt.export(path, store_distributions=False)
        return path

    def test_prediction_still_works(self, no_dist_model):
        """Prediction works on model without distributions."""
        from cartlet.runner import load_model, predict

        model = load_model(no_dist_model)
        result = predict(model, ["red", "small"])
        assert isinstance(result, str)
        assert result in ["apple", "ball", "box"]

    def test_cli_prediction_works(self, no_dist_model):
        """CLI prediction works on model without distributions."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", no_dist_model, "red", "small"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() in ["apple", "ball", "box"]

    def test_dist_flag_does_not_crash(self, no_dist_model):
        """--dist flag on a no-distribution model does not crash."""
        result = subprocess.run(
            [sys.executable, PREDICT_PY, "-m", no_dist_model, "--dist", "red", "small"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout.strip())
        # Without stored distributions, the runner returns the class label as a string
        assert isinstance(output, str)

    def test_no_dist_smaller_file(self, tmp_path):
        """Model without distributions is smaller or equal to model with."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        dt.load_data(X, y)
        dt.train()

        with_path = str(tmp_path / "with.cart")
        without_path = str(tmp_path / "without.cart")
        dt.export(with_path, store_distributions=True)
        dt.export(without_path, store_distributions=False)

        assert os.path.getsize(without_path) <= os.path.getsize(with_path)

    def test_parity_with_dist_model(self, tmp_path):
        """Best-class prediction is identical whether distributions are stored or not."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        dt.load_data(X, y)
        dt.train()

        with_path = str(tmp_path / "with.cart")
        without_path = str(tmp_path / "without.cart")
        dt.export(with_path, store_distributions=True)
        dt.export(without_path, store_distributions=False)

        from cartlet.runner import load_model, predict

        model_with = load_model(with_path)
        model_without = load_model(without_path)

        for vec in [["red", "small"], ["blue", "large"], ["red", "large"]]:
            assert predict(model_with, vec) == predict(model_without, vec)
