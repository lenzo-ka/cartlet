"""Tests for binary .cart format."""

import os
import subprocess
import sys
import tempfile

import pytest

from cartlet import TASK_REGRESSION, DecisionTree, RandomForest


class TestBytesExport:
    """Test binary format export."""

    def test_tree_classification_export(self):
        """Export classification tree to .cart format."""
        dt = DecisionTree(feature_names=["color", "size"])
        X = [["red", "small"], ["red", "large"], ["blue", "small"], ["blue", "large"]]
        y = ["apple", "apple", "ball", "box"]
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)
            assert os.path.exists(path)
            # Check magic bytes
            with open(path, "rb") as f:
                assert f.read(4) == b"CART"

    def test_tree_regression_export(self):
        """Export regression tree to .cart format."""
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[float(i)] for i in range(10)]
        y = [float(i * 2) for i in range(10)]
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)
            assert os.path.exists(path)

    def test_forest_classification_export(self):
        """Export classification forest to .cart format."""
        rf = RandomForest(n_estimators=5, feature_names=["color", "size"])
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
        ] * 10
        y = ["apple", "apple", "ball", "box"] * 10
        rf.load_data(X, y)
        rf.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.cart")
            rf.export(path)
            assert os.path.exists(path)
            with open(path, "rb") as f:
                magic = f.read(4)
                assert magic == b"CART"
                # Check is_forest flag
                f.read(2)  # version
                flags = int.from_bytes(f.read(2), "little")
                assert flags & 1  # is_forest flag set

    def test_auto_detect_bytes_format(self):
        """Export to .cart should auto-detect bytes format."""
        dt = DecisionTree(feature_names=["x"])
        dt.load_data([["a"], ["b"]], ["1", "2"])
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)  # no format specified, should detect from extension
            with open(path, "rb") as f:
                assert f.read(4) == b"CART"


class TestPythonRunner:
    """Test Python runner on exported models."""

    @pytest.fixture
    def runner_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "cartlet", "bundled", "predict.py"
        )

    def test_classification_tree(self, runner_path):
        """Python runner predicts correctly on classification tree."""
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

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)

            # Test predictions
            result = subprocess.run(
                [sys.executable, runner_path, path, "red", "small"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == "apple"

            result = subprocess.run(
                [sys.executable, runner_path, path, "blue", "large"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == "box"

    def test_numerical_features(self, runner_path):
        """Python runner handles numerical features."""
        dt = DecisionTree(
            features=[
                {"name": "age", "dtype": "int", "type": "num"},
                {"name": "income", "dtype": "float", "type": "num"},
            ],
        )
        X = [
            [25, 30000],
            [35, 50000],
            [55, 80000],
            [65, 90000],
        ] * 10
        y = ["young", "young", "old", "old"] * 10
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)

            result = subprocess.run(
                [sys.executable, runner_path, path, "30", "40000"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == "young"

    def test_forest_classification(self, runner_path):
        """Python runner handles forest classification."""
        rf = RandomForest(n_estimators=10, feature_names=["x"])
        X = [["a"], ["a"], ["b"], ["b"]] * 20
        y = ["yes", "yes", "no", "no"] * 20
        rf.load_data(X, y)
        rf.train(random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.cart")
            rf.export(path)

            result = subprocess.run(
                [sys.executable, runner_path, path, "a"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == "yes"


class TestBundledRunner:
    """Test bundled runner (model embedded in script)."""

    def test_bundle_python(self):
        """Bundle model with Python runner."""
        from cartlet.io.bytes import bundle

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

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.cart")
            bundle_path = os.path.join(tmpdir, "predictor.py")

            dt.export(model_path)
            bundle(model_path, bundle_path)

            assert os.path.exists(bundle_path)

            # Bundled runner takes features directly (no model path)
            result = subprocess.run(
                [sys.executable, bundle_path, "red", "small"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip() == "apple"

    def test_strip_cli_preserves_predict_py_layout(self):
        """_strip_cli_code must keep every top-level library symbol and drop
        only the CLI (W1-L4).

        The resume-on-top-level-def/class rule silently deletes anything that
        follows ``def main(`` up to EOF, so if predict.py ever grows a
        top-level symbol below main() this pins the layout and fails loudly.
        """
        import ast
        import pathlib

        from cartlet.io.bytes import _strip_cli_code

        src = pathlib.Path("cartlet/bundled/predict.py").read_bytes()
        stripped = _strip_cli_code(src)

        # Result must still be valid Python.
        tree = ast.parse(stripped)
        stripped_defs = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

        orig = ast.parse(src)
        orig_defs = {
            n.name for n in orig.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

        # Exactly main() is removed; every other top-level def/class survives.
        assert "main" in orig_defs
        assert stripped_defs == orig_defs - {"main"}

        # Core library API is intact; CLI entrypoint is gone.
        for sym in ("Predictor", "load_cart", "load_cart_from_bytes", "predict"):
            assert sym in stripped_defs
        text = stripped.decode("utf-8")
        assert "def main(" not in text
        assert '__name__ == "__main__"' not in text


class TestPythonRunnerRegression:
    """Verify the Python runner handles regression trees end-to-end."""

    @pytest.fixture
    def runner_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "cartlet", "bundled", "predict.py"
        )

    def test_regression_tree(self, runner_path):
        """Python runner returns near-monotone predictions on a regression tree."""
        dt = DecisionTree(
            features=[{"name": "x", "dtype": "float", "type": "num"}],
            task=TASK_REGRESSION,
        )
        X = [[float(i)] for i in range(50)]
        y = [float(i * 2 + 1) for i in range(50)]
        dt.load_data(X, y)
        dt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.cart")
            dt.export(path)

            preds = []
            for features in (["5.0"], ["15.0"], ["25.0"], ["45.0"]):
                result = subprocess.run(
                    [sys.executable, runner_path, path] + features,
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == 0, result.stderr
                preds.append(float(result.stdout.strip()))

            assert preds == sorted(preds)
