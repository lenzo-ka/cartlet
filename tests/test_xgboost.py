"""Tests for XGBoost integration."""

import os

import pytest

# Skip all tests if xgboost is unavailable. importorskip only catches
# ImportError, but a present-but-broken xgboost (e.g. missing libomp on macOS)
# raises XGBoostError at import time, so catch any exception and skip.
try:
    import xgboost  # noqa: F401
except Exception:  # pragma: no cover - environment-dependent
    pytest.skip("xgboost unavailable", allow_module_level=True)

from cartlet import XGBoostTree  # noqa: E402


class TestXGBoostTree:
    """Tests for XGBoostTree class."""

    @pytest.fixture
    def classification_data(self):
        """Simple classification dataset."""
        X = [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
            ["green", "small"],
            ["green", "large"],
        ] * 10  # Repeat for sufficient samples
        y = ["a", "b", "a", "b", "a", "b"] * 10
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Simple regression dataset."""
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 10
        y = [1.5, 2.5, 3.5, 4.5] * 10
        return X, y

    @pytest.fixture
    def mixed_data(self):
        """Dataset with both categorical and numerical features."""
        X = [
            ["red", 1.0],
            ["red", 2.0],
            ["blue", 1.0],
            ["blue", 2.0],
            ["green", 1.0],
            ["green", 2.0],
        ] * 10
        y = ["yes", "no", "yes", "no", "yes", "no"] * 10
        return X, y

    def test_train_classification(self, classification_data):
        """Test training a classification model."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        result = xgb_model.train(random_state=42)

        assert "n_trees" in result
        assert result["n_trees"] > 0
        assert xgb_model._xgb_model is not None
        # The trained model must actually predict a valid class label.
        assert xgb_model.predict(["red", "small"]) in ("a", "b")

    def test_train_regression(self, regression_data):
        """Test training a regression model."""
        X, y = regression_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["f0", "f1"],
            task="regression",
        )
        xgb_model.load_data(X, y)
        result = xgb_model.train(random_state=42)

        assert "n_trees" in result
        assert result["n_trees"] > 0
        # Prediction is a float roughly within the training-target range [1.5, 4.5].
        pred = xgb_model.predict([2.0, 3.0])
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 6.0

    def test_predict_classification(self, classification_data):
        """Test prediction for classification."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        pred = xgb_model.predict(["red", "small"])
        assert pred in ["a", "b"]

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        proba = xgb_model.predict_proba(["red", "small"])
        assert isinstance(proba, dict)
        assert len(proba) == 2  # Binary classification
        assert abs(sum(proba.values()) - 1.0) < 0.001  # Probabilities sum to 1

    def test_export_cart(self, classification_data, tmp_path):
        """Test exporting to .cart format."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "m.cart")
        xgb_model.export(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Roundtrip: the runner reading the .cart must agree with the model's
        # own predictions on every training row (classification labels).
        from cartlet import Predictor

        runner = Predictor(path)
        for row in X:
            assert runner.predict(row) == xgb_model.predict(row)

    def test_export_xgb(self, classification_data, tmp_path):
        """Test exporting to native .xgb format."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "m.xgb")
        xgb_model.export(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_load_xgb(self, classification_data, tmp_path):
        """Test loading from native .xgb format."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "m.xgb")
        xgb_model.export(path)

        # Load it back
        loaded = XGBoostTree.load(path)
        assert loaded._xgb_model is not None

    def test_mixed_features(self, mixed_data):
        """Test with mixed categorical and numerical features."""
        X, y = mixed_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "value"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        pred = xgb_model.predict(["red", 1.5])
        assert pred in ["yes", "no"]

    def test_cart_format_runner_compatible(self, classification_data, tmp_path):
        """Test that .cart export can be loaded by minimal runner."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["color", "size"],
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "m.cart")
        xgb_model.export(path)

        # Try to load with minimal runner
        from cartlet import load_model

        model = load_model(path)
        assert model is not None
        assert model.get("is_xgboost", False) is True
        # base_score round-trips through the trailing metadata blob
        assert model["meta"]["metadata"].get("base_score") == xgb_model.base_score

    def test_runner_base_score_parity_regression(self, regression_data, tmp_path):
        """Runner predictions must match XGBoostTree.predict for regression,
        which exercises the base_score additive offset (the runner used to
        hardcode 0.0 here)."""
        X, y = regression_data
        xgb_model = XGBoostTree(
            n_estimators=10,
            max_depth=3,
            feature_names=["f0", "f1"],
            task="regression",
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "m.cart")
        xgb_model.export(path)
        from cartlet import Predictor

        p = Predictor(path)
        for x in X[:10]:
            native = xgb_model.predict(x)
            runner = p.predict(x)
            # XGBoost adds base_score as raw-score offset; values can
            # differ slightly due to float32 round-trip in .cart.
            assert abs(native - runner) < 1e-3, (
                f"diverged for {x}: native={native} runner={runner}"
            )

    def test_xgboost_export_rejects_unsupported_extensions(
        self, classification_data, tmp_path
    ):
        """Exporting an XGBoostTree to .pkl/.skl/.jsonl should fail loudly,
        not silently write .cart bytes into a misnamed file."""
        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=5, max_depth=3, feature_names=["color", "size"]
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        for bad in ("model.pkl", "model.skl", "model.jsonl"):
            with pytest.raises(ValueError, match="XGBoostTree cannot export"):
                xgb_model.export(str(tmp_path / bad))

    def test_decisiontree_load_xgb_cart_raises(self, classification_data, tmp_path):
        """DecisionTree can't represent multi-tree XGBoost models — loading
        one should raise a clear error pointing at Predictor / XGBoostTree."""
        from cartlet import DecisionTree

        X, y = classification_data
        xgb_model = XGBoostTree(
            n_estimators=5, max_depth=3, feature_names=["color", "size"]
        )
        xgb_model.load_data(X, y)
        xgb_model.train(random_state=42)

        path = str(tmp_path / "xgb.cart")
        xgb_model.export(path)

        with pytest.raises(ValueError, match="XGBoost .cart export"):
            DecisionTree().load_model(path)
