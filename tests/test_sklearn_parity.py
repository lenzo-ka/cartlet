"""Tests verifying native RandomForest performs similarly to sklearn trainer."""

import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import (  # noqa: E402
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from cartlet import TASK_REGRESSION, RandomForest  # noqa: E402


class TestClassificationParity:
    """Verify native and sklearn trainers produce similar classification results."""

    @pytest.mark.parametrize(
        "loader,name",
        [
            (load_iris, "iris"),
            (load_wine, "wine"),
            (load_breast_cancer, "breast_cancer"),
        ],
    )
    def test_classification_accuracy_parity(self, loader, name):
        """Both trainers should achieve similar accuracy on standard datasets."""
        data = loader()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.tolist(),
            [str(c) for c in data.target.tolist()],
            test_size=0.3,
            random_state=42,
        )

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Train with native
        rf_native = RandomForest(
            n_estimators=50,
            max_depth=10,
            features=feature_specs,
        )
        rf_native.load_data(X_train, y_train)
        rf_native.train(trainer="native", random_state=42)
        native_preds = rf_native.predict_batch(X_test)
        native_acc = accuracy_score(y_test, native_preds)

        # Train with sklearn
        rf_sklearn = RandomForest(
            n_estimators=50,
            max_depth=10,
            features=feature_specs,
        )
        rf_sklearn.load_data(X_train, y_train)
        rf_sklearn.train(trainer="sklearn", random_state=42)
        sklearn_preds = rf_sklearn.predict_batch(X_test)
        sklearn_acc = accuracy_score(y_test, sklearn_preds)

        # Both should achieve reasonable accuracy (>70%)
        assert native_acc > 0.70, f"Native accuracy too low on {name}: {native_acc:.2%}"
        assert sklearn_acc > 0.70, (
            f"Sklearn accuracy too low on {name}: {sklearn_acc:.2%}"
        )

        # Accuracies should be within 15% of each other
        diff = abs(native_acc - sklearn_acc)
        assert diff < 0.15, (
            f"Accuracy gap too large on {name}: "
            f"native={native_acc:.2%}, sklearn={sklearn_acc:.2%}, diff={diff:.2%}"
        )


class TestRegressionParity:
    """Verify native and sklearn trainers produce similar regression results."""

    def test_diabetes_r2_parity(self):
        """Both trainers should achieve similar R^2 on diabetes dataset."""
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.tolist(),
            data.target.tolist(),
            test_size=0.3,
            random_state=42,
        )

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Train with native
        rf_native = RandomForest(
            n_estimators=50,
            max_depth=10,
            features=feature_specs,
            task=TASK_REGRESSION,
        )
        rf_native.load_data(X_train, y_train)
        rf_native.train(trainer="native", random_state=42)
        native_preds = rf_native.predict_batch(X_test)
        native_r2 = r2_score(y_test, native_preds)

        # Train with sklearn
        rf_sklearn = RandomForest(
            n_estimators=50,
            max_depth=10,
            features=feature_specs,
            task=TASK_REGRESSION,
        )
        rf_sklearn.load_data(X_train, y_train)
        rf_sklearn.train(trainer="sklearn", random_state=42)
        sklearn_preds = rf_sklearn.predict_batch(X_test)
        sklearn_r2 = r2_score(y_test, sklearn_preds)

        # Both should achieve positive R^2 (better than mean baseline)
        assert native_r2 > 0.0, f"Native R^2 too low: {native_r2:.3f}"
        assert sklearn_r2 > 0.0, f"Sklearn R^2 too low: {sklearn_r2:.3f}"

        # R^2 values should be within 0.2 of each other
        diff = abs(native_r2 - sklearn_r2)
        assert diff < 0.20, (
            f"R^2 gap too large: native={native_r2:.3f}, sklearn={sklearn_r2:.3f}, diff={diff:.3f}"
        )


class TestGiniParity:
    """Verify our Gini implementation matches sklearn's Gini (its default)."""

    @pytest.mark.parametrize(
        "loader,name",
        [
            (load_iris, "iris"),
            (load_wine, "wine"),
            (load_breast_cancer, "breast_cancer"),
        ],
    )
    def test_gini_single_tree_parity(self, loader, name):
        """Native Gini tree should match sklearn's default (Gini) tree exactly."""
        from cartlet import DecisionTree

        data = loader()
        X = data.data.tolist()
        y = [str(c) for c in data.target.tolist()]

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Our native trainer with Gini
        dt_native = DecisionTree(features=feature_specs, criterion="gini", max_depth=5)
        dt_native.load_data(X, y)
        dt_native.train(trainer="native", random_state=42)
        native_preds = dt_native.predict_batch(X)
        native_acc = accuracy_score(y, native_preds)

        # sklearn's default is Gini
        dt_sklearn = DecisionTree(features=feature_specs, max_depth=5)
        dt_sklearn.load_data(X, y)
        dt_sklearn.train(trainer="sklearn", random_state=42)
        sklearn_preds = dt_sklearn.predict_batch(X)
        sklearn_acc = accuracy_score(y, sklearn_preds)

        # Both should achieve high accuracy on training data
        assert native_acc > 0.80, (
            f"Native Gini accuracy too low on {name}: {native_acc:.2%}"
        )
        assert sklearn_acc > 0.80, (
            f"Sklearn Gini accuracy too low on {name}: {sklearn_acc:.2%}"
        )

        # Accuracies should be close (both using Gini, differ only in split search)
        diff = abs(native_acc - sklearn_acc)
        assert diff < 0.10, (
            f"Gini accuracy gap too large on {name}: "
            f"native={native_acc:.2%}, sklearn={sklearn_acc:.2%}, diff={diff:.2%}"
        )

    @pytest.mark.parametrize(
        "loader,name",
        [
            (load_iris, "iris"),
            (load_wine, "wine"),
        ],
    )
    def test_gini_forest_parity(self, loader, name):
        """Native Gini forest should perform similarly to sklearn's default forest."""
        data = loader()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.tolist(),
            [str(c) for c in data.target.tolist()],
            test_size=0.3,
            random_state=42,
        )

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Our native forest with Gini
        rf_native = RandomForest(
            n_estimators=50, max_depth=10, features=feature_specs, criterion="gini"
        )
        rf_native.load_data(X_train, y_train)
        rf_native.train(trainer="native", random_state=42)
        native_acc = accuracy_score(y_test, rf_native.predict_batch(X_test))

        # sklearn forest (default Gini)
        rf_sklearn = RandomForest(n_estimators=50, max_depth=10, features=feature_specs)
        rf_sklearn.load_data(X_train, y_train)
        rf_sklearn.train(trainer="sklearn", random_state=42)
        sklearn_acc = accuracy_score(y_test, rf_sklearn.predict_batch(X_test))

        assert native_acc > 0.70, (
            f"Native Gini forest too low on {name}: {native_acc:.2%}"
        )
        assert sklearn_acc > 0.70, (
            f"Sklearn forest too low on {name}: {sklearn_acc:.2%}"
        )

        diff = abs(native_acc - sklearn_acc)
        assert diff < 0.15, (
            f"Gini forest gap too large on {name}: "
            f"native={native_acc:.2%}, sklearn={sklearn_acc:.2%}, diff={diff:.2%}"
        )


class TestFeatureImportanceParity:
    """Verify feature importances are computed consistently."""

    def test_importances_identify_same_top_features(self):
        """Both trainers should agree on the most important features."""
        data = load_iris()
        X = data.data.tolist()
        y = [str(c) for c in data.target.tolist()]

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        rf_native = RandomForest(n_estimators=50, features=feature_specs)
        rf_native.load_data(X, y)
        rf_native.train(trainer="native", random_state=42)

        rf_sklearn = RandomForest(n_estimators=50, features=feature_specs)
        rf_sklearn.load_data(X, y)
        rf_sklearn.train(trainer="sklearn", random_state=42)

        native_imp = rf_native.feature_importances_
        sklearn_imp = rf_sklearn.feature_importances_

        # Both should have all features
        assert set(native_imp.keys()) == set(sklearn_imp.keys())

        # Both should sum to ~1.0
        assert abs(sum(native_imp.values()) - 1.0) < 0.01
        assert abs(sum(sklearn_imp.values()) - 1.0) < 0.01

        # Top 2 features should overlap (iris is known to have petal features most important)
        native_top2 = set(sorted(native_imp, key=native_imp.get, reverse=True)[:2])
        sklearn_top2 = set(sorted(sklearn_imp, key=sklearn_imp.get, reverse=True)[:2])

        # At least one of the top 2 should match
        assert len(native_top2 & sklearn_top2) >= 1, (
            f"Top features don't overlap: native={native_top2}, sklearn={sklearn_top2}"
        )


class TestRunnerVsSklearn:
    """Verify our runner produces identical predictions to sklearn."""

    def test_decision_tree_runner_matches_sklearn(self):
        """Runner predictions should match sklearn's raw predictions exactly."""
        import os
        import tempfile

        from cartlet import DecisionTree, load_model, predict

        data = load_iris()
        X = data.data.tolist()
        y = [str(c) for c in data.target.tolist()]

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Train with sklearn
        dt = DecisionTree(features=feature_specs)
        dt.load_data(X, y)
        dt.train(trainer="sklearn", random_state=42)

        # Get sklearn's raw predictions
        sklearn_preds = [dt.predict(x) for x in X]

        # Export to .cart and load with runner
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            cart_path = f.name

        try:
            dt.export(cart_path)
            model = load_model(cart_path)

            # Get runner predictions
            runner_preds = [predict(model, x) for x in X]

            # Should be identical
            mismatches = sum(1 for s, r in zip(sklearn_preds, runner_preds) if s != r)
            assert mismatches == 0, (
                f"Runner disagrees with sklearn on {mismatches}/{len(X)} samples"
            )
        finally:
            os.unlink(cart_path)

    def test_random_forest_runner_matches_sklearn(self):
        """Forest runner predictions should match sklearn's raw predictions."""
        import os
        import tempfile

        from cartlet import RandomForest, load_model, predict

        data = load_wine()
        X = data.data.tolist()
        y = [str(c) for c in data.target.tolist()]

        feature_specs = [
            {"name": n, "dtype": "float", "type": "num"} for n in data.feature_names
        ]

        # Train with sklearn
        rf = RandomForest(n_estimators=20, features=feature_specs)
        rf.load_data(X, y)
        rf.train(trainer="sklearn", random_state=42)

        # Get sklearn-trained model predictions
        sklearn_preds = [rf.predict(x) for x in X]

        # Export to .cart and load with runner
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as f:
            cart_path = f.name

        try:
            rf.export(cart_path)
            model = load_model(cart_path)

            # Get runner predictions
            runner_preds = [predict(model, x) for x in X]

            # Should be identical
            mismatches = sum(1 for s, r in zip(sklearn_preds, runner_preds) if s != r)
            assert mismatches == 0, (
                f"Runner disagrees with sklearn on {mismatches}/{len(X)} samples"
            )
        finally:
            os.unlink(cart_path)
