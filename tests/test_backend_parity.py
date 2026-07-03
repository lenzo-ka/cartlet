"""Native vs sklearn/xgboost backend parity and task-detection guards (W2).

These pin behaviors that used to silently diverge between backends: dropped
instance weights, ignored criterion / extra_trees, integer labels mistaken for
regression, and bool features broken under the sklearn forest.
"""

from __future__ import annotations

import importlib.util

import pytest

from cartlet import DecisionTree, RandomForest

_SKLEARN_MISSING = importlib.util.find_spec("sklearn") is None
requires_sklearn = pytest.mark.skipif(_SKLEARN_MISSING, reason="sklearn not installed")


# =============================================================================
# H5: integer class labels -> classification (not regression)
# =============================================================================


def test_int_labels_classify_not_regress():
    dt = DecisionTree(feature_names=["a", "b"])
    dt.load_data([["x", "y"], ["x", "z"], ["y", "y"], ["y", "z"]], [0, 1, 0, 1])
    dt.train()
    assert dt._effective_task() == "classification"
    # Classification labels are stringified for a single stable label type.
    assert dt.predict(["x", "y"]) == "0"


def test_many_valued_numeric_still_regresses():
    dt = DecisionTree(feature_names=["x"])
    dt.load_data([[i] for i in range(50)], [i * 1.5 for i in range(50)])
    dt.train()
    assert dt._effective_task() == "regression"
    assert isinstance(dt.predict([10]), float)


# =============================================================================
# H1: Sklearn is the real class (from_sklearn / isinstance), not a factory
# =============================================================================


def test_sklearn_name_is_the_class():
    # Does not require sklearn installed: the backend module imports sklearn
    # lazily, so attribute access resolves the class object itself.
    from cartlet.trainer import Sklearn

    assert isinstance(Sklearn, type)
    assert hasattr(Sklearn, "from_sklearn")


# =============================================================================
# H3: instance weights honored by every backend
# =============================================================================


@requires_sklearn
def test_sklearn_tree_honors_weights():
    # Two identical feature vectors, opposite labels: the tree is one leaf, so
    # the prediction is the weighted majority. Dropping weights would pick the
    # first class ("x") on the 1:1 tie.
    dt = DecisionTree(feature_names=["f"])
    dt.load_data([["a"], ["a"]], ["x", "y"], counts=[1, 100])
    dt.train(trainer="sklearn")
    assert dt.predict(["a"]) == "y"


def test_native_tree_honors_weights():
    dt = DecisionTree(feature_names=["f"])
    dt.load_data([["a"], ["a"]], ["x", "y"], counts=[1, 100])
    dt.train()
    assert dt.predict(["a"]) == "y"


@requires_sklearn
def test_sklearn_forest_honors_weights():
    rf = RandomForest(n_estimators=5, feature_names=["f"])
    rf.load_data([["a"], ["a"]], ["x", "y"], counts=[1, 100])
    rf.train(trainer="sklearn", random_state=0)
    assert rf.predict(["a"]) == "y"


# =============================================================================
# H4: sklearn backend honors criterion and extra_trees
# =============================================================================


@requires_sklearn
def test_sklearn_tree_forwards_criterion():
    dt = DecisionTree(feature_names=["f"], criterion="gini")
    dt.load_data([["a"], ["b"]], ["x", "y"])
    dt.train(trainer="sklearn")
    assert dt._sklearn_model.criterion == "gini"

    dt2 = DecisionTree(feature_names=["f"])  # cartlet default is entropy
    dt2.load_data([["a"], ["b"]], ["x", "y"])
    dt2.train(trainer="sklearn")
    assert dt2._sklearn_model.criterion == "entropy"


@requires_sklearn
def test_sklearn_forest_extra_trees_and_criterion():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    rf = RandomForest(
        n_estimators=5, extra_trees=True, criterion="gini", feature_names=["f"]
    )
    rf.load_data([["a"], ["b"]], ["x", "y"])
    rf.train(trainer="sklearn", random_state=0)
    assert isinstance(rf._sklearn_model, ExtraTreesClassifier)
    assert rf._sklearn_model.criterion == "gini"

    rf2 = RandomForest(n_estimators=5, feature_names=["f"])  # plain, entropy
    rf2.load_data([["a"], ["b"]], ["x", "y"])
    rf2.train(trainer="sklearn", random_state=0)
    assert isinstance(rf2._sklearn_model, RandomForestClassifier)
    assert rf2._sklearn_model.criterion == "entropy"


# =============================================================================
# M6: bool features work under the sklearn forest backend
# =============================================================================


@requires_sklearn
def test_sklearn_forest_bool_features():
    rf = RandomForest(
        n_estimators=10, features=[{"name": "flag", "dtype": "bool", "type": "cat"}]
    )
    rf.load_data([[True], [False]] * 10, ["on", "off"] * 10)
    rf.train(trainer="sklearn", random_state=0)
    assert rf.predict([True]) == "on"
    assert rf.predict([False]) == "off"
