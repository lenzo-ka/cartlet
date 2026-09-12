"""Regressions for numerical stability and consistent training constraints."""

import math

import pytest

from cartlet import DecisionTree, IsolationForest


@pytest.mark.parametrize("categorical", [False, True])
def test_regression_target_translation_preserves_split(categorical):
    features = [
        {
            "name": "x",
            "dtype": "str" if categorical else "float",
            "type": "cat" if categorical else "num",
        }
    ]
    predictions = []
    for offset in (0.0, 1e8, 1e10):
        tree = DecisionTree(
            features=features, task="regression", max_depth=1, categorical_split="fast"
        )
        X = [[str(i)] if categorical else [i] for i in range(4)]
        tree.load_data(X, [offset, offset, offset, offset + 1], [1, 2, 3, 4])
        tree.train()
        assert isinstance(tree.model, list) and len(tree.model) == 5
        predictions.append([tree.predict(row) - offset for row in X])
    for actual in predictions[1:]:
        assert actual == pytest.approx(predictions[0], abs=1e-6)


def test_extreme_finite_feature_midpoint_separates_classes():
    tree = DecisionTree(features=[{"name": "x", "dtype": "float", "type": "num"}])
    tree.load_data([[1e308], [1.1e308]], ["A", "B"])
    tree.train()
    assert math.isfinite(tree.model[2])
    assert tree.predict([1e308]) == "A"
    assert tree.predict([1.1e308]) == "B"


@pytest.mark.parametrize("feature_type", ["cat", "num"])
@pytest.mark.parametrize("categorical_split", ["exact", "fast"])
def test_minimum_leaf_counts_rows_not_frequency_weights(
    feature_type, categorical_split
):
    tree = DecisionTree(
        features=[{"name": "x", "dtype": "float", "type": feature_type}],
        min_samples_leaf=2,
        min_confidence=1.0,
        categorical_split=categorical_split,
    )
    tree.load_data([[0], [1]], ["A", "B"], [2, 2])
    tree.train()
    assert tree.model == {"A": 0.5, "B": 0.5}


def test_isolation_ignores_constant_columns_when_selecting_splits():
    forest = IsolationForest(n_estimators=100, max_samples=4, random_state=0)
    forest.load_data([[0, 0], [0, 1], [0, 2], [0, 100]])
    forest.train()
    assert all(isinstance(tree, list) and tree[0] == 1 for tree in forest.trees)
    assert forest.predict([0, 100]) > forest.predict([0, 1])
    constant = IsolationForest(n_estimators=2, random_state=0)
    constant.load_data([[0, 0], [0, 0]])
    constant.train()
    assert constant.trees == [2, 2]


def test_sparse_categorical_encoding_matches_dense_without_quadratic_storage():
    pytest.importorskip("scipy")
    from cartlet.trainer.sklearn import encode_categorical
    from cartlet.types import FeatureSpec

    X = [[str(i), 0.0] for i in range(100)]
    specs = [
        FeatureSpec(name="x", dtype="str", type="cat"),
        FeatureSpec(name="y", dtype="float", type="num"),
    ]
    dense, names, columns, values = encode_categorical(X, ["x", "y"], specs)
    sparse, snames, scolumns, svalues = encode_categorical(
        X, ["x", "y"], specs, sparse=True
    )
    assert sparse.shape == (100, 101)
    assert sparse.nnz == 100
    assert sparse.toarray().tolist() == dense
    assert (snames, scolumns, svalues) == (names, columns, values)


def test_xgboost_multiclass_training_supports_current_vector_intercepts():
    pytest.importorskip("xgboost")
    from cartlet.xgboost import XGBoostTree

    tree = XGBoostTree(
        task="classification", n_estimators=1, max_depth=1, min_child_weight=0
    )
    tree.load_data([[0.0], [1.0], [2.0], [3.0]], ["A", "B", "C", "A"])
    tree.train(random_state=0)
    assert len(tree.predict_proba([2.0])) == 3
    if isinstance(tree.base_score, list):
        assert len(tree.base_score) == 3
        assert all(math.isfinite(value) for value in tree.base_score)


def test_xgboost_dataset_is_detached_and_validated():
    from cartlet.xgboost import XGBoostTree

    tree = XGBoostTree(task="regression")
    X, counts = [[1.0], [2.0]], [1, 2]
    tree.load_data(X, [1.0, 2.0], counts)
    X[0][0] = 9.0
    counts[0] = 9
    assert tree.X == [[1.0], [2.0]] and tree.counts == [1, 2]
    with pytest.raises(ValueError, match="same length"):
        tree.load_data([[1], [2]], [1])
    assert tree.X == [[1.0], [2.0]]
    with pytest.raises(ValueError, match="positive total"):
        tree.load_data([[1], [2]], [1, 2], [0, 0])
    tree.load_data([[1], [2]], [1, 2], [0, 2])
    assert tree.X == [[2]] and tree.y == [2] and tree.counts == [2]


def test_isolation_training_options_validate_before_replacing_trees():
    tree = IsolationForest(n_estimators=1, max_depth=0)
    tree.load_data([[1], [2]])
    tree.train()
    assert tree.trees == [2]
    tree.n_estimators = 0
    with pytest.raises(ValueError, match="positive integer"):
        tree.train()
    assert tree.trees == [2]
    with pytest.raises(ValueError, match="rectangular"):
        tree.load_data([[1], [1, 2]])
    assert tree.X == [[1.0], [2.0]]
