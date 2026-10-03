"""Strict feature inference for direct supervised model APIs."""

import pytest

from cartlet import DecisionTree, RandomForest
from cartlet.runner import load_model

FLOAT_ROWS = [[float(i), float(i % 3)] for i in range(12)]
TARGETS = [str(i >= 6) for i in range(12)]


@pytest.mark.parametrize(
    "model",
    [
        DecisionTree(feature_names=["x", "noise"]),
        RandomForest(n_estimators=2, feature_names=["x", "noise"]),
    ],
)
def test_feature_names_infer_float_columns_and_reach_each_tree(model, tmp_path):
    model.load_data(FLOAT_ROWS, TARGETS)

    assert [(spec.dtype, spec.type) for spec in model.feature_specs] == [
        ("float", "num"),
        ("float", "num"),
    ]
    model.train(random_state=3)
    if isinstance(model, RandomForest):
        assert all(
            [spec.type for spec in tree.feature_specs] == ["num", "num"]
            for tree in model.trees
        )

    path = tmp_path / f"{type(model).__name__}.cart"
    model.export(str(path))
    exported = load_model(str(path))
    assert [feature["type"] for feature in exported["meta"]["features"]] == [
        "num",
        "num",
    ]
    assert exported["cat_vals"] == []
    assert exported["category_sets"] == []


def test_string_bool_and_integer_columns_use_strict_inference():
    tree = DecisionTree(feature_names=["label", "enabled", "count"])
    tree.load_data(
        [["a", True, 1], ["b", False, 2], ["a", True, 3]],
        ["x", "y", "x"],
    )

    assert [(spec.dtype, spec.type) for spec in tree.feature_specs] == [
        ("str", "cat"),
        ("str", "cat"),
        ("int", "num"),
    ]
    assert tree.feature_specs[1].values == {True, False}


@pytest.mark.parametrize("model_cls", [DecisionTree, RandomForest])
def test_ambiguous_column_requires_explicit_features(model_cls):
    kwargs = {"n_estimators": 2} if model_cls is RandomForest else {}
    model = model_cls(feature_names=["mixed"], **kwargs)
    with pytest.raises(ValueError, match=r"mixed.*features="):
        model.load_data([["1"], [2.0]], ["a", "b"])


def test_auto_names_infer_and_repeated_load_reinfers():
    tree = DecisionTree()
    tree.load_data([[1.0], [2.0]], ["a", "b"])
    assert tree.feature_names == ["0"]
    assert (tree.feature_specs[0].dtype, tree.feature_specs[0].type) == (
        "float",
        "num",
    )

    tree.load_data([["one"], ["two"]], ["a", "b"])
    assert (tree.feature_specs[0].dtype, tree.feature_specs[0].type) == (
        "str",
        "cat",
    )


def test_explicit_features_remain_unchanged():
    tree = DecisionTree(features=[{"name": "code", "dtype": "str", "type": "cat"}])
    tree.load_data([[1.0], [2.0]], ["a", "b"])
    assert (tree.feature_specs[0].dtype, tree.feature_specs[0].type) == (
        "str",
        "cat",
    )


def test_sklearn_backend_sees_inferred_numeric_columns():
    pytest.importorskip("sklearn")
    tree = DecisionTree(feature_names=["x", "noise"])
    tree.load_data(FLOAT_ROWS, TARGETS)
    tree.train(trainer="sklearn", random_state=3)

    assert [spec.type for spec in tree.feature_specs] == ["num", "num"]
    assert tree._sklearn_model.n_features_in_ == 2

    forest = RandomForest(n_estimators=2, feature_names=["x", "noise"])
    forest.load_data(FLOAT_ROWS, TARGETS)
    forest.train(trainer="sklearn", random_state=3)

    assert [spec.type for spec in forest.feature_specs] == ["num", "num"]
    assert forest._sklearn_model.n_features_in_ == 2


def test_loaded_forest_keeps_categorical_schema_for_reference_and_training_trees(
    tmp_path,
):
    source = RandomForest(
        n_estimators=3,
        features=[{"name": "code", "dtype": "str", "type": "cat"}],
    )
    rows = [[1], [2], [1], [2]]
    targets = ["a", "b", "a", "b"]
    source.load_data(rows, targets)
    source.train(random_state=7)
    path = tmp_path / "categorical.cart"
    source.export(str(path))

    loaded = RandomForest(n_estimators=3)
    loaded.load_model(str(path))
    loaded.load_data(rows, targets)
    loaded.train(random_state=7)

    assert [(spec.dtype, spec.type) for spec in loaded.feature_specs] == [
        ("str", "cat")
    ]
    assert all(
        [(spec.dtype, spec.type) for spec in tree.feature_specs] == [("str", "cat")]
        for tree in loaded.trees
    )


def test_forest_bootstrap_trees_reuse_inferred_float_schema():
    forest = RandomForest(n_estimators=12, feature_names=["value"])
    forest.load_data([[0], [1], [2], [3.5]], ["a", "b", "a", "b"])
    forest.train(random_state=11)

    assert [(spec.dtype, spec.type) for spec in forest.feature_specs] == [
        ("float", "num")
    ]
    assert all(
        [(spec.dtype, spec.type) for spec in tree.feature_specs] == [("float", "num")]
        for tree in forest.trees
    )


@pytest.mark.parametrize("extension", ["skl", "cart", "json"])
def test_loaded_tree_schema_survives_load_data(extension, tmp_path):
    pytest.importorskip("sklearn")
    if extension == "skl":
        pytest.importorskip("joblib")

    source = DecisionTree(feature_names=["value"])
    source.load_data([[1.0], [2.0], [3.0], [4.0]], ["a", "a", "b", "b"])
    source.train(trainer="sklearn", random_state=3)
    path = tmp_path / f"numeric.{extension}"
    source.export(str(path))

    loaded = DecisionTree()
    loaded.load_model(str(path))
    schema = [(spec.dtype, spec.type) for spec in loaded.feature_specs]
    loaded.load_data([["one"], ["two"]], ["a", "b"])

    assert [(spec.dtype, spec.type) for spec in loaded.feature_specs] == schema


@pytest.mark.parametrize("model_cls", [DecisionTree, RandomForest])
def test_failed_inference_preserves_trained_model_and_data(model_cls):
    kwargs = {"n_estimators": 2} if model_cls is RandomForest else {}
    model = model_cls(feature_names=["value"], **kwargs)
    model.load_data([[1], [2], [3], [4]], ["a", "a", "b", "b"])
    model.train(random_state=5)
    trained_model = model.trees if isinstance(model, RandomForest) else model.model
    original_rows = [row.copy() for row in model.X]
    original_targets = model.y.copy()
    original_counts = model.counts.copy()

    with pytest.raises(ValueError, match=r"value.*features="):
        model.load_data([["1"], [2.0]], ["a", "b"])

    current_model = model.trees if isinstance(model, RandomForest) else model.model
    assert current_model is trained_model
    assert original_rows == model.X
    assert original_targets == model.y
    assert original_counts == model.counts
