"""Decision-path parity across nested models and both .cart runners."""

import importlib.util
from pathlib import Path

import pytest

from cartlet import DecisionTree, MissingFeatureError, RandomForest
from cartlet.io.bytes import write_tree_bytes
from cartlet.runner import INDEX_MASK, LEAF_FLAG, Predictor, load_model
from cartlet.types import FeatureSpec


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("path_bundled_predict", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_steps(result, row):
    for tree in result["trees"]:
        assert isinstance(tree["leaf"], int)
        for step in tree["path"]:
            value = row[step["feature"]]
            if step["op"] == "<=":
                expected = "left" if float(value) <= step["value"] else "right"
            elif step["op"] == "<":
                expected = "left" if float(value) < step["value"] else "right"
            elif step["op"] == "=":
                expected = "left" if str(value) == step["value"] else "right"
            else:
                expected = step["branch"]
            assert step["branch"] == expected


def _roundtrips(model, rows, tmp_path, kind):
    json_path = tmp_path / f"{kind}.json"
    cart_path = tmp_path / f"{kind}.cart"
    model.export(str(json_path))
    model.export(str(cart_path))
    cls = RandomForest if isinstance(model, RandomForest) else DecisionTree
    json_model = cls()
    json_model.load_model(str(json_path))
    cart_model = cls()
    cart_model.load_model(str(cart_path))
    package = Predictor(str(cart_path))
    standalone = _bundled().Predictor(str(cart_path))
    reached = set()
    for row in rows:
        expected = model.predict_path(row)
        assert expected["prediction"] == model.predict(row)
        assert json_model.predict_path(row) == expected
        assert cart_model.predict_path(row) == expected
        assert package.predict_path(row) == expected
        assert standalone.predict_path(row) == expected
        _assert_steps(expected, row)
        reached.update((tree["tree"], tree["leaf"]) for tree in expected["trees"])
    assert len(reached) >= 3


@pytest.mark.parametrize("store_distributions", [False, True])
def test_classification_tree_paths_roundtrip(tmp_path, store_distributions):
    rows = [[x, y] for x in range(4) for y in range(3)]
    labels = [f"c{x // 2}-{y // 2}" for x, y in rows]
    model = DecisionTree(
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
        max_depth=4,
        store_distributions=store_distributions,
        min_confidence=1.0,
    )
    model.load_data(rows, labels)
    model.train(validation_split=0)
    _roundtrips(model, rows + [[0.5, 1.5], [2.5, 0.5]], tmp_path, "class")


def test_regression_tree_paths_roundtrip(tmp_path):
    rows = [[float(x), float(y)] for x in range(4) for y in range(3)]
    model = DecisionTree(
        task="regression",
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
        max_depth=4,
    )
    model.load_data(rows, [x * 10 + y for x, y in rows])
    model.train(validation_split=0)
    _roundtrips(model, rows, tmp_path, "regression")


def test_forest_paths_roundtrip(tmp_path):
    rows = [[x, y] for x in range(4) for y in range(3)]
    model = RandomForest(
        n_estimators=4,
        max_features=None,
        bootstrap=False,
        feature_names=["x", "y"],
        max_depth=4,
    )
    model.load_data(rows, [f"c{x // 2}-{y // 2}" for x, y in rows])
    model.train(random_state=4)
    _roundtrips(model, rows, tmp_path, "forest")


def test_regression_forest_path_prediction_matches_predict(tmp_path):
    rows = [[float(x), float(y)] for x in range(5) for y in range(3)]
    model = RandomForest(
        n_estimators=4,
        max_features=None,
        bootstrap=False,
        task="regression",
        feature_names=["x", "y"],
        max_depth=4,
    )
    model.load_data(rows, [x * 3.0 - y for x, y in rows])
    model.train(random_state=5)
    _roundtrips(model, rows + [[0.5, 1.5], [3.5, 0.25]], tmp_path, "forest-reg")


def test_xgboost_runner_paths(tmp_path):
    pytest.importorskip("xgboost")
    from cartlet import XGBoostTree

    rows = [[float(x), float(y)] for x in range(4) for y in range(3)] * 3
    model = XGBoostTree(
        n_estimators=3,
        max_depth=3,
        min_child_weight=0,
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
    )
    model.load_data(rows, ["low" if x < 2 else "high" for x, _ in rows])
    model.train(random_state=3)
    path = tmp_path / "xgb.cart"
    model.export(str(path))
    package = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))
    reached = set()
    for row in rows:
        package_result = package.predict_path(row)
        assert package_result == standalone.predict_path(row)
        assert package_result["prediction"] == package.predict(row)
        _assert_steps(package_result, row)
        reached.update((tree["tree"], tree["leaf"]) for tree in package_result["trees"])
    assert len(reached) >= 3


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_xgboost_path_prediction_matches_predict_many_rows(tmp_path, task):
    pytest.importorskip("xgboost")
    from cartlet import XGBoostTree

    rows = [[float(x), float(y)] for x in range(6) for y in range(4)] * 3
    labels = (
        ["a" if x < 2 else "b" if x < 4 else "c" for x, _ in rows]
        if task == "multiclass"
        else [x * 2.5 - y * 0.75 for x, y in rows]
    )
    model = XGBoostTree(
        task="classification" if task == "multiclass" else "regression",
        n_estimators=4,
        max_depth=3,
        min_child_weight=0,
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
    )
    model.load_data(rows, labels)
    model.train(random_state=6)
    path = tmp_path / f"xgb-{task}.cart"
    model.export(str(path))
    package = Predictor(str(path))
    standalone = _bundled().Predictor(str(path))
    probes = rows + [[x / 3.0, y / 5.0] for x in range(17) for y in range(7)]
    for row in probes:
        package_result = package.predict_path(row)
        standalone_result = standalone.predict_path(row)
        if task == "regression":
            assert package_result["prediction"] == pytest.approx(package.predict(row))
            assert standalone_result["prediction"] == pytest.approx(
                standalone.predict(row)
            )
        else:
            assert package_result["prediction"] == package.predict(row)
            assert standalone_result["prediction"] == standalone.predict(row)
        assert standalone_result == package_result


@pytest.mark.parametrize("missing_row", [[], [None], [float("nan")]])
def test_missing_default_errors_and_legacy_policy(tmp_path, missing_row):
    model = DecisionTree(features=[{"name": "g1@+1", "dtype": "float", "type": "num"}])
    model.load_data([[0.0], [1.0]], ["left", "right"])
    model.train(validation_split=0)
    path = tmp_path / "missing.cart"
    model.export(str(path))
    package = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))

    with pytest.raises(
        MissingFeatureError,
        match=r"feature 0 \('g1@\+1'\) is missing at tree 0 node 0",
    ):
        package.predict(missing_row)
    with pytest.raises(MissingFeatureError):
        model.predict(missing_row)
    with pytest.raises(bundled.MissingFeatureError):
        standalone.predict(missing_row)
    assert model.predict(missing_row, missing="right") == package.predict(
        missing_row, missing="right"
    )


def test_in_process_probability_helpers_honor_missing():
    tree = DecisionTree(
        features=[{"name": "x", "dtype": "float", "type": "num"}],
        store_distributions=True,
        min_confidence=1.0,
    )
    tree.load_data([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
    tree.train(validation_split=0)
    for method in (
        lambda: tree.predict([None], return_dist=True),
        lambda: tree.predict_with_confidence([None]),
        lambda: tree.predict_nbest([None]),
    ):
        with pytest.raises(MissingFeatureError):
            method()
    assert tree.predict([None], return_dist=True, missing="right")
    assert tree.predict_with_confidence([None], missing="right")
    assert tree.predict_nbest([None], missing="right")

    forest = RandomForest(
        n_estimators=3,
        max_features=None,
        bootstrap=False,
        features=[{"name": "x", "dtype": "float", "type": "num"}],
    )
    forest.load_data([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
    forest.train(random_state=0)
    with pytest.raises(MissingFeatureError):
        forest.predict_proba([None])
    assert forest.predict_proba([None], missing="right")


def test_forest_missing_error_names_tree_and_node(tmp_path):
    model = RandomForest(
        n_estimators=3,
        max_features=None,
        bootstrap=False,
        features=[{"name": "x", "dtype": "float", "type": "num"}],
    )
    model.load_data([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
    model.train(random_state=0)
    path = tmp_path / "missing-forest.cart"
    model.export(str(path))
    package = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))
    match = r"feature 0 \('x'\) is missing at tree 0 node 0"
    with pytest.raises(MissingFeatureError, match=match):
        package.predict([None])
    with pytest.raises(bundled.MissingFeatureError, match=match):
        standalone.predict([float("nan")])
    assert package.predict([None], missing="right") == standalone.predict(
        [None], missing="right"
    )


def test_xgboost_missing_error_names_tree_and_node(tmp_path):
    pytest.importorskip("xgboost")
    from cartlet import XGBoostTree

    model = XGBoostTree(
        n_estimators=2,
        max_depth=2,
        min_child_weight=0,
        features=[{"name": "x", "dtype": "float", "type": "num"}],
    )
    model.load_data([[0.0], [1.0], [2.0], [3.0]] * 4, ["a", "a", "b", "b"] * 4)
    model.train(random_state=0)
    path = tmp_path / "missing-xgb.cart"
    model.export(str(path))
    package = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))
    match = r"feature 0 \('x'\) is missing at tree 0 node 0"
    with pytest.raises(MissingFeatureError, match=match):
        package.predict([float("nan")])
    with pytest.raises(bundled.MissingFeatureError, match=match):
        standalone.predict([None])
    assert package.predict([None], missing="right") == standalone.predict(
        [None], missing="right"
    )


def test_nested_switch_path_uses_case_table_and_leaf_array_oracle(tmp_path):
    tree = [
        "color",
        "switch",
        {
            "red": ["shape", "=", "round", "red-round", "red-other"],
            "blue": "blue",
        },
        ["size", "<=", 1.5, "small-default", "large-default"],
    ]
    specs = [
        FeatureSpec("color", "str", "cat", {"red", "blue"}),
        FeatureSpec("shape", "str", "cat", {"round", "square"}),
        FeatureSpec("size", "float", "num"),
    ]
    path = tmp_path / "nested-switch.cart"
    write_tree_bytes(
        str(path),
        tree,
        specs,
        {"color": 0, "shape": 1, "size": 2},
        ["red-round", "red-other", "blue", "small-default", "large-default"],
        False,
    )
    loaded = load_model(str(path))
    predictor = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))

    root = loaded["decisions"][loaded["tree_offsets"][0]]
    table = loaded["case_tables"][root[2]]
    red_child = table["lookup"]["red"]
    default_child = table["default"]
    assert loaded["tree_offsets"][0] == 0
    assert default_child == 1
    assert red_child == 2

    def leaf_from_child(child, take_left):
        decision = loaded["decisions"][child]
        encoded = decision[3] if take_left else decision[4]
        assert encoded & LEAF_FLAG
        return encoded & INDEX_MASK

    # Writer preorder: root decision; default decision and leaves 0/1; red
    # decision and leaves 2/3; blue leaf 4.
    cases = [
        (["red", "round", 9.0], "case", leaf_from_child(red_child, True), 2),
        (["green", "square", 1.0], "default", leaf_from_child(default_child, True), 0),
    ]
    for row, branch, oracle_leaf, handwritten_leaf in cases:
        assert oracle_leaf == handwritten_leaf
        for result in (predictor.predict_path(row), standalone.predict_path(row)):
            assert result["trees"][0]["path"][0]["branch"] == branch
            assert result["trees"][0]["leaf"] == oracle_leaf

    for missing_value in (None, float("nan")):
        row = [missing_value, "square", 1.0]
        with pytest.raises(MissingFeatureError):
            predictor.predict_path(row)
        with pytest.raises(MissingFeatureError):
            predictor.predict(row)
        with pytest.raises(bundled.MissingFeatureError):
            standalone.predict_path(row)
        with pytest.raises(bundled.MissingFeatureError):
            standalone.predict(row)
        for result in (
            predictor.predict_path(row, missing="right"),
            standalone.predict_path(row, missing="right"),
        ):
            assert result["trees"][0]["path"][0]["branch"] == "default"
            assert result["trees"][0]["leaf"] == 0
        assert predictor.predict(row, missing="right") == "small-default"
        assert standalone.predict(row, missing="right") == "small-default"
