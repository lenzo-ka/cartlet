"""Decision-path parity across nested models and both .cart runners."""

import importlib.util
import math
import struct
from pathlib import Path

import pytest

from cartlet import DecisionTree, MissingFeatureError, RandomForest
from cartlet.io.bytes import write_forest_bytes, write_tree_bytes
from cartlet.runner import INDEX_MASK, LEAF_FLAG, Predictor, load_model
from cartlet.types import FeatureSpec
from cartlet.utils import build_tree_indices


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("path_bundled_predict", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_steps(result, row, switch_cases=None, *, xgboost=False):
    for tree in result["trees"]:
        assert isinstance(tree["leaf"], int)
        for step in tree["path"]:
            value = row[step["feature"]]
            if step["op"] == "<=":
                numeric = float(value)
                if xgboost:
                    numeric = struct.unpack("<f", struct.pack("<f", numeric))[0]
                expected = "left" if numeric <= step["value"] else "right"
            elif step["op"] == "<":
                numeric = float(value)
                if xgboost:
                    numeric = struct.unpack("<f", struct.pack("<f", numeric))[0]
                expected = "left" if numeric < step["value"] else "right"
            elif step["op"] == "=":
                expected = "left" if str(value) == step["value"] else "right"
            else:
                assert switch_cases is not None
                assert step["node"] in switch_cases
                expected = (
                    "case" if str(value) in switch_cases[step["node"]] else "default"
                )
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


@pytest.mark.parametrize(
    ("feature", "node", "row", "canonical_value"),
    [
        (
            {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]},
            ["flag", "=", True, "on", "off"],
            [True],
            "1",
        ),
        (
            {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]},
            ["flag", "=", "true", "on", "off"],
            [True],
            "1",
        ),
        (
            {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]},
            ["flag", "=", 1, "on", "off"],
            [True],
            "1",
        ),
        (
            {"name": "x", "dtype": "float", "type": "num"},
            ["x", "<=", "0.5", "low", "high"],
            [0.25],
            0.5,
        ),
    ],
)
def test_authored_path_values_match_writer_canonical_form(
    tmp_path, feature, node, row, canonical_value
):
    model = DecisionTree(features=[feature])
    model.model = node
    json_path = tmp_path / "authored.json"
    cart_path = tmp_path / "authored.cart"
    model.export(str(json_path))
    model.export(str(cart_path))

    json_model = DecisionTree()
    json_model.load_model(str(json_path))
    bundled = _bundled()
    results = [
        model.predict_path(row),
        json_model.predict_path(row),
        Predictor(str(cart_path)).predict_path(row),
        bundled.Predictor(str(cart_path)).predict_path(row),
    ]
    assert all(result == results[0] for result in results[1:])
    assert results[0]["trees"][0]["path"][0]["value"] == canonical_value


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


def _assert_edited_cart_parity(model, rows, tmp_path, filename):
    """Compare an edited nested model with a fresh binary export and reload."""
    path = tmp_path / filename
    model.export(str(path))
    cls = RandomForest if isinstance(model, RandomForest) else DecisionTree
    reloaded = cls()
    reloaded.load_model(str(path))
    package = Predictor(str(path))
    for row in rows:
        assert model.predict(row) == package.predict(row) == reloaded.predict(row)
        assert (
            model.predict_path(row)
            == package.predict_path(row)
            == reloaded.predict_path(row)
        )


def test_tree_in_place_subtree_edit_refreshes_prediction_and_path_ids(tmp_path):
    features = [{"name": "x", "dtype": "float", "type": "num"}]
    model = DecisionTree(features=features)
    model.load_data([[0.0], [2.0]], ["a", "b"])
    model.model = ["x", "<=", 1.0, "a", "b"]

    # Prime both former cache users, then replace the left leaf in place. The
    # edit is on the 0.5 path and off the 2.0 path, whose leaf ID still shifts.
    assert model.predict([0.5]) == "a"
    assert model.predict_path([2.0])["trees"][0]["leaf"] == 1
    model.model[3] = ["x", "<=", 0.0, "c", "d"]

    assert model.predict([0.5]) == "d"
    assert model.predict_path([2.0])["trees"][0]["leaf"] == 2
    _assert_edited_cart_parity(
        model, [[-0.5], [0.5], [2.0]], tmp_path, "edited-tree.cart"
    )


def test_deep_unvisited_branch_does_not_recurse_for_path_or_missing_error():
    depth = 1200
    root = "bottom"
    for level in range(depth, 0, -1):
        root = ["x", "<=", 0.0, root, f"right-{level}"]

    model = DecisionTree(features=[{"name": "x", "dtype": "float", "type": "num"}])
    model.model = root

    result = model.predict_path([1.0])
    assert result["prediction"] == "right-1"
    assert result["trees"] == [
        {
            "tree": 0,
            "leaf": depth,
            "path": [
                {
                    "node": 0,
                    "feature": 0,
                    "name": "x",
                    "op": "<=",
                    "value": 0.0,
                    "branch": "right",
                }
            ],
        }
    ]
    decisions, leaves = build_tree_indices([root])[0]
    assert decisions[()] == 0
    assert leaves[(1,)] == depth
    with pytest.raises(
        MissingFeatureError,
        match=r"feature 0 \('x'\) is missing at tree 0 node 0",
    ):
        model.predict([None])


def test_forest_in_place_member_edit_refreshes_prediction_and_path_ids(tmp_path):
    features = [{"name": "x", "dtype": "float", "type": "num"}]
    model = RandomForest(n_estimators=2, features=features)
    model.load_data([[0.0], [2.0]], ["a", "b"])
    first = model._make_tree()
    first.model = ["x", "<=", 1.0, "a", "b"]
    second = model._make_tree()
    second.model = ["x", "<=", 1.0, "a", "b"]
    model.trees = [first, second]

    assert model.predict([0.5]) == "a"
    primed = model.predict_path([2.0])
    assert [tree["leaf"] for tree in primed["trees"]] == [1, 3]
    first.model[3] = ["x", "<=", 0.0, "c", "d"]

    assert model.predict([0.5]) == "d"
    assert [tree["leaf"] for tree in model.predict_path([2.0])["trees"]] == [2, 4]
    _assert_edited_cart_parity(
        model, [[-0.5], [0.5], [2.0]], tmp_path, "edited-forest.cart"
    )


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
        _assert_steps(package_result, row, xgboost=True)
        reached.update((tree["tree"], tree["leaf"]) for tree in package_result["trees"])
    assert len(reached) >= 3


def test_xgboost_path_oracle_uses_float32_inputs_at_split_boundary(tmp_path):
    pytest.importorskip("xgboost")
    from cartlet import XGBoostTree

    model = XGBoostTree(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_child_weight=0,
        features=[{"name": "x", "dtype": "float", "type": "num"}],
    )
    model.load_data([[0.0], [0.0], [1.0], [1.0]], ["left", "left", "right", "right"])
    model.train(random_state=1)
    threshold = model.trees[0][2]
    assert threshold == struct.unpack("<f", struct.pack("<f", threshold))[0]
    assert threshold > 0

    path = tmp_path / "xgb-float32-boundary.cart"
    model.export(str(path))
    predictors = [Predictor(str(path)), _bundled().Predictor(str(path))]

    raw_below = math.nextafter(threshold, -math.inf)
    assert raw_below < threshold
    assert struct.unpack("<f", struct.pack("<f", raw_below))[0] == threshold

    threshold_bits = struct.unpack("<I", struct.pack("<f", threshold))[0]
    float32_below = struct.unpack("<f", struct.pack("<I", threshold_bits - 1))[0]
    assert float32_below < threshold

    for predictor in predictors:
        rounded_result = predictor.predict_path([raw_below])
        exact_result = predictor.predict_path([float32_below])
        assert rounded_result["trees"][0]["path"][0]["branch"] == "right"
        assert exact_result["trees"][0]["path"][0]["branch"] == "left"
        _assert_steps(rounded_result, [raw_below], xgboost=True)
        _assert_steps(exact_result, [float32_below], xgboost=True)
        assert rounded_result["prediction"] == model.predict([raw_below])
        assert exact_result["prediction"] == model.predict([float32_below])


@pytest.mark.parametrize(
    ("is_regression", "class_labels", "expected"),
    [
        (False, ["negative", "positive"], "negative"),
        (True, [], -0.5),
    ],
)
def test_xgboost_aggregation_preserves_base_first_tree_order(
    tmp_path, is_regression, class_labels, expected
):
    path = tmp_path / f"xgb-order-{is_regression}.cart"
    values = [1e16, -1e16, -0.5]
    write_forest_bytes(
        str(path),
        [[value, 0.0, 1] for value in values],
        [],
        {},
        class_labels,
        is_regression,
        metadata={"base_score": 0.65},
        is_xgboost=True,
    )
    bundled = _bundled()
    for predictor in (Predictor(str(path)), bundled.Predictor(str(path))):
        assert predictor.predict([]) == expected
        assert predictor.predict_path([])["prediction"] == expected


def test_regression_forest_aggregation_uses_sum_then_divide(tmp_path):
    path = tmp_path / "forest-order.cart"
    values = [1e16, -1e16, -0.5]
    forest = RandomForest(n_estimators=len(values), task="regression")
    forest.trees = []
    for value in values:
        tree = DecisionTree(task="regression")
        tree.model = [value, 0.0, 1]
        forest.trees.append(tree)
    write_forest_bytes(
        str(path),
        [[value, 0.0, 1] for value in values],
        [],
        {},
        [],
        True,
    )
    expected = sum(values) / len(values)
    assert forest.predict([]) == expected
    assert forest.predict_path([])["prediction"] == expected
    bundled = _bundled()
    for predictor in (Predictor(str(path)), bundled.Predictor(str(path))):
        assert predictor.predict([]) == expected
        assert predictor.predict_path([])["prediction"] == expected


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
    if task == "multiclass":
        from cartlet import runner

        bundled = _bundled()
        for loaded, predict, predict_path in (
            (runner.load_model(str(path)), runner.predict, runner.predict_path),
            (bundled.load_cart(str(path)), bundled.predict, bundled.predict_path),
        ):
            loaded["n_trees"] -= 1
            for call in (predict, predict_path):
                with pytest.raises(ValueError, match="not a multiple of 3 classes"):
                    call(loaded, rows[0])


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


def test_xgboost_learned_missing_route_overrides_runner_policy(tmp_path):
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
    expected = model.predict([float("nan")])
    for predictor in (package, standalone):
        assert predictor.predict([float("nan")]) == expected
        assert predictor.predict([None], missing="right") == expected
        path_result = predictor.predict_path([])
        assert path_result["prediction"] == expected
        assert all(
            step.get("missing") is True
            for tree in path_result["trees"]
            for step in tree["path"]
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
    table = loaded["case_tables"][root[3]]
    red_child = table["lookup"]["red"]
    blue_child = table["lookup"]["blue"]
    default_child = table["default"]
    assert loaded["tree_offsets"][0] == 0
    assert default_child == 1
    assert red_child == 2

    def leaf_from_child(child, take_left):
        decision = loaded["decisions"][child]
        encoded = decision[4] if take_left else decision[5]
        assert encoded & LEAF_FLAG
        return encoded & INDEX_MASK

    # Writer preorder: root decision; default decision and leaves 0/1; red
    # decision and leaves 2/3; blue leaf 4.
    assert blue_child & LEAF_FLAG
    blue_leaf = blue_child & INDEX_MASK
    cases = [
        (["red", "round", 9.0], "case", leaf_from_child(red_child, True), 2),
        (["blue", "square", 9.0], "case", blue_leaf, 4),
        (["green", "square", 1.0], "default", leaf_from_child(default_child, True), 0),
    ]
    for row, branch, oracle_leaf, handwritten_leaf in cases:
        assert oracle_leaf == handwritten_leaf
        for result in (predictor.predict_path(row), standalone.predict_path(row)):
            _assert_steps(result, row, {0: {"red", "blue"}})
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


def test_writer_rejects_duplicate_canonical_bool_switch_keys(tmp_path):
    tree = [
        "flag",
        "switch",
        {"yes": "first", True: "second"},
        "default",
    ]
    with pytest.raises(ValueError, match="duplicate canonical switch case key '1'"):
        write_tree_bytes(
            str(tmp_path / "collision.cart"),
            tree,
            [FeatureSpec("flag", "bool", "cat", {0, 1})],
            {"flag": 0},
            ["first", "second", "default"],
            False,
        )


def test_missing_policy_is_validated_before_any_row(tmp_path):
    from cartlet import runner

    model = DecisionTree(features=[{"name": "x", "dtype": "float", "type": "num"}])
    model.load_data([[0.0], [1.0]], ["left", "right"])
    model.train(validation_split=0)
    path = tmp_path / "policy.cart"
    model.export(str(path))
    bundled = _bundled()
    package_model = runner.load_model(str(path))
    bundled_model = bundled.load_cart(str(path))
    calls = [
        lambda: runner.predict_batch(package_model, [], missing="typo"),
        lambda: Predictor(str(path)).predict_batch([], missing="typo"),
        lambda: bundled.Predictor(str(path)).predict_batch([], missing="typo"),
        lambda: bundled.predict_tree(bundled_model, [None], missing="typo"),
        lambda: model.predict_batch([], missing="typo"),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="missing must be"):
            call()
