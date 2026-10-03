"""Decision-path parity across nested models and both .cart runners."""

import importlib.util
import math
import struct
from pathlib import Path

import pytest

import cartlet.runner as package_runner
from cartlet import DecisionTree, MissingFeatureError, RandomForest
from cartlet.cli import main as cli_main
from cartlet.io.bytes import write_forest_bytes, write_tree_bytes
from cartlet.runner import Predictor
from cartlet.types import FeatureSpec


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("path_bundled_predict", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_steps(result, row, switch_cases=None, *, xgboost=False):
    for tree in result["trees"]:
        assert isinstance(tree["leaf"], str)
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


@pytest.mark.parametrize(
    ("tree", "features", "row", "expected"),
    [
        (
            ["x", "<=", 0.0, "yes", "no"],
            [{"name": "x", "dtype": "float", "type": "num"}],
            [None],
            "feature 0 ('x') is missing at tree 0 node ''",
        ),
        (
            [
                "a",
                "<=",
                0.0,
                ["b", "<=", 0.0, "yes", ["x", "<=", 0.0, "yes", "no"]],
                "no",
            ],
            [
                {"name": "a", "dtype": "float", "type": "num"},
                {"name": "b", "dtype": "float", "type": "num"},
                {"name": "x", "dtype": "float", "type": "num"},
            ],
            [-1.0, 1.0, None],
            "feature 2 ('x') is missing at tree 0 node 'LR'",
        ),
        (
            ["route", "switch", {"red": ["x", "<=", 0.0, "yes", "no"]}, "no"],
            [
                {
                    "name": "route",
                    "dtype": "str",
                    "type": "cat",
                    "values": {"red", "blue"},
                },
                {"name": "x", "dtype": "float", "type": "num"},
            ],
            ["red", None],
            "feature 1 ('x') is missing at tree 0 node 'C3:red'",
        ),
        (
            ["route", "switch", {"red": "yes"}, ["x", "<=", 0.0, "yes", "no"]],
            [
                {
                    "name": "route",
                    "dtype": "str",
                    "type": "cat",
                    "values": {"red", "blue"},
                },
                {"name": "x", "dtype": "float", "type": "num"},
            ],
            ["blue", None],
            "feature 1 ('x') is missing at tree 0 node 'D'",
        ),
    ],
)
def test_missing_error_message_parity_across_all_predict_apis(
    tmp_path, monkeypatch, capsys, tree, features, row, expected
):
    nested = DecisionTree(features=features)
    nested.model = tree
    specs = [
        FeatureSpec(
            feature["name"],
            feature["dtype"],
            feature["type"],
            feature.get("values"),
        )
        for feature in features
    ]
    path = tmp_path / "missing-parity.cart"
    write_tree_bytes(
        str(path),
        tree,
        specs,
        {feature["name"]: i for i, feature in enumerate(features)},
        ["yes", "no"],
        False,
    )
    loaded = package_runner.load_model(str(path))
    predictor = Predictor(str(path))
    bundled = _bundled()
    bundled_loaded = bundled.load_cart(str(path))
    standalone = bundled.Predictor(str(path))

    calls = [
        (MissingFeatureError, lambda: nested.predict(row)),
        (MissingFeatureError, lambda: nested.predict_path(row)),
        (MissingFeatureError, lambda: nested.predict_batch([row])),
        (MissingFeatureError, lambda: predictor.predict(row)),
        (MissingFeatureError, lambda: predictor.predict_path(row)),
        (MissingFeatureError, lambda: predictor.predict_batch([row])),
        (MissingFeatureError, lambda: package_runner.predict(loaded, row)),
        (MissingFeatureError, lambda: package_runner.predict_path(loaded, row)),
        (MissingFeatureError, lambda: package_runner.predict_batch(loaded, [row])),
        (bundled.MissingFeatureError, lambda: standalone.predict(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_path(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_batch([row])),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict(bundled_loaded, row),
        ),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict_path(bundled_loaded, row),
        ),
    ]
    messages = []
    for error_type, call in calls:
        with pytest.raises(error_type) as exc_info:
            call()
        messages.append(str(exc_info.value))
    assert messages == [expected] * len(calls)

    values = ["" if value is None else str(value) for value in row]
    package_input = tmp_path / "package-input.csv"
    package_input.write_text(
        ",".join(feature["name"] for feature in features)
        + ",unused\n"
        + ",".join(values)
        + ",0\n"
    )
    assert cli_main(["predict", str(path), str(package_input)]) == 1
    assert capsys.readouterr().err.splitlines()[-1] == f"Error: {expected}"

    bundled_input = tmp_path / "bundled-input.csv"
    bundled_input.write_text(",".join(values) + ",0\n")
    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "-m", str(path), "-f", str(bundled_input), "-d", ","],
    )
    assert bundled.main() == 1
    assert capsys.readouterr().err == f"Error: {expected}\n"


def _clear_route_metadata(model):
    model.pop("_decision_parents", None)
    model.pop("_shared_decision_trees", None)


def _share_binary_child(model, tree_idx=0):
    root = model["tree_offsets"][tree_idx]
    decision = model["decisions"][root]
    model["decisions"][root] = (*decision[:5], decision[4])
    _clear_route_metadata(model)


def _share_switch_case_with_default(model, tree_idx=0):
    root = model["tree_offsets"][tree_idx]
    table_idx = model["decisions"][root][3]
    table = model["case_tables"][table_idx]
    case_value, _child = table["cases"][0]
    table["cases"][0] = (case_value, table["default"])
    key = model["strings"][model["cat_vals"][case_value]]
    table["lookup"][key] = table["default"]
    _clear_route_metadata(model)


@pytest.mark.parametrize("kind", ["binary", "switch"])
def test_shared_decision_error_uses_the_evaluated_route(tmp_path, kind):
    shared = ["x", "<=", 0.0, "yes", "no"]
    if kind == "binary":
        nested_tree = ["route", "<=", 0.0, shared, shared]
        flat_tree = [
            "route",
            "<=",
            0.0,
            shared,
            ["x", "<=", 0.0, "other", "no"],
        ]
        features = [
            {"name": "route", "dtype": "float", "type": "num"},
            {"name": "x", "dtype": "float", "type": "num"},
        ]
        row = [1.0, None]
        expected = "feature 1 ('x') is missing at tree 0 node 'R'"
        share = _share_binary_child
    else:
        nested_tree = ["route", "switch", {"red": shared}, shared]
        flat_tree = [
            "route",
            "switch",
            {"red": ["x", "<=", 0.0, "other", "no"]},
            shared,
        ]
        features = [
            {
                "name": "route",
                "dtype": "str",
                "type": "cat",
                "values": {"red", "blue"},
            },
            {"name": "x", "dtype": "float", "type": "num"},
        ]
        row = ["red", None]
        expected = "feature 1 ('x') is missing at tree 0 node 'C3:red'"
        share = _share_switch_case_with_default

    nested = DecisionTree(features=features)
    nested.model = nested_tree
    specs = [
        FeatureSpec(
            feature["name"],
            feature["dtype"],
            feature["type"],
            feature.get("values"),
        )
        for feature in features
    ]
    path = tmp_path / f"shared-{kind}.cart"
    write_tree_bytes(
        str(path),
        flat_tree,
        specs,
        {feature["name"]: i for i, feature in enumerate(features)},
        ["yes", "no", "other"],
        False,
    )
    package_model = package_runner.load_model(str(path))
    bundled = _bundled()
    bundled_model = bundled.load_cart(str(path))
    share(package_model)
    share(bundled_model)
    predictor = Predictor(package_model)
    standalone = bundled.Predictor()
    standalone._model = bundled_model

    calls = [
        (MissingFeatureError, lambda: nested.predict(row)),
        (MissingFeatureError, lambda: nested.predict_path(row)),
        (MissingFeatureError, lambda: predictor.predict(row)),
        (MissingFeatureError, lambda: predictor.predict_path(row)),
        (MissingFeatureError, lambda: predictor.predict_batch([row])),
        (MissingFeatureError, lambda: package_runner.predict(package_model, row)),
        (
            MissingFeatureError,
            lambda: package_runner.predict_path(package_model, row),
        ),
        (
            MissingFeatureError,
            lambda: package_runner.predict_batch(package_model, [row]),
        ),
        (bundled.MissingFeatureError, lambda: standalone.predict(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_path(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_batch([row])),
        (bundled.MissingFeatureError, lambda: bundled.predict(bundled_model, row)),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict_path(bundled_model, row),
        ),
    ]
    messages = []
    for error_type, call in calls:
        with pytest.raises(error_type) as exc_info:
            call()
        messages.append(str(exc_info.value))
    assert messages == [expected] * len(calls)


def test_shared_decision_error_reads_each_feature_once(tmp_path):
    tree = [
        "route",
        "<=",
        0.0,
        ["x", "<=", 0.0, "yes", "no"],
        ["x", "<=", 0.0, "other", "no"],
    ]
    specs = [
        FeatureSpec("route", "float", "num"),
        FeatureSpec("x", "float", "num"),
    ]
    path = tmp_path / "shared-lazy.cart"
    write_tree_bytes(
        str(path), tree, specs, {"route": 0, "x": 1}, ["yes", "no", "other"], False
    )
    package_model = package_runner.load_model(str(path))
    bundled = _bundled()
    bundled_model = bundled.load_cart(str(path))
    _share_binary_child(package_model)
    _share_binary_child(bundled_model)

    class CountingVector:
        def __init__(self):
            self.reads = {}

        def __len__(self):
            return 2

        def __getitem__(self, index):
            self.reads[index] = self.reads.get(index, 0) + 1
            return [1.0, None][index]

    for error_type, call in (
        (MissingFeatureError, lambda row: package_runner.predict(package_model, row)),
        (
            bundled.MissingFeatureError,
            lambda row: bundled.predict(bundled_model, row),
        ),
    ):
        row = CountingVector()
        with pytest.raises(error_type):
            call(row)
        assert row.reads == {0: 1, 1: 1}


def test_parent_table_ignores_unreachable_duplicate_switch_cases(tmp_path):
    tree = [
        "route",
        "switch",
        {"red": ["x", "<=", 0.0, "yes", "no"]},
        ["x", "<=", 0.0, "other", "no"],
    ]
    specs = [
        FeatureSpec("route", "str", "cat", {"red"}),
        FeatureSpec("x", "float", "num"),
    ]
    path = tmp_path / "duplicate-switch.cart"
    write_tree_bytes(
        str(path), tree, specs, {"route": 0, "x": 1}, ["yes", "no", "other"], False
    )
    package_model = package_runner.load_model(str(path))
    bundled = _bundled()
    bundled_model = bundled.load_cart(str(path))
    for model in (package_model, bundled_model):
        table = model["case_tables"][0]
        case_value, _case_child = table["cases"][0]
        table["cases"].append((case_value, table["default"]))
        _clear_route_metadata(model)

    expected = "feature 1 ('x') is missing at tree 0 node 'C3:red'"
    for error_type, call, model in (
        (
            MissingFeatureError,
            lambda: package_runner.predict(package_model, ["red", None]),
            package_model,
        ),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict(bundled_model, ["red", None]),
            bundled_model,
        ),
    ):
        with pytest.raises(error_type) as exc_info:
            call()
        assert str(exc_info.value) == expected
        assert model["_shared_decision_trees"] == [False]


def test_shared_decision_forest_error_names_later_tree(tmp_path):
    trees = [
        ["route", "<=", 0.0, "yes", "no"],
        [
            "route",
            "<=",
            0.0,
            ["x", "<=", 0.0, "yes", "no"],
            ["x", "<=", 0.0, "other", "no"],
        ],
    ]
    specs = [
        FeatureSpec("route", "float", "num"),
        FeatureSpec("x", "float", "num"),
    ]
    path = tmp_path / "shared-forest.cart"
    write_forest_bytes(
        str(path), trees, specs, {"route": 0, "x": 1}, ["yes", "no", "other"], False
    )
    package_model = package_runner.load_model(str(path))
    bundled = _bundled()
    bundled_model = bundled.load_cart(str(path))
    _share_binary_child(package_model, tree_idx=1)
    _share_binary_child(bundled_model, tree_idx=1)
    expected = "feature 1 ('x') is missing at tree 1 node 'R'"
    for error_type, call in (
        (
            MissingFeatureError,
            lambda: package_runner.predict(package_model, [1.0, None]),
        ),
        (
            MissingFeatureError,
            lambda: package_runner.predict_path(package_model, [1.0, None]),
        ),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict(bundled_model, [1.0, None]),
        ),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict_path(bundled_model, [1.0, None]),
        ),
    ):
        with pytest.raises(error_type) as exc_info:
            call()
        assert str(exc_info.value) == expected


def test_xgboost_cart_missing_error_message_parity(tmp_path):
    tree = [
        "a",
        "<",
        0.0,
        ["b", "<", 0.0, [1.0, 0.0, 1], ["x", "<", 0.0, [2.0, 0.0, 1], [3.0, 0.0, 1]]],
        [4.0, 0.0, 1],
    ]
    specs = [FeatureSpec(name, "float", "num") for name in ("a", "b", "x")]
    path = tmp_path / "missing-xgboost.cart"
    write_tree_bytes(
        str(path),
        tree,
        specs,
        {name: i for i, name in enumerate(("a", "b", "x"))},
        [],
        True,
        is_xgboost=True,
    )
    row = [-1.0, 1.0, None]
    expected = "feature 2 ('x') is missing at tree 0 node 'LR'"
    loaded = package_runner.load_model(str(path))
    predictor = Predictor(str(path))
    bundled = _bundled()
    bundled_loaded = bundled.load_cart(str(path))
    standalone = bundled.Predictor(str(path))
    calls = [
        (MissingFeatureError, lambda: predictor.predict(row)),
        (MissingFeatureError, lambda: predictor.predict_path(row)),
        (MissingFeatureError, lambda: predictor.predict_batch([row])),
        (MissingFeatureError, lambda: package_runner.predict(loaded, row)),
        (MissingFeatureError, lambda: package_runner.predict_path(loaded, row)),
        (MissingFeatureError, lambda: package_runner.predict_batch(loaded, [row])),
        (bundled.MissingFeatureError, lambda: standalone.predict(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_path(row)),
        (bundled.MissingFeatureError, lambda: standalone.predict_batch([row])),
        (bundled.MissingFeatureError, lambda: bundled.predict(bundled_loaded, row)),
        (
            bundled.MissingFeatureError,
            lambda: bundled.predict_path(bundled_loaded, row),
        ),
    ]
    messages = []
    for error_type, call in calls:
        with pytest.raises(error_type) as exc_info:
            call()
        messages.append(str(exc_info.value))
    assert messages == [expected] * len(calls)


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

    # Predict, then replace the left leaf in place. The edit is on the 0.5 path
    # and off the 2.0 path, whose path-based leaf ID remains unchanged.
    assert model.predict([0.5]) == "a"
    assert model.predict_path([2.0])["trees"][0]["leaf"] == "R"
    model.model[3] = ["x", "<=", 0.0, "c", "d"]

    assert model.predict([0.5]) == "d"
    assert model.predict_path([2.0])["trees"][0]["leaf"] == "R"
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
            "leaf": "R",
            "path": [
                {
                    "node": "",
                    "feature": 0,
                    "name": "x",
                    "op": "<=",
                    "value": 0.0,
                    "branch": "right",
                }
            ],
        }
    ]
    with pytest.raises(
        MissingFeatureError,
        match=r"feature 0 \('x'\) is missing at tree 0 node ''$",
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
    assert [tree["leaf"] for tree in primed["trees"]] == ["R", "R"]
    first.model[3] = ["x", "<=", 0.0, "c", "d"]

    assert model.predict([0.5]) == "d"
    assert [tree["leaf"] for tree in model.predict_path([2.0])["trees"]] == [
        "R",
        "R",
    ]
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
        match=r"feature 0 \('g1@\+1'\) is missing at tree 0 node ''",
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
    match = r"feature 0 \('x'\) is missing at tree 0 node ''"
    with pytest.raises(MissingFeatureError, match=match):
        model.predict([None])
    with pytest.raises(MissingFeatureError, match=match):
        model.predict_path([None])
    with pytest.raises(MissingFeatureError, match=match):
        package.predict([None])
    with pytest.raises(MissingFeatureError, match=match):
        package.predict_path([None])
    with pytest.raises(bundled.MissingFeatureError, match=match):
        standalone.predict([float("nan")])
    with pytest.raises(bundled.MissingFeatureError, match=match):
        standalone.predict_path([float("nan")])
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


def test_nested_switch_path_uses_canonical_case_keys(tmp_path):
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
    predictor = Predictor(str(path))
    bundled = _bundled()
    standalone = bundled.Predictor(str(path))

    cases = [
        (["red", "round", 9.0], "case", "C3:redL"),
        (["blue", "square", 9.0], "case", "C4:blue"),
        (["green", "square", 1.0], "default", "DL"),
    ]
    for row, branch, expected_leaf in cases:
        for result in (predictor.predict_path(row), standalone.predict_path(row)):
            _assert_steps(result, row, {"": {"red", "blue"}})
            assert result["trees"][0]["path"][0]["branch"] == branch
            assert result["trees"][0]["leaf"] == expected_leaf

    for missing_value in (None, float("nan")):
        row = [missing_value, "square", 1.0]
        expected = "feature 0 ('color') is missing at tree 0 node ''"
        with pytest.raises(MissingFeatureError) as exc_info:
            predictor.predict_path(row)
        assert str(exc_info.value) == expected
        with pytest.raises(MissingFeatureError) as exc_info:
            predictor.predict(row)
        assert str(exc_info.value) == expected
        with pytest.raises(bundled.MissingFeatureError) as exc_info:
            standalone.predict_path(row)
        assert str(exc_info.value) == expected
        with pytest.raises(bundled.MissingFeatureError) as exc_info:
            standalone.predict(row)
        assert str(exc_info.value) == expected
        for result in (
            predictor.predict_path(row, missing="right"),
            standalone.predict_path(row, missing="right"),
        ):
            assert result["trees"][0]["path"][0]["branch"] == "default"
            assert result["trees"][0]["leaf"] == "DL"
        assert predictor.predict(row, missing="right") == "small-default"
        assert standalone.predict(row, missing="right") == "small-default"


def test_switch_path_ids_escape_arbitrary_canonical_keys(tmp_path):
    keys = ["", "a:b", '"q"', "C1:xR", "snowman-☃"]
    cases = [(key, ["size", "<=", 0.0, f"{key}-left", f"{key}-right"]) for key in keys]
    model = DecisionTree(
        features=[
            {"name": "kind", "dtype": "str", "type": "cat"},
            {"name": "size", "dtype": "float", "type": "num"},
        ]
    )
    model.model = ["kind", "switch", cases, "default"]
    path = tmp_path / "arbitrary-switch-keys.cart"
    model.export(str(path))
    implementations = [
        model.predict_path,
        Predictor(str(path)).predict_path,
        _bundled().Predictor(str(path)).predict_path,
    ]

    for key in keys:
        token = f"C{len(key)}:{key}"
        expected = {
            "prediction": f"{key}-left",
            "trees": [
                {
                    "tree": 0,
                    "leaf": token + "L",
                    "path": [
                        {
                            "node": "",
                            "feature": 0,
                            "name": "kind",
                            "op": "switch",
                            "value": None,
                            "branch": "case",
                        },
                        {
                            "node": token,
                            "feature": 1,
                            "name": "size",
                            "op": "<=",
                            "value": 0.0,
                            "branch": "left",
                        },
                    ],
                }
            ],
        }
        assert [predict([key, -1.0]) for predict in implementations] == [
            expected,
            expected,
            expected,
        ]


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
