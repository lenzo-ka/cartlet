"""Held-out permutation importance and structural path inspection."""

import copy
import json
import struct

import pytest

from cartlet import (
    DecisionTree,
    MissingFeatureError,
    Predictor,
    RandomForest,
    decisive_leaves,
    leaf_paths,
    permutation_importance,
)
from cartlet.io.bytes import write_tree_bytes
from cartlet.runner import load_model, predict_path
from cartlet.types import FeatureSpec, normalize_bool

FEATURES = [
    {"name": "signal", "dtype": "float", "type": "num"},
    {"name": "noise", "dtype": "float", "type": "num"},
]


def _classification_data():
    rows = [[float(index), float((index * 7) % 5)] for index in range(40)]
    return rows, [int(row[0] >= 20) for row in rows]


def _regression_data():
    rows = [[float(index), float((index * 7) % 5)] for index in range(40)]
    return rows, [row[0] * 3.0 for row in rows]


def _train(kind, task):
    rows, targets = (
        _classification_data() if task == "classification" else _regression_data()
    )
    kwargs = {"features": FEATURES, "task": task, "max_depth": 5}
    if kind == "tree":
        model = DecisionTree(**kwargs)
    else:
        model = RandomForest(
            n_estimators=5,
            bootstrap=False,
            max_features=None,
            **kwargs,
        )
    model.load_data(rows, targets)
    if kind == "tree":
        model.train(validation_split=0, random_state=11)
    else:
        model.train(random_state=11)
    return model, rows, targets


@pytest.mark.parametrize("kind", ["tree", "forest"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_permutation_importance_finds_signal_and_is_deterministic(kind, task):
    model, rows, targets = _train(kind, task)
    original = copy.deepcopy(rows)
    first = permutation_importance(model, rows, targets, n_repeats=7, random_state=13)
    second = permutation_importance(model, rows, targets, n_repeats=7, random_state=13)
    assert first == second
    assert rows == original
    assert first["importances"][0]["name"] == "signal"
    assert first["importances"][0]["mean"] > 0
    noise = next(item for item in first["importances"] if item["name"] == "noise")
    assert abs(noise["mean"]) < first["importances"][0]["mean"]
    if task == "classification":
        # Held-out ints must match the model's canonical string labels.
        assert first["baseline"] == 0


def test_permutation_groups_seed_validation_and_missing_policy():
    model, rows, targets = _train("tree", "classification")
    grouped = permutation_importance(
        model,
        rows,
        targets,
        feature_groups={"both": ["signal", 1]},
        n_repeats=2,
        random_state=4,
    )
    assert grouped["importances"][0]["features"] == ["signal", "noise"]
    with pytest.raises(TypeError, match="random_state must be an int or None"):
        permutation_importance(model, rows, targets, random_state="4")
    with pytest.raises(TypeError, match="random_state must be an int or None"):
        permutation_importance(model, rows, targets, random_state=True)
    with pytest.raises(ValueError, match="appears more than once"):
        permutation_importance(
            model, rows, targets, feature_groups={"bad": [0, "signal"]}
        )

    authored = DecisionTree(features=[FEATURES[0]])
    authored.model = ["signal", "<=", 0.5, "0", "1"]
    with pytest.raises(MissingFeatureError):
        permutation_importance(authored, [[], []], [1, 1], n_repeats=1)
    result = permutation_importance(
        authored, [[], []], [1, 1], n_repeats=1, missing="right"
    )
    assert result["baseline"] == 0


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_native_oob_permutation_importance_finds_signal_and_is_deterministic(task):
    rows, targets = (
        _classification_data() if task == "classification" else _regression_data()
    )
    model = RandomForest(
        n_estimators=15,
        bootstrap=True,
        max_features=None,
        features=FEATURES,
        task=task,
        max_depth=5,
    )
    model.load_data(rows, targets)
    model.train(random_state=17)

    first = model.oob_permutation_importance(n_repeats=5, random_state=19)
    second = model.oob_permutation_importance(n_repeats=5, random_state=19)
    assert first == second
    assert first["baseline"] == first["oob_baseline"]
    assert first["importances"][0]["name"] == "signal"
    assert first["importances"][0]["mean"] > 0
    assert model._inbag_indices is not None
    assert len(model._inbag_indices) == len(model.trees)
    assert all(len(indices) == len(rows) for indices in model._inbag_indices)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_sklearn_oob_permutation_importance_finds_signal(task):
    pytest.importorskip("sklearn")
    rows, targets = (
        _classification_data() if task == "classification" else _regression_data()
    )
    model = RandomForest(
        n_estimators=15,
        bootstrap=True,
        max_features=None,
        features=FEATURES,
        task=task,
        max_depth=5,
    )
    model.load_data(rows, targets)
    model.train(trainer="sklearn", random_state=17, n_jobs=1)

    result = model.oob_permutation_importance(n_repeats=4, random_state=19)
    assert result["importances"][0]["name"] == "signal"
    assert model._inbag_indices is not None
    assert len(model._inbag_indices) == len(model.trees)


def test_oob_permutation_importance_requires_process_local_bootstraps(tmp_path):
    rows, targets = _classification_data()
    non_bootstrap = RandomForest(
        n_estimators=3,
        bootstrap=False,
        max_features=None,
        features=FEATURES,
    )
    non_bootstrap.load_data(rows, targets)
    non_bootstrap.train(random_state=3)
    with pytest.raises(ValueError, match="trained in this process with bootstrap=True"):
        non_bootstrap.oob_permutation_importance()

    model = RandomForest(
        n_estimators=3,
        bootstrap=True,
        max_features=None,
        features=FEATURES,
    )
    model.load_data(rows, targets)
    model.train(random_state=3)
    path = tmp_path / "forest.cart"
    model.export(str(path))
    loaded = RandomForest()
    loaded.load_model(str(path))
    with pytest.raises(ValueError, match="use held-out permutation_importance"):
        loaded.oob_permutation_importance()


def test_loaded_model_importance_and_classification_paths_match(tmp_path):
    model, rows, targets = _train("forest", "classification")
    expected_importance = permutation_importance(
        model, rows, targets, n_repeats=4, random_state=9
    )
    expected_paths = leaf_paths(model)
    for suffix in ("json", "cart"):
        path = tmp_path / f"model.{suffix}"
        model.export(str(path), store_distributions=True)
        if suffix == "cart":
            loaded = Predictor(str(path))
        else:
            loaded = RandomForest()
            loaded.load_model(str(path))
        assert (
            permutation_importance(loaded, rows, targets, n_repeats=4, random_state=9)
            == expected_importance
        )
        actual = leaf_paths(loaded)
        assert [
            (leaf["tree"], leaf["leaf"], leaf["path"], leaf["predicted_class"])
            for tree in actual["trees"]
            for leaf in tree["leaves"]
        ] == [
            (leaf["tree"], leaf["leaf"], leaf["path"], leaf["predicted_class"])
            for tree in expected_paths["trees"]
            for leaf in tree["leaves"]
        ]


def test_empirical_support_counts_purity_and_decisive_thresholds():
    model, rows, targets = _train("tree", "classification")
    export = leaf_paths(model, rows, targets)
    leaves = export["trees"][0]["leaves"]
    assert sum(leaf["data_support"] for leaf in leaves) == len(rows)
    assert all(leaf["support"] is None for leaf in leaves)
    assert all(leaf["data_purity"] == 1 for leaf in leaves)
    assert sum(sum(leaf["data_class_counts"].values()) for leaf in leaves) == len(rows)
    selected = decisive_leaves(model, 1, rows, targets, min_support=1, min_purity=1)
    assert selected
    assert all(leaf["predicted_class"] == "1" for leaf in selected)
    assert decisive_leaves(model, 1, rows, min_purity=0.1) == []


def test_regression_support_is_effective_weight_and_lost_in_cart(tmp_path):
    model = DecisionTree(
        features=[{"name": "x", "dtype": "float", "type": "num"}],
        task="regression",
    )
    model.load_data([[0.0], [0.0]], [1.0, 1.0], counts=[0.25, 0.25])
    model.train(validation_split=0)
    assert leaf_paths(model)["trees"][0]["leaves"][0]["support"] == 0.5
    empirical = leaf_paths(model, [[0.0], [0.0]], [1.0, 1.0])
    empirical_leaf = empirical["trees"][0]["leaves"][0]
    assert empirical_leaf["data_support"] == 2
    assert "data_class_counts" not in empirical_leaf

    json_path = tmp_path / "weighted.json"
    cart_path = tmp_path / "weighted.cart"
    model.export(str(json_path))
    model.export(str(cart_path))
    json_model = DecisionTree()
    json_model.load_model(str(json_path))
    cart_model = DecisionTree()
    cart_model.load_model(str(cart_path))
    assert leaf_paths(json_model)["trees"][0]["leaves"][0]["support"] == 0.5
    assert leaf_paths(cart_model)["trees"][0]["leaves"][0]["support"] is None


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_retraining_cart_model_restores_regression_leaf_support(kind, tmp_path):
    kwargs = {
        "features": [{"name": "x", "dtype": "float", "type": "num"}],
        "task": "regression",
    }
    if kind == "tree":
        model = DecisionTree(**kwargs)
    else:
        model = RandomForest(n_estimators=1, bootstrap=False, **kwargs)
    model.load_data([[0.0], [0.0]], [1.0, 1.0])
    if kind == "tree":
        model.train(validation_split=0)
    else:
        model.train(random_state=1)

    path = tmp_path / f"{kind}.cart"
    model.export(str(path))
    model.load_model(str(path))
    model.load_data([[0.0], [0.0]], [2.0, 2.0])
    if kind == "tree":
        model.train(validation_split=0)
    else:
        model.train(random_state=1)

    leaves = leaf_paths(model)["trees"][0]["leaves"]
    assert leaves[0]["support"] == 2.0


def test_switch_paths_group_nested_identity_and_flat_child_index(tmp_path):
    model = DecisionTree(features=[{"name": "kind", "dtype": "str", "type": "cat"}])
    shared = {"yes": 1.0}
    model.model = [
        "kind",
        "switch",
        [("a", shared), ("b", shared), ("a", "unreachable")],
        shared,
    ]
    direct = leaf_paths(model)["trees"][0]["leaves"]
    assert len(direct) == 2
    assert direct[0]["leaf"] == direct[1]["leaf"]
    assert direct[1]["path"][0]["value"] == ["a", "b"]

    # The writer deliberately expands identity-equal children and rejects
    # duplicate canonical keys. Build an ordinary artifact, then admit the two
    # flat-table cases the loader supports: shared child indexes and a later
    # duplicate key. Inspection must group by child index and keep first match.
    model.model = [
        "kind",
        "switch",
        [("a", "first"), ("b", "second")],
        "default",
    ]
    path = tmp_path / "switch.cart"
    model.export(str(path), store_distributions=True)
    data = load_model(str(path))
    table = data["case_tables"][0]
    first_case, second_case = table["cases"]
    table["cases"] = [
        first_case,
        (second_case[0], first_case[1]),
        (first_case[0], second_case[1]),
    ]
    table["lookup"] = {"a": first_case[1], "b": first_case[1]}
    export = leaf_paths(data)
    leaves = export["trees"][0]["leaves"]
    assert len(leaves) == 2
    assert leaves[0]["leaf"] != leaves[1]["leaf"]
    assert leaves[0]["path"][0]["op"] == "not in"
    assert leaves[0]["path"][0]["value"] == ["a", "b"]
    assert leaves[1]["path"][0]["op"] == "in"
    assert leaves[1]["path"][0]["value"] == ["a", "b"]
    assert predict_path(data, ["a"])["trees"][0]["leaf"] == leaves[1]["leaf"]
    assert predict_path(data, ["b"])["trees"][0]["leaf"] == leaves[1]["leaf"]
    assert predict_path(data, ["z"])["trees"][0]["leaf"] == leaves[0]["leaf"]


def _route_one(export, row, *, missing="error"):
    """Route using only exported conditions and the exported schema."""
    feature_schema = export["features"]
    float32 = export["numeric_input_float32"]
    matches = []
    for leaf in export["trees"][0]["leaves"]:
        accepted = True
        for condition in leaf["path"]:
            value = row[condition["feature"]]
            spec = feature_schema[condition["feature"]]
            if value is None:
                direction = condition["missing_direction"]
                if direction is None:
                    if missing == "error":
                        accepted = False
                        continue
                    direction = (
                        "default"
                        if condition["branch"] in ("case", "default")
                        else "right"
                    )
                if condition["branch"] in ("case", "default"):
                    predicate = (
                        direction == "left" and condition["branch"] == "case"
                    ) or (direction != "left" and condition["branch"] == "default")
                else:
                    predicate = direction == condition["branch"]
                accepted &= predicate
                continue
            if spec["dtype"] == "bool":
                value = normalize_bool(value)
            op = condition["op"]
            if op in ("<", "<="):
                number = float(value)
                if float32:
                    number = struct.unpack("<f", struct.pack("<f", number))[0]
                predicate = (
                    number < condition["value"]
                    if op == "<"
                    else number <= condition["value"]
                )
            elif op == "=":
                predicate = str(value) == condition["value"]
            elif op == "in":
                predicate = str(value) in condition["value"]
            else:
                predicate = str(value) not in condition["value"]
            if condition["branch"] == "right":
                predicate = not predicate
            accepted &= predicate
        if accepted:
            matches.append(leaf)
    assert len(matches) == 1
    return matches[0]


def test_exported_bool_schema_routes_like_predict_path(tmp_path):
    model = DecisionTree(features=[{"name": "flag", "dtype": "bool", "type": "cat"}])
    model.model = ["flag", "=", True, "yes", "no"]
    path = tmp_path / "bool.cart"
    model.export(str(path))
    data = load_model(str(path))
    export = leaf_paths(data)
    for row in ([True], ["yes"], [False]):
        expected = predict_path(data, list(row))["trees"][0]
        actual = _route_one(export, row)
        assert (actual["tree"], actual["leaf"]) == (
            expected["tree"],
            expected["leaf"],
        )


def test_exported_set_and_learned_missing_routes_match_predict_path(tmp_path):
    set_model = DecisionTree(features=[{"name": "kind", "dtype": "str", "type": "cat"}])
    set_model.model = ["kind", "in", ["b", "a"], "yes", "no"]
    set_path = tmp_path / "set.cart"
    set_model.export(str(set_path))
    set_data = load_model(str(set_path))
    set_export = leaf_paths(set_data)
    assert set_export["trees"][0]["leaves"][0]["path"][0]["value"] == [
        "a",
        "b",
    ]
    for row in (["a"], ["z"]):
        assert (
            _route_one(set_export, row)["leaf"]
            == predict_path(set_data, row)["trees"][0]["leaf"]
        )

    for direction in ("left", "right"):
        model = DecisionTree(features=[{"name": "x", "dtype": "float", "type": "num"}])
        model.model = [
            {"feature": "x", "missing": direction},
            "<=",
            0.5,
            "left",
            "right",
        ]
        path = tmp_path / f"missing-{direction}.cart"
        model.export(str(path))
        data = load_model(str(path))
        export = leaf_paths(data)
        expected = predict_path(data, [None])["trees"][0]
        actual = _route_one(export, [None])
        assert actual["leaf"] == expected["leaf"]
        assert all(
            leaf["path"][0]["missing_direction"] == direction
            for leaf in export["trees"][0]["leaves"]
        )


def test_xgboost_float32_boundary_routes_from_exported_schema(tmp_path):
    threshold = 1.0000001192092896
    path = tmp_path / "xgb.cart"
    write_tree_bytes(
        str(path),
        [{"feature": "x", "missing": "right"}, "<", threshold, [1, 0, 1], [2, 0, 1]],
        [FeatureSpec("x", dtype="float", type="num")],
        {"x": 0},
        [],
        True,
        metadata={"base_score": 0.0},
        is_xgboost=True,
    )
    data = load_model(str(path))
    export = leaf_paths(data)
    assert export["numeric_input_float32"] is True
    row = [1.00000007]
    assert row[0] < threshold
    assert struct.unpack("<f", struct.pack("<f", row[0]))[0] == threshold
    expected = predict_path(data, row)["trees"][0]
    actual = _route_one(export, row)
    assert actual["leaf"] == expected["leaf"]
    assert actual["prediction"] == 2.0


def test_leaf_paths_missing_error_and_right():
    model = DecisionTree(features=[FEATURES[0]])
    model.model = ["signal", "<=", 0.5, "a", "b"]
    with pytest.raises(MissingFeatureError):
        leaf_paths(model, [[None]])
    export = leaf_paths(model, [[None]], missing="right")
    leaves = export["trees"][0]["leaves"]
    assert [leaf["data_support"] for leaf in leaves] == [0, 1]


def test_stored_and_collapsed_classification_distributions(tmp_path):
    model = DecisionTree(feature_names=["x"])
    model.model = {"winner": 0.75, "other": 0.25}
    stored = leaf_paths(model)["trees"][0]["leaves"][0]
    assert stored["class_distribution"] == {"winner": 0.75, "other": 0.25}
    assert stored["purity"] == 0.75

    full_path = tmp_path / "full.cart"
    collapsed_path = tmp_path / "collapsed.cart"
    model.export(str(full_path), store_distributions=True)
    model.export(str(collapsed_path), store_distributions=False)
    full = leaf_paths(load_model(str(full_path)))["trees"][0]["leaves"][0]
    collapsed = leaf_paths(load_model(str(collapsed_path)))["trees"][0]["leaves"][0]
    assert full["class_distribution"] == pytest.approx(
        {"winner": 0.75, "other": 0.25}, abs=1 / 65535
    )
    assert full["purity"] == pytest.approx(0.75, abs=1 / 65535)
    assert collapsed["predicted_class"] == "winner"
    assert collapsed["class_distribution"] is None
    assert collapsed["purity"] is None


def test_exports_are_json_compatible():
    model, rows, targets = _train("tree", "classification")
    json.dumps(permutation_importance(model, rows, targets, random_state=1))
    json.dumps(leaf_paths(model, rows, targets), allow_nan=False)
