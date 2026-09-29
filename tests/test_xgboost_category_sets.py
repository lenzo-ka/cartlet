"""XGBoost categorical sets stay proportional and agree across runtimes."""

import json
import random

import pytest

from cartlet import DecisionTree, Predictor, XGBoostTree
from cartlet import runner as package_runner
from cartlet.bundled.predict import Predictor as BundledPredictor
from cartlet.io.cart_format import rebuild_tree_from_cart
from cartlet.utils import eval_tree

pytest.importorskip("xgboost")


def _categorical_training_data(categories, rows=1200):
    rng = random.Random(0)
    values = [f"c{i:02d}" for i in range(categories)]
    X = []
    y = []
    for _ in range(rows):
        row = [rng.choice(values) for _ in range(4)]
        numbers = [int(value[1:]) for value in row]
        X.append(row)
        y.append(str((numbers[0] // 3 + numbers[1] // 5 + numbers[2] % 3) % 2))
    return X, y


def _booster_counts(model):
    decisions = leaves = 0
    pending = [json.loads(raw) for raw in model._xgb_model.get_dump(dump_format="json")]
    while pending:
        node = pending.pop()
        if "leaf" in node:
            leaves += 1
        else:
            decisions += 1
            pending.extend(node.get("children", []))
    return decisions, leaves


def _train_size_model(categories, trees, depth):
    X, y = _categorical_training_data(categories)
    model = XGBoostTree(
        n_estimators=trees,
        max_depth=depth,
        min_child_weight=0,
        nthread=1,
        feature_names=["a", "b", "c", "d"],
    )
    model.load_data(X, y)
    model.train(random_state=0)
    return model


def test_category_set_exports_match_booster_nodes(tmp_path):
    for categories, trees, depth in ((20, 5, 4), (40, 50, 6)):
        model = _train_size_model(categories, trees, depth)
        booster_decisions, booster_leaves = _booster_counts(model)
        path = tmp_path / f"category-{categories}.cart"
        model.export(str(path))
        loaded = package_runner.load_model(str(path))
        assert len(loaded["decisions"]) == booster_decisions
        assert len(loaded["leaves"]) == booster_leaves
        assert loaded["category_sets"]
        X, _ = _categorical_training_data(categories, rows=200)
        for row in X:
            assert package_runner.predict(loaded, row) == model.predict(row)


def _prediction_model(task):
    categories = [f"c{i}" for i in range(12)]
    rows = []
    targets = []
    for i in range(480):
        a = categories[i % 12]
        b = categories[(i * 7 + i // 12) % 12]
        left, right = int(a[1:]), int(b[1:])
        rows.append([a, b])
        if task == "binary":
            targets.append("B" if (left // 3 + right % 2) % 2 else "A")
        elif task == "multiclass":
            targets.append(str((left // 2 + right // 3) % 3))
        else:
            targets.append(float(left * 0.7 - right * 0.3 + left % 3))
    model = XGBoostTree(
        task="regression" if task == "regression" else "classification",
        n_estimators=5,
        max_depth=3,
        min_child_weight=0,
        nthread=1,
        feature_names=["a", "b"],
    )
    model.load_data(rows, targets)
    model.train(random_state=3)
    return model, categories


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_category_set_predictions_match_booster_on_420_probes(tmp_path, task):
    model, categories = _prediction_model(task)
    path = tmp_path / f"{task}.cart"
    model.export(str(path))
    loaded = package_runner.load_model(str(path))
    package = Predictor(str(path))
    bundled = BundledPredictor(str(path))
    values = [*categories, "unseen", None, float("nan")]
    probes = [
        [values[i % len(values)], values[(i * 7 + 3) % len(values)]] for i in range(420)
    ]

    for row in probes:
        expected = model.predict(row)
        nested_values = [
            eval_tree(
                tree,
                row,
                model.name_to_col,
                feature_specs=model.feature_specs,
            )
            for tree in model.trees
        ]
        nested = package_runner._aggregate_xgboost(loaded, nested_values)
        actual = [
            nested,
            package.predict(row),
            bundled.predict(row),
            package.predict_path(row)["prediction"],
            bundled.predict_path(row)["prediction"],
        ]
        if task == "regression":
            assert actual == pytest.approx([expected] * len(actual), abs=1e-5)
        else:
            assert actual == [expected] * len(actual)
        assert package.predict_path(row) == bundled.predict_path(row)


def test_category_set_json_and_cart_round_trips_preserve_paths(tmp_path):
    tree = ["x", "in", ["a", "c"], "yes", ["x", "=", "b", "maybe", "no"]]
    model = DecisionTree(features=[{"name": "x", "dtype": "str", "type": "cat"}])
    model.model = tree
    model.class_labels = ["maybe", "no", "yes"]
    json_path = tmp_path / "membership.json"
    cart_path = tmp_path / "membership.cart"
    model.export(str(json_path))
    model.export(str(cart_path))

    json_model = DecisionTree()
    json_model.load_model(str(json_path))
    loaded = package_runner.load_model(str(cart_path))
    assert json_model.model == tree
    assert rebuild_tree_from_cart(loaded, ["x"]) == tree
    for row in (["a"], ["b"], ["z"]):
        expected = model.predict_path(row)
        assert json_model.predict_path(row) == expected
        assert Predictor(str(cart_path)).predict_path(row) == expected
        assert BundledPredictor(str(cart_path)).predict_path(row) == expected


def test_bool_category_set_aliases_report_the_canonical_set(tmp_path):
    model = DecisionTree(features=[{"name": "b", "type": "cat", "dtype": "bool"}])
    model.model = ["b", "in", ["1", "True"], "hit", "miss"]
    path = tmp_path / "bool-set.cart"
    model.export(str(path))
    expected = model.predict_path([True])
    assert expected["trees"][0]["path"][0]["value"] == ["1"]
    assert Predictor(str(path)).predict_path([True]) == expected
    assert BundledPredictor(str(path)).predict_path([True]) == expected
