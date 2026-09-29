"""XGBoost learned missing routes survive nested and binary inference."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

import cartlet.runner as package_runner
from cartlet import Predictor
from cartlet.io.cart_format import rebuild_tree_from_cart
from cartlet.types import FeatureSpec
from cartlet.utils import eval_tree, is_decision_node, is_switch_node
from cartlet.xgboost import XGBoostTree


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("missing_direction_bundled", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_data():
    rng = random.Random(731)
    rows = []
    scores = []
    for _ in range(180):
        a = rng.uniform(-2, 2)
        b = rng.uniform(-2, 2)
        missing_a = rng.random() < 0.28
        missing_b = rng.random() < 0.32
        rows.append(
            [
                float("nan") if missing_a else a,
                float("nan") if missing_b else b,
            ]
        )
        scores.append(
            1.4 * a - 0.9 * b + (1.8 if missing_a else 0) - (1.6 if missing_b else 0)
        )
    return rows, scores


def _trained_model(task):
    xgb = pytest.importorskip("xgboost")
    rows, scores = _training_data()
    if task == "regression":
        targets = scores
        params = {"objective": "reg:squarederror"}
        labels = []
    elif task == "binary":
        targets = [int(score > 0) for score in scores]
        params = {"objective": "binary:logistic"}
        labels = ["0", "1"]
    else:
        targets = [0 if score < -0.7 else 2 if score > 0.8 else 1 for score in scores]
        params = {"objective": "multi:softprob", "num_class": 3}
        labels = ["0", "1", "2"]
    params.update(
        {
            "max_depth": 3,
            "eta": 0.35,
            "min_child_weight": 0,
            "lambda": 0.1,
            "seed": 19,
            "nthread": 1,
        }
    )
    booster = xgb.train(
        params,
        xgb.DMatrix(rows, label=targets, feature_names=["a", "b"]),
        num_boost_round=8,
    )
    model = XGBoostTree(
        task="regression" if task == "regression" else "classification",
        n_estimators=8,
        learning_rate=0.35,
        max_depth=3,
        feature_names=["a", "b"],
    )
    model.feature_specs = [
        FeatureSpec(name="a", dtype="float", type="num"),
        FeatureSpec(name="b", dtype="float", type="num"),
    ]
    model._rebuild_name_to_col()
    model.class_labels = labels
    model._xgb_model = booster
    model.trees = model._extract_trees()
    model.base_score = model._extract_base_score()
    return model


def _missing_directions(trees):
    directions = set()
    pending = list(trees)
    while pending:
        node = pending.pop()
        if not (is_decision_node(node) or is_switch_node(node)):
            continue
        feature = node[0]
        if isinstance(feature, dict):
            directions.add(feature["missing"])
        if is_decision_node(node):
            pending.extend((node[3], node[4]))
        else:
            cases = node[2]
            pending.append(node[3])
            pending.extend(
                cases.values() if isinstance(cases, dict) else (v for _, v in cases)
            )
    return directions


def _assert_prediction_equal(task, expected, actual):
    if task == "regression":
        assert actual == pytest.approx(expected, abs=1e-4)
    else:
        assert actual == expected


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_learned_missing_directions_match_booster_everywhere(tmp_path, task):
    model = _trained_model(task)
    assert _missing_directions(model.trees) == {"left", "right"}
    assert _missing_directions(json.loads(json.dumps(model.trees))) == {
        "left",
        "right",
    }

    path = tmp_path / f"{task}.cart"
    model.export(str(path))
    loaded = package_runner.load_model(str(path))
    package = Predictor(str(path))
    standalone = _bundled().Predictor(str(path))
    rebuilt = [
        rebuild_tree_from_cart(loaded, ["a", "b"], tree_idx)
        for tree_idx in range(loaded["n_trees"])
    ]
    assert _missing_directions(rebuilt) == {"left", "right"}

    for row in ([float("nan"), float("nan")], [float("nan"), 0.4], [0.3, float("nan")]):
        native = model.predict(row)
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
        for actual in (
            nested,
            package.predict(row),
            package.predict(row, missing="right"),
            standalone.predict(row),
            standalone.predict(row, missing="right"),
            package.predict_path(row)["prediction"],
            standalone.predict_path(row)["prediction"],
        ):
            _assert_prediction_equal(task, native, actual)

    all_missing = [float("nan"), float("nan")]
    for predictor in (package, standalone):
        native = model.predict(all_missing)
        for row in ([], [None], [None, None]):
            _assert_prediction_equal(task, native, predictor.predict(row))
        missing_path = predictor.predict_path(all_missing)
        assert all(
            step.get("missing") is True
            for tree in missing_path["trees"]
            for step in tree["path"]
        )
        present_path = predictor.predict_path([0.0, 0.0])
        assert all(
            "missing" not in step
            for tree in present_path["trees"]
            for step in tree["path"]
        )
