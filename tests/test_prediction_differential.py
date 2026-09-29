"""Seeded differential checks for fast and path prediction traversals."""

import importlib.util
import random
from pathlib import Path

import pytest

from cartlet import DecisionTree, RandomForest
from cartlet.runner import Predictor


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("differential_bundled", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcome(call, row, missing):
    try:
        return ("ok", call(row, missing=missing))
    except Exception as exc:  # noqa: BLE001 - failure modes are the test subject
        return ("error", type(exc).__name__)


def _assert_fast_path_and_implementation_parity(implementations, rows):
    for missing in ("error", "right"):
        for row in rows:
            outcomes = []
            for predict, predict_path in implementations:
                fast = _outcome(predict, row, missing)
                path = _outcome(predict_path, row, missing)
                if path[0] == "ok":
                    path = ("ok", path[1]["prediction"])
                assert fast == path
                outcomes.append(fast)
            assert outcomes == [outcomes[0]] * len(outcomes)


def _mixed_training_data():
    rows = []
    labels = []
    for number in (0.0, 1.0):
        for category in ("a", "b"):
            for bool_category in (False, True):
                for bool_number in (False, True):
                    rows.append([number, category, bool_category, bool_number])
                    labels.append(
                        f"{int(number)}-{category}-{int(bool_category)}-{int(bool_number)}"
                    )
    return rows * 3, labels * 3


def _seeded_probe_matrix():
    rng = random.Random(20260929)
    choices = [
        [0.0, 1.0, -2.5, None, float("nan"), "not-numeric"],
        ["a", "b", "unknown", None, float("nan")],
        [False, True, "true", "0", "invalid-bool", None, float("nan")],
        [False, True, "yes", "no", "invalid-bool", None, float("nan")],
    ]
    rows = [
        [float("nan"), "a", False, False],
        [0.0, float("nan"), False, False],
        [0.0, "a", float("nan"), False],
        [0.0, "a", False, float("nan")],
        [],
        [0.0],
        [0.0, "a"],
        [0.0, "a", False],
    ]
    for _ in range(96):
        length = rng.randrange(5)
        rows.append([rng.choice(choices[index]) for index in range(length)])
    return rows


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_seeded_mixed_model_fast_path_differential(tmp_path, kind):
    features = [
        {"name": "number", "dtype": "float", "type": "num"},
        {"name": "category", "dtype": "str", "type": "cat"},
        {"name": "bool-category", "dtype": "bool", "type": "cat"},
        {"name": "bool-number", "dtype": "bool", "type": "num"},
    ]
    rows, labels = _mixed_training_data()
    if kind == "tree":
        model = DecisionTree(features=features, max_depth=6)
        model.load_data(rows, labels)
        model.train(validation_split=0)
    else:
        model = RandomForest(
            n_estimators=3,
            max_features=None,
            bootstrap=False,
            features=features,
            max_depth=6,
        )
        model.load_data(rows, labels)
        model.train(random_state=17)
    cart_path = tmp_path / f"differential-{kind}.cart"
    model.export(str(cart_path))
    package = Predictor(str(cart_path))
    standalone = _bundled().Predictor(str(cart_path))
    observed_features = {
        step["feature"]
        for row in rows
        for tree in model.predict_path(row)["trees"]
        for step in tree["path"]
    }
    assert observed_features == {0, 1, 2, 3}
    _assert_fast_path_and_implementation_parity(
        [
            (model.predict, model.predict_path),
            (package.predict, package.predict_path),
            (standalone.predict, standalone.predict_path),
        ],
        _seeded_probe_matrix(),
    )


def test_seeded_xgboost_runner_fast_path_differential(tmp_path):
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
    model.train(random_state=19)
    cart_path = tmp_path / "differential-xgb.cart"
    model.export(str(cart_path))
    package = Predictor(str(cart_path))
    standalone = _bundled().Predictor(str(cart_path))
    rng = random.Random(20260929)
    values = [0.0, 1.5, 4.0, None, float("nan"), "not-numeric"]
    probes = [[], [None], [float("nan"), 1.0]]
    probes.extend(
        [rng.choice(values) for _ in range(rng.randrange(3))] for _ in range(64)
    )
    _assert_fast_path_and_implementation_parity(
        [
            (package.predict, package.predict_path),
            (standalone.predict, standalone.predict_path),
        ],
        probes,
    )
