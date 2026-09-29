"""External expectations for scalar NaNs, bool predicates, and strict mode."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import cartlet.runner as package_runner
from cartlet import DecisionTree, MissingFeatureError, RandomForest
from cartlet.io.bytes import write_tree_bytes
from cartlet.types import FeatureSpec


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("missing_contract_bundled", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _models(tmp_path: Path, kind: str):
    if kind == "numeric":
        tree = ["x", "<=", 0.0, "left", "right"]
        feature = FeatureSpec("x", "float", "num")
        expected_right = "right"
    elif kind == "equality":
        tree = ["x", "=", "nan", "case", "default"]
        feature = FeatureSpec("x", "str", "cat")
        expected_right = "default"
    elif kind == "bool":
        tree = ["flag", "=", 1, "case", "default"]
        feature = FeatureSpec("flag", "bool", "cat", {0, 1})
        expected_right = "default"
    else:
        tree = ["x", "switch", {"nan": "case"}, "default"]
        feature = FeatureSpec("x", "str", "cat")
        expected_right = "default"

    nested = DecisionTree(
        features=[
            {
                "name": feature.name,
                "dtype": feature.dtype,
                "type": feature.type,
                "values": feature.values,
            }
        ]
    )
    nested.model = tree
    path = tmp_path / f"{kind}.cart"
    write_tree_bytes(
        str(path),
        tree,
        [feature],
        {feature.name: 0},
        ["left", "right", "case", "default"],
        False,
    )
    bundled = _bundled()
    return (
        (nested.predict, MissingFeatureError),
        (package_runner.Predictor(str(path)).predict, MissingFeatureError),
        (bundled.Predictor(str(path)).predict, bundled.MissingFeatureError),
        expected_right,
    )


@pytest.mark.parametrize("kind", ["numeric", "equality", "bool", "switch"])
@pytest.mark.parametrize("value", [Decimal("NaN")], ids=["decimal"])
def test_decimal_nan_is_missing_by_node_contract(tmp_path, kind, value):
    *implementations, expected = _models(tmp_path, kind)
    for predict, error in implementations:
        with pytest.raises(error):
            predict([value])
        assert predict([value], missing="right") == expected


@pytest.mark.parametrize("kind", ["numeric", "equality", "bool", "switch"])
def test_numpy_float32_nan_is_missing_by_node_contract(tmp_path, kind):
    numpy = pytest.importorskip("numpy")
    value = numpy.float32("nan")
    *implementations, expected = _models(tmp_path, kind)
    for predict, error in implementations:
        with pytest.raises(error):
            predict([value])
        assert predict([value], missing="right") == expected


def test_string_nan_is_missing_only_at_numeric_nodes(tmp_path):
    *numeric, expected = _models(tmp_path, "numeric")
    for predict, error in numeric:
        with pytest.raises(error):
            predict(["nan"])
        assert predict(["nan"], missing="right") == expected

    *equality, _ = _models(tmp_path, "equality")
    for predict, _error in equality:
        assert predict(["nan"]) == "case"

    *switch, _ = _models(tmp_path, "switch")
    for predict, _error in switch:
        assert predict(["nan"]) == "case"


@pytest.mark.parametrize("kind", ["equality", "switch"])
def test_categorical_nan_missing_right_uses_new_missing_definition(tmp_path, kind):
    *implementations, _ = _models(tmp_path, kind)
    for predict, _error in implementations:
        assert predict([float("nan")], missing="right") == "default"


def test_large_int_at_bool_numeric_split_raises_valueerror_everywhere(tmp_path):
    model = DecisionTree(
        features=[{"name": "flag", "dtype": "bool", "type": "num", "values": [0, 1]}]
    )
    model.model = ["flag", "<=", 0.5, "off", "on"]
    cart_path = tmp_path / "bool-numeric.cart"
    model.export(str(cart_path))
    package = package_runner.Predictor(str(cart_path))
    bundled = _bundled().Predictor(str(cart_path))
    value = 10**400

    routes = [
        model.predict,
        model.predict_path,
        package.predict,
        package.predict_path,
        bundled.predict,
        bundled.predict_path,
    ]
    for route in routes:
        with pytest.raises(ValueError, match="Cannot convert .* to bool"):
            route([value])
    with pytest.raises(ValueError, match="Cannot convert .* to bool"):
        model.predict([value], strict=True)


@pytest.mark.parametrize(
    ("predicate", "matching", "other"),
    [
        (True, True, False),
        ("true", True, False),
        (1, True, False),
        (0, False, True),
        (False, False, True),
    ],
)
def test_authored_bool_predicate_roundtrips(tmp_path, predicate, matching, other):
    model = DecisionTree(
        features=[{"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}]
    )
    model.model = ["flag", "=", predicate, "on", "off"]

    json_path = tmp_path / "bool.json"
    cart_path = tmp_path / "bool.cart"
    model.export(str(json_path))
    model.export(str(cart_path))
    loaded = DecisionTree()
    loaded.load_model(str(json_path))
    bundled = _bundled()
    implementations = [
        model.predict,
        loaded.predict,
        package_runner.Predictor(str(cart_path)).predict,
        bundled.Predictor(str(cart_path)).predict,
    ]
    for predict in implementations:
        assert predict([matching]) == "on"
        assert predict([other]) == "off"


@pytest.mark.parametrize("model_kind", ["tree", "forest"])
@pytest.mark.parametrize("value", [None, Decimal("NaN")])
@pytest.mark.parametrize("policy", ["error", "right"])
def test_strict_skips_missing_bool_before_evaluation(model_kind, value, policy):
    tree = DecisionTree(
        features=[{"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}]
    )
    tree.model = ["flag", "=", 1, "on", "off"]
    if model_kind == "forest":
        model: Any = RandomForest(
            n_estimators=1,
            features=[
                {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}
            ],
        )
        model.trees = [tree]
    else:
        model = tree

    if policy == "error":
        with pytest.raises(MissingFeatureError):
            model.predict([value], strict=True, missing=policy)
    else:
        assert model.predict([value], strict=True, missing=policy) == "off"


@pytest.mark.parametrize("model_kind", ["tree", "forest"])
@pytest.mark.parametrize("policy", ["error", "right"])
def test_strict_skips_absent_values_before_evaluation(model_kind, policy):
    tree = DecisionTree(
        features=[{"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}]
    )
    tree.model = ["flag", "=", 1, "on", "off"]
    if model_kind == "forest":
        model: Any = RandomForest(
            n_estimators=1,
            features=[
                {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}
            ],
        )
        model.trees = [tree]
    else:
        model = tree

    if policy == "error":
        with pytest.raises(MissingFeatureError):
            model.predict([], strict=True, missing=policy)
    else:
        assert model.predict([], strict=True, missing=policy) == "off"


def test_strict_missing_unvisited_bool_feature_is_ignored():
    model = DecisionTree(
        features=[
            {"name": "choice", "dtype": "str", "type": "cat", "values": ["a", "b"]},
            {"name": "unused", "dtype": "bool", "type": "cat", "values": [0, 1]},
        ]
    )
    model.model = ["choice", "=", "a", "yes", "no"]
    assert model.predict(["a", Decimal("NaN")], strict=True) == "yes"


@pytest.mark.parametrize("model_kind", ["tree", "forest"])
@pytest.mark.parametrize("policy", ["error", "right"])
def test_strict_still_rejects_present_invalid_bool(model_kind, policy):
    tree = DecisionTree(
        features=[{"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}]
    )
    tree.model = ["flag", "=", 1, "on", "off"]
    if model_kind == "forest":
        model: Any = RandomForest(
            n_estimators=1,
            features=[
                {"name": "flag", "dtype": "bool", "type": "cat", "values": [0, 1]}
            ],
        )
        model.trees = [tree]
    else:
        model = tree

    with pytest.raises(ValueError, match="Cannot convert 'maybe' to bool"):
        model.predict(["maybe"], strict=True, missing=policy)


@pytest.mark.parametrize("model_kind", ["tree", "forest"])
@pytest.mark.parametrize("policy", ["error", "right"])
def test_strict_still_rejects_present_oov_value(model_kind, policy):
    tree = DecisionTree(
        features=[{"name": "color", "dtype": "str", "type": "cat", "values": ["red"]}]
    )
    tree.model = ["color", "=", "red", "on", "off"]
    if model_kind == "forest":
        model: Any = RandomForest(
            n_estimators=1,
            features=[
                {"name": "color", "dtype": "str", "type": "cat", "values": ["red"]}
            ],
        )
        model.trees = [tree]
    else:
        model = tree

    with pytest.raises(ValueError, match="OOV values for features"):
        model.predict(["blue"], strict=True, missing=policy)


class _RaisingNe:
    def __ne__(self, other):
        raise RuntimeError("comparison failed")


class _NonBoolNe:
    def __ne__(self, other):
        return object()


@pytest.mark.parametrize("value", [_RaisingNe(), _NonBoolNe()])
def test_exotic_self_inequality_does_not_claim_missing(tmp_path, value):
    *implementations, _ = _models(tmp_path, "equality")
    for predict, _error in implementations:
        assert predict([value]) == "default"
