"""Prediction only reads features tested on evaluated decision paths."""

import importlib.util
from pathlib import Path

import pytest

from cartlet import DecisionTree, RandomForest
from cartlet.runner import Predictor


class FeatureNotYet(LookupError):
    pass


class LazyVector(list):
    def __init__(self, values, unavailable=()):
        super().__init__()
        self.values = values
        self.unavailable = set(unavailable)
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        assert isinstance(index, int), f"non-integer feature access: {index!r}"
        self.reads.append(index)
        if index in self.unavailable:
            raise FeatureNotYet(index)
        return self.values[index]

    def __iter__(self):
        raise AssertionError("feature vectors must not be iterated")


def _bundled():
    source = Path(__file__).parents[1] / "cartlet" / "bundled" / "predict.py"
    spec = importlib.util.spec_from_file_location("lazy_bundled_predict", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_reads(path):
    return [step["feature"] for tree in path["trees"] for step in tree["path"]]


def _exercise(predict, predict_path, row, untested):
    path_vector = LazyVector(row)
    result = predict_path(path_vector)
    assert path_vector.reads == _expected_reads(result)

    vector = LazyVector(row)
    assert predict(vector) == result["prediction"]
    assert vector.reads == _expected_reads(result)

    vector = LazyVector(row, unavailable={untested})
    assert predict(vector) == result["prediction"]

    tested = _expected_reads(result)[0]
    vector = LazyVector(row, unavailable={tested})
    with pytest.raises(FeatureNotYet):
        predict(vector)

    vector.unavailable.clear()
    assert predict(vector) == result["prediction"]


def _tree():
    model = DecisionTree(
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "unused", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
        max_depth=3,
    )
    rows = [[0, 9, 0], [0, 9, 1], [1, 9, 0], [1, 9, 1]] * 3
    model.load_data(rows, ["a", "b", "c", "d"] * 3)
    model.train(validation_split=0)
    return model


def test_tree_lazy_access_in_process_and_runners(tmp_path):
    model = _tree()
    row = [0, 999, 1]
    _exercise(model.predict, model.predict_path, row, 1)
    path = tmp_path / "tree.cart"
    model.export(str(path))
    for predictor in (Predictor(str(path)), _bundled().Predictor(str(path))):
        _exercise(predictor.predict, predictor.predict_path, row, 1)


def test_bool_feature_is_normalized_only_when_read(tmp_path):
    model = DecisionTree(
        features=[
            {"name": "flag", "dtype": "bool", "type": "cat"},
            {"name": "unused", "dtype": "float", "type": "num"},
        ],
        max_depth=2,
    )
    model.load_data([[False, 0.0], [True, 1.0]] * 5, ["off", "on"] * 5)
    model.train(validation_split=0)
    path = tmp_path / "bool-lazy.cart"
    model.export(str(path))

    predictors = [
        (model.predict, model.predict_path),
        (Predictor(str(path)).predict, Predictor(str(path)).predict_path),
    ]
    standalone = _bundled().Predictor(str(path))
    predictors.append((standalone.predict, standalone.predict_path))
    for predict, predict_path in predictors:
        vector = LazyVector([False, 999.0], unavailable={1})
        assert predict(vector) == "off"
        assert vector.reads == [0]
        path_vector = LazyVector(["true", 999.0], unavailable={1})
        result = predict_path(path_vector)
        assert result["prediction"] == "on"
        assert path_vector.reads == [0]


def test_forest_lazy_access_in_process_and_runners(tmp_path):
    base = _tree()
    rows = base.X
    model = RandomForest(
        n_estimators=3,
        max_features=2,
        bootstrap=False,
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "unused", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
        max_depth=3,
    )
    model.load_data(rows, ["a", "b", "c", "d"] * 3)
    model.train(random_state=8)
    row = [0, 999, 1]
    result = model.predict_path(row)
    used = set(_expected_reads(result))
    untested = next(index for index in range(3) if index not in used)
    _exercise(model.predict, model.predict_path, row, untested)
    path = tmp_path / "forest.cart"
    model.export(str(path))
    for predictor in (Predictor(str(path)), _bundled().Predictor(str(path))):
        _exercise(predictor.predict, predictor.predict_path, row, untested)


def test_xgboost_lazy_access_in_runners(tmp_path):
    pytest.importorskip("xgboost")
    from cartlet import XGBoostTree

    rows = [[0.0, 9.0, 0.0], [0.0, 9.0, 1.0], [1.0, 9.0, 0.0], [1.0, 9.0, 1.0]] * 5
    model = XGBoostTree(
        n_estimators=2,
        max_depth=2,
        min_child_weight=0,
        features=[
            {"name": "x", "dtype": "float", "type": "num"},
            {"name": "unused", "dtype": "float", "type": "num"},
            {"name": "y", "dtype": "float", "type": "num"},
        ],
    )
    model.load_data(rows, ["a", "b", "c", "d"] * 5)
    model.train(random_state=2)
    path = tmp_path / "xgb.cart"
    model.export(str(path))
    row = [0.0, 999.0, 1.0]
    for predictor in (Predictor(str(path)), _bundled().Predictor(str(path))):
        result = predictor.predict_path(row)
        used = set(_expected_reads(result))
        untested = next(index for index in range(3) if index not in used)
        _exercise(predictor.predict, predictor.predict_path, row, untested)
