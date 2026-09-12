"""Native Booster metadata and faithful model lifecycle."""

import pytest


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("suffix", [".json", ".ubj"])
def test_native_xgboost_roundtrip_restores_metadata_and_reexports_cart(
    tmp_path, task, suffix
):
    pytest.importorskip("xgboost")
    from cartlet import load_model, predict
    from cartlet.xgboost import XGBoostTree

    model = XGBoostTree(
        task=task,
        features=[{"name": "color", "dtype": "str", "type": "cat"}],
        n_estimators=2,
        min_child_weight=0,
    )
    X = [["red"], ["blue"], ["red"], ["blue"]]
    model.load_data(
        X, ["A", "B", "A", "B"] if task == "classification" else [1.0, 3.0, 1.0, 3.0]
    )
    model.train(random_state=0)
    native = tmp_path / ("model" + suffix)
    model.export(str(native))
    restored = XGBoostTree.load(str(native))
    assert restored.feature_names == model.feature_names
    assert restored.class_labels == model.class_labels
    assert restored._effective_task() == task
    for row in X:
        assert restored.predict(row) == model.predict(row)
    cart = tmp_path / "restored.cart"
    restored.export(str(cart))
    data = load_model(str(cart))
    for row in X:
        actual, expected = predict(data, row), model.predict(row)
        assert (
            actual == pytest.approx(expected)
            if task == "regression"
            else actual == expected
        )
    assert model._xgb_model.attr("cartlet_metadata") is None


def test_external_native_xgboost_without_metadata_fails_without_replacing_model(
    tmp_path,
):
    xgb = pytest.importorskip("xgboost")
    from cartlet.xgboost import XGBoostTree

    external = xgb.train(
        {"objective": "reg:squarederror"},
        xgb.DMatrix([[1.0], [2.0]], label=[1.0, 2.0]),
        num_boost_round=1,
    )
    path = tmp_path / "external.json"
    external.save_model(path)
    model = XGBoostTree(task="regression")
    model.load_data([[1.0], [2.0]], [1.0, 2.0])
    model.train(random_state=0)
    previous = model.predict([1.0])
    with pytest.raises(ValueError, match="lacks versioned"):
        model.load_model(str(path))
    assert model.predict([1.0]) == previous


def test_malformed_native_metadata_is_rejected_atomically(tmp_path):
    xgb = pytest.importorskip("xgboost")
    import json

    from cartlet.xgboost import XGBoostTree

    original = XGBoostTree(
        task="classification", features=[{"name": "x", "type": "cat"}], n_estimators=1
    )
    original.load_data([["a"], ["b"]], ["A", "B"])
    original.train(random_state=0)
    path = tmp_path / "valid.json"
    original.export(str(path))
    booster = xgb.Booster()
    booster.load_model(str(path))
    payload = json.loads(booster.attr("cartlet_metadata"))
    payload["feature_specs"][0]["values"] = {"a": 1, "b": 2}
    booster.set_attr(cartlet_metadata=json.dumps(payload))
    invalid = tmp_path / "invalid.json"
    booster.save_model(str(invalid))
    before = original.predict(["a"])
    with pytest.raises(ValueError, match="string list"):
        original.load_model(str(invalid))
    assert original.predict(["a"]) == before


@pytest.mark.parametrize("suffix", [".json", ".ubj"])
def test_single_class_xgboost_native_roundtrip(tmp_path, suffix):
    pytest.importorskip("xgboost")
    from cartlet import load_model, predict
    from cartlet.xgboost import XGBoostTree

    original = XGBoostTree(task="classification", n_estimators=1)
    original.load_data([[0.0], [1.0]], ["A", "A"])
    original.train(random_state=0)
    native = tmp_path / ("single" + suffix)
    original.export(str(native))
    restored = XGBoostTree.load(str(native))
    assert restored.predict([0.0]) == "A"
    assert restored.predict_proba([0.0]) == {"A": 1.0}
    cart = tmp_path / "single.cart"
    restored.export(str(cart))
    assert predict(load_model(str(cart)), [0.0], return_dist=True) == {"A": 1.0}


def test_xgboost_new_data_invalidates_old_model_and_bad_data_is_atomic(tmp_path):
    pytest.importorskip("xgboost")
    from cartlet.xgboost import XGBoostTree

    model = XGBoostTree(
        task="classification", features=[{"name": "x", "type": "cat"}], n_estimators=1
    )
    model.load_data([["a"], ["b"]], ["A", "B"])
    model.train(random_state=0)
    old = model.predict(["a"])
    with pytest.raises(ValueError, match="same length"):
        model.load_data([["c"], ["d"]], ["C"])
    assert model.predict(["a"]) == old
    model.load_data([["c"], ["d"]], ["C", "D"])
    with pytest.raises(ValueError, match="not trained"):
        model.predict(["c"])
    with pytest.raises(ValueError, match="No model"):
        model.export(str(tmp_path / "stale.json"))
    with pytest.raises(ValueError, match="No trees"):
        model.export(str(tmp_path / "stale.cart"))
    model.train(random_state=0)
    assert model.predict(["c"]) in {"C", "D"}
