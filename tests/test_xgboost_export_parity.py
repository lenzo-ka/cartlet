"""Actual Booster precision and exported inference parity controls."""

from typing import Any

import pytest


@pytest.mark.parametrize(
    "task", ["regression", "binary", "multiclass", "binary-vector"]
)
def test_xgboost_export_matches_booster_at_float32_boundaries(tmp_path, task):
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    import json

    from cartlet import load_model, predict
    from cartlet.bundled.predict import load_cart
    from cartlet.bundled.predict import predict as bundled_predict
    from cartlet.xgboost import XGBoostTree

    if task == "binary-vector" and int(xgb.__version__.split(".")[0]) < 3:
        pytest.skip("vector intercept parameter requires current XGBoost")
    regression = task == "regression"
    labels: list[Any] = (
        [0.0, 0.0, 10.0, 10.0]
        if regression
        else ["A", "B", "C", "A"]
        if task == "multiclass"
        else ["A", "A", "B", "B"]
    )
    params: dict[str, Any] = (
        {"base_score": [0.2] if task == "binary-vector" else 0.2}
        if task in {"binary", "binary-vector"}
        else {}
    )
    tree = XGBoostTree(
        task="regression" if regression else "classification",
        n_estimators=2,
        max_depth=1,
        learning_rate=1.0,
        min_child_weight=0,
        **params,
    )
    tree.load_data([[0.0], [1.0], [2.0], [3.0]], labels)
    tree.train(random_state=0)
    artifact = tmp_path / "model.cart"
    tree.export(str(artifact))
    runtime, bundled = load_model(str(artifact)), load_cart(str(artifact))
    values = {0.0, 1.0, 2.0, 3.0}
    for raw in tree._xgb_model.get_dump(dump_format="json"):
        node = json.loads(raw)
        if "split_condition" not in node:
            continue
        threshold = np.float32(node["split_condition"])
        previous = np.nextafter(threshold, np.float32(-np.inf))
        following = np.nextafter(threshold, np.float32(np.inf))
        midpoint = (float(previous) + float(threshold)) / 2
        values.update(
            [
                float(threshold),
                float(previous),
                float(following),
                midpoint,
                float(np.nextafter(midpoint, -np.inf)),
                float(np.nextafter(midpoint, np.inf)),
                float(np.nextafter(float(threshold), -np.inf)),
                float(np.nextafter(float(threshold), np.inf)),
            ]
        )
    assert len(values) > 4
    for value in values:
        row = [value]
        expected = tree.predict(row)
        for model, predictor in [(runtime, predict), (bundled, bundled_predict)]:
            actual = predictor(model, row)
            if regression:
                assert actual == pytest.approx(expected, abs=1e-6)
            else:
                assert actual == expected
                assert predictor(model, row, return_dist=True) == pytest.approx(
                    tree.predict_proba(row), abs=1e-6
                )
