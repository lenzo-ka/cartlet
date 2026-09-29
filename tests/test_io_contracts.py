"""Discriminating IO and standalone parity regressions."""

import gzip
from collections.abc import Callable
from typing import Any

import pytest

from cartlet import DecisionTree
from cartlet.bundled.predict import load_cart
from cartlet.bundled.predict import predict as bundled_predict
from cartlet.bundled.predict import predict_path as bundled_predict_path
from cartlet.io.bytes import ByteWriter, write_tree_bytes
from cartlet.io.cart_format import rebuild_tree_from_cart
from cartlet.io.loader import iter_vectors, load_training_data, read_vectors
from cartlet.io.writer import write_vectors
from cartlet.runner import load_model, predict, predict_path
from cartlet.types import FeatureSpec
from cartlet.utils import eval_tree


def test_format_2_is_refused_with_migration_message(tmp_path):
    import struct

    current = tmp_path / "current.cart"
    old = tmp_path / "format-2.cart"
    tree = DecisionTree(feature_names=["x"])
    tree.model = ["x", "=", "a", "A", "B"]
    tree.export(str(current))
    data = bytearray(current.read_bytes())
    data[4:6] = struct.pack("<H", 2)
    old.write_bytes(data)

    message = (
        "Unsupported format version 2: format 2 was written by Cartlet 0.6.0; "
        "re-export from the training model or retrain"
    )
    for loader in (load_model, load_cart):
        with pytest.raises(ValueError) as exc_info:
            loader(str(old))
        assert str(exc_info.value) == message


def test_default_training_preserves_97_3_leaf_distribution(tmp_path):
    rows = [["same"]] * 100
    labels = ["A"] * 97 + ["B"] * 3
    tree = DecisionTree(feature_names=["x"], max_depth=0)
    tree.load_data(rows, labels)
    tree.train(validation_split=0)

    json_path = str(tmp_path / "model.json")
    cart_path = str(tmp_path / "model.cart")
    tree.export(json_path)
    tree.export(cart_path)
    json_tree = DecisionTree()
    json_tree.load_model(json_path)

    expected = {"A": 0.97, "B": 0.03}
    assert tree.predict(["same"], return_dist=True) == pytest.approx(expected)
    assert json_tree.predict(["same"], return_dist=True) == pytest.approx(expected)
    normalized_q16_bound = 1 / (2 * 65535 - 2)
    assert predict(load_model(cart_path), ["same"], return_dist=True) == pytest.approx(
        expected, abs=normalized_q16_bound
    )
    assert bundled_predict(
        load_cart(cart_path), ["same"], return_dist=True
    ) == pytest.approx(expected, abs=normalized_q16_bound)


def test_q16_distributions_roundtrip_best_class_and_deduplicate(tmp_path):
    left = {"A": 0.400001, "B": 0.4, "C": 0.199999}
    right = {"B": 0.400001, "A": 0.4, "C": 0.199999}
    tree = DecisionTree(feature_names=["x"])
    tree.model = ["x", "=", "left", left, right]
    path = str(tmp_path / "q16.cart")
    tree.export(path, store_distributions=True)

    package_model = load_model(path)
    bundled_model = load_cart(path)
    rebuilt = rebuild_tree_from_cart(package_model, ["x"])
    normalized_q16_bound = 2 / (2 * 65535 - 3)

    for row, expected in [(["left"], left), (["right"], right)]:
        assert predict(package_model, row) == tree.predict(row)
        assert bundled_predict(bundled_model, row) == tree.predict(row)
        assert predict(package_model, row, return_dist=True) == pytest.approx(
            expected, abs=normalized_q16_bound
        )
        assert bundled_predict(bundled_model, row, return_dist=True) == pytest.approx(
            expected, abs=normalized_q16_bound
        )
        rebuilt_leaf = rebuilt[3] if row[0] == "left" else rebuilt[4]
        assert rebuilt_leaf == pytest.approx(expected, abs=normalized_q16_bound)

    writer = ByteWriter(store_distributions=True)
    first = writer._add_distribution({"A": 0.6, "B": 0.4})
    second = writer._add_distribution({"A": 0.6000001, "B": 0.3999999})
    assert first == second
    assert len(writer.distributions) == 1


def test_wide_categorical_model_roundtrip(tmp_path):
    n_features = 112

    def value(feature: int, category: int) -> str:
        return f"f{feature:03d}-v{category:05d}-" + "x" * 72

    features = [
        FeatureSpec(
            name=f"feature_{feature:03d}",
            dtype="str",
            type="cat",
            values={value(feature, category) for category in range(256)},
        )
        for feature in range(n_features)
    ]
    tree = DecisionTree(features=features)
    tree.model = [features[-1].name, "=", value(111, 0), "hit", "miss"]
    path = tmp_path / "wide.cart"
    tree.export(str(path))

    package_model = load_model(str(path))
    bundled_model = load_cart(str(path))
    assert len(package_model["strings"]) >= n_features * 256
    assert sum(len(item.encode("utf-8")) + 1 for item in package_model["strings"]) > (
        2_350_000
    )
    assert package_model["decisions"][0][0] == 111
    assert bundled_model["decisions"][0][0] == 111

    rows = [
        [value(feature, (row + feature) % 256) for feature in range(n_features)]
        for row in range(300)
    ]
    rows[0][-1] = value(111, 0)
    for row in rows:
        expected = tree.predict(row)
        expected_path = tree.predict_path(row)
        assert predict(package_model, row) == expected
        assert bundled_predict(bundled_model, row) == expected
        assert predict_path(package_model, row) == expected_path
        assert bundled_predict_path(bundled_model, row) == expected_path


def test_explicit_min_confidence_still_collapses_97_3_leaf():
    tree = DecisionTree(feature_names=["x"], max_depth=0, min_confidence=0.95)
    tree.load_data([["same"]] * 100, ["A"] * 97 + ["B"] * 3)
    tree.train(validation_split=0)

    assert tree.model == "A"
    assert tree.predict(["same"], return_dist=True) == {"A": 1.0}


def test_runners_return_distribution_for_bare_class_leaf(tmp_path):
    tree = DecisionTree(feature_names=["x"], max_depth=0)
    tree.load_data([["same"]] * 3, ["only"] * 3)
    tree.train(validation_split=0)
    cart_path = str(tmp_path / "single.cart")
    tree.export(cart_path)

    expected = tree.predict(["same"], return_dist=True)
    assert expected == {"only": 1.0}
    assert predict(load_model(cart_path), ["same"], return_dist=True) == expected
    assert bundled_predict(load_cart(cart_path), ["same"], return_dist=True) == expected


def test_nonfinite_export_preserves_existing_output(tmp_path):
    path = tmp_path / "model.cart"
    path.write_bytes(b"KEEP")
    with pytest.raises(ValueError, match="float64"):
        write_tree_bytes(str(path), [float("inf"), 0, 1], [], {}, [], True)
    assert path.read_bytes() == b"KEEP"


@pytest.mark.parametrize("label", ["A\0B", "\0"])
def test_nul_strings_rejected_before_replace(tmp_path, label):
    path = tmp_path / "model.cart"
    path.write_bytes(b"KEEP")
    with pytest.raises(ValueError, match="NUL"):
        write_tree_bytes(str(path), label, [], {}, [label], False)
    assert path.read_bytes() == b"KEEP"


def test_headerless_jsonl_roundtrips_batch_stream(tmp_path):
    path = str(tmp_path / "rows.jsonl")
    write_vectors(path, [["e\u0301", "a"], ["b", "c"]], format="jsonl")
    x, y, names, target = read_vectors(path)
    assert (x, y, names, target) == ([["é"], ["b"]], ["a", "c"], ["1"], "2")
    assert list(iter_vectors(path)) == list(zip(x, y, strict=True))


@pytest.mark.parametrize(
    "text", ["[]\n", "{}\n", '{"x":1,"y":"a"}\n{"x":2}\n', '{"x":1,"y":null}\n']
)
def test_jsonl_schema_has_line_errors(tmp_path, text):
    path = str(tmp_path / "bad.jsonl")
    with open(path, "w") as f:
        f.write(text)
    readers: list[Callable[[str], Any]] = [
        load_training_data,
        read_vectors,
        lambda p: list(iter_vectors(p)),
    ]
    for reader in readers:
        with pytest.raises(ValueError, match="line [12]"):
            reader(path)


def test_jsonl_malformed_is_not_silently_skipped(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"x":1,"y":"a"}\nnot-json\n')
    with pytest.raises(ValueError, match="line 2"):
        list(iter_vectors(str(path)))


def test_writer_rejects_ragged_rows_without_replacing(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("KEEP")
    with pytest.raises(ValueError, match="columns"):
        write_vectors(str(path), [[1, 2, 3]], header=["x", "y"])
    assert path.read_text() == "KEEP"


def test_numeric_invalid_nested_exported_parity(tmp_path):
    node = ["x", "<=", 1.0, "L", "R"]
    path = str(tmp_path / "model.cart")
    write_tree_bytes(
        path, node, [FeatureSpec("x", dtype="float")], {"x": 0}, ["L", "R"], False
    )
    assert eval_tree(node, ["oops"], {"x": 0}) == "R"
    assert predict(load_model(path), ["oops"]) == "R"
    assert bundled_predict(load_cart(path), ["oops"]) == "R"


def test_gzip_content_detected_without_suffix(tmp_path):
    path = tmp_path / "model.cart"
    write_tree_bytes(str(path), "A", [], {}, ["A"], False)
    path.write_bytes(gzip.compress(path.read_bytes()))
    assert predict(load_model(str(path)), []) == "A"
    assert bundled_predict(load_cart(str(path)), []) == "A"


def test_strict_threshold_roundtrip(tmp_path):
    from cartlet.io.cart_format import rebuild_tree_from_cart

    path = str(tmp_path / "strict.cart")
    node = ["x", "<", 1.0, "L", "R"]
    write_tree_bytes(
        path, node, [FeatureSpec("x", dtype="float")], {"x": 0}, ["L", "R"], False
    )
    model = load_model(path)
    assert rebuild_tree_from_cart(model, ["x"]) == node
    for value, expected in [(0.5, "L"), (1.0, "R"), (2.0, "R")]:
        assert eval_tree(node, [value], {"x": 0}) == expected
        assert predict(model, [value]) == expected
        assert bundled_predict(load_cart(path), [value]) == expected


def test_multiclass_vector_intercepts(tmp_path):
    import math

    from cartlet.io.bytes import write_forest_bytes

    path = str(tmp_path / "xgb.cart")
    margins = [0.5, -0.25, -0.25]
    write_forest_bytes(
        path,
        [[0.0, 0, 1]] * 3,
        [],
        {},
        ["a", "b", "c"],
        False,
        metadata={"base_score": margins},
        is_xgboost=True,
    )
    expected = {
        key: math.exp(value) / sum(math.exp(x) for x in margins)
        for key, value in zip(["a", "b", "c"], margins, strict=True)
    }
    assert predict(load_model(path), [], return_dist=True) == pytest.approx(expected)
    assert bundled_predict(load_cart(path), [], return_dist=True) == pytest.approx(
        expected
    )


def test_atomic_serializer_failure_preserves_output(tmp_path):
    from cartlet.io.utils import write_with_optional_gzip

    path = tmp_path / "output.json"
    path.write_bytes(b"KEEP")

    def broken(output):
        with open(output, "wb") as stream:
            stream.write(b"partial")
        raise OSError("injected write failure")

    for compressed in [False, True]:
        with pytest.raises(OSError, match="injected"):
            write_with_optional_gzip(str(path), compressed, broken)
        assert path.read_bytes() == b"KEEP"
    assert sorted(f.name for f in tmp_path.iterdir()) == ["output.json"]


def test_xgboost_input_precision_matches_float32_threshold(tmp_path):
    import math
    import struct

    path = str(tmp_path / "xgb.cart")
    node = ["x", "<", 1.0, [10.0, 0, 1], [20.0, 0, 1]]
    write_tree_bytes(
        path,
        node,
        [FeatureSpec("x", dtype="float")],
        {"x": 0},
        [],
        True,
        is_xgboost=True,
    )
    previous_f32 = struct.unpack("<f", struct.pack("<I", 0x3F7FFFFF))[0]
    for value, expected in [
        (math.nextafter(1.0, 0.0), 20.0),
        (math.nextafter(1.0, 2.0), 20.0),
        (previous_f32, 10.0),
        (1.0, 20.0),
    ]:
        assert predict(load_model(path), [value]) == expected
        assert bundled_predict(load_cart(path), [value]) == expected


def test_bundle_and_gzip_reject_input_alias(tmp_path):
    from cartlet.io.bytes import bundle
    from cartlet.io.utils import gzip_file

    path = tmp_path / "model.cart"
    write_tree_bytes(str(path), "A", [], {}, ["A"], False)
    original = path.read_bytes()
    alias = tmp_path / "alias.cart"
    alias.symlink_to(path)
    for output in [path, alias]:
        for operation in [
            lambda output=output: bundle(str(path), str(output)),
            lambda output=output: gzip_file(str(path), str(output)),
        ]:
            with pytest.raises(ValueError):
                operation()
            assert path.read_bytes() == original


def test_native_float64_boundary_and_mean_preserved(tmp_path):
    threshold = 1.00000001
    path = str(tmp_path / "native.cart")
    node = ["x", "<=", threshold, [1e100, 0, 1], [2.0, 0, 1]]
    write_tree_bytes(path, node, [FeatureSpec("x", dtype="float")], {"x": 0}, [], True)
    for value in [1.0, threshold, 1.00000002]:
        expected = eval_tree(node, [value], {"x": 0})
        assert predict(load_model(path), [value]) == expected
        assert bundled_predict(load_cart(path), [value]) == expected


@pytest.mark.parametrize("dtype,values", [("bool", [False, True]), ("int", [1, 2])])
def test_feature_dtype_and_typed_vocabulary_roundtrip(tmp_path, dtype, values):
    from cartlet.bundled.predict import Predictor as BundledPredictor
    from cartlet.runner import Predictor

    path = str(tmp_path / "model.cart")
    node = ["x", "=", values[0], "A", "B"]
    write_tree_bytes(
        path,
        node,
        [FeatureSpec("x", dtype=dtype, type="cat", values=set(values))],
        {"x": 0},
        ["A", "B"],
        False,
    )
    model = load_model(path)
    assert model["meta"]["features"][0].get("dtype") == dtype
    assert set(model["meta"]["features"][0]["values"]) == set(values)
    assert load_cart(path)["features"][0]["dtype"] == dtype
    predictors: list[Any] = [Predictor(path), BundledPredictor(path)]
    for predictor in predictors:
        for value, expected in zip(values, ["A", "B"], strict=True):
            assert predictor.predict([value]) == expected
            assert not predictor.is_oov(0, value)


@pytest.mark.parametrize("forest", [False, True])
def test_bundle_json_decodes_once_and_preserves_predictions(
    tmp_path, monkeypatch, forest
):
    import json
    import runpy

    from cartlet import DecisionTree, RandomForest, bundle

    model = RandomForest(n_estimators=2) if forest else DecisionTree()
    model.load_data([["a"], ["b"], ["a"], ["b"]], ["A", "B", "A", "B"])
    model.train(random_state=1)
    source = tmp_path / "model.json"
    output = tmp_path / "predictor.py"
    model.export(str(source))
    original = json.load
    decoded = []

    def record(stream, *args, **kwargs):
        decoded.append(stream.name)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(json, "load", record)
    bundle(str(source), str(output), library_only=True)
    assert decoded == [str(source)]
    namespace = runpy.run_path(str(output))
    embedded = namespace["load_embedded"]()
    for row in [["a"], ["b"]]:
        assert namespace["predict"](embedded, row) == model.predict(row)


def test_bundle_rejected_schema_preserves_existing_output(tmp_path):
    from cartlet import bundle

    source = tmp_path / "bad.json"
    output = tmp_path / "existing.py"
    source.write_text(
        '{"schema_version":3,"model":[99,"=","a","A","B"],"feature_names":["x"]}'
    )
    output.write_bytes(b"KEEP")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="decision reference"):
        bundle(str(source), str(output))
    assert output.read_bytes() == b"KEEP"
    assert source.read_bytes() == before


def test_writer_refuses_models_beyond_loader_caps(tmp_path):
    import cartlet.bundled.predict as bundled
    import cartlet.runner as package

    assert (
        package._CART_MAX_FEATURES,
        package._CART_MAX_CLASSES,
        package._CART_MAX_TREES,
        package._CART_MAX_NODES,
    ) == (
        bundled._MAX_FEATURES,
        bundled._MAX_CLASSES,
        bundled._MAX_TREES,
        bundled._MAX_NODES,
    )
    classes = [f"c{i}" for i in range(package._CART_MAX_CLASSES + 1)]
    with pytest.raises(ValueError, match="number of classes"):
        write_tree_bytes(
            "c0",
            str(tmp_path / "classes.cart"),
            [FeatureSpec("x", "str", "cat", None)],
            {"x": 0},
            classes,
            False,
        )
    n = package._CART_MAX_FEATURES + 1
    with pytest.raises(ValueError, match="number of features"):
        write_tree_bytes(
            "c0",
            str(tmp_path / "features.cart"),
            [FeatureSpec(f"f{i}", "str", "cat", None) for i in range(n)],
            {f"f{i}": i for i in range(n)},
            ["c0"],
            False,
        )
