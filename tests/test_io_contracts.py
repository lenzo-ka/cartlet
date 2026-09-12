"""Discriminating IO and standalone parity regressions."""

import gzip

import pytest

from cartlet.bundled.predict import load_cart
from cartlet.bundled.predict import predict as bundled_predict
from cartlet.io.bytes import write_tree_bytes
from cartlet.io.loader import iter_vectors, load_training_data, read_vectors
from cartlet.io.writer import write_vectors
from cartlet.runner import load_model, predict
from cartlet.types import FeatureSpec
from cartlet.utils import eval_tree


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
    for reader in [load_training_data, read_vectors, lambda p: list(iter_vectors(p))]:
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
    node = ["x", "<", 1.0, "L", "R"]
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
    node = ["x", "lt", 1.0, "L", "R"]
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
    node = ["x", "lt", 1.0, [10.0, 0, 1], [20.0, 0, 1]]
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
    node = ["x", "<", threshold, [1e100, 0, 1], [2.0, 0, 1]]
    write_tree_bytes(path, node, [FeatureSpec("x", dtype="float")], {"x": 0}, [], True)
    for value in [1.0, threshold, 1.00000002]:
        expected = eval_tree(node, [value], {"x": 0})
        assert predict(load_model(path), [value]) == expected
        assert bundled_predict(load_cart(path), [value]) == expected
