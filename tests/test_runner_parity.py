"""Cross-runner parity + adversarial-model corpus.

The package runner (``cartlet.runner``) and the zero-dependency bundled runner
(``cartlet/bundled/predict.py``) intentionally duplicate the ``.cart`` traversal
logic, so they must agree byte-for-byte on every input. This module builds a
small corpus of models that historically broke one runner or the other --
non-ASCII string tables, n-ary switch nodes, short/missing vectors, gzip -- and
asserts both runners return the same value (or fail the same way) for each.

It also pins the writer's hard limits (feature index, pool sizes) so oversized
models fail with a clear ValueError rather than silently corrupting output.
"""

from __future__ import annotations

import gzip
import importlib.util
import os

import pytest

import cartlet.runner as pkg_runner
from cartlet.io.bytes import write_tree_bytes
from cartlet.types import FeatureSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_PY = os.path.join(REPO_ROOT, "cartlet", "bundled", "predict.py")


def _load_bundled_module():
    spec = importlib.util.spec_from_file_location("bundled_predict", PREDICT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bundled = _load_bundled_module()


def _outcome(fn):
    """Return ('ok', value) or ('err', ExceptionType) without raising."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001 - we are comparing failure modes
        return ("err", type(e))


def _assert_parity(cart_path, vector):
    """Both runners must return the same value or the same error type."""
    pkg = _outcome(lambda: pkg_runner.predict(pkg_runner.load_model(cart_path), vector))
    bnd = _outcome(lambda: bundled.predict(bundled.load_cart(cart_path), vector))
    assert pkg[0] == bnd[0], f"one runner errored, the other did not: {pkg} vs {bnd}"
    if pkg[0] == "ok":
        assert pkg[1] == bnd[1], f"runners disagree: {pkg[1]!r} vs {bnd[1]!r}"
    return pkg


# =============================================================================
# H2: non-ASCII string table (last interned string is non-ASCII)
# =============================================================================


def test_non_ascii_feature_name_roundtrips(tmp_path):
    """A non-ASCII feature name (interned last) must not corrupt the parse."""
    # ["categoría", "=", "café", "sí", "no"]: the feature name is the last
    # string added to the pool, so a char-vs-byte length bug misparses here.
    tree = ["categoría", "=", "café", "sí", "no"]
    specs = [FeatureSpec(name="categoría", dtype="str", type="cat", values={"café"})]
    path = str(tmp_path / "unicode.cart")
    write_tree_bytes(path, tree, specs, {"categoría": 0}, ["sí", "no"], False)

    assert _assert_parity(path, ["café"]) == ("ok", "sí")
    assert _assert_parity(path, ["té"]) == ("ok", "no")


# =============================================================================
# H3: missing feature at a switch node -> default branch (not decision 0)
# =============================================================================


def _write_switch_model(path):
    # color == red/blue -> warm/cool; anything else (or missing) -> "unknown".
    tree = ["color", "switch", {"red": "warm", "blue": "cool"}, "unknown"]
    specs = [FeatureSpec(name="color", dtype="str", type="cat", values={"red", "blue"})]
    write_tree_bytes(
        path, tree, specs, {"color": 0}, ["warm", "cool", "unknown"], False
    )


def test_switch_known_values(tmp_path):
    path = str(tmp_path / "switch.cart")
    _write_switch_model(path)
    assert _assert_parity(path, ["red"]) == ("ok", "warm")
    assert _assert_parity(path, ["blue"]) == ("ok", "cool")


def test_switch_unknown_value_takes_default(tmp_path):
    path = str(tmp_path / "switch.cart")
    _write_switch_model(path)
    assert _assert_parity(path, ["green"]) == ("ok", "unknown")


def test_switch_missing_feature_takes_default(tmp_path):
    """Short vector (feature absent) must hit the default, not decision node 0."""
    path = str(tmp_path / "switch.cart")
    _write_switch_model(path)
    # Empty vector: feature index 0 is past the end of the input.
    assert _assert_parity(path, []) == ("ok", "unknown")
    assert _assert_parity(path, [None]) == ("ok", "unknown")


# =============================================================================
# M1: non-numeric value at a numeric (OP_LT) node -> go right, both runners
# =============================================================================


def test_numeric_node_non_numeric_value(tmp_path):
    tree = ["x", "<", 5.0, "small", "big"]
    specs = [FeatureSpec(name="x", dtype="float", type="num")]
    path = str(tmp_path / "num.cart")
    write_tree_bytes(path, tree, specs, {"x": 0}, ["small", "big"], False)

    assert _assert_parity(path, [1.0]) == ("ok", "small")
    assert _assert_parity(path, [9.0]) == ("ok", "big")
    # A non-numeric string at a numeric node: comparison fails -> right ("big").
    assert _assert_parity(path, ["not-a-number"]) == ("ok", "big")
    assert _assert_parity(path, [None]) == ("ok", "big")


# =============================================================================
# M2: gzipped model loads in BOTH runners
# =============================================================================


def test_gzip_parity(tmp_path):
    plain = str(tmp_path / "switch.cart")
    _write_switch_model(plain)
    gz = str(tmp_path / "switch.cart.gz")
    with open(plain, "rb") as f, gzip.open(gz, "wb") as out:
        out.write(f.read())

    assert _assert_parity(gz, ["red"]) == ("ok", "warm")
    assert _assert_parity(gz, ["green"]) == ("ok", "unknown")


# =============================================================================
# H1: > 63 features must be rejected at export (no silent corruption)
# =============================================================================


def test_wide_feature_index_rejected(tmp_path):
    # A split on feature index 64 cannot fit the 6-bit packed field.
    names = [f"f{i}" for i in range(65)]
    tree = ["f64", "=", "yes", "a", "b"]
    specs = [FeatureSpec(name=n, dtype="str", type="cat") for n in names]
    name_to_col = {n: i for i, n in enumerate(names)}
    path = str(tmp_path / "wide.cart")
    with pytest.raises(ValueError, match="feature index"):
        write_tree_bytes(path, tree, specs, name_to_col, ["a", "b"], False)


# =============================================================================
# H4: a .cart with a switch node re-imports without crashing
# =============================================================================


def test_switch_model_reimport(tmp_path):
    from cartlet.io.cart_format import rebuild_tree_from_cart

    path = str(tmp_path / "switch.cart")
    _write_switch_model(path)
    model_data = pkg_runner.load_model(path)
    # This used to raise "too many values to unpack" on the switch node.
    rebuilt = rebuild_tree_from_cart(model_data, ["color"], tree_idx=0)
    assert rebuilt[0] == "color"
    assert rebuilt[1] == "switch"
