"""Real model/workflow counterexamples for shared API contracts."""

import json

import pytest

from cartlet import (
    DecisionTree,
    RandomForest,
    confusion_matrix,
    cross_validate,
    per_class_metrics,
)
from cartlet.cli import main


@pytest.mark.parametrize("model_class", [DecisionTree, RandomForest])
@pytest.mark.parametrize(
    "X,y,weights",
    [
        ([[1], [2, 3]], ["a", "b"], None),
        ([[float("nan")]], ["a"], None),
        ([[1]], ["a"], [-1]),
        ([[1]], ["a"], [0]),
    ],
)
def test_models_reject_invalid_training_data(model_class, X, y, weights):
    with pytest.raises(ValueError):
        model_class().load_data(X, y, weights)


@pytest.mark.parametrize("metrics", [confusion_matrix, per_class_metrics])
def test_metrics_reject_misaligned_labels(metrics):
    with pytest.raises(ValueError):
        metrics(["a", "b"], ["a"])


def test_cli_eval_named_feature_alignment(tmp_path, capsys):
    model = tmp_path / "model.cart"
    tree = DecisionTree(
        features=[{"name": "x", "type": "cat"}, {"name": "z", "type": "cat"}]
    )
    tree.model = ["x", "=", "a", "A", "B"]
    tree.export(str(model))
    data = tmp_path / "test.csv"
    data.write_text("z,x,label\nq,a,A\nq,a,A\nq,b,B\n")
    assert main(["evaluate", str(model), str(data), "-J"]) == 0
    assert json.loads(capsys.readouterr().out)["correct"] == 3


def test_stale_sklearn_estimator_cannot_be_reexported(tmp_path):
    pytest.importorskip("sklearn")
    tree = DecisionTree(task="classification")
    tree.load_data([["a"], ["b"]], ["OLD", "OLD"])
    tree.train(trainer="sklearn")
    replacement = DecisionTree()
    replacement.model = "NEW"
    source = tmp_path / "new.json"
    replacement.export(str(source))
    tree.load_model(str(source))
    assert tree.predict(["a"]) == "NEW"
    with pytest.raises(ValueError):
        tree.export(str(tmp_path / "stale.skl"))


def test_cross_validation_balances_real_folds():
    sizes = []

    class RecordingTree(DecisionTree):
        def load_data(self, X, y, counts=None):
            sizes.append(6 - len(X))
            return super().load_data(X, y, counts)

    cross_validate(
        RecordingTree, [[str(i)] for i in range(6)], ["a"] * 6, n_folds=4, shuffle=False
    )
    assert sorted(sizes) == [1, 1, 2, 2]


def test_cli_outputs_cannot_overwrite_training_input(tmp_path, capsys):
    source = tmp_path / "data.jsonl"
    text = '{"x":"a","y":"A"}\n{"x":"b","y":"B"}\n'
    source.write_text(text)
    for option in ("-o", "--save-config"):
        assert main(["train", str(source), option, str(source)]) == 1
        assert source.read_text() == text
        capsys.readouterr()


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"max_dept": 1},
        {"max_depth": "oops"},
        {"trainer": "unknown"},
        {"forest": "false"},
        {"output": []},
    ],
)
def test_cli_config_schema_preserves_output(tmp_path, capsys, config):
    source = tmp_path / "data.csv"
    source.write_text("x,label\na,A\nb,B\n")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    output = tmp_path / "model.cart"
    output.write_bytes(b"keep")
    assert main(["train", str(source), "-c", str(cfg), "-o", str(output)]) == 1
    assert output.read_bytes() == b"keep"
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("fraction", ["1", "-1", "nan", "inf"])
def test_cli_split_bounds(tmp_path, capsys, fraction):
    source = tmp_path / "data.csv"
    source.write_text("x,label\na,A\nb,B\n")
    assert main(["train", str(source), "-S", fraction]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [
        {},
        [],
        None,
        {"model": []},
        {"model": {}},
        {"model": "A", "task": []},
        {"model": "A", "feature_specs": [{"name": "x", "values": [[]]}]},
    ],
)
def test_model_schema_errors(tmp_path, data):
    model = tmp_path / "bad.json"
    model.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        DecisionTree().load_model(str(model))


def test_zero_weights_are_removed_and_inputs_detached():
    X = [["a"], ["b"]]
    y = ["A", "B"]
    weights = [0, 2]
    tree = DecisionTree(task="classification")
    tree.load_data(X, y, weights)
    X[1][0] = "changed"
    y[1] = "changed"
    weights[1] = 100
    assert tree.X == [["b"]] and tree.y == ["B"] and tree.counts == [2]
    tree.train()
    assert tree.predict(["b"]) == "B"


def test_total_weight_must_be_finite():
    with pytest.raises(ValueError, match="total training weight"):
        DecisionTree().load_data([[1], [2]], ["a", "b"], [1e308, 1e308])


def test_python_row_sequences_and_explicit_array_boundary():
    tree = DecisionTree()
    tree.load_data([range(2), range(2, 4)], ["a", "b"])
    assert tree.X == [[0, 1], [2, 3]]
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="tolist"):
        tree.load_data(np.array([[1], [2]]), ["a", "b"])


def test_failed_model_load_is_transactional(tmp_path):
    pytest.importorskip("sklearn")
    tree = DecisionTree(task="classification")
    tree.load_data([["a"], ["b"]], ["OLD", "OLD"])
    tree.train(trainer="sklearn")
    estimator = tree._sklearn_model
    before = (tree.model, tree.X, tree.y, tree.feature_specs, tree.task)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        tree.load_model(str(bad))
    assert tree._sklearn_model is estimator
    assert (tree.model, tree.X, tree.y, tree.feature_specs, tree.task) == before
    assert tree.predict(["a"]) == "OLD"
    tree.export(str(tmp_path / "still-valid.skl"))


@pytest.mark.parametrize("suffix", ["json", "jsonl", "pkl", "json.gz"])
def test_schema_versioned_official_roundtrip(tmp_path, suffix):
    from cartlet.validation import MODEL_SCHEMA_VERSION

    tree = DecisionTree()
    tree.model = [0, "=", "a", "A", {"B": 0.4, "C": 0.6}]
    model = tmp_path / f"model.{suffix}"
    tree.export(str(model))
    loaded = DecisionTree()
    metadata = loaded.load_model(str(model))
    assert metadata["schema_version"] == MODEL_SCHEMA_VERSION
    assert loaded.predict(["a"]) == "A"
    assert loaded.predict(["b"], return_dist=True) == {"B": 0.4, "C": 0.6}


def test_missing_schema_rejected_actionably(tmp_path):
    model = tmp_path / "old.json"
    model.write_text('{"model":"A"}')
    with pytest.raises(ValueError, match="retrain or re-export"):
        DecisionTree().load_model(str(model))


@pytest.mark.parametrize(
    "changes",
    [
        {"model": [99, "=", "a", "A", "B"]},
        {"feature_specs": [{"name": "other", "dtype": "str", "type": "cat"}]},
        {
            "feature_specs": [
                {"name": "x", "dtype": "float", "type": "cat", "values": [float("nan")]}
            ]
        },
        {
            "feature_specs": [
                {"name": "x", "dtype": "float", "type": "cat", "values": [float("inf")]}
            ]
        },
    ],
)
def test_schema_feature_contract_failure_preserves_live_model(tmp_path, changes):
    data = {
        "schema_version": 2,
        "feature_names": ["x"],
        "feature_specs": [],
        "model": "NEW",
    }
    data.update(changes)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    tree = DecisionTree()
    tree.model = "OLD"
    with pytest.raises(ValueError):
        tree.load_model(str(path))
    assert tree.predict(["a"]) == "OLD"


def test_named_feature_positional_index_roundtrip(tmp_path):
    tree = DecisionTree(feature_names=["x"])
    tree.model = [0, "=", "a", "A", "B"]
    path = tmp_path / "valid.json"
    tree.export(str(path))
    loaded = DecisionTree()
    loaded.load_model(str(path))
    assert loaded.predict(["a"]) == "A"
