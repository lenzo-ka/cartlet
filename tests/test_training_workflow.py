"""Actual operational workflows, not parser-only delegation tests."""

import json
from dataclasses import replace

import pytest

from cartlet import DecisionTree, TrainingSettings, convert, train_file, train_model
from cartlet.cli import main


def test_quiet_training_export_and_actual_populations(tmp_path, capsys):
    settings = TrainingSettings(
        test_split=0.2, validation_split=0.2, prune=True, random_state=3
    )
    output = tmp_path / "model.json"
    result = train_model(
        [["a"], ["b"]] * 10, ["A", "B"] * 10, settings=settings, output=output
    )
    assert result.training_samples == 12
    assert result.validation_samples == 4
    assert result.test_samples == 4
    assert result.source_samples == 20
    assert result.metrics["accuracy"] == 1
    assert capsys.readouterr() == ("", "")
    loaded = DecisionTree()
    metadata = loaded.load_model(str(output))
    assert loaded.predict(["a"]) == "A"
    assert metadata["metadata"]["training"]["training_samples"] == 12
    json.dumps(result.to_dict(), allow_nan=False)


def test_named_test_columns_and_cli_share_workflow(tmp_path, capsys):
    source = tmp_path / "train.csv"
    source.write_text(
        "color,size,label\nred,small,R\nblue,large,B\nred,large,R\nblue,small,B\n"
    )
    test = tmp_path / "test.csv"
    test.write_text("size,color,label\nsmall,red,R\nlarge,blue,B\n")
    result = train_file(source, settings=TrainingSettings(test_split=0), test_file=test)
    assert result.metrics["accuracy"] == 1
    assert capsys.readouterr() == ("", "")
    output = tmp_path / "model.json"
    assert (
        main(
            [
                "train",
                str(source),
                "--test-file",
                str(test),
                "-S",
                "0",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    saved = DecisionTree()
    metadata = saved.load_model(str(output))
    assert metadata["metadata"]["training"]["metrics"] == result.metrics


def test_isolation_uses_every_unlabeled_column_and_is_quiet(capsys):
    result = train_model(
        [[1, 2], [2, 3], [100, 200]],
        settings=TrainingSettings(
            model_type="isolation", n_estimators=3, random_state=1
        ),
    )
    assert len(result.model.feature_names) == 2
    assert result.training_samples == 3 and result.test_samples == 0
    assert result.metrics == {}
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "changes",
    [
        {"n_estimators": 0},
        {"test_split": 1},
        {"random_state": True},
        {"n_jobs": 0},
        {"model_type": "isolation", "prune": True},
    ],
)
def test_invalid_settings_preserve_export(tmp_path, changes):
    output = tmp_path / "existing.json"
    output.write_text("KEEP")
    with pytest.raises(ValueError):
        train_model(
            [["a"], ["b"]],
            ["A", "B"],
            settings=replace(TrainingSettings(), **changes),
            output=output,
        )
    assert output.read_text() == "KEEP"


def test_file_output_alias_rejected_before_writing(tmp_path):
    source = tmp_path / "train.csv"
    source.write_text("x,y\na,A\nb,B\n")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        train_file(source, output=source)
    assert source.read_bytes() == before


def test_conversion_decodes_json_once_and_returns_serializable_report(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    tree = DecisionTree()
    tree.model = {"A": 0.4, "B": 0.6}
    tree.export(str(source))
    original = json.load
    calls = []

    def recording_load(stream, *args, **kwargs):
        calls.append(stream.name)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(json, "load", recording_load)
    result = convert(str(source), str(target))
    assert calls == [str(source)]
    assert result.to_dict() == {
        "model_type": "DecisionTree",
        "input_path": str(source),
        "output_path": str(target),
    }
    loaded = DecisionTree()
    loaded.load_model(str(target))
    assert loaded.predict(["a"], return_dist=True) == {"A": 0.4, "B": 0.6}


def test_long_inline_feature_json_and_json_cli_report(tmp_path, capsys):
    names = [f"feature_{i}" for i in range(30)]
    source = tmp_path / "wide.csv"
    source.write_text(
        ",".join([*names, "label"]) + "\n" + ",".join([*["a"] * 30, "A"]) + "\n"
    )
    specs = json.dumps(dict.fromkeys(names, "cat"))
    assert len(specs) > 255
    result = train_file(source, features=specs, settings=TrainingSettings(test_split=0))
    assert result.training_samples == 1
    assert capsys.readouterr() == ("", "")
    assert main(["train", str(source), "--features", specs, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_samples"] == 1
    assert report["settings"]["model_type"] == "tree"


def test_weighted_workflow_counts_only_active_rows(capsys):
    result = train_model(
        [["a"], ["b"], ["a"]],
        ["A", "B", "A"],
        counts=[2, 0, 1],
        settings=TrainingSettings(test_split=0),
    )
    assert result.source_samples == result.training_samples == 2
    assert result.model.predict(["b"]) == "A"
    assert result.output_path is None
    assert capsys.readouterr() == ("", "")


def test_isolation_file_preserves_all_columns_and_explicit_names(tmp_path):
    source = tmp_path / "unlabeled.csv"
    source.write_text("1,2\n2,3\n10,20\n")
    result = train_file(
        source,
        has_header=False,
        column_names=["left", "right"],
        settings=TrainingSettings(model_type="isolation", n_estimators=2),
    )
    assert result.model.feature_names == ["left", "right"]
    assert len(result.model.X[0]) == 2


def test_regression_pruning_does_not_remove_training_rows(capsys):
    result = train_model(
        [[1], [2], [3], [4]],
        [1.0, 2.0, 3.0, 4.0],
        settings=TrainingSettings(task="regression", prune=True, test_split=0),
    )
    assert result.training_samples == 4
    assert result.validation_samples == 0
    assert result.warnings and "unsupported" in result.warnings[0]
    assert capsys.readouterr() == ("", "")


def test_python_feature_specs_supply_named_model_columns():
    result = train_model(
        [["a"], ["b"]],
        ["A", "B"],
        feature_specs=[{"name": "letter", "dtype": "str", "type": "cat"}],
        settings=TrainingSettings(test_split=0),
    )
    assert result.model.feature_names == ["letter"]
    assert result.model.predict(["a"]) == "A"
