"""CLI coverage for importance and leaf-path export."""

import json

from cartlet import DecisionTree
from cartlet.cli import main


def _model_and_data(tmp_path):
    model = DecisionTree(
        features=[
            {"name": "signal", "dtype": "float", "type": "num"},
            {"name": "noise", "dtype": "float", "type": "num"},
        ],
        task="classification",
        max_depth=2,
    )
    rows = [[float(index), float(index % 2)] for index in range(12)]
    targets = [int(index >= 6) for index in range(12)]
    model.load_data(rows, targets)
    model.train(validation_split=0)
    model_path = tmp_path / "model.cart"
    model.export(str(model_path))
    data_path = tmp_path / "heldout.csv"
    data_path.write_text(
        "noise,label,signal\n"
        + "".join(
            f"{row[1]},{target},{row[0]}\n"
            for row, target in zip(rows, targets, strict=True)
        ),
        encoding="utf-8",
    )
    return model_path, data_path


def test_importance_cli_json_groups_and_stderr(tmp_path, capsys):
    model, data = _model_and_data(tmp_path)
    result = main(
        [
            "importance",
            str(model),
            str(data),
            "--target",
            "label",
            "--groups",
            '{"joint":["signal","noise"]}',
            "--repeats",
            "2",
            "--random-seed",
            "3",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["baseline"] == 0
    assert report["importances"][0]["features"] == ["signal", "noise"]
    assert "Loading model" in captured.err


def test_importance_cli_tsv(tmp_path, capsys):
    model, data = _model_and_data(tmp_path)
    assert (
        main(
            [
                "importance",
                str(model),
                str(data),
                "--target",
                "label",
                "--repeats",
                "1",
                "--random-seed",
                "2",
                "--format",
                "tsv",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.startswith("name\tfeatures\tbaseline\tmean\tstd\tvalues\r\n")
    assert "signal" in captured.out
    assert captured.err


def test_leaves_cli_json_data_filter_and_tsv(tmp_path, capsys):
    model, data = _model_and_data(tmp_path)
    assert (
        main(
            [
                "leaves",
                str(model),
                "--data",
                str(data),
                "--target",
                "label",
                "--class",
                "1",
                "--min-support",
                "1",
                "--min-purity",
                "1",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    leaves = [leaf for tree in report["trees"] for leaf in tree["leaves"]]
    assert leaves
    assert all(leaf["predicted_class"] == "1" for leaf in leaves)
    assert all(leaf["data_purity"] == 1 for leaf in leaves)
    assert "Loading inspection data" in captured.err

    assert main(["leaves", str(model), "--format", "tsv"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("tree\tleaf\tprediction\tpredicted_class")


def test_leaves_cli_unlabeled_data_uses_all_columns(tmp_path, capsys):
    model, _ = _model_and_data(tmp_path)
    data = tmp_path / "features.jsonl"
    data.write_text(
        '{"noise": 0, "signal": 1}\n{"noise": 1, "signal": 10}\n',
        encoding="utf-8",
    )
    assert main(["leaves", str(model), "--data", str(data)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (
        sum(leaf["data_support"] for tree in report["trees"] for leaf in tree["leaves"])
        == 2
    )
