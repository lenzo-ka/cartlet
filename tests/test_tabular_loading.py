"""Batch and streaming tabular loading share record and text contracts."""

import csv
from io import StringIO
from types import GeneratorType

import pytest

from cartlet.io import loader


@pytest.mark.parametrize(
    "delimiter,format", [(",", "csv"), ("\t", "tsv"), (" ", "ssv")]
)
@pytest.mark.parametrize("has_header", [True, False])
@pytest.mark.parametrize("labeled", [True, False])
def test_quoted_records_and_blank_ragged_rows(
    tmp_path, delimiter, format, has_header, labeled
):
    source = StringIO()
    source.write("\n")  # Empty records do not define the schema.
    writer = csv.writer(source, delimiter=delimiter, lineterminator="\n")
    if has_header:
        writer.writerow(["cafe\u0301", "value", "target"])
    writer.writerow([f"cafe\u0301{delimiter}quoted\ntext", "003", "yes"])
    writer.writerow([])
    writer.writerow(["ragged"])
    writer.writerow(["", "4.5", "no"])
    path = tmp_path / f"data.{format}"
    path.write_text(source.getvalue(), encoding="utf-8")

    features = [[f"café{delimiter}quoted\ntext", "003"], ["", "4.5"]]
    labels = ["yes", "no"]
    names = ["café", "value"] if has_header else ["1", "2"]
    expected_X = (
        features
        if labeled
        else [row + [label] for row, label in zip(features, labels, strict=True)]
    )
    expected_names = names if labeled else names + ["target" if has_header else "3"]
    batch = loader.read_vectors(str(path), has_header=has_header, labeled=labeled)
    assert batch == (
        expected_X,
        labels if labeled else None,
        expected_names,
        ("target" if has_header else "3") if labeled else None,
    )
    assert list(
        loader.iter_vectors(str(path), has_header=has_header, labeled=labeled)
    ) == list(zip(expected_X, labels if labeled else [None, None], strict=True))
    training = loader.load_training_data(
        str(path), delimiter=delimiter, has_header=has_header
    )
    assert training == (
        [[features[0][0], 3], ["", 4.5]],
        labels,
        names,
        "target" if has_header else "3",
    )


@pytest.mark.parametrize("content", ["", "\n\n", "x,target\n", "\nx,target\n\n"])
def test_empty_or_header_only_batch_errors_stream_exhausts(tmp_path, content):
    path = tmp_path / "data.csv"
    path.write_text(content)
    for read in (loader.read_vectors, loader.load_training_data):
        with pytest.raises(ValueError, match="Empty input|No data rows"):
            read(str(path))
    assert list(loader.iter_vectors(str(path))) == []


def test_only_ragged_data_retains_empty_batch_contract(tmp_path, caplog):
    path = tmp_path / "data.csv"
    path.write_text("x,target\nragged\n")
    assert loader.read_vectors(str(path)) == ([], [], ["x"], "target")
    assert loader.load_training_data(str(path)) == ([], [], ["x"], "target")
    assert list(loader.iter_vectors(str(path))) == []
    assert "row 2: expected 2 columns, got 1" in caplog.text


def test_quoted_empty_field_is_a_record(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text('target\n""\n')
    assert loader.read_vectors(str(path)) == ([[]], [""], [], "target")
    assert list(loader.iter_vectors(str(path))) == [([], "")]
    assert loader.load_training_data(str(path)) == ([[]], [""], [], "target")


@pytest.mark.parametrize("has_header", [True, False])
@pytest.mark.parametrize(
    "target,expected_y,names,target_name",
    [
        ("2", [20], ["left", "right"], "2"),
        (2, [30], ["left", "2"], "right"),
        ("0", [10], ["2", "right"], "left"),
    ],
)
def test_name_overrides_and_target_precedence(
    tmp_path, has_header, target, expected_y, names, target_name
):
    path = tmp_path / "data.csv"
    path.write_text(("old,header,names\n" if has_header else "") + "10,20,30\n")
    X, y, actual_names, actual_target = loader.load_training_data(
        str(path),
        has_header=has_header,
        column_names=["left", "2", "right"],
        target_col=target,
    )
    assert y == expected_y
    assert [[value for value in [10, 20, 30] if value != expected_y[0]]] == X
    assert actual_names == names
    assert actual_target == target_name
    with pytest.raises(ValueError, match="Column names count"):
        loader.load_training_data(
            str(path), has_header=has_header, column_names=["short"]
        )


def test_training_converts_before_reading_remaining_records(tmp_path, monkeypatch):
    """A batch result must not require a second complete raw-row population."""
    path = tmp_path / "data.csv"
    path.write_text("x,target\n1,a\n2,b\n")
    original_reader = csv.reader
    original_convert = loader.try_numeric
    converted = []

    def convert(value):
        converted.append(value)
        return original_convert(value)

    def checked_reader(*args, **kwargs):
        for index, row in enumerate(original_reader(*args, **kwargs)):
            if index == 2:
                assert converted == ["1", "a"], "read ahead before converting first row"
            yield row

    monkeypatch.setattr(loader, "try_numeric", convert)
    monkeypatch.setattr(csv, "reader", checked_reader)
    assert loader.load_training_data(str(path), delimiter=",") == (
        [[1], [2]],
        ["a", "b"],
        ["x"],
        "target",
    )


def test_stream_does_not_consume_next_data_record():
    source = StringIO("x,target\n1,a\n2,b\n")
    rows = loader.iter_vectors(source)
    assert next(rows) == (["1"], "a")
    assert source.readline() == "2,b\n"
    assert isinstance(rows, GeneratorType)
    rows.close()
    assert not source.closed
