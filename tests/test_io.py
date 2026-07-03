"""Tests for IO modules: loader, writer, utils."""

import json

from cartlet.io.loader import iter_vectors, load_training_data, read_vectors
from cartlet.io.utils import detect_delimiter, detect_format, format_to_delimiter
from cartlet.io.writer import write_vectors


class TestWriter:
    """Tests for write_vectors."""

    def test_write_csv(self, tmp_path):
        """Write CSV format."""
        path = tmp_path / "test.csv"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data, header=["col1", "col2"])

        content = path.read_text()
        assert "col1,col2" in content
        assert "a,1" in content

    def test_write_tsv(self, tmp_path):
        """Write TSV format."""
        path = tmp_path / "test.tsv"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data, header=["col1", "col2"])

        content = path.read_text()
        assert "col1\tcol2" in content

    def test_write_json(self, tmp_path):
        """Write JSON format."""
        path = tmp_path / "test.json"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data, header=["col1", "col2"])

        content = json.loads(path.read_text())
        assert content == [{"col1": "a", "col2": "1"}, {"col1": "b", "col2": "2"}]

    def test_write_jsonl(self, tmp_path):
        """Write JSONL format."""
        path = tmp_path / "test.jsonl"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data, header=["col1", "col2"])

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"col1": "a", "col2": "1"}

    def test_write_single_values(self, tmp_path):
        """Write list of single values (predictions)."""
        path = tmp_path / "test.csv"
        data = ["yes", "no", "yes"]
        write_vectors(str(path), data)

        content = path.read_text()
        assert "yes" in content
        assert "no" in content

    def test_write_no_header(self, tmp_path):
        """Write without header."""
        path = tmp_path / "test.csv"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data)

        content = path.read_text()
        assert "a,1" in content

    def test_write_json_no_header(self, tmp_path):
        """Write JSON without header (raw lists)."""
        path = tmp_path / "test.json"
        data = [["a", "1"], ["b", "2"]]
        write_vectors(str(path), data)

        content = json.loads(path.read_text())
        assert content == [["a", "1"], ["b", "2"]]

    def test_write_to_file_object(self, tmp_path):
        """Write to file object instead of path."""
        path = tmp_path / "test.csv"
        data = [["a", "1"], ["b", "2"]]
        with open(path, "w") as f:
            write_vectors(f, data, format="csv")

        content = path.read_text()
        assert "a,1" in content


class TestLoader:
    """Tests for load_training_data."""

    def test_load_csv(self, tmp_path):
        """Load CSV data."""
        path = tmp_path / "test.csv"
        path.write_text("a,b,target\n1,2,yes\n3,4,no\n")

        X, y, names, target_name = load_training_data(str(path))
        # Loader auto-converts numeric strings
        assert X == [[1, 2], [3, 4]]
        assert y == ["yes", "no"]
        assert names == ["a", "b"]
        assert target_name == "target"

    def test_load_tsv(self, tmp_path):
        """Load TSV data."""
        path = tmp_path / "test.tsv"
        path.write_text("a\tb\ttarget\n1\t2\tyes\n3\t4\tno\n")

        X, y, names, target_name = load_training_data(str(path))
        # Loader auto-converts numeric strings
        assert X == [[1, 2], [3, 4]]
        assert y == ["yes", "no"]

    def test_load_jsonl(self, tmp_path):
        """Load JSONL data."""
        path = tmp_path / "test.jsonl"
        path.write_text(
            '{"a": 1, "b": 2, "target": "yes"}\n{"a": 3, "b": 4, "target": "no"}\n'
        )

        X, y, names, target_name = load_training_data(str(path), target_col="target")
        assert len(X) == 2
        assert y == ["yes", "no"]
        assert target_name == "target"

    def test_load_no_header(self, tmp_path):
        """Load data without header."""
        path = tmp_path / "test.csv"
        path.write_text("1,2,yes\n3,4,no\n")

        X, y, names, target_name = load_training_data(str(path), has_header=False)
        # Loader auto-converts numeric strings
        assert X == [[1, 2], [3, 4]]
        assert y == ["yes", "no"]
        # Column names are 1-indexed when no header
        assert names == ["1", "2"]
        assert target_name == "3"

    def test_load_custom_target(self, tmp_path):
        """Load with custom target column."""
        path = tmp_path / "test.csv"
        path.write_text("target,a,b\nyes,1,2\nno,3,4\n")

        X, y, names, target_name = load_training_data(str(path), target_col="target")
        assert y == ["yes", "no"]
        assert "target" not in names
        assert target_name == "target"


class TestUtils:
    """Tests for IO utilities."""

    def test_detect_format_csv(self):
        """Detect CSV format."""
        assert detect_format("test.csv") == "csv"

    def test_detect_format_tsv(self):
        """Detect TSV format."""
        assert detect_format("test.tsv") == "tsv"

    def test_detect_format_jsonl(self):
        """Detect JSONL format."""
        assert detect_format("test.jsonl") == "jsonl"

    def test_detect_format_default(self, tmp_path):
        """Default to CSV for unknown extensions."""
        # detect_format sniffs file content for unknown extensions
        path = tmp_path / "test.txt"
        path.write_text("a,b,c\n1,2,3\n")
        assert detect_format(str(path)) == "csv"

    def test_detect_delimiter_comma(self, tmp_path):
        """Detect comma delimiter."""
        path = tmp_path / "test.csv"
        path.write_text("a,b,c\n1,2,3\n")
        assert detect_delimiter(str(path)) == ","

    def test_detect_delimiter_tab(self, tmp_path):
        """Detect tab delimiter."""
        path = tmp_path / "test.tsv"
        path.write_text("a\tb\tc\n1\t2\t3\n")
        assert detect_delimiter(str(path)) == "\t"

    def test_format_to_delimiter(self):
        """Convert format to delimiter."""
        assert format_to_delimiter("csv") == ","
        assert format_to_delimiter("tsv") == "\t"
        assert format_to_delimiter("ssv") == " "


class TestWriterUnknownExtension:
    """write_vectors must not content-sniff a not-yet-existing output path."""

    def test_write_unknown_extension_defaults_csv(self, tmp_path):
        # A .txt destination that does not exist used to raise FileNotFoundError
        # via detect_format's content sniff; it should default to CSV instead.
        out = tmp_path / "preds.txt"
        write_vectors(str(out), [["a", 1], ["b", 2]], header=["x", "y"])
        assert out.read_text().splitlines()[0] == "x,y"


class TestTargetColumnResolution:
    """Target-column precedence must be identical across loaders (M7)."""

    def test_digit_named_column_resolves_by_name(self, tmp_path):
        # A column literally named "2" must win over positional index 2.
        p = tmp_path / "d.csv"
        p.write_text("a,2,c\n10,20,30\n40,50,60\n")
        X, y, feature_names, target_name = load_training_data(str(p), target_col="2")
        assert target_name == "2"
        assert feature_names == ["a", "c"]
        assert y == [20, 50]

        rv_X, rv_y, rv_names, rv_target = read_vectors(str(p), target_col="2")
        assert rv_target == "2"
        assert rv_names == ["a", "c"]

    def test_jsonl_int_target_indexes_keys(self, tmp_path):
        # An int target_col must index the keys, not become the string "2".
        p = tmp_path / "d.jsonl"
        p.write_text('{"a": 1, "b": 2, "label": "x"}\n{"a": 3, "b": 4, "label": "y"}\n')
        X, y, feature_names, target_name = load_training_data(str(p), target_col=2)
        assert target_name == "label"
        assert y == ["x", "y"]


class TestStreamingBatchParity:
    """iter_vectors must agree with read_vectors on the same file (M8)."""

    def test_csv_streaming_matches_batch(self, tmp_path):
        p = tmp_path / "d.csv"
        # Includes a non-ASCII value (NFC normalization) and a ragged row.
        p.write_text("x,y,label\ncafé,1,a\nbad,row\nb,2,c\n")
        batch_X, batch_y, _, _ = read_vectors(str(p))
        stream = list(iter_vectors(str(p)))
        stream_features = [f for f, _ in stream]
        stream_labels = [lbl for _, lbl in stream]
        assert stream_features == batch_X
        assert stream_labels == batch_y
        # The malformed 2-column row was skipped by both paths.
        assert len(batch_X) == 2
