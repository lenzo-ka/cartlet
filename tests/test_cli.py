"""Tests for CLI."""

import json
import os
import tempfile

import pytest

from cartlet.cli import main
from cartlet.io import (
    detect_delimiter,
    detect_format,
    load_training_data,
)


class TestDetection:
    """Test format and delimiter detection."""

    def test_detect_format_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n")
            f.flush()
            assert detect_format(f.name) == "csv"
            os.unlink(f.name)

    def test_detect_format_tsv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("a\tb\tc\n1\t2\t3\n")
            f.flush()
            assert detect_format(f.name) == "tsv"
            os.unlink(f.name)

    def test_detect_format_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cart", delete=False) as f:
            f.write('{"a": 1, "b": 2}\n')
            f.flush()
            assert detect_format(f.name) == "jsonl"
            os.unlink(f.name)

    def test_detect_format_jsonl_by_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write('{"a": 1, "b": 2}\n')
            f.flush()
            assert detect_format(f.name) == "jsonl"
            os.unlink(f.name)

    def test_detect_delimiter_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n")
            f.flush()
            assert detect_delimiter(f.name) == ","
            os.unlink(f.name)

    def test_detect_delimiter_tsv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("a\tb\tc\n1\t2\t3\n")
            f.flush()
            assert detect_delimiter(f.name) == "\t"
            os.unlink(f.name)

    def test_detect_format_ssv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ssv", delete=False) as f:
            f.write("a b c\n1 2 3\n")
            f.flush()
            assert detect_format(f.name) == "ssv"
            os.unlink(f.name)

    def test_detect_delimiter_ssv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ssv", delete=False) as f:
            f.write("a b c\n1 2 3\n")
            f.flush()
            assert detect_delimiter(f.name) == " "
            os.unlink(f.name)


class TestLoadData:
    """Test data loading functions."""

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,target\n1,2,x\n3,4,y\n")
            f.flush()
            X, y, names, target_name = load_training_data(f.name)
            assert names == ["a", "b"]
            assert target_name == "target"
            assert len(X) == 2
            assert y == ["x", "y"]
            os.unlink(f.name)

    def test_load_csv_no_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1,2,x\n3,4,y\n")
            f.flush()
            X, y, names, target_name = load_training_data(f.name, has_header=False)
            # 1-indexed like Unix cut/paste
            assert names == ["1", "2"]
            assert target_name == "3"
            assert len(X) == 2
            os.unlink(f.name)

    def test_load_csv_target_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("target,a,b\nx,1,2\ny,3,4\n")
            f.flush()
            X, y, names, target_name = load_training_data(f.name, target_col="target")
            assert names == ["a", "b"]
            assert target_name == "target"
            assert y == ["x", "y"]
            os.unlink(f.name)

    def test_load_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cart", delete=False) as f:
            f.write('{"a": 1, "b": 2, "target": "x"}\n')
            f.write('{"a": 3, "b": 4, "target": "y"}\n')
            f.flush()
            X, y, names, target_name = load_training_data(f.name, target_col="target")
            assert set(names) == {"a", "b"}
            assert target_name == "target"
            assert len(X) == 2
            assert y == ["x", "y"]
            os.unlink(f.name)

    def test_load_jsonl_default_target(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cart", delete=False) as f:
            f.write('{"a": 1, "b": 2, "target": "x"}\n')
            f.flush()
            X, y, names, target_name = load_training_data(f.name)
            # Default target is last key
            assert "target" not in names
            assert target_name == "target"
            assert y == ["x"]
            os.unlink(f.name)


class TestTrainCommand:
    """Test train command."""

    @pytest.fixture
    def csv_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("color,size,fruit\n")
            f.write("red,small,apple\n")
            f.write("red,large,apple\n")
            f.write("blue,small,blueberry\n")
            f.write("blue,large,blueberry\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def jsonl_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cart", delete=False) as f:
            f.write('{"color": "red", "size": "small", "fruit": "apple"}\n')
            f.write('{"color": "red", "size": "large", "fruit": "apple"}\n')
            f.write('{"color": "blue", "size": "small", "fruit": "blueberry"}\n')
            f.write('{"color": "blue", "size": "large", "fruit": "blueberry"}\n')
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_train_basic(self, csv_data):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
            result = main(["train", csv_data, "-o", out.name])
            assert result == 0
            assert os.path.exists(out.name)
            os.unlink(out.name)

    def test_train_with_target(self, csv_data):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
            result = main(["train", csv_data, "-o", out.name, "-t", "fruit"])
            assert result == 0
            os.unlink(out.name)

    def test_train_with_test_file(self, csv_data):
        # Create a separate test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as test_f:
            test_f.write("a,b,label\n5,6,unknown\n")
            test_f.flush()
            with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
                result = main(["train", csv_data, "-o", out.name, "-e", test_f.name])
                assert result == 0
            os.unlink(out.name)
            os.unlink(test_f.name)

    def test_train_with_test_file_cross_format(self, csv_data):
        # Test CSV train with JSONL test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cart", delete=False
        ) as test_f:
            test_f.write('{"color": "red", "size": "small", "fruit": "apple"}\n')
            test_f.flush()
            with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
                result = main(
                    [
                        "train",
                        csv_data,
                        "-o",
                        out.name,
                        "-e",
                        test_f.name,
                        "-t",
                        "fruit",
                    ]
                )
                assert result == 0
            os.unlink(out.name)
            os.unlink(test_f.name)

    def test_train_forest(self, csv_data):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
            result = main(["train", csv_data, "-o", out.name, "-F", "-n", "5"])
            assert result == 0
            # Verify it's a valid .cart file with CART magic
            with open(out.name, "rb") as f:
                magic = f.read(4)
            assert magic == b"CART"
            os.unlink(out.name)

    def test_train_jsonl(self, jsonl_data):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
            result = main(["train", jsonl_data, "-o", out.name, "-t", "fruit"])
            assert result == 0
            os.unlink(out.name)

    def test_train_with_test_split(self, csv_data):
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as out:
            result = main(["train", csv_data, "-o", out.name, "-S", "0.5"])
            assert result == 0
            os.unlink(out.name)


class TestPredictCommand:
    """Test predict command."""

    @pytest.fixture
    def model_and_data(self):
        # Create training data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as train_f:
            train_f.write("color,size,fruit\n")
            train_f.write("red,small,apple\n")
            train_f.write("red,large,apple\n")
            train_f.write("blue,small,blueberry\n")
            train_f.write("blue,large,blueberry\n")
            train_f.flush()
            train_path = train_f.name

        # Create test data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as test_f:
            test_f.write("color,size,fruit\n")
            test_f.write("red,small,unknown\n")
            test_f.write("blue,large,unknown\n")
            test_f.flush()
            test_path = test_f.name

        # Train model
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", train_path, "-o", model_path])

        yield model_path, test_path

        os.unlink(train_path)
        os.unlink(test_path)
        os.unlink(model_path)

    def test_predict_values_mode(self, model_and_data, capsys):
        model_path, test_path = model_and_data
        result = main(["predict", model_path, test_path, "-m", "values"])
        assert result == 0
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().split("\n") if ln]
        assert len(lines) == 2
        assert "apple" in lines[0]
        assert "blueberry" in lines[1]

    def test_predict_append_mode(self, model_and_data, capsys):
        model_path, test_path = model_and_data
        result = main(["predict", model_path, test_path, "-m", "append"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "prediction" in lines[0]  # Header has prediction column
        assert "apple" in lines[1]

    def test_predict_inplace_mode(self, model_and_data, capsys):
        model_path, test_path = model_and_data
        result = main(["predict", model_path, test_path, "-m", "inplace"])
        assert result == 0
        captured = capsys.readouterr()
        assert "unknown" not in captured.out  # "unknown" replaced with prediction
        assert "apple" in captured.out

    def test_predict_to_file(self, model_and_data):
        model_path, test_path = model_and_data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
            result = main(
                ["predict", model_path, test_path, "-m", "append", "-o", out.name]
            )
            assert result == 0
            with open(out.name) as f:
                content = f.read()
            assert "prediction" in content
            os.unlink(out.name)

    def test_predict_custom_column(self, model_and_data, capsys):
        model_path, test_path = model_and_data
        result = main(
            ["predict", model_path, test_path, "-m", "append", "-c", "predicted_fruit"]
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "predicted_fruit" in captured.out


class TestPredictJsonl:
    """Test predict command with JSONL."""

    @pytest.fixture
    def model_and_jsonl_data(self):
        # Create training data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cart", delete=False
        ) as train_f:
            train_f.write('{"color": "red", "size": "small", "fruit": "apple"}\n')
            train_f.write('{"color": "red", "size": "large", "fruit": "apple"}\n')
            train_f.write('{"color": "blue", "size": "small", "fruit": "blueberry"}\n')
            train_f.write('{"color": "blue", "size": "large", "fruit": "blueberry"}\n')
            train_f.flush()
            train_path = train_f.name

        # Create test data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cart", delete=False
        ) as test_f:
            test_f.write('{"color": "red", "size": "small", "fruit": "unknown"}\n')
            test_f.write('{"color": "blue", "size": "large", "fruit": "unknown"}\n')
            test_f.flush()
            test_path = test_f.name

        # Train model
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", train_path, "-o", model_path, "-t", "fruit"])

        yield model_path, test_path

        os.unlink(train_path)
        os.unlink(test_path)
        os.unlink(model_path)

    def test_predict_jsonl_values(self, model_and_jsonl_data, capsys):
        model_path, test_path = model_and_jsonl_data
        result = main(["predict", model_path, test_path])
        assert result == 0
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().split("\n") if ln]
        # JSONL values mode outputs {"prediction": "..."}
        for line in lines:
            record = json.loads(line)
            assert "prediction" in record

    def test_predict_jsonl_append(self, model_and_jsonl_data, capsys):
        model_path, test_path = model_and_jsonl_data
        result = main(["predict", model_path, test_path, "-m", "append"])
        assert result == 0
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().split("\n") if ln]
        for line in lines:
            record = json.loads(line)
            assert "color" in record
            assert "prediction" in record

    def test_predict_jsonl_to_csv(self, model_and_jsonl_data, capsys):
        model_path, test_path = model_and_jsonl_data
        result = main(["predict", model_path, test_path, "-m", "append", "-f", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "," in lines[0]  # CSV format
        assert "prediction" in lines[0]

    def test_predict_jsonl_to_tsv(self, model_and_jsonl_data, capsys):
        model_path, test_path = model_and_jsonl_data
        result = main(["predict", model_path, test_path, "-m", "append", "-f", "tsv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "\t" in lines[0]  # TSV format
        assert "prediction" in lines[0]


class TestCrossFormatConversion:
    """Test format conversion between CSV, TSV, and JSONL."""

    @pytest.fixture
    def model_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,label\n")
            f.write("1,2,x\n")
            f.write("3,4,y\n")
            f.flush()
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", data_path, "-o", model_path])
        os.unlink(data_path)

        yield model_path
        os.unlink(model_path)

    @pytest.fixture
    def csv_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,label\n")
            f.write("1,2,unknown\n")
            f.write("3,4,unknown\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def tsv_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("a\tb\tlabel\n")
            f.write("1\t2\tunknown\n")
            f.write("3\t4\tunknown\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def jsonl_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cart", delete=False) as f:
            f.write('{"a": 1, "b": 2, "label": "unknown"}\n')
            f.write('{"a": 3, "b": 4, "label": "unknown"}\n')
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_csv_to_tsv(self, model_file, csv_data, capsys):
        result = main(["predict", model_file, csv_data, "-m", "append", "-f", "tsv"])
        assert result == 0
        captured = capsys.readouterr()
        assert "\t" in captured.out
        assert "," not in captured.out.split("\n")[0]  # No commas in first line

    def test_csv_to_jsonl(self, model_file, csv_data, capsys):
        result = main(["predict", model_file, csv_data, "-m", "append", "-f", "jsonl"])
        assert result == 0
        captured = capsys.readouterr()
        for line in captured.out.strip().split("\n"):
            if line:
                record = json.loads(line)
                assert "prediction" in record

    def test_tsv_to_csv(self, model_file, tsv_data, capsys):
        result = main(["predict", model_file, tsv_data, "-m", "append", "-f", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        assert "," in captured.out

    def test_tsv_to_jsonl(self, model_file, tsv_data, capsys):
        result = main(["predict", model_file, tsv_data, "-m", "append", "-f", "jsonl"])
        assert result == 0
        captured = capsys.readouterr()
        for line in captured.out.strip().split("\n"):
            if line:
                record = json.loads(line)
                assert "prediction" in record

    def test_jsonl_to_csv(self, model_file, jsonl_data, capsys):
        result = main(["predict", model_file, jsonl_data, "-m", "append", "-f", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "," in lines[0]
        assert "prediction" in lines[0]

    def test_jsonl_to_tsv(self, model_file, jsonl_data, capsys):
        result = main(["predict", model_file, jsonl_data, "-m", "append", "-f", "tsv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "\t" in lines[0]
        assert "prediction" in lines[0]

    @pytest.fixture
    def ssv_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ssv", delete=False) as f:
            f.write("a b label\n")
            f.write("1 2 unknown\n")
            f.write("3 4 unknown\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_ssv_to_csv(self, model_file, ssv_data, capsys):
        result = main(["predict", model_file, ssv_data, "-m", "append", "-f", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        assert "," in captured.out

    def test_ssv_to_jsonl(self, model_file, ssv_data, capsys):
        result = main(["predict", model_file, ssv_data, "-m", "append", "-f", "jsonl"])
        assert result == 0
        captured = capsys.readouterr()
        for line in captured.out.strip().split("\n"):
            if line:
                record = json.loads(line)
                assert "prediction" in record

    def test_csv_to_ssv(self, model_file, csv_data, capsys):
        result = main(["predict", model_file, csv_data, "-m", "append", "-f", "ssv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # SSV should have spaces, not commas or tabs
        assert " " in lines[0]
        assert "prediction" in lines[0]

    def test_jsonl_to_ssv(self, model_file, jsonl_data, capsys):
        result = main(["predict", model_file, jsonl_data, "-m", "append", "-f", "ssv"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert " " in lines[0]
        assert "prediction" in lines[0]


class TestEvaluateCommand:
    """Test evaluate command."""

    @pytest.fixture
    def model_and_test_data(self):
        # Create training data with string targets (classification)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as train_f:
            train_f.write("x,label\n")
            train_f.write("a,cat\n")
            train_f.write("b,dog\n")
            train_f.write("c,cat\n")
            train_f.flush()
            train_path = train_f.name

        # Create test data (same as training for perfect accuracy)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as test_f:
            test_f.write("x,label\n")
            test_f.write("a,cat\n")
            test_f.write("b,dog\n")
            test_f.flush()
            test_path = test_f.name

        # Train model
        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", train_path, "-o", model_path])

        yield model_path, test_path

        os.unlink(train_path)
        os.unlink(test_path)
        os.unlink(model_path)

    def test_evaluate_basic(self, model_and_test_data, capsys):
        model_path, test_path = model_and_test_data
        result = main(["evaluate", model_path, test_path])
        assert result == 0
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out

    def test_evaluate_verbose(self, model_and_test_data, capsys):
        model_path, test_path = model_and_test_data
        result = main(["evaluate", model_path, test_path, "-v"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Precision" in captured.out
        assert "Recall" in captured.out

    def test_evaluate_alias(self, model_and_test_data, capsys):
        model_path, test_path = model_and_test_data
        result = main(["eval", model_path, test_path])
        assert result == 0


class TestStatsCommand:
    """Test stats command."""

    @pytest.fixture
    def model_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,x\n3,4,y\n5,6,x\n")
            f.flush()
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", data_path, "-o", model_path])
        os.unlink(data_path)

        yield model_path
        os.unlink(model_path)

    @pytest.fixture
    def forest_model_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,x\n3,4,y\n5,6,x\n7,8,y\n")
            f.flush()
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".cart", delete=False) as model_f:
            model_path = model_f.name
        main(["train", data_path, "-o", model_path, "-F", "-n", "5"])
        os.unlink(data_path)

        yield model_path
        os.unlink(model_path)

    def test_stats_basic(self, model_file, capsys):
        result = main(["stats", model_file])
        assert result == 0
        captured = capsys.readouterr()
        assert "MODEL STATISTICS" in captured.out
        assert "DecisionTree" in captured.out
        assert "Total nodes" in captured.out

    def test_stats_verbose(self, model_file, capsys):
        result = main(["stats", model_file, "-v"])
        assert result == 0
        captured = capsys.readouterr()
        # Verbose output for single tree shows total nodes
        assert "Total nodes" in captured.out

    def test_stats_forest(self, forest_model_file, capsys):
        result = main(["stats", forest_model_file])
        assert result == 0
        captured = capsys.readouterr()
        assert "RandomForest" in captured.out
        assert "Trees:" in captured.out

    def test_stats_forest_verbose(self, forest_model_file, capsys):
        result = main(["stats", forest_model_file, "-v"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Total nodes" in captured.out
        assert "Avg nodes/tree" in captured.out

    def test_stats_alias(self, model_file, capsys):
        result = main(["info", model_file])
        assert result == 0


class TestErrorHandling:
    """Test error handling."""

    def test_no_command(self, capsys):
        result = main([])
        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_train_missing_file(self, capsys):
        # main() converts expected errors to a clean message + exit code 1
        # rather than surfacing a raw traceback.
        result = main(["train", "/nonexistent/file.csv"])
        assert result == 1
        assert "Error:" in capsys.readouterr().err

    def test_predict_missing_model(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n")
            f.flush()
            result = main(["predict", "/nonexistent/model.cart", f.name])
            assert result == 1
            assert "Error:" in capsys.readouterr().err
            os.unlink(f.name)


class TestMalformedTrainingData:
    """Test handling of malformed training data rows."""

    def test_malformed_rows_skipped_with_warning(self, caplog):
        """Rows with wrong column count should be skipped with warning."""
        import logging

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,label\n")
            f.write("1,2,yes\n")  # Valid
            f.write("3,4\n")  # Malformed - missing column
            f.write("5,6,no\n")  # Valid
            f.write("7\n")  # Malformed - missing columns
            f.flush()

            with caplog.at_level(logging.WARNING, logger="cartlet"):
                X, y, features, target = load_training_data(f.name)

            # Should have 2 valid rows
            assert len(X) == 2
            assert len(y) == 2
            assert y == ["yes", "no"]

            # Should have logged warnings for malformed rows
            assert len(caplog.records) == 2
            assert "malformed row" in caplog.records[0].message.lower()
            assert "row 3" in caplog.records[0].message

            os.unlink(f.name)

    def test_all_valid_rows_no_warning(self, caplog):
        """Valid data should not produce warnings."""
        import logging

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,label\n")
            f.write("1,2,yes\n")
            f.write("3,4,no\n")
            f.flush()

            with caplog.at_level(logging.WARNING, logger="cartlet"):
                X, y, features, target = load_training_data(f.name)

            assert len(X) == 2
            assert len(caplog.records) == 0

            os.unlink(f.name)


class TestConfigPresets:
    """Test config preset handling."""

    def test_train_with_builtin_preset(self, capsys):
        """Train command with --config preset should work."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y,label\n")
            f.write("a,1,yes\n")
            f.write("b,2,no\n")
            f.write("c,3,yes\n")
            f.write("d,4,no\n")
            f.flush()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cart", delete=False
            ) as out:
                # Use 'fast' preset which sets max_depth=10
                result = main(["train", f.name, "-o", out.name, "-c", "fast"])

                assert result == 0
                captured = capsys.readouterr()
                assert "Using preset config: fast" in captured.err

                os.unlink(out.name)
            os.unlink(f.name)

    def test_train_without_config_no_error(self, capsys):
        """Train command without --config should work (tests P0 bug fix)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,label\n")
            f.write("a,yes\n")
            f.write("b,no\n")
            f.flush()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cart", delete=False
            ) as out:
                # This should NOT raise NameError for undefined config_name
                result = main(["train", f.name, "-o", out.name])
                assert result == 0

                os.unlink(out.name)
            os.unlink(f.name)

    def test_train_cli_args_override_preset(self, capsys):
        """CLI args should override preset values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,label\n")
            for i in range(10):
                f.write(f"{i},{i % 2}\n")
            f.flush()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cart", delete=False
            ) as out:
                # 'fast' preset sets max_depth=10, but we override to 2
                result = main(
                    ["train", f.name, "-o", out.name, "-c", "fast", "-D", "2"]
                )
                assert result == 0

                # Verify it's a valid .cart file
                with open(out.name, "rb") as rf:
                    magic = rf.read(4)
                assert magic == b"CART"

                os.unlink(out.name)
            os.unlink(f.name)

    def test_train_with_equals_form_preset(self, tmp_path, capsys):
        """--config=NAME (equals form) must be honored, not silently ignored."""
        data = tmp_path / "d.csv"
        data.write_text("x,label\na,yes\nb,no\nc,yes\nd,no\n")
        out = tmp_path / "m.cart"
        result = main(["train", str(data), "-o", str(out), "--config=fast"])
        assert result == 0
        assert "Using preset config: fast" in capsys.readouterr().err

    def test_train_unknown_preset_clean_error(self, tmp_path, capsys):
        """A mistyped preset yields a clean error listing valid presets."""
        data = tmp_path / "d.csv"
        data.write_text("x,label\na,yes\nb,no\n")
        result = main(
            ["train", str(data), "-o", str(tmp_path / "m.cart"), "-c", "fastt"]
        )
        assert result == 1
        err = capsys.readouterr().err
        assert "Unknown config preset" in err and "fastt" in err
