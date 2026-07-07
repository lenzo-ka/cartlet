"""Tests for CLI."""

import json

import pytest

from cartlet.cli import main
from cartlet.io import (
    detect_delimiter,
    detect_format,
    load_training_data,
)


class TestDetection:
    """Test format and delimiter detection."""

    def test_detect_format_csv(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b,c\n1,2,3\n")
        assert detect_format(str(p)) == "csv"

    def test_detect_format_tsv(self, tmp_path):
        p = tmp_path / "d.tsv"
        p.write_text("a\tb\tc\n1\t2\t3\n")
        assert detect_format(str(p)) == "tsv"

    def test_detect_format_jsonl(self, tmp_path):
        p = tmp_path / "d.cart"
        p.write_text('{"a": 1, "b": 2}\n')
        assert detect_format(str(p)) == "jsonl"

    def test_detect_format_jsonl_by_content(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text('{"a": 1, "b": 2}\n')
        assert detect_format(str(p)) == "jsonl"

    def test_detect_delimiter_csv(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b,c\n1,2,3\n")
        assert detect_delimiter(str(p)) == ","

    def test_detect_delimiter_tsv(self, tmp_path):
        p = tmp_path / "d.tsv"
        p.write_text("a\tb\tc\n1\t2\t3\n")
        assert detect_delimiter(str(p)) == "\t"

    def test_detect_format_ssv(self, tmp_path):
        p = tmp_path / "d.ssv"
        p.write_text("a b c\n1 2 3\n")
        assert detect_format(str(p)) == "ssv"

    def test_detect_delimiter_ssv(self, tmp_path):
        p = tmp_path / "d.ssv"
        p.write_text("a b c\n1 2 3\n")
        assert detect_delimiter(str(p)) == " "


class TestLoadData:
    """Test data loading functions."""

    def test_load_csv(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b,target\n1,2,x\n3,4,y\n")
        X, y, names, target_name = load_training_data(str(p))
        assert names == ["a", "b"]
        assert target_name == "target"
        assert len(X) == 2
        assert y == ["x", "y"]

    def test_load_csv_no_header(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("1,2,x\n3,4,y\n")
        X, y, names, target_name = load_training_data(str(p), has_header=False)
        # 1-indexed like Unix cut/paste
        assert names == ["1", "2"]
        assert target_name == "3"
        assert len(X) == 2

    def test_load_csv_target_column(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("target,a,b\nx,1,2\ny,3,4\n")
        X, y, names, target_name = load_training_data(str(p), target_col="target")
        assert names == ["a", "b"]
        assert target_name == "target"
        assert y == ["x", "y"]

    def test_load_jsonl(self, tmp_path):
        p = tmp_path / "d.cart"
        p.write_text(
            '{"a": 1, "b": 2, "target": "x"}\n{"a": 3, "b": 4, "target": "y"}\n'
        )
        X, y, names, target_name = load_training_data(str(p), target_col="target")
        assert set(names) == {"a", "b"}
        assert target_name == "target"
        assert len(X) == 2
        assert y == ["x", "y"]

    def test_load_jsonl_default_target(self, tmp_path):
        p = tmp_path / "d.cart"
        p.write_text('{"a": 1, "b": 2, "target": "x"}\n')
        X, y, names, target_name = load_training_data(str(p))
        # Default target is last key
        assert "target" not in names
        assert target_name == "target"
        assert y == ["x"]


class TestTrainCommand:
    """Test train command."""

    @pytest.fixture
    def csv_data(self, tmp_path):
        p = tmp_path / "train.csv"
        p.write_text(
            "color,size,fruit\n"
            "red,small,apple\n"
            "red,large,apple\n"
            "blue,small,blueberry\n"
            "blue,large,blueberry\n"
        )
        return str(p)

    @pytest.fixture
    def jsonl_data(self, tmp_path):
        p = tmp_path / "train.cart"
        p.write_text(
            '{"color": "red", "size": "small", "fruit": "apple"}\n'
            '{"color": "red", "size": "large", "fruit": "apple"}\n'
            '{"color": "blue", "size": "small", "fruit": "blueberry"}\n'
            '{"color": "blue", "size": "large", "fruit": "blueberry"}\n'
        )
        return str(p)

    def test_train_basic(self, csv_data, tmp_path):
        out = tmp_path / "m.cart"
        result = main(["train", csv_data, "-o", str(out)])
        assert result == 0
        assert out.exists()

    def test_train_with_target(self, csv_data, tmp_path):
        out = tmp_path / "m.cart"
        result = main(["train", csv_data, "-o", str(out), "-t", "fruit"])
        assert result == 0

    def test_train_with_test_file(self, csv_data, tmp_path):
        test_f = tmp_path / "test.csv"
        test_f.write_text("a,b,label\n5,6,unknown\n")
        out = tmp_path / "m.cart"
        result = main(["train", csv_data, "-o", str(out), "-e", str(test_f)])
        assert result == 0

    def test_train_with_test_file_cross_format(self, csv_data, tmp_path):
        # CSV train with JSONL test file
        test_f = tmp_path / "test.cart"
        test_f.write_text('{"color": "red", "size": "small", "fruit": "apple"}\n')
        out = tmp_path / "m.cart"
        result = main(
            ["train", csv_data, "-o", str(out), "-e", str(test_f), "-t", "fruit"]
        )
        assert result == 0

    def test_train_forest(self, csv_data, tmp_path):
        out = tmp_path / "m.cart"
        result = main(["train", csv_data, "-o", str(out), "-F", "-n", "5"])
        assert result == 0
        # Verify it's a valid .cart file with CART magic
        with open(out, "rb") as f:
            magic = f.read(4)
        assert magic == b"CART"

    def test_train_jsonl(self, jsonl_data, tmp_path):
        out = tmp_path / "m.cart"
        result = main(["train", jsonl_data, "-o", str(out), "-t", "fruit"])
        assert result == 0

    def test_train_with_test_split(self, csv_data, tmp_path):
        out = tmp_path / "m.cart"
        result = main(["train", csv_data, "-o", str(out), "-S", "0.5"])
        assert result == 0


class TestPredictCommand:
    """Test predict command."""

    @pytest.fixture
    def model_and_data(self, tmp_path):
        train_path = tmp_path / "train.csv"
        train_path.write_text(
            "color,size,fruit\n"
            "red,small,apple\n"
            "red,large,apple\n"
            "blue,small,blueberry\n"
            "blue,large,blueberry\n"
        )
        test_path = tmp_path / "test.csv"
        test_path.write_text(
            "color,size,fruit\nred,small,unknown\nblue,large,unknown\n"
        )
        model_path = tmp_path / "m.cart"
        main(["train", str(train_path), "-o", str(model_path)])
        return str(model_path), str(test_path)

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

    def test_predict_to_file(self, model_and_data, tmp_path):
        model_path, test_path = model_and_data
        out = tmp_path / "out.csv"
        result = main(
            ["predict", model_path, test_path, "-m", "append", "-o", str(out)]
        )
        assert result == 0
        assert "prediction" in out.read_text()

    def test_predict_custom_column(self, model_and_data, capsys):
        model_path, test_path = model_and_data
        result = main(
            ["predict", model_path, test_path, "-m", "append", "-p", "predicted_fruit"]
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "predicted_fruit" in captured.out


class TestPredictJsonl:
    """Test predict command with JSONL."""

    @pytest.fixture
    def model_and_jsonl_data(self, tmp_path):
        train_path = tmp_path / "train.cart"
        train_path.write_text(
            '{"color": "red", "size": "small", "fruit": "apple"}\n'
            '{"color": "red", "size": "large", "fruit": "apple"}\n'
            '{"color": "blue", "size": "small", "fruit": "blueberry"}\n'
            '{"color": "blue", "size": "large", "fruit": "blueberry"}\n'
        )
        test_path = tmp_path / "test.cart"
        test_path.write_text(
            '{"color": "red", "size": "small", "fruit": "unknown"}\n'
            '{"color": "blue", "size": "large", "fruit": "unknown"}\n'
        )
        model_path = tmp_path / "m.cart"
        main(["train", str(train_path), "-o", str(model_path), "-t", "fruit"])
        return str(model_path), str(test_path)

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
    def model_file(self, tmp_path):
        data_path = tmp_path / "conv_train.csv"
        data_path.write_text("a,b,label\n1,2,x\n3,4,y\n")
        model_path = tmp_path / "conv_model.cart"
        main(["train", str(data_path), "-o", str(model_path)])
        return str(model_path)

    @pytest.fixture
    def csv_data(self, tmp_path):
        p = tmp_path / "conv.csv"
        p.write_text("a,b,label\n1,2,unknown\n3,4,unknown\n")
        return str(p)

    @pytest.fixture
    def tsv_data(self, tmp_path):
        p = tmp_path / "conv.tsv"
        p.write_text("a\tb\tlabel\n1\t2\tunknown\n3\t4\tunknown\n")
        return str(p)

    @pytest.fixture
    def jsonl_data(self, tmp_path):
        p = tmp_path / "conv.cart"
        p.write_text(
            '{"a": 1, "b": 2, "label": "unknown"}\n'
            '{"a": 3, "b": 4, "label": "unknown"}\n'
        )
        return str(p)

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
    def ssv_data(self, tmp_path):
        p = tmp_path / "conv.ssv"
        p.write_text("a b label\n1 2 unknown\n3 4 unknown\n")
        return str(p)

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
    def model_and_test_data(self, tmp_path):
        # Training data with string targets (classification)
        train_path = tmp_path / "train.csv"
        train_path.write_text("x,label\na,cat\nb,dog\nc,cat\n")
        # Test data (subset of training for perfect accuracy)
        test_path = tmp_path / "test.csv"
        test_path.write_text("x,label\na,cat\nb,dog\n")
        model_path = tmp_path / "m.cart"
        main(["train", str(train_path), "-o", str(model_path)])
        return str(model_path), str(test_path)

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
    def model_file(self, tmp_path):
        data_path = tmp_path / "stats_train.csv"
        data_path.write_text("a,b,c\n1,2,x\n3,4,y\n5,6,x\n")
        model_path = tmp_path / "stats_model.cart"
        main(["train", str(data_path), "-o", str(model_path)])
        return str(model_path)

    @pytest.fixture
    def forest_model_file(self, tmp_path):
        data_path = tmp_path / "stats_forest_train.csv"
        data_path.write_text("a,b,c\n1,2,x\n3,4,y\n5,6,x\n7,8,y\n")
        model_path = tmp_path / "stats_forest.cart"
        main(["train", str(data_path), "-o", str(model_path), "-F", "-n", "5"])
        return str(model_path)

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

    def test_predict_missing_model(self, capsys, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b\n1,2\n")
        result = main(["predict", "/nonexistent/model.cart", str(p)])
        assert result == 1
        assert "Error:" in capsys.readouterr().err


class TestMalformedTrainingData:
    """Test handling of malformed training data rows."""

    def test_malformed_rows_skipped_with_warning(self, caplog, tmp_path):
        """Rows with wrong column count should be skipped with warning."""
        import logging

        p = tmp_path / "d.csv"
        p.write_text(
            "a,b,label\n"
            "1,2,yes\n"  # Valid
            "3,4\n"  # Malformed - missing column
            "5,6,no\n"  # Valid
            "7\n"  # Malformed - missing columns
        )

        with caplog.at_level(logging.WARNING, logger="cartlet"):
            X, y, features, target = load_training_data(str(p))

        # Should have 2 valid rows
        assert len(X) == 2
        assert len(y) == 2
        assert y == ["yes", "no"]

        # Should have logged warnings for malformed rows
        assert len(caplog.records) == 2
        assert "malformed row" in caplog.records[0].message.lower()
        assert "row 3" in caplog.records[0].message

    def test_all_valid_rows_no_warning(self, caplog, tmp_path):
        """Valid data should not produce warnings."""
        import logging

        p = tmp_path / "d.csv"
        p.write_text("a,b,label\n1,2,yes\n3,4,no\n")

        with caplog.at_level(logging.WARNING, logger="cartlet"):
            X, y, features, target = load_training_data(str(p))

        assert len(X) == 2
        assert len(caplog.records) == 0


class TestConfigPresets:
    """Test config preset handling."""

    def test_train_with_builtin_preset(self, capsys, tmp_path):
        """Train command with --config preset should work."""
        data = tmp_path / "d.csv"
        data.write_text("x,y,label\na,1,yes\nb,2,no\nc,3,yes\nd,4,no\n")
        out = tmp_path / "m.cart"
        # Use 'fast' preset which sets max_depth=10
        result = main(["train", str(data), "-o", str(out), "-c", "fast"])
        assert result == 0
        assert "Using preset config: fast" in capsys.readouterr().err

    def test_train_without_config_no_error(self, capsys, tmp_path):
        """Train command without --config should work (tests P0 bug fix)."""
        data = tmp_path / "d.csv"
        data.write_text("x,label\na,yes\nb,no\n")
        out = tmp_path / "m.cart"
        # This should NOT raise NameError for undefined config_name
        result = main(["train", str(data), "-o", str(out)])
        assert result == 0

    def test_train_cli_args_override_preset(self, capsys, tmp_path):
        """CLI args should override preset values."""
        data = tmp_path / "d.csv"
        data.write_text("x,label\n" + "".join(f"{i},{i % 2}\n" for i in range(10)))
        out = tmp_path / "m.cart"
        # 'fast' preset sets max_depth=10, but we override to 2
        result = main(["train", str(data), "-o", str(out), "-c", "fast", "-D", "2"])
        assert result == 0
        # Verify it's a valid .cart file
        with open(out, "rb") as rf:
            magic = rf.read(4)
        assert magic == b"CART"

    def test_preset_value_applied_and_overridable(self, tmp_path):
        """The preset value reaches args (config-as-default), and an explicit
        flag overrides it -- verified via --save-config, which dumps the
        effective args (guards the set_defaults-based config handling)."""

        from cartlet.cli import load_config

        data = tmp_path / "d.csv"
        data.write_text("x,label\n" + "".join(f"{i},{i % 2}\n" for i in range(10)))

        # 'fast' preset -> max_depth 10 flows through to the saved config.
        cfg1 = tmp_path / "c1.json"
        assert (
            main(
                [
                    "train",
                    str(data),
                    "-o",
                    str(tmp_path / "a.cart"),
                    "-c",
                    "fast",
                    "--save-config",
                    str(cfg1),
                ]
            )
            == 0
        )
        assert load_config(str(cfg1))["max_depth"] == 10

        # Explicit -D 2 overrides the preset.
        cfg2 = tmp_path / "c2.json"
        assert (
            main(
                [
                    "train",
                    str(data),
                    "-o",
                    str(tmp_path / "b.cart"),
                    "-c",
                    "fast",
                    "-D",
                    "2",
                    "--save-config",
                    str(cfg2),
                ]
            )
            == 0
        )
        assert load_config(str(cfg2))["max_depth"] == 2

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


class TestShortFlagConsistency:
    """Guard against short flags meaning different things across subcommands.

    A short flag reused for two unrelated options (historically ``-c`` was
    ``--config`` in ``train`` but ``--prediction-column`` in ``predict``, and
    ``-D`` was ``--max-depth`` vs ``--output-delimiter``) is a muscle-memory
    trap and a breaking change to fix after 1.0. This test pins the audited
    mapping so any future reuse fails loudly.

    A short flag may pair with more than one *long* name only when the names
    denote the same concept; the sole audited exception is ``-f``, used for
    both ``--format`` (inspect) and ``--output-format`` (predict), which are
    both "output format".
    """

    _ALLOWED_MULTI = {"-f": {"--format", "--output-format"}}

    def _iter_subparsers(self):
        import argparse

        from cartlet.cli import _build_parser

        parser = _build_parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                # choices maps every name *and alias* to its subparser; dedupe
                # by identity so aliases are not double-counted.
                seen = set()
                for name, sub in action.choices.items():
                    if id(sub) in seen:
                        continue
                    seen.add(id(sub))
                    yield name, sub

    def test_short_flags_have_one_meaning(self):
        short_to_longs: dict[str, set[str]] = {}
        for _name, sub in self._iter_subparsers():
            for act in sub._actions:
                shorts = [o for o in act.option_strings if len(o) == 2]
                longs = [o for o in act.option_strings if o.startswith("--")]
                for s in shorts:
                    short_to_longs.setdefault(s, set()).update(longs)

        offenders = {
            s: sorted(longs)
            for s, longs in short_to_longs.items()
            if len(longs) > 1 and longs != self._ALLOWED_MULTI.get(s)
        }
        assert not offenders, f"Short flags with inconsistent meanings: {offenders}"

    def test_known_collisions_are_resolved(self):
        short_to_longs: dict[str, set[str]] = {}
        for _name, sub in self._iter_subparsers():
            for act in sub._actions:
                for s in (o for o in act.option_strings if len(o) == 2):
                    longs = {o for o in act.option_strings if o.startswith("--")}
                    short_to_longs.setdefault(s, set()).update(longs)

        # -c is config-only; -D is max-depth-only; prediction-column moved to -p.
        assert short_to_longs.get("-c") == {"--config"}
        assert short_to_longs.get("-D") == {"--max-depth"}
        assert short_to_longs.get("-p") == {"--prediction-column"}


class TestConfigFile:
    """Save/load of config *files* (W3-6) — previously untested."""

    def _data(self, tmp_path):
        d = tmp_path / "d.csv"
        d.write_text("x,label\na,yes\nb,no\nc,yes\nd,no\n")
        return str(d)

    def test_save_config_json_then_train_with_it(self, tmp_path):
        from cartlet.cli import load_config

        data = self._data(tmp_path)
        cfg = tmp_path / "cfg.json"
        # --save-config writes the config and still trains.
        rc = main(
            [
                "train",
                data,
                "-o",
                str(tmp_path / "m.cart"),
                "-D",
                "3",
                "--save-config",
                str(cfg),
            ]
        )
        assert rc == 0
        assert cfg.exists()

        loaded = load_config(str(cfg))
        assert loaded["max_depth"] == 3

        # The saved file is usable as a -c config.
        rc2 = main(["train", data, "-o", str(tmp_path / "m2.cart"), "-c", str(cfg)])
        assert rc2 == 0

    def test_save_config_yaml_roundtrip(self, tmp_path):
        pytest.importorskip("yaml")
        from cartlet.cli import load_config

        data = self._data(tmp_path)
        cfg = tmp_path / "cfg.yaml"
        rc = main(
            [
                "train",
                data,
                "-o",
                str(tmp_path / "m.cart"),
                "-D",
                "4",
                "--save-config",
                str(cfg),
            ]
        )
        assert rc == 0
        assert load_config(str(cfg))["max_depth"] == 4


class TestIsolationForestFlagWarnings:
    """W3-8(b): isolation-forest warns on supervised flags it ignores."""

    def test_warns_on_prune(self, tmp_path, capsys):
        data = tmp_path / "d.csv"
        data.write_text("x,y\n1,2\n3,4\n5,6\n7,8\n")
        rc = main(
            [
                "train",
                str(data),
                "--isolation-forest",
                "--prune",
                "-o",
                str(tmp_path / "if.json"),
            ]
        )
        assert rc == 0
        err = capsys.readouterr().err
        assert "ignores" in err and "--prune" in err


class TestValidationFileRemoved:
    """W3-8(c): the never-implemented --validation-file flag is gone."""

    def test_validation_file_flag_rejected(self, tmp_path):
        data = tmp_path / "d.csv"
        data.write_text("x,label\na,yes\nb,no\n")
        # argparse exits(2) on an unknown option.
        with pytest.raises(SystemExit) as exc:
            main(["train", str(data), "-E", "val.csv"])
        assert exc.value.code == 2


class TestStatsHonesty:
    """W3-8(a): stats reports the real format version and no fake dtype."""

    def _model(self, tmp_path):
        data = tmp_path / "d.csv"
        data.write_text("x,label\na,yes\nb,no\nc,yes\nd,no\n")
        model = tmp_path / "m.cart"
        assert main(["train", str(data), "-o", str(model)]) == 0
        return str(model)

    def test_format_version_reflects_header(self, tmp_path, capsys):
        from cartlet.io.cart_format import VERSION

        model = self._model(tmp_path)
        capsys.readouterr()  # drop training output
        assert main(["stats", model, "--json"]) == 0
        out = capsys.readouterr().out
        stats = json.loads(out)
        assert stats["format_version"] == f"cart-{VERSION}"

    def test_human_stats_no_fake_str_dtype(self, tmp_path, capsys):
        model = self._model(tmp_path)
        capsys.readouterr()
        assert main(["stats", model]) == 0
        out = capsys.readouterr().out
        # .cart does not store dtype; the feature rows must not claim "str".
        assert "Features" in out
        assert " str " not in out

    def test_stats_output_file_has_no_status_line(self, tmp_path, capsys):
        """The 'Loading model from' status must go to stderr, not into the
        stats written to -o (it would otherwise head the saved file)."""
        model = self._model(tmp_path)
        capsys.readouterr()
        out_file = tmp_path / "stats.txt"
        assert main(["stats", model, "-o", str(out_file)]) == 0
        content = out_file.read_text()
        assert "Loading model from" not in content
        assert content.lstrip().startswith("=")  # the stats banner
        assert "Loading model from" in capsys.readouterr().err


class TestCliEndToEndCoverage:
    """End-to-end CLI flows that were previously untested (T5.3)."""

    def test_regression_train_predict_evaluate(self, tmp_path, capsys):
        """A numeric-target flow reports regression metrics (mse/mae) via CLI."""
        data = tmp_path / "reg.csv"
        data.write_text(
            "x,y,target\n1,2,10.5\n2,4,20.5\n3,6,30.5\n4,8,40.5\n5,10,50.5\n6,12,60.5\n"
        )
        model = tmp_path / "reg.cart"
        assert main(["train", str(data), "-o", str(model), "-T", "regression"]) == 0

        capsys.readouterr()
        assert main(["predict", str(model), str(data)]) == 0
        pred_out = capsys.readouterr().out.strip().splitlines()
        assert len(pred_out) == 6
        # Predictions parse as floats in the training-target range.
        for line in pred_out:
            assert 10.0 <= float(line) <= 61.0

        capsys.readouterr()
        assert main(["evaluate", str(model), str(data)]) == 0
        eval_out = capsys.readouterr().out
        assert "MSE" in eval_out and "MAE" in eval_out

    def test_unicode_labels_through_cli(self, tmp_path, capsys):
        """Unicode class labels survive a CLI train -> predict round trip."""
        data = tmp_path / "u.csv"
        data.write_text("x,label\na,café\nb,naïve\na,café\nb,naïve\n")
        model = tmp_path / "u.cart"
        assert main(["train", str(data), "-o", str(model)]) == 0

        capsys.readouterr()
        assert main(["predict", str(model), str(data)]) == 0
        out = capsys.readouterr().out
        assert "café" in out and "naïve" in out

    def test_isolation_forest_cli_train_json(self, tmp_path, capsys):
        """--isolation-forest trains, reports anomaly stats, and saves JSON."""
        data = tmp_path / "iso.csv"
        data.write_text("a,b\n1,1\n2,2\n3,3\n2,3\n1,2\n100,100\n")
        model = tmp_path / "iso.json"
        assert main(["train", str(data), "--isolation-forest", "-o", str(model)]) == 0
        assert model.exists()
        assert "Anomaly scores" in capsys.readouterr().err

    def test_cart_gz_predict_via_cli(self, tmp_path, capsys):
        """A gzipped .cart model trains and predicts through the CLI."""
        data = tmp_path / "d.csv"
        data.write_text("color,label\nred,apple\nblue,berry\nred,apple\nblue,berry\n")
        model = tmp_path / "m.cart.gz"
        assert main(["train", str(data), "-o", str(model)]) == 0
        assert model.exists()

        capsys.readouterr()
        assert main(["predict", str(model), str(data)]) == 0
        out = capsys.readouterr().out
        assert "apple" in out and "berry" in out
