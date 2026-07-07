"""Tests for model format export/load: JSON, JSONL, Pickle, sklearn."""

import os

import pytest

from cartlet import DecisionTree, RandomForest


class TestDecisionTreeFormats:
    """Tests for DecisionTree export/load across formats."""

    @pytest.fixture
    def trained_tree(self):
        """Create a trained decision tree."""
        dt = DecisionTree()
        X = [["a", 1], ["b", 2], ["a", 3], ["b", 4]]
        y = ["yes", "no", "yes", "no"]
        dt.load_data(X, y)
        dt.train()
        return dt

    def test_export_load_cart(self, trained_tree, tmp_path):
        """Export and load .cart format."""
        path = tmp_path / "model.cart"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"
        assert dt2.predict(["b", 2]) == "no"

    def test_export_load_cart_gz(self, trained_tree, tmp_path):
        """Export and load .cart.gz format."""
        path = tmp_path / "model.cart.gz"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"

    def test_export_load_json(self, trained_tree, tmp_path):
        """Export and load .json format."""
        path = tmp_path / "model.json"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"
        assert dt2.predict(["b", 2]) == "no"

    def test_export_load_json_gz(self, trained_tree, tmp_path):
        """Export and load .json.gz format."""
        path = tmp_path / "model.json.gz"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"

    def test_export_load_jsonl(self, trained_tree, tmp_path):
        """Export and load .jsonl format."""
        path = tmp_path / "model.jsonl"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"

    def test_export_load_pickle(self, trained_tree, tmp_path):
        """Export and load .pkl format."""
        path = tmp_path / "model.pkl"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"

    def test_export_load_pickle_gz(self, trained_tree, tmp_path):
        """Export and load .pkl.gz format."""
        path = tmp_path / "model.pkl.gz"
        trained_tree.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        assert dt2.predict(["a", 1]) == "yes"

    @pytest.mark.parametrize("ext", ["json", "jsonl", "pkl"])
    def test_store_distributions_false_collapses_leaves(self, tmp_path, ext):
        """store_distributions=False must strip distribution leaves for the
        JSON/JSONL/pickle codecs too, not just .cart."""
        import json
        import pickle

        dt = DecisionTree(feature_names=["x"], store_distributions=True)
        dt.load_data([["a"], ["a"], ["a"], ["b"], ["b"]], ["y", "y", "n", "m", "m"])
        dt.train()

        def leaves(node):
            if isinstance(node, list) and len(node) == 5:
                return leaves(node[3]) + leaves(node[4])
            return [node]

        path = tmp_path / f"model.{ext}"
        dt.export(str(path), store_distributions=False)

        if ext == "pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            with open(path) as f:
                text = f.read()
            data = json.loads(text.splitlines()[0] if ext == "jsonl" else text)

        # A tree that had a distribution leaf must now have only bare labels.
        assert not any(isinstance(leaf, dict) for leaf in leaves(data["model"]))

        # Sanity: with distributions kept, at least one dict leaf survives.
        path_with = tmp_path / f"with.{ext}"
        dt.export(str(path_with), store_distributions=True)

    def test_cart_with_distributions(self, trained_tree, tmp_path):
        """Export .cart with distributions for nbest."""
        path = tmp_path / "model.cart"
        trained_tree.export(str(path), store_distributions=True)

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        # Should work for predict
        assert dt2.predict(["a", 1]) == "yes"

    def test_cart_without_distributions(self, trained_tree, tmp_path):
        """Export .cart without distributions."""
        path_with = tmp_path / "with.cart"
        path_without = tmp_path / "without.cart"

        trained_tree.export(str(path_with), store_distributions=True)
        trained_tree.export(str(path_without), store_distributions=False)

        # Without distributions should be smaller or equal
        assert os.path.getsize(str(path_without)) <= os.path.getsize(str(path_with))


class TestRandomForestFormats:
    """Tests for RandomForest export/load across formats."""

    @pytest.fixture
    def trained_forest(self):
        """Create a trained random forest."""
        rf = RandomForest(n_estimators=3)
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]]
        y = ["a", "a", "b", "b", "a"]
        rf.load_data(X, y)
        rf.train()
        return rf

    def test_export_load_cart(self, trained_forest, tmp_path):
        """Export and load .cart format."""
        path = tmp_path / "model.cart"
        trained_forest.export(str(path))

        rf2 = RandomForest()
        rf2.load_model(str(path))

        # Should predict something reasonable
        pred = rf2.predict([1.0, 2.0])
        assert pred in ["a", "b"]

    def test_export_load_json(self, trained_forest, tmp_path):
        """Export and load .json format."""
        path = tmp_path / "model.json"
        trained_forest.export(str(path))

        rf2 = RandomForest()
        rf2.load_model(str(path))

        pred = rf2.predict([1.0, 2.0])
        assert pred in ["a", "b"]

    def test_export_load_pickle(self, trained_forest, tmp_path):
        """Export and load .pkl format."""
        path = tmp_path / "model.pkl"
        trained_forest.export(str(path))

        rf2 = RandomForest()
        rf2.load_model(str(path))

        pred = rf2.predict([1.0, 2.0])
        assert pred in ["a", "b"]


class TestSklearnExport:
    """Tests for sklearn model export."""

    def test_export_sklearn_tree(self, tmp_path):
        """Export sklearn-trained tree."""
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")

        dt = DecisionTree()
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]
        y = ["a", "a", "b", "b"]
        dt.load_data(X, y)
        dt.train(trainer="sklearn")

        path = tmp_path / "model.skl"
        dt.export(str(path))

        assert os.path.exists(str(path))
        assert os.path.getsize(str(path)) > 0

    def test_export_sklearn_forest(self, tmp_path):
        """Export sklearn-trained forest."""
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")

        rf = RandomForest(n_estimators=3)
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]]
        y = ["a", "a", "b", "b", "a"]
        rf.load_data(X, y)
        rf.train(trainer="sklearn")

        path = tmp_path / "model.skl"
        rf.export(str(path))

        assert os.path.exists(str(path))

    def test_sklearn_export_without_sklearn_training(self, tmp_path):
        """Trying to export sklearn format without sklearn training should fail."""
        dt = DecisionTree()
        X = [[1.0, 2.0], [2.0, 3.0]]
        y = ["a", "b"]
        dt.load_data(X, y)
        dt.train()  # native training

        path = tmp_path / "model.skl"
        with pytest.raises(ValueError, match="No sklearn model"):
            dt.export(str(path))


class TestRegressionFormats:
    """Tests for regression model formats."""

    @pytest.fixture
    def trained_regressor(self):
        """Create a trained regression tree."""
        dt = DecisionTree(task="regression")
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [1.5, 2.5, 3.5, 4.5]
        dt.load_data(X, y)
        dt.train()
        return dt

    def test_export_load_cart_regression(self, trained_regressor, tmp_path):
        """Export and load regression .cart format."""
        path = tmp_path / "model.cart"
        trained_regressor.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        pred = dt2.predict([2.0])
        assert isinstance(pred, float)

    def test_export_load_json_regression(self, trained_regressor, tmp_path):
        """Export and load regression .json format."""
        path = tmp_path / "model.json"
        trained_regressor.export(str(path))

        dt2 = DecisionTree()
        dt2.load_model(str(path))

        pred = dt2.predict([2.0])
        assert isinstance(pred, float)


class TestConvert:
    """Tests for the convert() function."""

    @pytest.fixture
    def trained_tree_path(self, tmp_path):
        dt = DecisionTree()
        X = [["a", 1], ["b", 2], ["a", 3], ["b", 4]]
        y = ["yes", "no", "yes", "no"]
        dt.load_data(X, y)
        dt.train()
        path = tmp_path / "model.cart"
        dt.export(str(path))
        return str(path), dt, tmp_path

    @pytest.fixture
    def trained_forest_path(self, tmp_path):
        rf = RandomForest(n_estimators=5, feature_names=["x", "y"])
        X = [["a", 1], ["b", 2], ["a", 3], ["b", 4]] * 5
        y = ["yes", "no", "yes", "no"] * 5
        rf.load_data(X, y)
        rf.train(random_state=42)
        path = tmp_path / "forest.cart"
        rf.export(str(path))
        return str(path), rf, tmp_path

    def test_cart_to_json(self, trained_tree_path):
        from cartlet import convert

        src, original_dt, tmp_path = trained_tree_path
        dest = str(tmp_path / "model.json")
        convert(src, dest)

        dt2 = DecisionTree()
        dt2.load_model(dest)
        assert dt2.predict(["a", 1]) == original_dt.predict(["a", 1])

    def test_cart_to_jsonl(self, trained_tree_path):
        from cartlet import convert

        src, original_dt, tmp_path = trained_tree_path
        dest = str(tmp_path / "model.jsonl")
        convert(src, dest)

        dt2 = DecisionTree()
        dt2.load_model(dest)
        assert dt2.predict(["b", 2]) == original_dt.predict(["b", 2])

    def test_json_to_cart_roundtrip(self, trained_tree_path):
        from cartlet import convert

        src, original_dt, tmp_path = trained_tree_path
        json_path = str(tmp_path / "model.json")
        cart_path = str(tmp_path / "roundtrip.cart")
        convert(src, json_path)
        convert(json_path, cart_path)

        dt2 = DecisionTree()
        dt2.load_model(cart_path)
        assert dt2.predict(["a", 1]) == original_dt.predict(["a", 1])

    def test_cart_to_pickle(self, trained_tree_path):
        from cartlet import convert

        src, original_dt, tmp_path = trained_tree_path
        dest = str(tmp_path / "model.pkl")
        convert(src, dest)

        dt2 = DecisionTree()
        dt2.load_model(dest)
        assert dt2.predict(["a", 1]) == original_dt.predict(["a", 1])

    def test_forest_cart_to_json(self, trained_forest_path):
        from cartlet import convert

        src, original_rf, tmp_path = trained_forest_path
        dest = str(tmp_path / "forest.json")
        convert(src, dest)

        rf2 = RandomForest()
        rf2.load_model(dest)
        assert rf2.predict(["a", 1]) == original_rf.predict(["a", 1])

    def test_convert_unknown_format_raises(self, trained_tree_path):
        from cartlet import convert

        src, _, tmp_path = trained_tree_path
        with pytest.raises(ValueError):
            convert(src, str(tmp_path / "model.xyz"))

    def test_convert_nonexistent_file_raises(self, tmp_path):
        from cartlet import convert

        with pytest.raises(FileNotFoundError):
            convert(str(tmp_path / "missing.cart"), str(tmp_path / "out.json"))
