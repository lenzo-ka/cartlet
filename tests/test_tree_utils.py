"""Tests for tree utility functions."""

from cartlet import count_leaves, count_nodes, max_depth, tree_stats


class TestCountNodes:
    """Test count_nodes function."""

    def test_string_leaf(self):
        assert count_nodes("leaf") == 1

    def test_dict_leaf(self):
        assert count_nodes({"a": 0.5, "b": 0.5}) == 1

    def test_regression_leaf(self):
        assert count_nodes([10.5, 2.3, 100]) == 1  # [mean, var, n]

    def test_single_decision(self):
        # [feature, op, value, left, right]
        tree = ["feat", "=", "val", "left", "right"]
        assert count_nodes(tree) == 3

    def test_nested_tree(self):
        tree = ["f1", "=", "v1", ["f2", "=", "v2", "a", "b"], "c"]
        assert count_nodes(tree) == 5


class TestCountLeaves:
    """Test count_leaves function."""

    def test_string_leaf(self):
        assert count_leaves("leaf") == 1

    def test_dict_leaf(self):
        assert count_leaves({"a": 0.5}) == 1

    def test_regression_leaf(self):
        assert count_leaves([10.5, 2.3, 100]) == 1

    def test_single_decision(self):
        tree = ["feat", "=", "val", "left", "right"]
        assert count_leaves(tree) == 2

    def test_nested_tree(self):
        tree = ["f1", "=", "v1", ["f2", "=", "v2", "a", "b"], "c"]
        assert count_leaves(tree) == 3


class TestMaxDepth:
    """Test max_depth function."""

    def test_leaf_depth_zero(self):
        assert max_depth("leaf") == 0

    def test_single_decision_depth_one(self):
        tree = ["feat", "=", "val", "left", "right"]
        assert max_depth(tree) == 1

    def test_nested_tree(self):
        tree = ["f1", "=", "v1", ["f2", "=", "v2", "a", "b"], "c"]
        assert max_depth(tree) == 2

    def test_unbalanced_tree(self):
        tree = ["f1", "=", "v1", ["f2", "=", "v2", ["f3", "<", 5, "a", "b"], "c"], "d"]
        assert max_depth(tree) == 3


class TestTreeStats:
    """Test tree_stats function."""

    def test_leaf_stats(self):
        stats = tree_stats("leaf")
        assert stats["total_nodes"] == 1
        assert stats["leaf_nodes"] == 1
        assert stats["decision_nodes"] == 0
        assert stats["max_depth"] == 0

    def test_tree_stats(self):
        tree = ["f1", "=", "v1", ["f2", "=", "v2", "a", "b"], "c"]
        stats = tree_stats(tree)
        assert stats["total_nodes"] == 5
        assert stats["leaf_nodes"] == 3
        assert stats["decision_nodes"] == 2
        assert stats["max_depth"] == 2
