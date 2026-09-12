"""Direct models and operational settings share parameter contracts."""

from dataclasses import replace

import pytest

from cartlet import DecisionTree, RandomForest, TrainingSettings, train_model


@pytest.mark.parametrize("model_type", [DecisionTree, RandomForest])
@pytest.mark.parametrize(
    "settings",
    [
        {"max_depth": -1},
        {"max_depth": True},
        {"min_samples_leaf": 0},
        {"min_samples_leaf": 0.5},
        {"min_samples_split": 0},
        {"min_samples_split": True},
        {"criterion": "typo"},
        {"categorical_split": "typo"},
    ],
)
def test_constructor_and_workflow_reject_same_tree_parameters(model_type, settings):
    with pytest.raises(ValueError):
        model_type(**settings)
    with pytest.raises(ValueError):
        replace(TrainingSettings(), **settings).validate()


@pytest.mark.parametrize("value", [0, -1, 0.5, True, None])
def test_forest_count_is_positive_integer(value):
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForest(n_estimators=value)
    with pytest.raises(ValueError, match="n_estimators"):
        replace(TrainingSettings(model_type="forest"), n_estimators=value).validate()


@pytest.mark.parametrize(
    "options",
    [{"n_jobs": 0}, {"n_jobs": True}, {"random_state": True}, {"trainer": "typo"}],
)
def test_invalid_train_options_preserve_trained_forest(options):
    model = RandomForest(n_estimators=2)
    model.load_data([["a"], ["b"]], ["A", "B"])
    model.train(random_state=2)
    trees = model.trees
    with pytest.raises(ValueError):
        model.train(**options)
    assert model.trees is trees
    with pytest.raises(ValueError):
        replace(TrainingSettings(model_type="forest"), **options).validate()


def test_mutated_direct_parameter_rejected_before_retraining():
    tree = DecisionTree()
    tree.load_data([["a"], ["b"]], ["A", "B"])
    tree.train()
    old_model = tree.model
    tree.min_samples_leaf = 0
    with pytest.raises(ValueError):
        tree.train()
    assert tree.model is old_model


@pytest.mark.parametrize(
    "settings", [{"max_depth": 0}, {"min_samples_split": 1}, {"random_state": -1}]
)
def test_sklearn_bounds_are_shared_before_backend_is_invoked(settings):
    tree = DecisionTree(
        **{key: value for key, value in settings.items() if key != "random_state"}
    )
    tree.load_data([["a"], ["b"]], ["A", "B"])
    with pytest.raises(ValueError, match="sklearn"):
        tree.train(
            trainer="sklearn",
            **{key: value for key, value in settings.items() if key == "random_state"},
        )
    with pytest.raises(ValueError, match="sklearn"):
        replace(TrainingSettings(trainer="sklearn"), **settings).validate()


def test_native_leaf_depth_remains_supported():
    result = train_model(
        [["a"], ["b"]], ["A", "B"], settings=TrainingSettings(max_depth=0, test_split=0)
    )
    assert result.model.predict(["a"], return_dist=True) == {"A": 0.5, "B": 0.5}


def test_supported_pruning_default_and_explicit_zero_contract():
    X, y = [["a"], ["b"]] * 20, ["A", "B"] * 20
    tree = DecisionTree()
    tree.load_data(X, y)
    tree.train(prune=True, random_state=1)
    assert tree.training_summary["validation_samples"] == 2
    old_model = tree.model
    with pytest.raises(ValueError, match="positive validation_split"):
        tree.train(prune=True, validation_split=0)
    assert tree.model is old_model
    with pytest.raises(ValueError, match="positive validation_split"):
        train_model(X, y, settings=TrainingSettings(prune=True, validation_split=0))


def test_pruning_off_never_holds_out_validation():
    tree = DecisionTree()
    tree.load_data([["a"], ["b"]] * 10, ["A", "B"] * 10)
    tree.train(validation_split=0.9)
    assert tree.training_summary == {
        "training_samples": 20,
        "validation_samples": 0,
        "test_samples": 0,
    }
    result = train_model(
        [["a"], ["b"]] * 10,
        ["A", "B"] * 10,
        settings=TrainingSettings(validation_split=0.9, test_split=0.2),
    )
    assert result.validation_samples == 0 and result.training_samples == 16


def test_unsupported_pruning_zero_does_not_hold_out_rows():
    tree = DecisionTree(task="regression")
    tree.load_data([[1], [2], [3]], [1.0, 2.0, 3.0])
    tree.train(prune=True, validation_split=0)
    assert tree.training_summary["training_samples"] == 3
    result = train_model(
        [[1], [2], [3]],
        [1.0, 2.0, 3.0],
        settings=TrainingSettings(
            task="regression", prune=True, validation_split=0, test_split=0
        ),
    )
    assert result.validation_samples == 0 and result.training_samples == 3


def test_labeled_isolation_workflow_rejected_before_export(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("KEEP")
    with pytest.raises(ValueError, match="unlabeled"):
        train_model(
            [[0], [1]],
            ["A", "B"],
            settings=TrainingSettings(model_type="isolation"),
            output=output,
        )
    assert output.read_text() == "KEEP"


def test_source_fraction_uses_exact_integer_validation_count():
    X, y = [["a"], ["b"]] * 9 + [["a"]], ["A", "B"] * 9 + ["A"]
    result = train_model(
        X,
        y,
        settings=TrainingSettings(
            prune=True, test_split=0.2, validation_split=0.2, random_state=3
        ),
    )
    assert result.source_samples == 19
    assert (
        result.training_samples,
        result.validation_samples,
        result.test_samples,
    ) == (13, 3, 3)
