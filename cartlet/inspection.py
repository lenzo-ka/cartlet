"""Structural path export and decisive-leaf inspection."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Any, cast

from .io.cart_format import (
    CATEGORY_SET,
    INDEX_MASK,
    LEAF_CLASS,
    LEAF_CLASS_DIST,
    LEAF_FLAG,
    MISSING_LEFT,
    MISSING_MASK,
    MISSING_RIGHT,
    OP_EQ,
    OP_LE,
    OP_LT,
    OP_SWITCH,
)
from .runner import Predictor
from .runner import predict_path as runner_predict_path
from .types import TASK_CLASSIFICATION, TASK_REGRESSION, ModelData, normalize_bool
from .utils import (
    is_decision_node,
    is_leaf,
    is_switch_node,
    split_feature_and_missing,
)


def _model_data(model: Any) -> ModelData | None:
    if isinstance(model, Predictor):
        return model.model
    if isinstance(model, dict) and {
        "meta",
        "decisions",
        "leaves",
        "tree_offsets",
    }.issubset(model):
        return cast(ModelData, model)
    return None


def _nested_model(model: Any) -> tuple[list[Any], str, list[dict[str, str]], bool]:
    from .forest import RandomForest
    from .tree import DecisionTree

    if isinstance(model, DecisionTree):
        if model.model is None:
            raise ValueError("Model not trained. Call train() first.")
        trees = [model.model]
    elif isinstance(model, RandomForest):
        if not model.trees:
            raise ValueError("Forest not trained. Call train() first.")
        trees = [tree.model for tree in model.trees]
    else:
        raise TypeError(
            "model must be a DecisionTree, RandomForest, Predictor, or ModelData"
        )
    features = [
        {"name": spec.name, "dtype": spec.dtype, "type": str(spec.type)}
        for spec in model.feature_specs
    ]
    statistics_available = bool(getattr(model, "_leaf_statistics_available", True))
    return trees, model._effective_task(), features, statistics_available


def _feature_schema(model: Any) -> tuple[str, list[dict[str, str]], bool]:
    data = _model_data(model)
    if data is not None:
        features = [
            {
                "name": str(feature["name"]),
                "dtype": str(feature.get("dtype", "str")),
                "type": str(feature.get("type", "cat")),
            }
            for feature in data["meta"].get("features", [])
        ]
        return data["meta"]["task"], features, bool(data.get("is_xgboost"))
    _, task, features, _ = _nested_model(model)
    return task, features, False


def _missing_direction(flags: int) -> str | None:
    direction = flags & MISSING_MASK
    if direction == MISSING_LEFT:
        return "left"
    if direction == MISSING_RIGHT:
        return "right"
    return None


def _canonical_category(value: Any, feature: dict[str, str]) -> str:
    if feature["dtype"] == "bool":
        value = normalize_bool(value)
    return str(value)


def _condition(
    *,
    node: int,
    feature_index: int,
    feature: dict[str, str],
    op: str,
    value: Any,
    branch: str,
    missing_direction: str | None,
) -> dict[str, Any]:
    if op in ("<", "<="):
        canonical: Any = float(value)
    elif op in ("in", "not in"):
        canonical = sorted({_canonical_category(item, feature) for item in value})
    else:
        canonical = _canonical_category(value, feature)
    return {
        "node": node,
        "feature": feature_index,
        "name": feature["name"],
        "op": op,
        "value": canonical,
        "branch": branch,
        "missing_direction": missing_direction,
    }


def _leaf_record(
    leaf: Any,
    *,
    task: str,
    tree: int,
    leaf_id: int,
    path: list[dict[str, Any]],
    statistics_available: bool,
    xgboost: bool = False,
) -> dict[str, Any]:
    prediction: Any
    predicted_class: str | None = None
    distribution: dict[str, float] | None = None
    support: float | None = None
    purity: float | None = None
    if xgboost:
        prediction = float(leaf)
    elif task == TASK_REGRESSION:
        prediction = float(leaf[0]) if isinstance(leaf, list) else float(leaf)
        if statistics_available and isinstance(leaf, list) and len(leaf) == 3:
            support = float(leaf[2])
    elif isinstance(leaf, dict):
        distribution = {str(label): float(prob) for label, prob in leaf.items()}
        predicted_class = max(distribution, key=distribution.__getitem__)
        prediction = predicted_class
        purity = distribution[predicted_class]
    else:
        predicted_class = str(leaf)
        prediction = predicted_class
    return {
        "tree": tree,
        "leaf": leaf_id,
        "path": path,
        "prediction": prediction,
        "predicted_class": predicted_class,
        "class_distribution": distribution,
        "class_counts": None,
        "support": support,
        "purity": purity,
    }


def _flat_leaf_value(data: ModelData, leaf_id: int) -> Any:
    leaf_type, value_index = data["leaves"][leaf_id]
    if leaf_type == LEAF_CLASS:
        return data["strings"][value_index]
    if leaf_type == LEAF_CLASS_DIST:
        return {
            data["strings"][class_index]: probability
            for class_index, probability in data["distributions"][value_index]
        }
    return data["floats"][value_index]


def _flat_paths(data: ModelData) -> list[dict[str, Any]]:
    task = data["meta"]["task"]
    features = [
        {
            "name": str(feature["name"]),
            "dtype": str(feature.get("dtype", "str")),
            "type": str(feature.get("type", "cat")),
        }
        for feature in data["meta"]["features"]
    ]
    records: list[dict[str, Any]] = []
    for tree_index, root in enumerate(data["tree_offsets"]):
        stack: list[tuple[int, list[dict[str, Any]], frozenset[int]]] = [
            (root, [], frozenset())
        ]
        while stack:
            index, path, ancestors = stack.pop()
            if index & LEAF_FLAG:
                leaf_id = index & INDEX_MASK
                records.append(
                    _leaf_record(
                        _flat_leaf_value(data, leaf_id),
                        task=task,
                        tree=tree_index,
                        leaf_id=leaf_id,
                        path=path,
                        statistics_available=False,
                        xgboost=bool(data.get("is_xgboost")),
                    )
                )
                continue
            if index in ancestors:
                raise ValueError("model structure contains a decision cycle")
            feat, op_code, flags, value_index, left, right = data["decisions"][index]
            feature = features[feat]
            direction = _missing_direction(flags)
            next_ancestors = ancestors | {index}
            if op_code == OP_SWITCH:
                table = data["case_tables"][value_index]
                all_values: list[str] = []
                grouped: dict[int, list[str]] = {}
                seen: set[str] = set()
                for cat_value_index, child in table["cases"]:
                    case_value = data["strings"][data["cat_vals"][cat_value_index]]
                    if case_value in seen:
                        continue
                    seen.add(case_value)
                    all_values.append(case_value)
                    grouped.setdefault(child, []).append(case_value)
                # Push cases in reverse so traversal remains default, then
                # first-encountered child groups in writer order.
                for child, values in reversed(list(grouped.items())):
                    condition = _condition(
                        node=index,
                        feature_index=feat,
                        feature=feature,
                        op="in",
                        value=values,
                        branch="case",
                        missing_direction=direction,
                    )
                    stack.append((child, [*path, condition], next_ancestors))
                default_condition = _condition(
                    node=index,
                    feature_index=feat,
                    feature=feature,
                    op="not in",
                    value=all_values,
                    branch="default",
                    missing_direction=direction,
                )
                stack.append(
                    (table["default"], [*path, default_condition], next_ancestors)
                )
                continue
            if op_code in (OP_LE, OP_LT):
                op = "<" if op_code == OP_LT else "<="
                decision_value: Any = data["floats"][value_index]
            elif op_code == OP_EQ:
                if flags & CATEGORY_SET:
                    op = "in"
                    decision_value = data["category_sets"][value_index]
                else:
                    op = "="
                    decision_value = data["strings"][data["cat_vals"][value_index]]
            else:
                raise ValueError(f"unknown decision operation: {op_code}")
            right_condition = _condition(
                node=index,
                feature_index=feat,
                feature=feature,
                op=op,
                value=decision_value,
                branch="right",
                missing_direction=direction,
            )
            left_condition = _condition(
                node=index,
                feature_index=feat,
                feature=feature,
                op=op,
                value=decision_value,
                branch="left",
                missing_direction=direction,
            )
            stack.append((right, [*path, right_condition], next_ancestors))
            stack.append((left, [*path, left_condition], next_ancestors))
    return records


def _nested_paths(model: Any) -> list[dict[str, Any]]:
    trees, task, features, statistics_available = _nested_model(model)
    indices = model._tree_indices()
    records: list[dict[str, Any]] = []
    for tree_index, root in enumerate(trees):
        decision_ids, leaf_ids = indices[tree_index]
        stack: list[tuple[Any, tuple[Any, ...], list[dict[str, Any]]]] = [
            (root, (), [])
        ]
        while stack:
            node, address, path = stack.pop()
            if is_leaf(node):
                records.append(
                    _leaf_record(
                        node,
                        task=task,
                        tree=tree_index,
                        leaf_id=leaf_ids[address],
                        path=path,
                        statistics_available=statistics_available,
                    )
                )
                continue
            if is_switch_node(node):
                feature_ref, _, cases, default = node
                feature_ref, direction = split_feature_and_missing(feature_ref)
                feat = (
                    model.name_to_col[feature_ref]
                    if isinstance(feature_ref, str)
                    else int(feature_ref)
                )
                feature = features[feat]
                items = list(cases.items()) if isinstance(cases, dict) else list(cases)
                seen: set[str] = set()
                all_values: list[str] = []
                child_groups: dict[int, tuple[int | str, Any, list[str]]] = {
                    id(default): ("default", default, [])
                }
                for case_index, (value, child) in enumerate(items):
                    canonical = _canonical_category(value, feature)
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    all_values.append(canonical)
                    identity = id(child)
                    if identity in child_groups:
                        child_groups[identity][2].append(canonical)
                    else:
                        child_groups[identity] = (case_index, child, [canonical])
                case_groups = [group for group in child_groups.values() if group[2]]
                for selector, child, values in reversed(case_groups):
                    condition = _condition(
                        node=decision_ids[address],
                        feature_index=feat,
                        feature=feature,
                        op="in",
                        value=values,
                        branch="case",
                        missing_direction=direction,
                    )
                    child_address = (
                        address + ("default",)
                        if selector == "default"
                        else address + (("case", selector),)
                    )
                    stack.append(
                        (
                            child,
                            child_address,
                            [*path, condition],
                        )
                    )
                default_condition = _condition(
                    node=decision_ids[address],
                    feature_index=feat,
                    feature=feature,
                    op="not in",
                    value=all_values,
                    branch="default",
                    missing_direction=direction,
                )
                stack.append(
                    (default, address + ("default",), [*path, default_condition])
                )
                continue
            if not is_decision_node(node):
                raise ValueError(f"unknown node type: {type(node)}")
            feature_ref, op, value, left, right = node
            feature_ref, direction = split_feature_and_missing(feature_ref)
            feat = (
                model.name_to_col[feature_ref]
                if isinstance(feature_ref, str)
                else int(feature_ref)
            )
            feature = features[feat]
            right_condition = _condition(
                node=decision_ids[address],
                feature_index=feat,
                feature=feature,
                op=op,
                value=value,
                branch="right",
                missing_direction=direction,
            )
            left_condition = _condition(
                node=decision_ids[address],
                feature_index=feat,
                feature=feature,
                op=op,
                value=value,
                branch="left",
                missing_direction=direction,
            )
            stack.append((right, address + (1,), [*path, right_condition]))
            stack.append((left, address + (0,), [*path, left_condition]))
    return records


def _is_missing(value: Any, numeric: bool) -> tuple[bool, float | None]:
    if value is None:
        return True, None
    if numeric:
        try:
            number = float(value)
        except OverflowError:
            raise
        except (TypeError, ValueError):
            return False, None
        return math.isnan(number), number
    if isinstance(value, str):
        return False, None
    try:
        unequal = value != value
    except Exception:
        return False, None
    if isinstance(unequal, bool):
        return unequal, None
    unequal_type = type(unequal)
    if unequal_type.__module__.split(".", 1)[
        0
    ] == "numpy" and unequal_type.__name__ in (
        "bool",
        "bool_",
    ):
        try:
            return bool(unequal), None
        except Exception:
            pass
    return False, None


def _condition_matches(
    condition: dict[str, Any],
    row: Sequence[Any],
    features: list[dict[str, str]],
    *,
    float32_numeric: bool,
    missing: str,
) -> bool:
    index = condition["feature"]
    value = row[index] if index < len(row) else None
    op = condition["op"]
    numeric = op in ("<", "<=")
    absent, numeric_value = _is_missing(value, numeric)
    direction = condition["missing_direction"]
    branch = condition["branch"]
    if absent:
        selected = direction
        if selected is None:
            if missing == "error":
                return False
            selected = "default" if branch in ("case", "default") else "right"
        if branch in ("case", "default"):
            return (
                selected == "left"
                and branch == "case"
                or selected != "left"
                and branch == "default"
            )
        return selected == branch
    if features[index]["dtype"] == "bool":
        value = normalize_bool(value)
        if numeric:
            numeric_value = float(value)
    if numeric and numeric_value is not None and float32_numeric:
        try:
            numeric_value = struct.unpack("<f", struct.pack("<f", numeric_value))[0]
        except OverflowError as exc:
            raise ValueError(f"XGBoost input {value!r} exceeds float32 range") from exc
    if op == "<":
        selected_left = numeric_value is not None and numeric_value < condition["value"]
    elif op == "<=":
        selected_left = (
            numeric_value is not None and numeric_value <= condition["value"]
        )
    elif op == "=":
        selected_left = str(value) == condition["value"]
    elif op == "in":
        selected_left = str(value) in condition["value"]
        if branch == "case":
            return selected_left
    elif op == "not in":
        return str(value) not in condition["value"]
    else:
        raise ValueError(f"unknown exported condition operation: {op!r}")
    return selected_left if branch == "left" else not selected_left


def _predict_path(model: Any, row: list[Any], missing: str) -> dict[str, Any]:
    data = _model_data(model)
    if data is not None:
        return runner_predict_path(data, row, missing=missing)
    return model.predict_path(row, missing=missing)


def _add_data_statistics(
    export: dict[str, Any],
    model: Any,
    X: Sequence[Sequence[Any]],
    y: Sequence[Any] | None,
    missing: str,
) -> None:
    if not isinstance(X, Sequence) or isinstance(X, (str, bytes)):
        raise ValueError("X must be a sequence of feature rows")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in X):
        raise ValueError("data rows must be sequences of feature values")
    if y is not None and len(y) != len(X):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    targets = (
        [str(value) for value in y]
        if y is not None and export["task"] == TASK_CLASSIFICATION
        else None
    )
    records = [leaf for tree in export["trees"] for leaf in tree["leaves"]]
    candidates: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        record["data_support"] = 0
        if targets is not None:
            record["data_class_counts"] = {}
            record["data_purity"] = None
        candidates.setdefault((record["tree"], record["leaf"]), []).append(record)

    for row_index, source_row in enumerate(X):
        row = list(source_row)
        routed = _predict_path(model, row, missing)
        for tree_path in routed["trees"]:
            matches = candidates.get((tree_path["tree"], tree_path["leaf"]), [])
            if len(matches) > 1:
                matches = [
                    record
                    for record in matches
                    if all(
                        _condition_matches(
                            condition,
                            row,
                            export["features"],
                            float32_numeric=export["numeric_input_float32"],
                            missing=missing,
                        )
                        for condition in record["path"]
                    )
                ]
            if len(matches) != 1:
                raise RuntimeError(
                    "predict_path did not identify exactly one exported path for "
                    f"tree {tree_path['tree']} leaf {tree_path['leaf']}"
                )
            record = matches[0]
            record["data_support"] += 1
            if targets is not None:
                label = targets[row_index]
                counts = record["data_class_counts"]
                counts[label] = counts.get(label, 0) + 1

    if targets is not None:
        for record in records:
            support = record["data_support"]
            record["data_class_counts"] = dict(
                sorted(record["data_class_counts"].items())
            )
            if support and record["predicted_class"] is not None:
                record["data_purity"] = (
                    record["data_class_counts"].get(record["predicted_class"], 0)
                    / support
                )


def leaf_paths(
    model: Any,
    X: Sequence[Sequence[Any]] | None = None,
    y: Sequence[Any] | None = None,
    *,
    missing: str = "error",
) -> dict[str, Any]:
    """Export every root-to-leaf path and optional empirical path statistics.

    ``support`` on an in-memory or JSON regression leaf is its effective
    training weight. It is unavailable after `.cart` reload. Empirical fields
    are separate: ``data_support`` is the number of supplied rows routed to the
    path, while classification ``data_class_counts`` and ``data_purity`` use
    optional held-out labels.
    """
    if missing not in ("error", "right"):
        raise ValueError("missing must be 'error' or 'right'")
    if y is not None and X is None:
        raise ValueError("y requires X")
    task, features, float32_numeric = _feature_schema(model)
    data = _model_data(model)
    records = _flat_paths(data) if data is not None else _nested_paths(model)
    trees = []
    tree_count = data["n_trees"] if data is not None else len(_nested_model(model)[0])
    for tree_index in range(tree_count):
        trees.append(
            {
                "tree": tree_index,
                "leaves": [
                    record for record in records if record["tree"] == tree_index
                ],
            }
        )
    result = {
        "task": task,
        "features": features,
        "numeric_input_float32": float32_numeric,
        "trees": trees,
    }
    if X is not None:
        _add_data_statistics(result, model, X, y, missing)
    return result


def _threshold(value: Any, name: str, *, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or maximum is not None
        and value > maximum
    ):
        suffix = f" and at most {maximum:g}" if maximum is not None else ""
        raise ValueError(f"{name} must be finite, nonnegative{suffix}")
    return float(value)


def decisive_leaves(
    model: Any,
    predicted_class: Any,
    X: Sequence[Sequence[Any]] | None = None,
    y: Sequence[Any] | None = None,
    *,
    min_support: float | None = None,
    min_purity: float | None = None,
    missing: str = "error",
) -> list[dict[str, Any]]:
    """Return classification paths meeting inclusive support/purity thresholds.

    With ``X``, thresholds use ``data_support`` and ``data_purity``. Without
    data, they use model-stored ``support`` and ``purity``. An unavailable value
    never satisfies a requested threshold.
    """
    data = _model_data(model)
    if data is not None and data.get("is_xgboost"):
        raise ValueError("decisive_leaves does not support XGBoost-derived models")
    support_threshold = _threshold(min_support, "min_support")
    purity_threshold = _threshold(min_purity, "min_purity", maximum=1.0)
    export = leaf_paths(model, X, y, missing=missing)
    if export["task"] != TASK_CLASSIFICATION:
        raise ValueError("decisive_leaves requires a classification model")
    expected = str(predicted_class)
    support_field = "data_support" if X is not None else "support"
    purity_field = "data_purity" if X is not None else "purity"
    selected = []
    for tree in export["trees"]:
        for record in tree["leaves"]:
            if record["predicted_class"] != expected:
                continue
            support = record.get(support_field)
            purity = record.get(purity_field)
            if support_threshold is not None and (
                support is None or support < support_threshold
            ):
                continue
            if purity_threshold is not None and (
                purity is None or purity < purity_threshold
            ):
                continue
            selected.append(record)
    return selected


__all__ = ["decisive_leaves", "leaf_paths"]
