"""Scale inspection APIs; run with ``python -m benchmarks.inspection_costs``."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from cartlet import RandomForest, leaf_paths, permutation_importance
from cartlet.utils import count_nodes


@dataclass(frozen=True)
class CostCase:
    """One generated inspection workload."""

    name: str
    rows: int
    features: int
    trees: int
    repeats: int
    depth: int = 5


DEFAULT_CASES = (
    CostCase("rows-40", 40, 4, 6, 2),
    CostCase("baseline", 80, 4, 6, 2),
    CostCase("rows-160", 160, 4, 6, 2),
    CostCase("features-2", 80, 2, 6, 2),
    CostCase("features-8", 80, 8, 6, 2),
    CostCase("trees-3", 80, 4, 3, 2),
    CostCase("trees-12", 80, 4, 12, 2),
    CostCase("repeats-1", 80, 4, 6, 1),
    CostCase("repeats-4", 80, 4, 6, 4),
)


def _dataset(case: CostCase, seed: int) -> tuple[list[list[float]], list[str]]:
    rng = random.Random(seed)
    rows = [[rng.random() for _ in range(case.features)] for _ in range(case.rows)]
    labels = [str(int(row[0] + 0.25 * row[1] > 0.625)) for row in rows]
    return rows, labels


def _median_seconds(operation: Callable[[], Any], timing_repeats: int) -> float:
    operation()
    samples = []
    for _ in range(timing_repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def run_benchmarks(
    cases: Sequence[CostCase] = DEFAULT_CASES,
    *,
    timing_repeats: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return timings and dominant work-unit counts for each case."""
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be positive")
    results = []
    for case in cases:
        rows, labels = _dataset(case, seed)
        features = [
            {"name": f"x{index}", "dtype": "float", "type": "num"}
            for index in range(case.features)
        ]
        forest = RandomForest(
            n_estimators=case.trees,
            max_features=None,
            bootstrap=True,
            features=features,
            max_depth=case.depth,
        )
        forest.load_data(rows, labels)
        forest.train(random_state=seed)
        node_count = sum(count_nodes(tree.model) for tree in forest.trees)
        assert forest._inbag_indices is not None
        oob_rows = sum(
            case.rows - len(set(inbag)) for inbag in forest._inbag_indices.values()
        )
        held_out_units = case.rows * case.trees * (1 + case.features * case.repeats)
        oob_units = oob_rows * (1 + case.features * case.repeats)
        held_out = partial(
            permutation_importance,
            forest,
            rows,
            labels,
            n_repeats=case.repeats,
            random_state=seed,
        )
        oob = partial(
            forest.oob_permutation_importance,
            n_repeats=case.repeats,
            random_state=seed,
        )
        paths = partial(leaf_paths, forest)
        results.append(
            {
                **asdict(case),
                "nodes": node_count,
                "held_out_prediction_units": held_out_units,
                "oob_prediction_units": oob_units,
                "held_out_seconds": _median_seconds(held_out, timing_repeats),
                "oob_seconds": _median_seconds(oob, timing_repeats),
                "path_seconds": _median_seconds(paths, timing_repeats),
            }
        )
    return results


def render_markdown(results: Sequence[dict[str, Any]]) -> str:
    """Render a compact cost table."""
    lines = [
        "| case | rows | features | trees | repeats | nodes | held-out ms | OOB ms | paths ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {name} | {rows} | {features} | {trees} | {repeats} | "
            "{nodes} | {held_out:.3f} | {oob:.3f} | {paths:.3f} |".format(
                **result,
                held_out=result["held_out_seconds"] * 1000,
                oob=result["oob_seconds"] * 1000,
                paths=result["path_seconds"] * 1000,
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    results = run_benchmarks(
        timing_repeats=args.timing_repeats,
        seed=args.seed,
    )
    if args.format == "json":
        print(json.dumps(results, indent=2, allow_nan=False))
    else:
        print(render_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
