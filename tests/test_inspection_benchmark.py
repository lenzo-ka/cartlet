"""Smoke coverage for the inspection cost benchmark."""

from benchmarks.inspection_costs import CostCase, render_markdown, run_benchmarks


def test_inspection_cost_benchmark_smoke():
    case = CostCase("smoke", rows=16, features=2, trees=2, repeats=1, depth=2)
    results = run_benchmarks([case], timing_repeats=1)

    assert len(results) == 1
    result = results[0]
    assert result["held_out_prediction_units"] == 16 * 2 * (1 + 2)
    assert result["oob_prediction_units"] > 0
    assert result["nodes"] > 0
    assert result["held_out_seconds"] > 0
    assert result["oob_seconds"] > 0
    assert result["path_seconds"] > 0
    assert "| smoke | 16 | 2 | 2 | 1 |" in render_markdown(results)
