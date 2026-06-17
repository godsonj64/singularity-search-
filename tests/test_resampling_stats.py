from __future__ import annotations

from singularity_search.evaluation.runner import paired_bootstrap_delta


def test_resampling_comparison_runs() -> None:
    system_a = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    system_b = {"q1": ["d2", "d1"], "q2": ["d4", "d3"]}
    judgments = {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}
    out = paired_bootstrap_delta(system_a, system_b, judgments, samples=20, seed=1)
    assert "delta_mean" in out
    assert "ci95_low" in out
    assert "ci95_high" in out
