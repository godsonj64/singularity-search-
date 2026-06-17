from __future__ import annotations

import math

from singularity_search.evaluation.metrics import evaluate_run, ndcg_at, recall_at, mrr_at, average_precision_at


def test_metrics_perfect_ranking() -> None:
    qrel = {"d1": 1.0, "d2": 1.0, "d3": 0.0}
    ranking = ["d1", "d2", "d3"]
    assert math.isclose(ndcg_at(ranking, qrel, 2), 1.0)
    assert math.isclose(recall_at(ranking, qrel, 2), 1.0)
    assert math.isclose(mrr_at(ranking, qrel, 10), 1.0)
    assert math.isclose(average_precision_at(ranking, qrel, 10), 1.0)


def test_evaluate_run_overlap() -> None:
    run = {"q1": ["d2", "d1"], "q2": ["d3"]}
    qrels = {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}
    metrics = evaluate_run(run, qrels)
    assert metrics.queries_evaluated == 2
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0
    assert 0.0 <= metrics.recall_at_100 <= 1.0
